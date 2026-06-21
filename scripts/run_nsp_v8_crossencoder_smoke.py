import argparse
import csv
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate
from src.eval.partial_proxy_metrics import (
    partial_delivered_answer_hit,
    partial_field_keep_ratio,
)
from src.methods.neural_semantic_tokenizer_v2 import (
    load_nst_v2_model,
    predict_keep_mask_v2,
)
from src.methods.nsp_v5_semantic_features import triplet_text
from src.methods.qtc_rule import tx_field_keep_ratio
from src.semantic.compressed_sg_codec import (
    decode_compressed_sg_triplets,
    encode_compressed_sg_triplets,
)
from src.utils.config import load_yaml

from scripts.run_ranking_sweep import transmit_bits


FIELDS = ["subject", "relation", "object"]


def budgeted_masks(prob_lists, target_bits, n_top):
    field_budget = int((target_bits - 3 * n_top) // 16)
    field_budget = max(n_top, field_budget)

    keep = [
        {"subject": False, "relation": False, "object": False}
        for _ in prob_lists
    ]

    used = 0

    for i, probs in enumerate(prob_lists):
        best = max(range(3), key=lambda j: float(probs[j]))
        keep[i][FIELDS[best]] = True
        used += 1

    rest = []
    for i, probs in enumerate(prob_lists):
        for j, field in enumerate(FIELDS):
            if not keep[i][field]:
                rest.append((float(probs[j]), i, field))

    rest.sort(reverse=True, key=lambda x: x[0])

    for _, i, field in rest:
        if used >= field_budget:
            break
        keep[i][field] = True
        used += 1

    return keep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="nsp_v8_crossencoder")
    parser.add_argument("--start-index", type=int, default=8000)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--crossencoder", required=True)
    parser.add_argument("--nst-model", required=True)
    parser.add_argument("--target-bits", type=int, default=299)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_yaml("configs/experiment.yaml")
    seed = int(cfg["project"]["seed"])
    perfect_csi = bool(cfg["channel"]["perfect_csi"])

    ds = GQACommSubset(Path(cfg["data"]["root"]))
    all_samples = ds.load_samples()
    all_sg_rows = ds.load_sg_triplets()

    samples = all_samples[args.start_index: args.start_index + args.num_samples]
    sg_rows = all_sg_rows[args.start_index: args.start_index + args.num_samples]
    sample_by_qid = {s["question_id"]: s for s in samples}

    ranker = CrossEncoder(args.crossencoder)
    nst = load_nst_v2_model(args.nst_model)

    ber_list = []
    answer_list = []
    bits_list = []
    tx_keep_ratio_list = []
    rx_keep_ratio_list = []

    for sample_idx, row in enumerate(sg_rows):
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]
        question = sample.get("question", "")
        candidates = list(row.get("triplets", []))[: args.max_candidates]

        if not candidates:
            continue

        pairs = [[question, triplet_text(t)] for t in candidates]
        scores = ranker.predict(pairs, show_progress_bar=False)

        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        selected_indices = order[: args.n_top]
        selected = [candidates[i] for i in selected_indices]

        prob_lists = []

        for rank_index, t in enumerate(selected):
            _, probs = predict_keep_mask_v2(
                model=nst,
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                rank_index=rank_index,
                n_top=args.n_top,
                threshold=0.5,
            )
            prob_lists.append(probs)

        keep_masks = budgeted_masks(
            prob_lists=prob_lists,
            target_bits=args.target_bits,
            n_top=len(selected),
        )

        tx_bits = encode_compressed_sg_triplets(selected, keep_masks)

        rx_bits = transmit_bits(
            bits=tx_bits,
            channel_type=args.channel,
            snr_db=args.snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
        )

        rx_packets = decode_compressed_sg_triplets(
            rx_bits,
            num_triplets=len(selected),
        )

        ber_list.append(bit_error_rate(tx_bits, rx_bits))
        bits_list.append(len(tx_bits))
        tx_keep_ratio_list.append(tx_field_keep_ratio(keep_masks))
        rx_keep_ratio_list.append(partial_field_keep_ratio(rx_packets))
        answer_list.append(
            partial_delivered_answer_hit(
                selected_triplets=selected,
                rx_partial_packets=rx_packets,
                answer=sample.get("answer", ""),
            )
        )

    if not answer_list:
        raise RuntimeError("No V8 rows were produced.")

    avg_bits = sum(bits_list) / len(bits_list)
    delivered_answer = sum(answer_list) / len(answer_list)

    row = {
        "method": args.method_name,
        "crossencoder": args.crossencoder,
        "nst_model": args.nst_model,
        "semantic_type": "sg",
        "channel": args.channel,
        "snr_db": args.snr_db,
        "n_top": args.n_top,
        "num_samples": len(answer_list),
        "avg_source_bits": avg_bits,
        "ber": sum(ber_list) / len(ber_list),
        "delivered_answer_hit_rate": delivered_answer,
        "tx_field_keep_ratio": sum(tx_keep_ratio_list) / len(tx_keep_ratio_list),
        "rx_field_keep_ratio": sum(rx_keep_ratio_list) / len(rx_keep_ratio_list),
        "answer_per_kbit": delivered_answer / max(1e-12, avg_bits / 1000.0),
        "target_bits": args.target_bits,
        "start_index": args.start_index,
    }

    print(row)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
