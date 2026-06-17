import argparse
import csv
from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate
from src.eval.partial_proxy_metrics import (
    partial_delivered_answer_hit,
    partial_field_keep_ratio,
)
from src.methods.neural_set_dpp_planner import (
    load_setdpp_model,
    predict_set,
    setdpp_select,
)
from src.methods.qtc_rule import tx_field_keep_ratio
from src.semantic.compressed_sg_codec import (
    decode_compressed_sg_triplets,
    encode_compressed_sg_triplets,
)
from src.utils.config import load_yaml

from scripts.run_ranking_sweep import transmit_bits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="nsp_v4_setdpp")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--mask-threshold", type=float, default=0.30)
    parser.add_argument("--diversity-beta", type=float, default=0.30)
    parser.add_argument("--model", default="results/nsp/nsp_v4_setdpp.pt")
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

    start = args.start_index
    end = start + args.num_samples

    samples = all_samples[start:end]
    sg_rows = all_sg_rows[start:end]
    sample_by_qid = {s["question_id"]: s for s in samples}

    model = load_setdpp_model(args.model)

    ber_list = []
    answer_list = []
    bits_list = []
    tx_keep_ratio_list = []
    rx_keep_ratio_list = []
    candidate_count_list = []

    for sample_idx, row in enumerate(sg_rows):
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]
        triplets = list(row.get("triplets", []))[: args.max_candidates]

        if not triplets:
            continue

        quality, mask_probs, diversity = predict_set(
            model=model,
            triplets=triplets,
            question=sample.get("question", ""),
            keywords=sample.get("keywords", []),
            question_type=sample.get("question_type", ""),
            max_candidates=args.max_candidates,
        )

        selected_indices = setdpp_select(
            quality=quality,
            diversity=diversity,
            n_top=args.n_top,
            beta=args.diversity_beta,
        )

        selected = []
        keep_masks = []

        for idx in selected_indices:
            t = triplets[idx]
            probs = mask_probs[idx].tolist()

            keep = {
                "subject": bool(probs[0] >= args.mask_threshold),
                "relation": bool(probs[1] >= args.mask_threshold),
                "object": bool(probs[2] >= args.mask_threshold),
            }

            if not any(keep.values()):
                keep["object"] = True

            selected.append(t)
            keep_masks.append(keep)

        if not selected:
            continue

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
        candidate_count_list.append(len(triplets))
        answer_list.append(
            partial_delivered_answer_hit(
                selected_triplets=selected,
                rx_partial_packets=rx_packets,
                answer=sample.get("answer", ""),
            )
        )

    if not answer_list:
        raise RuntimeError("No NSP-v4 SetDPP rows were produced.")

    avg_bits = sum(bits_list) / len(bits_list)
    delivered_answer = sum(answer_list) / len(answer_list)

    row = {
        "method": args.method_name,
        "semantic_type": "sg",
        "channel": args.channel,
        "snr_db": args.snr_db,
        "n_top": args.n_top,
        "num_samples": len(answer_list),
        "avg_candidates": sum(candidate_count_list) / len(candidate_count_list),
        "avg_source_bits": avg_bits,
        "ber": sum(ber_list) / len(ber_list),
        "delivered_answer_hit_rate": delivered_answer,
        "tx_field_keep_ratio": sum(tx_keep_ratio_list) / len(tx_keep_ratio_list),
        "rx_field_keep_ratio": sum(rx_keep_ratio_list) / len(rx_keep_ratio_list),
        "answer_per_kbit": delivered_answer / max(1e-12, avg_bits / 1000.0),
        "mask_threshold": args.mask_threshold,
        "diversity_beta": args.diversity_beta,
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
