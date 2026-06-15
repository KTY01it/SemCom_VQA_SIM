from pathlib import Path
import argparse
import csv

from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate
from src.eval.partial_proxy_metrics import (
    partial_delivered_answer_hit,
    partial_field_keep_ratio,
)
from src.methods.dbss import dbss_select_triplets
from src.methods.neural_semantic_tokenizer import load_nst_model, predict_keep_mask
from src.semantic.compressed_sg_codec import (
    decode_compressed_sg_triplets,
    encode_compressed_sg_triplets,
)
from src.utils.config import load_yaml

from scripts.run_ranking_sweep import transmit_bits

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", default="results/nst/nst_model.pt")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser.parse_args()

def tx_field_keep_ratio(keep_masks):
    total = 3 * len(keep_masks)
    kept = 0
    for m in keep_masks:
        kept += int(bool(m["subject"]))
        kept += int(bool(m["relation"]))
        kept += int(bool(m["object"]))
    return kept / max(1, total)

def main():
    args = parse_args()
    cfg = load_yaml("configs/experiment.yaml")

    seed = int(cfg["project"]["seed"])
    perfect_csi = bool(cfg["channel"]["perfect_csi"])

    ds = GQACommSubset(Path(cfg["data"]["root"]))

    num_samples = args.num_samples
    n_top = args.n_top
    snr_db = args.snr_db
    channel_type = args.channel
    model = load_nst_model(args.model)

    all_samples = ds.load_samples()
    all_sg_rows = ds.load_sg_triplets()

    start = args.start_index
    end = start + args.num_samples

    samples = all_samples[start:end]
    sg_rows = all_sg_rows[start:end]
    sample_by_qid = {s["question_id"]: s for s in samples}

    model = load_nst_model(args.model)

    ber_list = []
    answer_list = []
    bits_list = []
    keep_ratio_list = []

    for sample_idx, row in enumerate(sg_rows):
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]

        ranked = dbss_select_triplets(
            triplets=row.get("triplets", []),
            question=sample.get("question", ""),
            keywords=sample.get("keywords", []),
            n_top=n_top,
            snr_db=snr_db,
            channel_type=channel_type,
        )

        selected = ranked[:n_top]
        if not selected:
            continue

        keep_masks = []
        for t in selected:
            keep, _ = predict_keep_mask(model, t, threshold=args.threshold)
            keep_masks.append(keep)

        tx_keep_ratio_list = []
        tx_keep_ratio_list.append(tx_field_keep_ratio(keep_masks))
        
        tx_bits = encode_compressed_sg_triplets(selected, keep_masks)

        rx_bits = transmit_bits(
            bits=tx_bits,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
        )

        rx_packets = decode_compressed_sg_triplets(
            rx_bits,
            num_triplets=len(selected),
        )

        ber_list.append(bit_error_rate(tx_bits, rx_bits))
        bits_list.append(len(tx_bits))
        keep_ratio_list.append(partial_field_keep_ratio(rx_packets))
        answer_list.append(
            partial_delivered_answer_hit(
                selected_triplets=selected,
                rx_partial_packets=rx_packets,
                answer=sample.get("answer", ""),
            )
        )

    if not answer_list:
        raise RuntimeError("No DBSS-NST smoke rows were produced.")

    avg_bits = sum(bits_list) / len(bits_list)
    delivered_answer = sum(answer_list) / len(answer_list)

    row = {
        "method": "dbss_nst",
        "model": args.model,
        "semantic_type": "sg",
        "channel": args.channel,
        "snr_db": args.snr_db,
        "n_top": args.n_top,
        "num_samples": len(answer_list),
        "avg_source_bits": avg_bits,
        "ber": sum(ber_list) / len(ber_list),
        "delivered_answer_hit_rate": delivered_answer,
        "tx_field_keep_ratio": sum(tx_keep_ratio_list) / len(tx_keep_ratio_list),
        "rx_field_keep_ratio": sum(keep_ratio_list) / len(keep_ratio_list),
        "answer_per_kbit": delivered_answer / max(1e-12, avg_bits / 1000.0),
        "threshold": args.threshold,
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
