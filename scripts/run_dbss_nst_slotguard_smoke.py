import argparse
import csv
from email import parser
from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate
from src.eval.partial_proxy_metrics import (
    partial_delivered_answer_hit,
    partial_field_keep_ratio,
)
from src.methods.dbss import dbss_select_triplets
from src.methods.neural_semantic_tokenizer_v2 import (
    load_nst_v2_model,
    predict_keep_mask_v2,
)
from src.methods.qtc_rule import tx_field_keep_ratio
from src.methods.slot_guard import (
    apply_slot_guard,
    repair_masks_to_budget,
    total_compressed_bits,
)
from src.semantic.compressed_sg_codec import (
    decode_compressed_sg_triplets,
    encode_compressed_sg_triplets,
)
from src.utils.config import load_yaml

from scripts.run_ranking_sweep import transmit_bits


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--method-name", default="dbss_nst_v4_slotguard")
    parser.add_argument("--start-index", type=int, default=8000)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])

    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument(
        "--guard-mode",
        default="slot",
        choices=[
            "off",
            "slot",
            "slot_relation",
            "answer_slot",
            "answer_slot_relation",
            "full",
        ],
    )
    parser.add_argument("--min-guard-prob", type=float, default=0.0)

    # 0 disables repair.
    # Useful targets for n_top=9:
    #   283 bits = 16 retained fields total
    #   299 bits = 17 retained fields total
    #   315 bits = 18 retained fields total
    parser.add_argument("--target-bits", type=int, default=299)
    parser.add_argument("--allow-guard-drop", action="store_true")
    parser.add_argument("--guard-drop-penalty", type=float, default=0.0)
    parser.add_argument("--model", default="results/nst/nst_v2_model.pt")
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

    model = load_nst_v2_model(args.model)

    ber_list = []
    answer_list = []
    bits_list = []
    tx_keep_ratio_list = []
    rx_keep_ratio_list = []
    pre_repair_bits_list = []
    post_repair_bits_list = []

    for sample_idx, row in enumerate(sg_rows):
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]

        ranked = dbss_select_triplets(
            triplets=row.get("triplets", []),
            question=sample.get("question", ""),
            keywords=sample.get("keywords", []),
            n_top=args.n_top,
            snr_db=args.snr_db,
            channel_type=args.channel,
        )

        selected = ranked[: args.n_top]

        if not selected:
            continue

        keep_masks = []
        guard_masks = []
        prob_lists = []

        for rank_index, t in enumerate(selected):
            keep, probs = predict_keep_mask_v2(
                model=model,
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                rank_index=rank_index,
                n_top=args.n_top,
                threshold=args.threshold,
            )

            if args.guard_mode != "off":
                keep, guard = apply_slot_guard(
                    keep=keep,
                    probs=probs,
                    triplet=t,
                    question=sample.get("question", ""),
                    keywords=sample.get("keywords", []),
                    question_type=sample.get("question_type", ""),
                    mode=args.guard_mode,
                    min_guard_prob=args.min_guard_prob,
                )
            else:
                guard = {
                    "subject": False,
                    "relation": False,
                    "object": False,
                }

            keep_masks.append(keep)
            guard_masks.append(guard)
            prob_lists.append(probs)

        pre_repair_bits = total_compressed_bits(keep_masks)

        keep_masks = repair_masks_to_budget(
            keep_masks=keep_masks,
            prob_lists=prob_lists,
            guard_masks=guard_masks,
            target_bits=args.target_bits,
            allow_guard_drop=args.allow_guard_drop,
            guard_drop_penalty=args.guard_drop_penalty,
        )

        post_repair_bits = total_compressed_bits(keep_masks)

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
        pre_repair_bits_list.append(pre_repair_bits)
        post_repair_bits_list.append(post_repair_bits)
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
        raise RuntimeError("No DBSS-NST-SlotGuard rows were produced.")

    avg_bits = sum(bits_list) / len(bits_list)
    delivered_answer = sum(answer_list) / len(answer_list)

    row = {
        "method": args.method_name,
        "model": args.model,
        "semantic_type": "sg",
        "channel": args.channel,
        "snr_db": args.snr_db,
        "n_top": args.n_top,
        "num_samples": len(answer_list),
        "avg_source_bits": avg_bits,
        "avg_pre_repair_bits": sum(pre_repair_bits_list) / len(pre_repair_bits_list),
        "avg_post_repair_bits": sum(post_repair_bits_list) / len(post_repair_bits_list),
        "ber": sum(ber_list) / len(ber_list),
        "delivered_answer_hit_rate": delivered_answer,
        "tx_field_keep_ratio": sum(tx_keep_ratio_list) / len(tx_keep_ratio_list),
        "rx_field_keep_ratio": sum(rx_keep_ratio_list) / len(rx_keep_ratio_list),
        "answer_per_kbit": delivered_answer / max(1e-12, avg_bits / 1000.0),
        "threshold": args.threshold,
        "guard_mode": args.guard_mode,
        "min_guard_prob": args.min_guard_prob,
        "target_bits": args.target_bits,
        "allow_guard_drop": args.allow_guard_drop,
        "guard_drop_penalty": args.guard_drop_penalty,
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
