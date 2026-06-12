import argparse
import csv
from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.semantic.ranking import build_object_frequency, build_relation_frequency
from src.utils.config import load_yaml

from scripts.run_benchmark_ldpc_snr import run_sg_ldpc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LDPC-coded Ntop benchmark for SG semantic selection methods."
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "original", "do", "go", "dbss"],
    )

    parser.add_argument(
        "--channels",
        nargs="+",
        default=["awgn", "rayleigh"],
        choices=["awgn", "rayleigh"],
    )

    parser.add_argument("--snr-db", type=float, default=8.0)

    parser.add_argument(
        "--n-tops",
        nargs="+",
        type=int,
        default=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    )

    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--out",
        type=str,
        default="results/benchmark_ldpc/main_ntop_sg_ldpc_dbss.csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_yaml("configs/experiment.yaml")

    seed = int(cfg["project"]["seed"]) if args.seed is None else args.seed
    perfect_csi = bool(cfg["channel"]["perfect_csi"])
    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])

    ds = GQACommSubset(Path(cfg["data"]["root"]))

    samples = ds.load_samples(limit=args.num_samples)
    sample_by_qid = {s["question_id"]: s for s in samples}

    sg_rows = ds.load_sg_triplets(limit=args.num_samples)

    all_samples_for_freq = ds.load_samples()
    object_freq = build_object_frequency(all_samples_for_freq)
    relation_freq = build_relation_frequency(all_samples_for_freq)

    out_rows = []

    for channel_type in args.channels:
        for n_top in args.n_tops:
            for method in args.methods:
                row = run_sg_ldpc(
                    sg_rows=sg_rows,
                    sample_by_qid=sample_by_qid,
                    object_freq=object_freq,
                    relation_freq=relation_freq,
                    method=method,
                    channel_type=channel_type,
                    snr_db=args.snr_db,
                    n_top=n_top,
                    seed=seed,
                    perfect_csi=perfect_csi,
                    bandwidth_hz=bandwidth_hz,
                )
                out_rows.append(row)
                print(row)

    if not out_rows:
        raise RuntimeError("No LDPC Ntop benchmark rows were produced.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
