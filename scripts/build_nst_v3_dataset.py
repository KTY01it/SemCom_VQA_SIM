import argparse
import csv
from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.methods.dbss import dbss_select_triplets
from src.methods.nst_budget_labeling import infer_budget_keep_label
from src.methods.nst_features import build_nst_v2_features, nst_v2_feature_names
from src.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--utility-lambda", type=float, default=1.0)
    parser.add_argument("--out", default="results/nst/nst_v3_train.csv")
    args = parser.parse_args()

    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    all_samples = ds.load_samples()
    all_sg_rows = ds.load_sg_triplets()

    start = args.start_index
    end = start + args.num_samples

    samples = all_samples[start:end]
    sg_rows = all_sg_rows[start:end]
    sample_by_qid = {s["question_id"]: s for s in samples}

    feat_names = nst_v2_feature_names()
    rows = []

    label_sum = {"subject": 0, "relation": 0, "object": 0}
    n_label = 0

    for row in sg_rows:
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

        for rank_index, t in enumerate(selected):
            labels = infer_budget_keep_label(
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                answer=sample.get("answer", ""),
                question_type=sample.get("question_type", ""),
                utility_lambda=args.utility_lambda,
            )

            feats = build_nst_v2_features(
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                rank_index=rank_index,
                n_top=args.n_top,
            )

            base = {
                "question_id": qid,
                "question": sample.get("question", ""),
                "keywords": " ".join(map(str, sample.get("keywords", []))),
                "question_type": sample.get("question_type", ""),
                "subject": t.get("subject", ""),
                "relation": t.get("relation", ""),
                "object": t.get("object", ""),
                "subject_id": int(t.get("subject_id", -1)),
                "relation_id": int(t.get("relation_id", -1)),
                "object_id": int(t.get("object_id", -1)),
                "rank_index": rank_index,
                "keep_subject": labels["subject"],
                "keep_relation": labels["relation"],
                "keep_object": labels["object"],
            }

            for k in label_sum:
                label_sum[k] += labels[k]
            n_label += 1

            for name, val in zip(feat_names, feats):
                base[name] = val

            rows.append(base)

    if not rows:
        raise RuntimeError("No NST-v3 rows were produced.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")
    print(f"feature_dim: {len(feat_names)}")
    print("avg_label_keep:", {
        k: label_sum[k] / max(1, n_label)
        for k in label_sum
    })
    print("avg_total_keep:", sum(label_sum.values()) / max(1, 3 * n_label))


if __name__ == "__main__":
    main()
