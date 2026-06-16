import argparse
import csv
from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.methods.dbss import dbss_select_triplets
from src.methods.nst_budget_labeling import infer_budget_keep_label
from src.methods.nst_features import build_nst_v2_features, nst_v2_feature_names
from src.utils.config import load_yaml


def triplet_key(t):
    return (
        int(t.get("subject_id", -1)),
        int(t.get("relation_id", -1)),
        int(t.get("object_id", -1)),
        str(t.get("subject", "")),
        str(t.get("relation", "")),
        str(t.get("object", "")),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--utility-lambda", type=float, default=0.10)
    parser.add_argument("--out", default="results/nsp/nsp_v2_train.csv")
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

    total_candidates = 0
    total_selected = 0
    total_mask_keep = 0

    for row in sg_rows:
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]
        triplets = list(row.get("triplets", []))[: args.max_candidates]

        if not triplets:
            continue

        # DBSS teacher selection.
        dbss_ranked = dbss_select_triplets(
            triplets=triplets,
            question=sample.get("question", ""),
            keywords=sample.get("keywords", []),
            n_top=args.n_top,
            snr_db=args.snr_db,
            channel_type=args.channel,
        )
        teacher_selected = set(triplet_key(t) for t in dbss_ranked[: args.n_top])

        for rank_index, t in enumerate(triplets):
            select_label = int(triplet_key(t) in teacher_selected)

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
                n_top=max(1, min(args.max_candidates, len(triplets))),
            )

            base = {
                "question_id": qid,
                "question": sample.get("question", ""),
                "keywords": " ".join(map(str, sample.get("keywords", []))),
                "question_type": sample.get("question_type", ""),
                "candidate_index": rank_index,
                "subject": t.get("subject", ""),
                "relation": t.get("relation", ""),
                "object": t.get("object", ""),
                "subject_id": int(t.get("subject_id", -1)),
                "relation_id": int(t.get("relation_id", -1)),
                "object_id": int(t.get("object_id", -1)),
                "select_label": select_label,
                "keep_subject": labels["subject"],
                "keep_relation": labels["relation"],
                "keep_object": labels["object"],
            }

            for name, val in zip(feat_names, feats):
                base[name] = val

            rows.append(base)

            total_candidates += 1
            total_selected += select_label
            total_mask_keep += labels["subject"] + labels["relation"] + labels["object"]

    if not rows:
        raise RuntimeError("No NSP-v2 rows were produced.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")
    print("select_positive_ratio:", total_selected / max(1, total_candidates))
    print("avg_mask_keep_ratio:", total_mask_keep / max(1, 3 * total_candidates))
    print("feature_dim:", len(feat_names))


if __name__ == "__main__":
    main()
