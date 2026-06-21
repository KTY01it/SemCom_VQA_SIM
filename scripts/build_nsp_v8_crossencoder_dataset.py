import argparse
import json
import random
from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.methods.dbss import dbss_select_triplets
from src.methods.nsp_v5_semantic_features import (
    has_answer_overlap,
    question_triplet_jaccard,
    triplet_text,
)
from src.utils.config import load_yaml


def normalize_triplets(raw_triplets):
    """
    Return a list of triplet dictionaries.

    Some loaders may return:
      - list[dict]
      - dict[id -> dict]
      - a single dict triplet
    This helper prevents accidentally iterating over dict keys as strings.
    """
    if raw_triplets is None:
        return []

    if isinstance(raw_triplets, list):
        return [t for t in raw_triplets if isinstance(t, dict)]

    if isinstance(raw_triplets, dict):
        # Single triplet dict.
        if any(k in raw_triplets for k in ["subject", "relation", "object", "subject_id", "relation_id", "object_id"]):
            return [raw_triplets]

        # Mapping id -> triplet dict.
        vals = list(raw_triplets.values())
        return [t for t in vals if isinstance(t, dict)]

    return []



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--neg-per-pos", type=int, default=4)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/nsp/nsp_v8a_crossencoder_answer_hn_train.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    samples = ds.load_samples()
    sg_rows = ds.load_sg_triplets()

    samples = samples[args.start_index: args.start_index + args.num_samples]
    sg_rows = sg_rows[args.start_index: args.start_index + args.num_samples]
    sample_by_qid = {s["question_id"]: s for s in samples}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_pos = 0
    n_neg = 0
    n_groups = 0
    n_groups_with_pos = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in sg_rows:
            qid = row["question_id"]
            if qid not in sample_by_qid:
                continue

            sample = sample_by_qid[qid]
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            keywords = sample.get("keywords", [])

            candidates = normalize_triplets(row.get("triplets", []))[: args.max_candidates]
            if not candidates:
                continue

            teacher_ranked = dbss_select_triplets(
                triplets=candidates,
                question=question,
                keywords=keywords,
                n_top=args.n_top,
                snr_db=args.snr_db,
                channel_type=args.channel,
            )
            teacher_selected = teacher_ranked[: args.n_top]
            teacher_ids = {id(t) for t in teacher_selected}

            positives = []
            hard_negatives = []
            easy_negatives = []

            for rank_idx, t in enumerate(candidates):
                is_answer = has_answer_overlap(t, answer)
                is_teacher = id(t) in teacher_ids
                qjac = question_triplet_jaccard(t, question, keywords)

                rec = {
                    "question_id": qid,
                    "question": question,
                    "answer": answer,
                    "triplet_text": triplet_text(t),
                    "candidate_rank": rank_idx,
                    "teacher_select_label": 1 if is_teacher else 0,
                    "question_triplet_jaccard": float(qjac),
                }

                if is_answer:
                    positives.append(rec)
                else:
                    if is_teacher or qjac > 0:
                        hard_negatives.append(rec)
                    else:
                        easy_negatives.append(rec)

            n_groups += 1
            if positives:
                n_groups_with_pos += 1

            # Write all positives.
            for rec in positives:
                rec["label"] = 1.0
                rec["negative_type"] = "positive"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rows += 1
                n_pos += 1

            # Sample negatives per group.
            target_neg = max(args.neg_per_pos, args.neg_per_pos * len(positives))

            random.shuffle(hard_negatives)
            random.shuffle(easy_negatives)

            selected_negs = []

            # Prefer hard negatives.
            selected_negs.extend(hard_negatives[: target_neg])

            if len(selected_negs) < target_neg:
                remain = target_neg - len(selected_negs)
                selected_negs.extend(easy_negatives[:remain])

            # If group has no positive, still keep a few hard negatives.
            if not positives:
                selected_negs = hard_negatives[:args.neg_per_pos]
                if len(selected_negs) < args.neg_per_pos:
                    selected_negs.extend(easy_negatives[: args.neg_per_pos - len(selected_negs)])

            for rec in selected_negs:
                rec["label"] = 0.0
                rec["negative_type"] = "hard" if rec["teacher_select_label"] or rec["question_triplet_jaccard"] > 0 else "easy"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rows += 1
                n_neg += 1

    print("Saved:", out_path)
    print("num_rows:", n_rows)
    print("num_groups:", n_groups)
    print("groups_with_answer_positive:", n_groups_with_pos)
    print("group_answer_positive_ratio:", n_groups_with_pos / max(1, n_groups))
    print("positive_rows:", n_pos)
    print("negative_rows:", n_neg)
    print("row_positive_ratio:", n_pos / max(1, n_rows))


if __name__ == "__main__":
    main()
