import argparse
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.data.gqa_subset import GQACommSubset
from src.methods.dbss import dbss_select_triplets
from src.methods.nsp_v5_semantic_features import (
    has_answer_overlap,
    triplet_text,
)
from src.utils.config import load_yaml


def normalize_triplets(raw_triplets):
    if raw_triplets is None:
        return []

    if isinstance(raw_triplets, list):
        return [t for t in raw_triplets if isinstance(t, dict)]

    if isinstance(raw_triplets, dict):
        if any(k in raw_triplets for k in ["subject", "relation", "object"]):
            return [raw_triplets]
        return [t for t in raw_triplets.values() if isinstance(t, dict)]

    return []


def hit_answer(triplets, answer):
    return any(has_answer_overlap(t, answer) for t in triplets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=8000)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--crossencoder", required=True)
    args = parser.parse_args()

    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    all_samples = ds.load_samples()
    all_sg_rows = ds.load_sg_triplets()

    samples = all_samples[args.start_index: args.start_index + args.num_samples]
    sg_rows = all_sg_rows[args.start_index: args.start_index + args.num_samples]
    sample_by_qid = {s["question_id"]: s for s in samples}

    ranker = CrossEncoder(args.crossencoder)

    total = 0

    oracle_hit = 0

    original_top1_hit = 0
    original_topk_hit = 0

    dbss_top1_hit = 0
    dbss_topk_hit = 0

    ce_top1_hit = 0
    ce_topk_hit = 0

    ce_topk_pos_count = 0
    dbss_topk_pos_count = 0
    original_topk_pos_count = 0

    for row in sg_rows:
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        keywords = sample.get("keywords", [])

        candidates = normalize_triplets(row.get("triplets", []))[:args.max_candidates]
        if not candidates:
            continue

        total += 1

        if hit_answer(candidates, answer):
            oracle_hit += 1

        original_selected = candidates[:args.n_top]
        original_top1 = candidates[:1]

        original_top1_hit += int(hit_answer(original_top1, answer))
        original_topk_hit += int(hit_answer(original_selected, answer))
        original_topk_pos_count += sum(int(has_answer_overlap(t, answer)) for t in original_selected)

        dbss_ranked = dbss_select_triplets(
            triplets=candidates,
            question=question,
            keywords=keywords,
            n_top=args.n_top,
            snr_db=args.snr_db,
            channel_type=args.channel,
        )
        dbss_selected = dbss_ranked[:args.n_top]
        dbss_top1 = dbss_ranked[:1]

        dbss_top1_hit += int(hit_answer(dbss_top1, answer))
        dbss_topk_hit += int(hit_answer(dbss_selected, answer))
        dbss_topk_pos_count += sum(int(has_answer_overlap(t, answer)) for t in dbss_selected)

        pairs = [[question, triplet_text(t)] for t in candidates]
        scores = ranker.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)

        ce_selected = [candidates[i] for i in order[:args.n_top]]
        ce_top1 = [candidates[order[0]]]

        ce_top1_hit += int(hit_answer(ce_top1, answer))
        ce_topk_hit += int(hit_answer(ce_selected, answer))
        ce_topk_pos_count += sum(int(has_answer_overlap(t, answer)) for t in ce_selected)

    def rate(x):
        return x / max(1, total)

    print({
        "total": total,
        "oracle_candidate_answer_hit": rate(oracle_hit),

        "original_top1_answer_hit": rate(original_top1_hit),
        "original_topk_answer_hit": rate(original_topk_hit),
        "original_avg_answer_positive_in_topk": original_topk_pos_count / max(1, total),

        "dbss_top1_answer_hit": rate(dbss_top1_hit),
        "dbss_topk_answer_hit": rate(dbss_topk_hit),
        "dbss_avg_answer_positive_in_topk": dbss_topk_pos_count / max(1, total),

        "crossencoder_top1_answer_hit": rate(ce_top1_hit),
        "crossencoder_topk_answer_hit": rate(ce_topk_hit),
        "crossencoder_avg_answer_positive_in_topk": ce_topk_pos_count / max(1, total),
    })


if __name__ == "__main__":
    main()
