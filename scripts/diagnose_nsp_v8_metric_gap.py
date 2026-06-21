import argparse
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.data.gqa_subset import GQACommSubset
from src.eval.partial_proxy_metrics import partial_delivered_answer_hit
from src.methods.dbss import dbss_select_triplets
from src.methods.nsp_v5_semantic_features import has_answer_overlap, triplet_text
from src.semantic.compressed_sg_codec import (
    decode_compressed_sg_triplets,
    encode_compressed_sg_triplets,
)
from src.utils.config import load_yaml


FULL_KEEP = {"subject": True, "relation": True, "object": True}


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


def overlap_hit(triplets, answer):
    return any(has_answer_overlap(t, answer) for t in triplets)


def metric_hit_fullfield(triplets, answer):
    if not triplets:
        return 0

    keep_masks = [dict(FULL_KEEP) for _ in triplets]
    tx_bits = encode_compressed_sg_triplets(triplets, keep_masks)
    rx_packets = decode_compressed_sg_triplets(tx_bits, num_triplets=len(triplets))

    return int(
        partial_delivered_answer_hit(
            selected_triplets=triplets,
            rx_partial_packets=rx_packets,
            answer=answer,
        )
    )


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

    oracle_overlap = 0
    oracle_metric = 0

    dbss_overlap = 0
    dbss_metric = 0

    ce_overlap = 0
    ce_metric = 0

    original_overlap = 0
    original_metric = 0

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

        oracle_overlap += int(overlap_hit(candidates, answer))
        oracle_metric += metric_hit_fullfield(candidates, answer)

        original_selected = candidates[:args.n_top]
        original_overlap += int(overlap_hit(original_selected, answer))
        original_metric += metric_hit_fullfield(original_selected, answer)

        dbss_ranked = dbss_select_triplets(
            triplets=candidates,
            question=question,
            keywords=keywords,
            n_top=args.n_top,
            snr_db=args.snr_db,
            channel_type=args.channel,
        )
        dbss_selected = dbss_ranked[:args.n_top]

        dbss_overlap += int(overlap_hit(dbss_selected, answer))
        dbss_metric += metric_hit_fullfield(dbss_selected, answer)

        pairs = [[question, triplet_text(t)] for t in candidates]
        scores = ranker.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        ce_selected = [candidates[i] for i in order[:args.n_top]]

        ce_overlap += int(overlap_hit(ce_selected, answer))
        ce_metric += metric_hit_fullfield(ce_selected, answer)

    def rate(x):
        return x / max(1, total)

    print({
        "total": total,

        "oracle_overlap_hit": rate(oracle_overlap),
        "oracle_metric_hit": rate(oracle_metric),

        "original_overlap_hit": rate(original_overlap),
        "original_metric_hit": rate(original_metric),

        "dbss_overlap_hit": rate(dbss_overlap),
        "dbss_metric_hit": rate(dbss_metric),

        "crossencoder_overlap_hit": rate(ce_overlap),
        "crossencoder_metric_hit": rate(ce_metric),
    })


if __name__ == "__main__":
    main()
