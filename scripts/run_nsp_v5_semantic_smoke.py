import argparse
import csv
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate
from src.eval.partial_proxy_metrics import (
    partial_delivered_answer_hit,
    partial_field_keep_ratio,
)
from src.methods.neural_semantic_reranker import (
    load_semantic_reranker,
    safe_shifted_id,
)
from src.methods.nsp_v5_semantic_features import (
    build_v5_dense_features,
    cosine_scalar,
    triplet_text,
    slot_aware_features,
    noanswer_proxy_score,
    question_triplet_jaccard,
)
from src.methods.qtc_rule import tx_field_keep_ratio
from src.semantic.compressed_sg_codec import (
    decode_compressed_sg_triplets,
    encode_compressed_sg_triplets,
)
from src.utils.config import load_yaml

from scripts.run_ranking_sweep import transmit_bits


def semantic_mmr_select(scores, trip_embs, n_top, beta):
    remaining = list(range(len(scores)))
    selected = []

    # Normalize score to stabilize beta.
    s = torch.tensor(scores, dtype=torch.float32)
    s = (s - s.mean()) / (s.std() + 1e-6)

    while remaining and len(selected) < n_top:
        best_idx = remaining[0]
        best_score = -1e18

        for idx in remaining:
            if not selected:
                red = 0.0
            else:
                sims = []
                for j in selected:
                    sims.append(float(torch.dot(trip_embs[idx], trip_embs[j]).item()))
                red = max(sims) if sims else 0.0

            score = float(s[idx].item()) - float(beta) * red

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected

def zscore_list(xs):
    if not xs:
        return xs

    import math

    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs))
    std = math.sqrt(var) + 1e-6

    return [(x - mean) / std for x in xs]


def build_augmented_scores(
    raw_scores,
    semantic_cosines,
    triplets,
    question,
    keywords,
    question_type,
    gamma_slot,
    gamma_proxy,
    gamma_semantic,
    gamma_jaccard,
):
    """
    Inference-safe score augmentation.
    Does not use answer.
    """
    neural_z = zscore_list([float(x) for x in raw_scores])
    sem_z = zscore_list([float(x) for x in semantic_cosines])

    proxy_scores = []
    jaccard_scores = []
    slot_scores = []

    for t in triplets:
        slot = slot_aware_features(
            triplet=t,
            question=question,
            keywords=keywords,
            question_type=question_type,
        )

        subject_as_answer_slot = slot[3]
        object_as_answer_slot = slot[4]
        relation_as_answer_slot = slot[5]
        relation_bridge = slot[6]
        spatial_match = slot[9]

        # Evidence-density proxy:
        # - answer-slot side is important
        # - relation bridge/spatial match improves VQA evidence quality
        slot_score = (
            1.0 * max(subject_as_answer_slot, object_as_answer_slot)
            + 0.5 * relation_as_answer_slot
            + 0.4 * relation_bridge
            + 0.4 * spatial_match
        )

        slot_scores.append(float(slot_score))

        proxy_scores.append(
            float(
                noanswer_proxy_score(
                    triplet=t,
                    question=question,
                    keywords=keywords,
                    question_type=question_type,
                )
                / 4.0
            )
        )

        jaccard_scores.append(
            float(
                question_triplet_jaccard(
                    triplet=t,
                    question=question,
                    keywords=keywords,
                )
            )
        )

    proxy_z = zscore_list(proxy_scores)
    jaccard_z = zscore_list(jaccard_scores)

    out = []
    for i in range(len(raw_scores)):
        s = (
            neural_z[i]
            + gamma_slot * slot_scores[i]
            + gamma_proxy * proxy_z[i]
            + gamma_semantic * sem_z[i]
            + gamma_jaccard * jaccard_z[i]
        )
        out.append(float(s))

    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="nsp_v5_semantic")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--mask-threshold", type=float, default=0.40)
    parser.add_argument("--selection-mode", default="topk", choices=["topk", "mmr"])
    parser.add_argument("--semantic-mmr-beta", type=float, default=0.30)
    parser.add_argument("--model", default="results/nsp/nsp_v5_semantic.pt")
    parser.add_argument("--score-augment", action="store_true")
    parser.add_argument("--gamma-slot", type=float, default=0.0)
    parser.add_argument("--gamma-proxy", type=float, default=0.0)
    parser.add_argument("--gamma-semantic", type=float, default=0.0)
    parser.add_argument("--gamma-jaccard", type=float, default=0.0)    
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

    model = load_semantic_reranker(args.model)
    ckpt = torch.load(args.model, map_location="cpu")
    text_model_name = ckpt.get("text_model", "all-MiniLM-L6-v2")

    encoder = SentenceTransformer(text_model_name)

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

        q_text = sample.get("question", "")
        t_texts = [triplet_text(t) for t in triplets]

        q_emb = encoder.encode(
            [q_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].cpu()

        t_embs = encoder.encode(
            t_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).cpu()

        subject_ids = []
        relation_ids = []
        object_ids = []
        dense_features = []

        for i, t in enumerate(triplets):
            cos = cosine_scalar(q_emb, t_embs[i])

            dense = build_v5_dense_features(
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                rank_index=i,
                n_candidates=len(triplets),
                semantic_cosine=cos,
            )

            subject_ids.append(safe_shifted_id(t.get("subject_id", 0), model.object_vocab_size))
            relation_ids.append(safe_shifted_id(t.get("relation_id", 0), model.relation_vocab_size))
            object_ids.append(safe_shifted_id(t.get("object_id", 0), model.object_vocab_size))
            dense_features.append(dense)

        with torch.no_grad():
            s = torch.tensor(subject_ids, dtype=torch.long)
            r = torch.tensor(relation_ids, dtype=torch.long)
            o = torch.tensor(object_ids, dtype=torch.long)
            dense = torch.tensor(dense_features, dtype=torch.float32)
            q_batch = q_emb[None, :].repeat(len(triplets), 1).float()
            t_batch = t_embs.float()

            scores, mask_logits = model(s, r, o, dense, q_batch, t_batch)
            scores = scores.cpu().tolist()
            mask_probs = torch.sigmoid(mask_logits).cpu()

        semantic_cosines = []
        for i in range(len(triplets)):
            semantic_cosines.append(cosine_scalar(q_emb, t_embs[i]))

        if args.score_augment:
            scores = build_augmented_scores(
                raw_scores=scores,
                semantic_cosines=semantic_cosines,
                triplets=triplets,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                gamma_slot=args.gamma_slot,
                gamma_proxy=args.gamma_proxy,
                gamma_semantic=args.gamma_semantic,
                gamma_jaccard=args.gamma_jaccard,
            )

        if args.selection_mode == "mmr":
            selected_indices = semantic_mmr_select(
                scores=scores,
                trip_embs=t_embs,
                n_top=args.n_top,
                beta=args.semantic_mmr_beta,
            )
        else:
            selected_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.n_top]

        selected = []
        keep_masks = []

        for idx in selected_indices:
            probs = mask_probs[idx].tolist()
            keep = {
                "subject": bool(probs[0] >= args.mask_threshold),
                "relation": bool(probs[1] >= args.mask_threshold),
                "object": bool(probs[2] >= args.mask_threshold),
            }

            if not any(keep.values()):
                keep["object"] = True

            selected.append(triplets[idx])
            keep_masks.append(keep)

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
        raise RuntimeError("No NSP-v5 rows were produced.")

    avg_bits = sum(bits_list) / len(bits_list)
    delivered_answer = sum(answer_list) / len(answer_list)

    out_row = {
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
        "selection_mode": args.selection_mode,
        "semantic_mmr_beta": args.semantic_mmr_beta,
        "start_index": args.start_index,
        "score_augment": args.score_augment,
        "gamma_slot": args.gamma_slot,
        "gamma_proxy": args.gamma_proxy,
        "gamma_semantic": args.gamma_semantic,
        "gamma_jaccard": args.gamma_jaccard,
    }

    print(out_row)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(out_row)


if __name__ == "__main__":
    main()
