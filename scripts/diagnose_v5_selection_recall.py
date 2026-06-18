import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.data.gqa_subset import GQACommSubset
from src.methods.dbss import dbss_select_triplets
from src.methods.neural_semantic_reranker import (
    load_semantic_reranker,
    safe_shifted_id,
)
from src.methods.nsp_v5_semantic_features import (
    answer_tokens,
    build_v5_dense_features,
    cosine_scalar,
    field_tokens,
    has_answer_overlap,
    triplet_text,
)
from src.utils.config import load_yaml

from scripts.run_nsp_v5_semantic_smoke import semantic_mmr_select


def answer_field_kept(triplet, keep, answer):
    ans = answer_tokens(answer)
    if not ans:
        return False

    for field in ["subject", "relation", "object"]:
        if keep.get(field, False) and (field_tokens(triplet, field) & ans):
            return True

    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=8000)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--selection-mode", default="mmr", choices=["topk", "mmr"])
    parser.add_argument("--semantic-mmr-beta", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    samples = ds.load_samples()[args.start_index: args.start_index + args.num_samples]
    sg_rows = ds.load_sg_triplets()[args.start_index: args.start_index + args.num_samples]
    sample_by_qid = {s["question_id"]: s for s in samples}

    model = load_semantic_reranker(args.model)
    ckpt = torch.load(args.model, map_location="cpu")
    encoder = SentenceTransformer(ckpt.get("text_model", "all-MiniLM-L6-v2"))

    total_samples = 0
    samples_with_answer_candidate = 0

    dbss_answer_recall = 0
    v5_answer_recall = 0

    v5_selected_answer_triplets = 0
    v5_answer_field_kept = 0

    dbss_selected_answer_triplets = 0

    for row in sg_rows:
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]
        triplets = list(row.get("triplets", []))[: args.max_candidates]
        if not triplets:
            continue

        answer = sample.get("answer", "")
        answer_flags = [has_answer_overlap(t, answer) for t in triplets]

        total_samples += 1

        if not any(answer_flags):
            continue

        samples_with_answer_candidate += 1

        # DBSS selection recall.
        dbss_selected = dbss_select_triplets(
            triplets=triplets,
            question=sample.get("question", ""),
            keywords=sample.get("keywords", []),
            n_top=args.n_top,
            snr_db=args.snr_db,
            channel_type=args.channel,
        )

        dbss_hit = any(has_answer_overlap(t, answer) for t in dbss_selected)
        if dbss_hit:
            dbss_answer_recall += 1

        dbss_selected_answer_triplets += sum(
            int(has_answer_overlap(t, answer)) for t in dbss_selected
        )

        # V5 scores.
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
            scores_list = scores.cpu().tolist()
            mask_probs = torch.sigmoid(mask_logits).cpu()

        if args.selection_mode == "mmr":
            selected_indices = semantic_mmr_select(
                scores=scores_list,
                trip_embs=t_embs,
                n_top=args.n_top,
                beta=args.semantic_mmr_beta,
            )
        else:
            selected_indices = sorted(
                range(len(scores_list)),
                key=lambda i: scores_list[i],
                reverse=True,
            )[: args.n_top]

        selected_has_answer = False

        for idx in selected_indices:
            t = triplets[idx]

            if has_answer_overlap(t, answer):
                selected_has_answer = True
                v5_selected_answer_triplets += 1

                probs = mask_probs[idx].tolist()
                keep = {
                    "subject": probs[0] >= args.mask_threshold,
                    "relation": probs[1] >= args.mask_threshold,
                    "object": probs[2] >= args.mask_threshold,
                }

                if answer_field_kept(t, keep, answer):
                    v5_answer_field_kept += 1

        if selected_has_answer:
            v5_answer_recall += 1

    out = {
        "total_samples": total_samples,
        "samples_with_answer_candidate": samples_with_answer_candidate,
        "answer_candidate_ratio": samples_with_answer_candidate / max(1, total_samples),

        "dbss_answer_recall_at_k": dbss_answer_recall / max(1, samples_with_answer_candidate),
        "v5_answer_recall_at_k": v5_answer_recall / max(1, samples_with_answer_candidate),

        "dbss_avg_selected_answer_triplets": dbss_selected_answer_triplets / max(1, samples_with_answer_candidate),
        "v5_avg_selected_answer_triplets": v5_selected_answer_triplets / max(1, samples_with_answer_candidate),

        "v5_answer_field_keep_rate_given_selected_answer": (
            v5_answer_field_kept / max(1, v5_selected_answer_triplets)
        ),

        "selection_mode": args.selection_mode,
        "semantic_mmr_beta": args.semantic_mmr_beta,
        "mask_threshold": args.mask_threshold,
    }

    print(out)


if __name__ == "__main__":
    main()
