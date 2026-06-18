import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.data.gqa_subset import GQACommSubset
from src.methods.neural_evidence_predictor import safe_shifted_id
from src.methods.nsp_v5_semantic_features import (
    build_v5_dense_features,
    cosine_scalar,
    has_answer_overlap,
    nsp_v5_dense_feature_names,
    triplet_text,
)
from src.methods.nst_budget_labeling import infer_budget_keep_label
from src.utils.config import load_yaml


def encode_unique(model, texts, batch_size=256):
    unique = sorted(set(texts))
    emb = model.encode(
        unique,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return {t: emb[i].cpu() for i, t in enumerate(unique)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--utility-lambda", type=float, default=0.10)
    parser.add_argument("--text-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", default="results/nsp/nsp_v6_rams_train.pt")
    parser.add_argument("--object-vocab-size", type=int, default=1118)
    parser.add_argument("--relation-vocab-size", type=int, default=208)
    args = parser.parse_args()

    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    all_samples = ds.load_samples()
    all_sg_rows = ds.load_sg_triplets()

    samples = all_samples[args.start_index: args.start_index + args.num_samples]
    sg_rows = all_sg_rows[args.start_index: args.start_index + args.num_samples]
    sample_by_qid = {s["question_id"]: s for s in samples}

    encoder = SentenceTransformer(args.text_model)
    text_dim = int(encoder.get_sentence_embedding_dimension())

    question_texts = []
    triplet_texts = []
    valid_rows = []

    for row in sg_rows:
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]
        triplets = list(row.get("triplets", []))[: args.max_candidates]

        if not triplets:
            continue

        q = sample.get("question", "")
        question_texts.append(q)

        for t in triplets:
            triplet_texts.append(triplet_text(t))

        valid_rows.append((sample, triplets))

    print("encoding questions:", len(set(question_texts)))
    q_cache = encode_unique(encoder, question_texts)

    print("encoding triplets:", len(set(triplet_texts)))
    t_cache = encode_unique(encoder, triplet_texts)

    group_ids = []
    subject_ids = []
    relation_ids = []
    object_ids = []
    dense_features = []
    q_embs = []
    t_embs = []
    answer_labels = []
    mask_labels = []

    total_candidates = 0
    total_answer_positive = 0
    total_mask_keep = 0

    for group_id, (sample, triplets) in enumerate(valid_rows):
        q = sample.get("question", "")
        q_emb = q_cache[q]
        answer = sample.get("answer", "")

        for i, t in enumerate(triplets):
            tt = triplet_text(t)
            t_emb = t_cache[tt]
            cos = cosine_scalar(q_emb, t_emb)

            dense = build_v5_dense_features(
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                rank_index=i,
                n_candidates=len(triplets),
                semantic_cosine=cos,
            )

            labels = infer_budget_keep_label(
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                answer=answer,
                question_type=sample.get("question_type", ""),
                utility_lambda=args.utility_lambda,
            )

            answer_y = 1.0 if has_answer_overlap(t, answer) else 0.0

            group_ids.append(group_id)
            subject_ids.append(safe_shifted_id(t.get("subject_id", 0), args.object_vocab_size))
            relation_ids.append(safe_shifted_id(t.get("relation_id", 0), args.relation_vocab_size))
            object_ids.append(safe_shifted_id(t.get("object_id", 0), args.object_vocab_size))
            dense_features.append(dense)
            q_embs.append(q_emb)
            t_embs.append(t_emb)
            answer_labels.append(answer_y)
            mask_labels.append([
                float(labels["subject"]),
                float(labels["relation"]),
                float(labels["object"]),
            ])

            total_candidates += 1
            total_answer_positive += int(answer_y > 0.5)
            total_mask_keep += labels["subject"] + labels["relation"] + labels["object"]

    out = {
        "group_id": torch.tensor(group_ids, dtype=torch.long),
        "subject_id": torch.tensor(subject_ids, dtype=torch.long),
        "relation_id": torch.tensor(relation_ids, dtype=torch.long),
        "object_id": torch.tensor(object_ids, dtype=torch.long),
        "dense_features": torch.tensor(dense_features, dtype=torch.float32),
        "q_emb": torch.stack(q_embs).to(torch.float16),
        "t_emb": torch.stack(t_embs).to(torch.float16),
        "answer_label": torch.tensor(answer_labels, dtype=torch.float32),
        "mask_label": torch.tensor(mask_labels, dtype=torch.float32),
        "dense_feature_names": nsp_v5_dense_feature_names(),
        "text_model": args.text_model,
        "text_dim": text_dim,
        "object_vocab_size": args.object_vocab_size,
        "relation_vocab_size": args.relation_vocab_size,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    print(f"Saved: {out_path}")
    print("num_rows:", len(group_ids))
    print("num_groups:", len(valid_rows))
    print("answer_positive_ratio:", total_answer_positive / max(1, total_candidates))
    print("avg_mask_keep_ratio:", total_mask_keep / max(1, 3 * total_candidates))
    print("dense_dim:", len(nsp_v5_dense_feature_names()))
    print("text_dim:", text_dim)


if __name__ == "__main__":
    main()
