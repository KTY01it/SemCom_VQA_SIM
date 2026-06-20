import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.data.gqa_subset import GQACommSubset
from src.methods.dbss import dbss_select_triplets
from src.methods.neural_semantic_tokenizer_v2 import (
    load_nst_v2_model,
    predict_keep_mask_v2,
    safe_shifted_id,
)
from src.methods.nsp_v5_semantic_features import (
    build_v5_dense_features,
    cosine_scalar,
    has_answer_overlap,    
    nsp_v5_dense_feature_names,
    triplet_text,
)
from src.methods.slot_guard import (
    apply_slot_guard,
    repair_masks_to_budget,
)
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
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])

    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-threshold", type=float, default=0.5)
    parser.add_argument("--teacher-guard-mode", default="slot")
    parser.add_argument("--teacher-target-bits", type=int, default=299)
    parser.add_argument("--teacher-guard-drop-penalty", type=float, default=0.2)

    parser.add_argument("--text-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--object-vocab-size", type=int, default=1118)
    parser.add_argument("--relation-vocab-size", type=int, default=208)
    parser.add_argument("--out", default="results/nsp/nsp_v7_teacher_train.pt")
    args = parser.parse_args()

    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    all_samples = ds.load_samples()
    all_sg_rows = ds.load_sg_triplets()

    samples = all_samples[args.start_index: args.start_index + args.num_samples]
    sg_rows = all_sg_rows[args.start_index: args.start_index + args.num_samples]
    sample_by_qid = {s["question_id"]: s for s in samples}

    teacher_nst = load_nst_v2_model(args.teacher_model)
    encoder = SentenceTransformer(args.text_model)
    text_dim = int(encoder.get_sentence_embedding_dimension())

    valid_items = []
    question_texts = []
    triplet_texts = []

    for row in sg_rows:
        qid = row["question_id"]
        if qid not in sample_by_qid:
            continue

        sample = sample_by_qid[qid]
        candidates = list(row.get("triplets", []))[: args.max_candidates]

        if not candidates:
            continue

        valid_items.append((sample, candidates))
        question_texts.append(sample.get("question", ""))

        for t in candidates:
            triplet_texts.append(triplet_text(t))

    print("encoding questions:", len(set(question_texts)))
    q_cache = encode_unique(encoder, question_texts)

    print("encoding triplets:", len(set(triplet_texts)))
    t_cache = encode_unique(encoder, triplet_texts)

    max_n = args.max_candidates
    dense_names = nsp_v5_dense_feature_names()
    dense_dim = len(dense_names)

    subject_id_rows = []
    relation_id_rows = []
    object_id_rows = []
    dense_rows = []
    q_rows = []
    t_rows = []
    candidate_mask_rows = []
    select_label_rows = []
    mask_label_rows = []
    answer_label_rows = []   
    teacher_bits_rows = []

    total_groups = 0
    total_candidates = 0
    total_selected = 0
    total_kept_fields = 0

    for group_idx, (sample, candidates) in enumerate(valid_items):
        q_text = sample.get("question", "")
        q_emb = q_cache[q_text]

        ranked = dbss_select_triplets(
            triplets=candidates,
            question=sample.get("question", ""),
            keywords=sample.get("keywords", []),
            n_top=args.n_top,
            snr_db=args.snr_db,
            channel_type=args.channel,
        )

        selected = ranked[: args.n_top]
        selected_ids = {id(t): i for i, t in enumerate(selected)}
        candidate_index_by_id = {id(t): i for i, t in enumerate(candidates)}

        keep_masks = []
        guard_masks = []
        prob_lists = []

        for rank_index, t in enumerate(selected):
            keep, probs = predict_keep_mask_v2(
                model=teacher_nst,
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                rank_index=rank_index,
                n_top=args.n_top,
                threshold=args.teacher_threshold,
            )

            keep, guard = apply_slot_guard(
                keep=keep,
                probs=probs,
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                mode=args.teacher_guard_mode,
                min_guard_prob=0.0,
            )

            keep_masks.append(keep)
            guard_masks.append(guard)
            prob_lists.append(probs)

        keep_masks = repair_masks_to_budget(
            keep_masks=keep_masks,
            prob_lists=prob_lists,
            guard_masks=guard_masks,
            target_bits=args.teacher_target_bits,
            allow_guard_drop=True,
            guard_drop_penalty=args.teacher_guard_drop_penalty,
        )

        select_label = torch.zeros(max_n, dtype=torch.float32)
        mask_label = torch.zeros(max_n, 3, dtype=torch.float32)
        answer_label = torch.zeros(max_n, dtype=torch.float32)
        
        for selected_pos, t in enumerate(selected):
            cidx = candidate_index_by_id.get(id(t), None)
            if cidx is None or cidx >= max_n:
                continue

            select_label[cidx] = 1.0
            m = keep_masks[selected_pos]
            mask_label[cidx, 0] = float(m["subject"])
            mask_label[cidx, 1] = float(m["relation"])
            mask_label[cidx, 2] = float(m["object"])

        sid = torch.zeros(max_n, dtype=torch.long)
        rid = torch.zeros(max_n, dtype=torch.long)
        oid = torch.zeros(max_n, dtype=torch.long)
        dense_mat = torch.zeros(max_n, dense_dim, dtype=torch.float32)
        t_mat = torch.zeros(max_n, text_dim, dtype=torch.float16)
        cand_mask = torch.zeros(max_n, dtype=torch.bool)

        for i, t in enumerate(candidates[:max_n]):
            tt = triplet_text(t)
            t_emb = t_cache[tt]
            cos = cosine_scalar(q_emb, t_emb)

            dense = build_v5_dense_features(
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                rank_index=i,
                n_candidates=len(candidates),
                semantic_cosine=cos,
            )

            sid[i] = safe_shifted_id(t.get("subject_id", 0), args.object_vocab_size)
            rid[i] = safe_shifted_id(t.get("relation_id", 0), args.relation_vocab_size)
            oid[i] = safe_shifted_id(t.get("object_id", 0), args.object_vocab_size)
            dense_mat[i] = torch.tensor(dense, dtype=torch.float32)
            t_mat[i] = t_emb.to(torch.float16)
            cand_mask[i] = True
            answer_label[i] = 1.0 if has_answer_overlap(
                t,
                sample.get("answer", ""),
            ) else 0.0

        teacher_bits = int(args.n_top * 3 + int(mask_label.sum().item()) * 16)

        subject_id_rows.append(sid)
        relation_id_rows.append(rid)
        object_id_rows.append(oid)
        dense_rows.append(dense_mat)
        q_rows.append(q_emb.to(torch.float16))
        t_rows.append(t_mat)
        candidate_mask_rows.append(cand_mask)
        select_label_rows.append(select_label)
        mask_label_rows.append(mask_label)
        answer_label_rows.append(answer_label)
        teacher_bits_rows.append(teacher_bits)

        total_groups += 1
        total_candidates += int(cand_mask.sum().item())
        total_selected += int(select_label.sum().item())
        total_kept_fields += int(mask_label.sum().item())

    out = {
        "subject_id": torch.stack(subject_id_rows),
        "relation_id": torch.stack(relation_id_rows),
        "object_id": torch.stack(object_id_rows),
        "dense_features": torch.stack(dense_rows),
        "q_emb": torch.stack(q_rows),
        "t_emb": torch.stack(t_rows),
        "candidate_mask": torch.stack(candidate_mask_rows),
        "select_label": torch.stack(select_label_rows),
        "mask_label": torch.stack(mask_label_rows),
        "answer_label": torch.stack(answer_label_rows),
        "teacher_bits": torch.tensor(teacher_bits_rows, dtype=torch.float32),
        "dense_feature_names": dense_names,
        "text_model": args.text_model,
        "text_dim": text_dim,
        "dense_dim": dense_dim,
        "object_vocab_size": args.object_vocab_size,
        "relation_vocab_size": args.relation_vocab_size,
        "max_candidates": args.max_candidates,
        "n_top": args.n_top,
        "teacher_config": vars(args),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    print(f"Saved: {out_path}")
    print("num_groups:", total_groups)
    print("avg_candidates:", total_candidates / max(1, total_groups))
    print("avg_selected:", total_selected / max(1, total_groups))
    total_answer = float(torch.stack(answer_label_rows).sum().item())
    total_valid = float(torch.stack(candidate_mask_rows).sum().item())
    print("answer_positive_ratio:", total_answer / max(1.0, total_valid))    
    print("avg_kept_fields:", total_kept_fields / max(1, total_groups))
    print("avg_teacher_bits:", sum(teacher_bits_rows) / max(1, len(teacher_bits_rows)))
    print("dense_dim:", dense_dim)
    print("text_dim:", text_dim)


if __name__ == "__main__":
    main()
