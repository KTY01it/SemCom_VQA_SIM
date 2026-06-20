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
from src.methods.neural_semantic_policy_v7 import load_v7_policy
from src.methods.neural_semantic_tokenizer_v2 import safe_shifted_id
from src.methods.nsp_v5_semantic_features import (
    build_v5_dense_features,
    cosine_scalar,
    triplet_text,
)
from src.methods.qtc_rule import tx_field_keep_ratio
from src.semantic.compressed_sg_codec import (
    decode_compressed_sg_triplets,
    encode_compressed_sg_triplets,
)
from src.utils.config import load_yaml

from scripts.run_ranking_sweep import transmit_bits


FIELDS = ["subject", "relation", "object"]


def threshold_masks(mask_probs, threshold):
    keep_masks = []
    for probs in mask_probs:
        keep = {
            "subject": bool(float(probs[0]) >= threshold),
            "relation": bool(float(probs[1]) >= threshold),
            "object": bool(float(probs[2]) >= threshold),
        }
        if not any(keep.values()):
            best = max(range(3), key=lambda i: float(probs[i]))
            keep[FIELDS[best]] = True
        keep_masks.append(keep)
    return keep_masks


def budgeted_masks(mask_probs, field_budget):
    """
    Pure neural budget allocator:
      - uses only model mask probabilities
      - no DBSS
      - no Slot-Guard
      - no answer

    Ensures at least one field per selected triplet, then allocates remaining
    fields to highest mask probabilities.
    """
    n = len(mask_probs)
    field_budget = max(n, int(field_budget))

    keep = [
        {"subject": False, "relation": False, "object": False}
        for _ in range(n)
    ]

    used = 0

    # First, keep one best field for each selected triplet.
    for i, probs in enumerate(mask_probs):
        best = max(range(3), key=lambda j: float(probs[j]))
        keep[i][FIELDS[best]] = True
        used += 1

    remaining = []

    for i, probs in enumerate(mask_probs):
        for j, field in enumerate(FIELDS):
            if not keep[i][field]:
                remaining.append((float(probs[j]), i, field))

    remaining.sort(reverse=True, key=lambda x: x[0])

    for _, i, field in remaining:
        if used >= field_budget:
            break
        keep[i][field] = True
        used += 1

    return keep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="nsp_v7_td_setpolicy")
    parser.add_argument("--start-index", type=int, default=8000)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--select-temperature", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--target-bits", type=int, default=299)
    parser.add_argument("--teacher-weight", type=float, default=0.4)
    parser.add_argument("--evidence-weight", type=float, default=0.6)
    parser.add_argument("--model", required=True)
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

    samples = all_samples[args.start_index: args.start_index + args.num_samples]
    sg_rows = all_sg_rows[args.start_index: args.start_index + args.num_samples]
    sample_by_qid = {s["question_id"]: s for s in samples}

    model = load_v7_policy(args.model)
    ckpt = torch.load(args.model, map_location="cpu")
    encoder = SentenceTransformer(ckpt.get("text_model", "all-MiniLM-L6-v2"))

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
        candidates = list(row.get("triplets", []))[: args.max_candidates]

        if not candidates:
            continue

        q_text = sample.get("question", "")
        t_texts = [triplet_text(t) for t in candidates]

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

        max_n = args.max_candidates
        dense_dim = int(model.dense_dim)
        text_dim = int(model.text_dim)

        sid = torch.zeros(1, max_n, dtype=torch.long)
        rid = torch.zeros(1, max_n, dtype=torch.long)
        oid = torch.zeros(1, max_n, dtype=torch.long)
        dense = torch.zeros(1, max_n, dense_dim, dtype=torch.float32)
        t_mat = torch.zeros(1, max_n, text_dim, dtype=torch.float32)
        cand_mask = torch.zeros(1, max_n, dtype=torch.bool)

        for i, t in enumerate(candidates[:max_n]):
            cos = cosine_scalar(q_emb, t_embs[i])

            feats = build_v5_dense_features(
                triplet=t,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
                question_type=sample.get("question_type", ""),
                rank_index=i,
                n_candidates=len(candidates),
                semantic_cosine=cos,
            )

            sid[0, i] = safe_shifted_id(t.get("subject_id", 0), model.object_vocab_size)
            rid[0, i] = safe_shifted_id(t.get("relation_id", 0), model.relation_vocab_size)
            oid[0, i] = safe_shifted_id(t.get("object_id", 0), model.object_vocab_size)
            dense[0, i] = torch.tensor(feats, dtype=torch.float32)
            t_mat[0, i] = t_embs[i].float()
            cand_mask[0, i] = True

        q_batch = q_emb[None, :].float()

        with torch.no_grad():
            select_logits, evidence_logits, mask_logits = model(
                sid, rid, oid, dense, q_batch, t_mat, cand_mask
            )

            temp = max(1e-6, float(args.select_temperature))
            teacher_scores = torch.sigmoid(select_logits[0] / temp)
            evidence_scores = torch.sigmoid(evidence_logits[0] / temp)
            select_scores = (
                args.teacher_weight * teacher_scores
                + args.evidence_weight * evidence_scores
            )
            mask_probs = torch.sigmoid(mask_logits[0])

        valid_n = len(candidates)
        topk = min(args.n_top, valid_n)
        selected_indices = torch.topk(select_scores[:valid_n], k=topk).indices.tolist()
        selected = [candidates[i] for i in selected_indices]

        selected_mask_probs = [mask_probs[i].cpu().tolist() for i in selected_indices]

        if args.target_bits > 0:
            # compressed bits = 3*n_top + 16*num_fields
            field_budget = int((args.target_bits - 3 * topk) // 16)
            keep_masks = budgeted_masks(selected_mask_probs, field_budget)
        else:
            keep_masks = threshold_masks(selected_mask_probs, args.mask_threshold)

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
        candidate_count_list.append(valid_n)
        answer_list.append(
            partial_delivered_answer_hit(
                selected_triplets=selected,
                rx_partial_packets=rx_packets,
                answer=sample.get("answer", ""),
            )
        )

    if not answer_list:
        raise RuntimeError("No V7 rows were produced.")

    avg_bits = sum(bits_list) / len(bits_list)
    delivered_answer = sum(answer_list) / len(answer_list)

    row = {
        "method": args.method_name,
        "model": args.model,
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
        "select_temperature": args.select_temperature,
        "mask_threshold": args.mask_threshold,
        "target_bits": args.target_bits,
        "start_index": args.start_index,
        "teacher_weight": args.teacher_weight,
        "evidence_weight": args.evidence_weight,
    }

    print(row)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
