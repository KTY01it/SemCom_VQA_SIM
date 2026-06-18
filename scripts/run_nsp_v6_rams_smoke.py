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
from src.methods.neural_evidence_predictor import (
    load_evidence_predictor,
    safe_shifted_id,
)
from src.methods.nsp_v5_semantic_features import (
    build_v5_dense_features,
    cosine_scalar,
    triplet_text,
)
from src.methods.qtc_rule import tx_field_keep_ratio
from src.methods.rams_selector import rams_select_triplets
from src.semantic.compressed_sg_codec import (
    decode_compressed_sg_triplets,
    encode_compressed_sg_triplets,
)
from src.utils.config import load_yaml

from scripts.run_ranking_sweep import transmit_bits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="nsp_v6_rams")
    parser.add_argument("--start-index", type=int, default=8000)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--mask-threshold", type=float, default=0.20)
    parser.add_argument("--evidence-temperature", type=float, default=1.0)

    parser.add_argument("--lambda-survival", type=float, default=4.0)
    parser.add_argument("--lambda-relevance", type=float, default=0.5)
    parser.add_argument("--lambda-coverage", type=float, default=0.5)
    parser.add_argument("--lambda-redundancy", type=float, default=0.2)
    parser.add_argument("--lambda-cost", type=float, default=0.01)

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

    model = load_evidence_predictor(args.model)
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

            evidence_logits, mask_logits = model(s, r, o, dense, q_batch, t_batch)

            temp = max(1e-6, float(args.evidence_temperature))
            evidence_probs = torch.sigmoid(evidence_logits / temp).cpu().tolist()
            mask_probs = torch.sigmoid(mask_logits).cpu().tolist()

        selected_indices, keep_masks = rams_select_triplets(
            triplets=triplets,
            evidence_probs=evidence_probs,
            mask_probs=mask_probs,
            question=sample.get("question", ""),
            keywords=sample.get("keywords", []),
            n_top=args.n_top,
            snr_db=args.snr_db,
            channel_type=args.channel,
            mask_threshold=args.mask_threshold,
            lambda_survival=args.lambda_survival,
            lambda_relevance=args.lambda_relevance,
            lambda_coverage=args.lambda_coverage,
            lambda_redundancy=args.lambda_redundancy,
            lambda_cost=args.lambda_cost,
        )

        selected = [triplets[i] for i in selected_indices]

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
        raise RuntimeError("No RAMS rows were produced.")

    avg_bits = sum(bits_list) / len(bits_list)
    delivered_answer = sum(answer_list) / len(answer_list)

    row = {
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
        "evidence_temperature": args.evidence_temperature,
        "lambda_survival": args.lambda_survival,
        "lambda_relevance": args.lambda_relevance,
        "lambda_coverage": args.lambda_coverage,
        "lambda_redundancy": args.lambda_redundancy,
        "lambda_cost": args.lambda_cost,
        "start_index": args.start_index,
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
