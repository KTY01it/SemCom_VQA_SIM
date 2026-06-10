import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.run_answerability_sweep import get_vocab_info
from src.comm.latency import communication_latency_sec
from src.comm.ldpc_codec import LDPCConfig, SystematicLDPC
from src.comm.transmission import transmit_ldpc_bits, transmit_uncoded_bits
from src.data.gqa_subset import GQACommSubset
from src.eval.answerability import (
    bbox_answerability,
    is_supported_answer,
    sg_answerability_loose,
    sg_answerability_strict,
)
from src.eval.delivered_proxy import (
    delivered_bboxes_by_object_recovery,
    delivered_triplets_by_exact_recovery,
)
from src.eval.metrics import packet_error_rate
from src.semantic.packet_codec import (
    BBoxPacket,
    SGTripletPacket,
    decode_bboxes,
    decode_sg_triplets,
    encode_bboxes,
    encode_sg_triplets,
)
from src.semantic.packet_validation import (
    summarize_validation_results,
    validate_bbox_packet,
    validate_sg_packet,
    vocab_to_valid_id_set,
)
from src.semantic.ranking import (
    build_object_frequency,
    build_relation_frequency,
    rank_bboxes_do,
    rank_bboxes_go,
    rank_bboxes_original,
    rank_triplets_do,
    rank_triplets_go,
    rank_triplets_original,
)
from src.utils.config import load_yaml


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_vocab_sizes(cfg, data_root: Path) -> tuple[int, int]:
    object_vocab_name = cfg["data"].get("object_vocab", "object_vocab.json")
    relation_vocab_name = cfg["data"].get("relation_vocab", "relation_vocab.json")

    object_vocab = load_json(data_root / object_vocab_name)
    relation_vocab = load_json(data_root / relation_vocab_name)

    return len(object_vocab), len(relation_vocab)


def filter_valid_aligned(selected_units, tx_packets, rx_packets, validation_results):
    selected_valid = []
    tx_valid = []
    rx_valid = []

    for unit, tx_pkt, rx_pkt, status in zip(
        selected_units,
        tx_packets,
        rx_packets,
        validation_results,
    ):
        if status.valid:
            selected_valid.append(unit)
            tx_valid.append(tx_pkt)
            rx_valid.append(rx_pkt)

    return selected_valid, tx_valid, rx_valid


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def write_csv_union_fieldnames(output_path: Path, rows: list[dict]) -> None:
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_bboxes(
    bboxes: List[Dict[str, Any]],
    method: str,
    keywords: List[str],
    object_freq,
) -> List[Dict[str, Any]]:
    if method == "original":
        return rank_bboxes_original(bboxes)
    if method == "do":
        return rank_bboxes_do(bboxes, object_freq)
    if method == "go":
        return rank_bboxes_go(bboxes, keywords, object_freq)

    raise ValueError(f"Unknown bbox ranking method: {method}")


def select_triplets(
    triplets: List[Dict[str, Any]],
    method: str,
    keywords: List[str],
    object_freq,
    relation_freq,
) -> List[Dict[str, Any]]:
    if method == "original":
        return rank_triplets_original(triplets)
    if method == "do":
        return rank_triplets_do(triplets, object_freq, relation_freq)
    if method == "go":
        return rank_triplets_go(triplets, keywords, object_freq, relation_freq)

    raise ValueError(f"Unknown sg ranking method: {method}")


def coding_metadata_for_mode(coding_mode, codec):
    if coding_mode == "uncoded":
        return {
            "coding_family": "uncoded",
            "decoder_type": "hard_bpsk_decision",
            "soft_llr_decoder": False,
            "standard_ldpc_bp": False,
            "k": "",
            "m": "",
            "n": "",
            "nominal_code_rate": 1.0,
            "col_weight": "",
            "max_iter": "",
        }

    if coding_mode == "ldpc_like":
        if codec is None:
            return {
                "coding_family": "ldpc_like_sparse_systematic",
                "decoder_type": "hard_bit_flipping",
                "soft_llr_decoder": False,
                "standard_ldpc_bp": False,
                "k": "",
                "m": "",
                "n": "",
                "nominal_code_rate": "",
                "col_weight": "",
                "max_iter": "",
            }
        return codec.metadata()

    return {
        "coding_family": coding_mode,
        "decoder_type": "unknown",
        "soft_llr_decoder": False,
        "standard_ldpc_bp": False,
        "k": "",
        "m": "",
        "n": "",
        "nominal_code_rate": "",
        "col_weight": "",
        "max_iter": "",
    }
    
    
def transmit_bits_by_mode(
    bits,
    coding_mode,
    channel_type,
    snr_db,
    seed,
    perfect_csi,
    codec,
):
    if coding_mode == "uncoded":
        return transmit_uncoded_bits(
            bits=bits,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed,
            perfect_csi=perfect_csi,
        )

    if coding_mode == "ldpc_like":
        return transmit_ldpc_bits(
            bits=bits,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed,
            perfect_csi=perfect_csi,
            codec=codec,
        )

    raise ValueError(f"Unknown coding_mode: {coding_mode}")


def run_sg(
    sg_rows,
    sample_by_qid,
    object_freq,
    relation_freq,
    method,
    coding_mode,
    channel_type,
    snr_db,
    n_top,
    seed,
    perfect_csi,
    bandwidth_hz,
    codec,
    object_vocab_size,
    relation_vocab_size,
    valid_object_ids,
    valid_relation_ids,
):
    before_strict_list = []
    after_strict_list = []
    after_strict_validated_list = []

    before_loose_list = []
    after_loose_list = []
    after_loose_validated_list = []

    source_bits_list = []
    coded_bits_list = []
    channel_ber_list = []
    decoded_ber_list = []
    per_list = []
    ldpc_success_list = []
    source_latency_list = []
    coded_latency_list = []

    validation_results_all = []

    used_samples = 0

    for sample_idx, row in enumerate(sg_rows):
        qid = row["question_id"]
        sample = sample_by_qid[qid]
        answer = sample["answer"]
        keywords = sample["keywords"]

        if not is_supported_answer(answer):
            continue

        ranked = select_triplets(
            triplets=row["triplets"],
            method=method,
            keywords=keywords,
            object_freq=object_freq,
            relation_freq=relation_freq,
        )
        selected = ranked[:n_top]

        if not selected:
            continue

        tx_packets = [
            SGTripletPacket(
                subject_id=t["subject_id"],
                relation_id=t["relation_id"],
                object_id=t["object_id"],
            )
            for t in selected
        ]

        tx_bits = encode_sg_triplets(tx_packets)

        rx_bits, stats = transmit_bits_by_mode(
            bits=tx_bits,
            coding_mode=coding_mode,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
            codec=codec,
        )

        rx_packets = decode_sg_triplets(
            rx_bits,
            num_triplets=len(tx_packets),
        )

        validation_results = [
            validate_sg_packet(
                pkt,
                object_vocab_size=object_vocab_size,
                relation_vocab_size=relation_vocab_size,
                valid_object_ids=valid_object_ids,
                valid_relation_ids=valid_relation_ids,
            )
            for pkt in rx_packets
        ]
        validation_results_all.extend(validation_results)

        selected_valid, tx_packets_valid, rx_packets_valid = filter_valid_aligned(
            selected_units=selected,
            tx_packets=tx_packets,
            rx_packets=rx_packets,
            validation_results=validation_results,
        )

        delivered = delivered_triplets_by_exact_recovery(
            selected_triplets=selected,
            tx_packets=tx_packets,
            rx_packets=rx_packets,
        )

        delivered_validated = delivered_triplets_by_exact_recovery(
            selected_triplets=selected_valid,
            tx_packets=tx_packets_valid,
            rx_packets=rx_packets_valid,
        )

        before_strict_list.append(sg_answerability_strict(selected, answer, keywords))
        after_strict_list.append(sg_answerability_strict(delivered, answer, keywords))
        after_strict_validated_list.append(
            sg_answerability_strict(delivered_validated, answer, keywords)
        )

        before_loose_list.append(sg_answerability_loose(selected, answer))
        after_loose_list.append(sg_answerability_loose(delivered, answer))
        after_loose_validated_list.append(
            sg_answerability_loose(delivered_validated, answer)
        )

        source_bits = int(stats["source_bits"])
        coded_bits = int(stats["coded_bits"])

        source_bits_list.append(source_bits)
        coded_bits_list.append(coded_bits)
        channel_ber_list.append(stats["channel_ber"])
        decoded_ber_list.append(stats["decoded_ber"])
        ldpc_success_list.append(stats["ldpc_block_success_rate"])

        per_list.append(packet_error_rate(tx_bits, rx_bits, packet_size_bits=48))

        source_latency_list.append(
            communication_latency_sec(source_bits, bandwidth_hz, snr_db)
        )
        coded_latency_list.append(
            communication_latency_sec(coded_bits, bandwidth_hz, snr_db)
        )

        used_samples += 1

    validation_summary = summarize_validation_results(validation_results_all)

    coding_metadata = coding_metadata_for_mode(coding_mode, codec)
    
    out = {
        "semantic_type": "sg",
        "ranking_method": method,
        "coding_mode": coding_mode,
        "coding_family": coding_metadata.get("coding_family", "uncoded"),
        "decoder_type": coding_metadata.get("decoder_type", "none"),
        "soft_llr_decoder": coding_metadata.get("soft_llr_decoder", False),
        "standard_ldpc_bp": coding_metadata.get("standard_ldpc_bp", False),
        "ldpc_k": coding_metadata.get("k", ""),
        "ldpc_m": coding_metadata.get("m", ""),
        "ldpc_n": coding_metadata.get("n", ""),
        "ldpc_nominal_code_rate": coding_metadata.get("nominal_code_rate", ""),
        "ldpc_col_weight": coding_metadata.get("col_weight", ""),
        "ldpc_max_iter": coding_metadata.get("max_iter", ""),        
        "channel": channel_type,
        "snr_db": snr_db,
        "n_top": n_top,
        "num_samples": used_samples,
        "avg_source_bits": mean_or_zero(source_bits_list),
        "avg_coded_bits": mean_or_zero(coded_bits_list),
        "avg_code_rate": mean_or_zero(source_bits_list) / mean_or_zero(coded_bits_list),
        "channel_ber": mean_or_zero(channel_ber_list),
        "decoded_ber": mean_or_zero(decoded_ber_list),
        "packet_error_rate": mean_or_zero(per_list),
        "ldpc_block_success_rate": mean_or_zero(ldpc_success_list),
        "answerability_before_strict": mean_or_zero(before_strict_list),
        "answerability_after_strict": mean_or_zero(after_strict_list),
        "answerability_after_strict_validated": mean_or_zero(
            after_strict_validated_list
        ),
        "answerability_before_loose": mean_or_zero(before_loose_list),
        "answerability_after_loose": mean_or_zero(after_loose_list),
        "answerability_after_loose_validated": mean_or_zero(after_loose_validated_list),
        "source_t_com_sec": mean_or_zero(source_latency_list),
        "coded_t_com_sec": mean_or_zero(coded_latency_list),
    }
    out.update(validation_summary)

    return out


def run_bbox(
    bbox_rows,
    sample_by_qid,
    object_freq,
    method,
    coding_mode,
    channel_type,
    snr_db,
    n_top,
    seed,
    perfect_csi,
    bandwidth_hz,
    codec,
    object_vocab_size,
    valid_object_ids,
):
    before_list = []
    after_list = []
    after_validated_list = []

    source_bits_list = []
    coded_bits_list = []
    channel_ber_list = []
    decoded_ber_list = []
    per_list = []
    ldpc_success_list = []
    source_latency_list = []
    coded_latency_list = []

    validation_results_all = []

    used_samples = 0

    for sample_idx, row in enumerate(bbox_rows):
        qid = row["question_id"]
        sample = sample_by_qid[qid]
        answer = sample["answer"]
        keywords = sample["keywords"]

        if not is_supported_answer(answer):
            continue

        ranked = select_bboxes(
            bboxes=row["bboxes"],
            method=method,
            keywords=keywords,
            object_freq=object_freq,
        )
        selected = ranked[:n_top]

        if not selected:
            continue

        tx_packets = [
            BBoxPacket(
                object_id=b["object_id"],
                x1=b["bbox"][0],
                y1=b["bbox"][1],
                x2=b["bbox"][2],
                y2=b["bbox"][3],
            )
            for b in selected
        ]

        tx_bits = encode_bboxes(tx_packets)

        rx_bits, stats = transmit_bits_by_mode(
            bits=tx_bits,
            coding_mode=coding_mode,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
            codec=codec,
        )

        rx_packets = decode_bboxes(
            rx_bits,
            num_bboxes=len(tx_packets),
        )

        validation_results = [
            validate_bbox_packet(
                pkt,
                object_vocab_size=object_vocab_size,
                valid_object_ids=valid_object_ids,
            )
            for pkt in rx_packets
        ]
        validation_results_all.extend(validation_results)

        selected_valid, tx_packets_valid, rx_packets_valid = filter_valid_aligned(
            selected_units=selected,
            tx_packets=tx_packets,
            rx_packets=rx_packets,
            validation_results=validation_results,
        )

        delivered = delivered_bboxes_by_object_recovery(
            selected_bboxes=selected,
            tx_packets=tx_packets,
            rx_packets=rx_packets,
        )

        delivered_validated = delivered_bboxes_by_object_recovery(
            selected_bboxes=selected_valid,
            tx_packets=tx_packets_valid,
            rx_packets=rx_packets_valid,
        )

        before_list.append(bbox_answerability(selected, answer))
        after_list.append(bbox_answerability(delivered, answer))
        after_validated_list.append(bbox_answerability(delivered_validated, answer))

        source_bits = int(stats["source_bits"])
        coded_bits = int(stats["coded_bits"])

        source_bits_list.append(source_bits)
        coded_bits_list.append(coded_bits)
        channel_ber_list.append(stats["channel_ber"])
        decoded_ber_list.append(stats["decoded_ber"])
        ldpc_success_list.append(stats["ldpc_block_success_rate"])

        per_list.append(packet_error_rate(tx_bits, rx_bits, packet_size_bits=80))

        source_latency_list.append(
            communication_latency_sec(source_bits, bandwidth_hz, snr_db)
        )
        coded_latency_list.append(
            communication_latency_sec(coded_bits, bandwidth_hz, snr_db)
        )

        used_samples += 1

    validation_summary = summarize_validation_results(validation_results_all)
    
    coding_metadata = coding_metadata_for_mode(coding_mode, codec)
    
    out = {
        "semantic_type": "bbox",
        "ranking_method": method,
        "coding_mode": coding_mode,
        "coding_family": coding_metadata.get("coding_family", "uncoded"),
        "decoder_type": coding_metadata.get("decoder_type", "none"),
        "soft_llr_decoder": coding_metadata.get("soft_llr_decoder", False),
        "standard_ldpc_bp": coding_metadata.get("standard_ldpc_bp", False),
        "ldpc_k": coding_metadata.get("k", ""),
        "ldpc_m": coding_metadata.get("m", ""),
        "ldpc_n": coding_metadata.get("n", ""),
        "ldpc_nominal_code_rate": coding_metadata.get("nominal_code_rate", ""),
        "ldpc_col_weight": coding_metadata.get("col_weight", ""),
        "ldpc_max_iter": coding_metadata.get("max_iter", ""),        
        "channel": channel_type,
        "snr_db": snr_db,
        "n_top": n_top,
        "num_samples": used_samples,
        "avg_source_bits": mean_or_zero(source_bits_list),
        "avg_coded_bits": mean_or_zero(coded_bits_list),
        "avg_code_rate": mean_or_zero(source_bits_list) / mean_or_zero(coded_bits_list),
        "channel_ber": mean_or_zero(channel_ber_list),
        "decoded_ber": mean_or_zero(decoded_ber_list),
        "packet_error_rate": mean_or_zero(per_list),
        "ldpc_block_success_rate": mean_or_zero(ldpc_success_list),
        "answerability_before_strict": "",
        "answerability_after_strict": "",
        "answerability_after_strict_validated": "",
        "answerability_before_loose": mean_or_zero(before_list),
        "answerability_after_loose": mean_or_zero(after_list),
        "answerability_after_loose_validated": mean_or_zero(after_validated_list),
        "source_t_com_sec": mean_or_zero(source_latency_list),
        "coded_t_com_sec": mean_or_zero(coded_latency_list),
    }
    out.update(validation_summary)

    return out


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")

    seed = int(cfg["project"]["seed"])
    perfect_csi = bool(cfg["channel"]["perfect_csi"])
    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])

    max_samples = 300
    snr_db_list = [4.0, 6.0, 8.0, 10.0, 12.0]
    n_top_list = [3, 6, 9, 12]
    ranking_methods = ["go"]
    channels = ["awgn", "rayleigh"]
    coding_modes = ["uncoded", "ldpc_like"]

    data_root = Path(cfg["data"]["root"])
    ds = GQACommSubset(data_root)

    vocab_info = get_vocab_info(cfg, data_root)
    object_vocab_size = vocab_info["object_vocab_size"]
    relation_vocab_size = vocab_info["relation_vocab_size"]
    valid_object_ids = vocab_info["valid_object_ids"]
    valid_relation_ids = vocab_info["valid_relation_ids"]

    print(
        "Vocab ID ranges:",
        {
            "object_vocab_size": object_vocab_size,
            "relation_vocab_size": relation_vocab_size,
            "min_object_id": vocab_info["min_object_id"],
            "max_object_id": vocab_info["max_object_id"],
            "min_relation_id": vocab_info["min_relation_id"],
            "max_relation_id": vocab_info["max_relation_id"],
        },
    )

    samples = ds.load_samples(limit=max_samples)
    sample_by_qid = {s["question_id"]: s for s in samples}

    bbox_rows = ds.load_bbox_packets(limit=max_samples)
    sg_rows = ds.load_sg_triplets(limit=max_samples)

    all_samples_for_freq = ds.load_samples()
    object_freq = build_object_frequency(all_samples_for_freq)
    relation_freq = build_relation_frequency(all_samples_for_freq)

    codec = SystematicLDPC(
        LDPCConfig(
            k=256,
            m=256,
            col_weight=3,
            max_iter=30,
            seed=123,
        )
    )

    out_rows = []

    for coding_mode in coding_modes:
        for channel_type in channels:
            for snr_db in snr_db_list:
                for n_top in n_top_list:
                    for method in ranking_methods:
                        sg_row = run_sg(
                            sg_rows=sg_rows,
                            sample_by_qid=sample_by_qid,
                            object_freq=object_freq,
                            relation_freq=relation_freq,
                            method=method,
                            coding_mode=coding_mode,
                            channel_type=channel_type,
                            snr_db=snr_db,
                            n_top=n_top,
                            seed=seed,
                            perfect_csi=perfect_csi,
                            bandwidth_hz=bandwidth_hz,
                            codec=codec,
                            object_vocab_size=object_vocab_size,
                            relation_vocab_size=relation_vocab_size,
                            valid_object_ids=valid_object_ids,
                            valid_relation_ids=valid_relation_ids,
                        )
                        out_rows.append(sg_row)
                        print(sg_row)

                        bbox_row = run_bbox(
                            bbox_rows=bbox_rows,
                            sample_by_qid=sample_by_qid,
                            object_freq=object_freq,
                            method=method,
                            coding_mode=coding_mode,
                            channel_type=channel_type,
                            snr_db=snr_db,
                            n_top=n_top,
                            seed=seed,
                            perfect_csi=perfect_csi,
                            bandwidth_hz=bandwidth_hz,
                            codec=codec,
                            object_vocab_size=object_vocab_size,
                            valid_object_ids=valid_object_ids,
                        )
                        out_rows.append(bbox_row)
                        print(bbox_row)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "answerability_sweep_ldpc.csv"

    write_csv_union_fieldnames(output_path, out_rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()