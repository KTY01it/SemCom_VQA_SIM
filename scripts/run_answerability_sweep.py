import csv
from pathlib import Path
from typing import Any, Dict, List

from src.comm.bpsk import bpsk_demodulate_hard, bpsk_modulate
from src.comm.channel import awgn_channel, rayleigh_channel
from src.comm.latency import communication_latency_sec
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
from src.eval.metrics import bit_error_rate, packet_error_rate
from src.semantic.packet_codec import (
    BBoxPacket,
    SGTripletPacket,
    decode_bboxes,
    decode_sg_triplets,
    encode_bboxes,
    encode_sg_triplets,
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


def transmit_bits(bits, channel_type, snr_db, seed, perfect_csi):
    symbols = bpsk_modulate(bits)

    if channel_type == "awgn":
        rx_symbols = awgn_channel(symbols, snr_db=snr_db, seed=seed)
    elif channel_type == "rayleigh":
        rx_symbols = rayleigh_channel(
            symbols,
            snr_db=snr_db,
            seed=seed,
            perfect_csi=perfect_csi,
        )
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}")

    return bpsk_demodulate_hard(rx_symbols)


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


def run_sg_answerability(
    sg_rows,
    sample_by_qid,
    object_freq,
    relation_freq,
    method,
    channel_type,
    snr_db,
    n_top,
    seed,
    perfect_csi,
    bandwidth_hz,
):
    before_strict_list = []
    after_strict_list = []
    before_loose_list = []
    after_loose_list = []

    ber_list = []
    per_list = []
    latency_list = []
    bit_list = []

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
        rx_bits = transmit_bits(
            tx_bits,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
        )
        rx_packets = decode_sg_triplets(rx_bits, num_triplets=len(tx_packets))

        delivered = delivered_triplets_by_exact_recovery(
            selected_triplets=selected,
            tx_packets=tx_packets,
            rx_packets=rx_packets,
        )

        before_strict = sg_answerability_strict(selected, answer, keywords)
        after_strict = sg_answerability_strict(delivered, answer, keywords)

        before_loose = sg_answerability_loose(selected, answer)
        after_loose = sg_answerability_loose(delivered, answer)

        before_strict_list.append(before_strict)
        after_strict_list.append(after_strict)
        before_loose_list.append(before_loose)
        after_loose_list.append(after_loose)

        ber_list.append(bit_error_rate(tx_bits, rx_bits))
        per_list.append(packet_error_rate(tx_bits, rx_bits, packet_size_bits=48))
        latency_list.append(
            communication_latency_sec(
                num_bits=len(tx_bits),
                bandwidth_hz=bandwidth_hz,
                snr_db=snr_db,
            )
        )
        bit_list.append(len(tx_bits))

        used_samples += 1

    return {
        "semantic_type": "sg",
        "ranking_method": method,
        "channel": channel_type,
        "snr_db": snr_db,
        "n_top": n_top,
        "num_samples": used_samples,
        "avg_source_bits": sum(bit_list) / len(bit_list),
        "avg_num_units": (sum(bit_list) / len(bit_list)) / 48.0,
        "ber": sum(ber_list) / len(ber_list),
        "packet_error_rate": sum(per_list) / len(per_list),
        "answerability_before_strict": sum(before_strict_list) / len(before_strict_list),
        "answerability_after_strict": sum(after_strict_list) / len(after_strict_list),
        "answerability_before_loose": sum(before_loose_list) / len(before_loose_list),
        "answerability_after_loose": sum(after_loose_list) / len(after_loose_list),
        "t_com_sec": sum(latency_list) / len(latency_list),
    }


def run_bbox_answerability(
    bbox_rows,
    sample_by_qid,
    object_freq,
    method,
    channel_type,
    snr_db,
    n_top,
    seed,
    perfect_csi,
    bandwidth_hz,
):
    before_list = []
    after_list = []

    ber_list = []
    per_list = []
    latency_list = []
    bit_list = []

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
        rx_bits = transmit_bits(
            tx_bits,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
        )
        rx_packets = decode_bboxes(rx_bits, num_bboxes=len(tx_packets))

        delivered = delivered_bboxes_by_object_recovery(
            selected_bboxes=selected,
            tx_packets=tx_packets,
            rx_packets=rx_packets,
        )

        before = bbox_answerability(selected, answer)
        after = bbox_answerability(delivered, answer)

        before_list.append(before)
        after_list.append(after)

        ber_list.append(bit_error_rate(tx_bits, rx_bits))
        per_list.append(packet_error_rate(tx_bits, rx_bits, packet_size_bits=80))
        latency_list.append(
            communication_latency_sec(
                num_bits=len(tx_bits),
                bandwidth_hz=bandwidth_hz,
                snr_db=snr_db,
            )
        )
        bit_list.append(len(tx_bits))

        used_samples += 1

    return {
        "semantic_type": "bbox",
        "ranking_method": method,
        "channel": channel_type,
        "snr_db": snr_db,
        "n_top": n_top,
        "num_samples": used_samples,
        "avg_source_bits": sum(bit_list) / len(bit_list),
        "avg_num_units": (sum(bit_list) / len(bit_list)) / 80.0,
        "ber": sum(ber_list) / len(ber_list),
        "packet_error_rate": sum(per_list) / len(per_list),
        "answerability_before_strict": "",
        "answerability_after_strict": "",
        "answerability_before_loose": sum(before_list) / len(before_list),
        "answerability_after_loose": sum(after_list) / len(after_list),
        "t_com_sec": sum(latency_list) / len(latency_list),
    }


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")

    seed = int(cfg["project"]["seed"])
    perfect_csi = bool(cfg["channel"]["perfect_csi"])
    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])

    max_samples = 500
    snr_db = 8.0
    n_top_list = [3, 6, 9, 12, 15, 18]
    ranking_methods = ["original", "do", "go"]
    channels = ["awgn", "rayleigh"]

    ds = GQACommSubset(Path(cfg["data"]["root"]))

    samples = ds.load_samples(limit=max_samples)
    sample_by_qid = {s["question_id"]: s for s in samples}

    bbox_rows = ds.load_bbox_packets(limit=max_samples)
    sg_rows = ds.load_sg_triplets(limit=max_samples)

    all_samples_for_freq = ds.load_samples()
    object_freq = build_object_frequency(all_samples_for_freq)
    relation_freq = build_relation_frequency(all_samples_for_freq)

    out_rows = []

    for channel_type in channels:
        for n_top in n_top_list:
            for method in ranking_methods:
                sg_row = run_sg_answerability(
                    sg_rows=sg_rows,
                    sample_by_qid=sample_by_qid,
                    object_freq=object_freq,
                    relation_freq=relation_freq,
                    method=method,
                    channel_type=channel_type,
                    snr_db=snr_db,
                    n_top=n_top,
                    seed=seed,
                    perfect_csi=perfect_csi,
                    bandwidth_hz=bandwidth_hz,
                )
                out_rows.append(sg_row)
                print(sg_row)

                bbox_row = run_bbox_answerability(
                    bbox_rows=bbox_rows,
                    sample_by_qid=sample_by_qid,
                    object_freq=object_freq,
                    method=method,
                    channel_type=channel_type,
                    snr_db=snr_db,
                    n_top=n_top,
                    seed=seed,
                    perfect_csi=perfect_csi,
                    bandwidth_hz=bandwidth_hz,
                )
                out_rows.append(bbox_row)
                print(bbox_row)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "answerability_sweep.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
