import csv
from pathlib import Path

from src.comm.bpsk import bpsk_demodulate_hard, bpsk_modulate
from src.comm.channel import awgn_channel, rayleigh_channel
from src.comm.latency import communication_latency_sec
from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate, packet_error_rate
from src.eval.semantic_metrics import (
    bbox_exact_match_rate,
    bbox_mean_l1_error,
    bbox_object_accuracy,
    sg_field_accuracy,
    sg_triplet_exact_match_rate,
)
from src.semantic.packet_codec import (
    BBoxPacket,
    SGTripletPacket,
    decode_bboxes,
    decode_sg_triplets,
    encode_bboxes,
    encode_sg_triplets,
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


def run_sg_mode(ds, channel_type, snr_db, n_top, max_samples, seed, perfect_csi, bandwidth_hz):
    rows_raw = ds.load_sg_triplets(limit=max_samples)

    ber_list = []
    per_list = []
    triplet_match_list = []
    field_acc_list = []
    latency_list = []
    bit_list = []

    for sample_idx, row in enumerate(rows_raw):
        triplets = row["triplets"][:n_top]

        if not triplets:
            continue

        tx_packets = [
            SGTripletPacket(
                subject_id=t["subject_id"],
                relation_id=t["relation_id"],
                object_id=t["object_id"],
            )
            for t in triplets
        ]

        tx_bits = encode_sg_triplets(tx_packets)

        rx_bits = transmit_bits(
            bits=tx_bits,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
        )

        rx_packets = decode_sg_triplets(rx_bits, num_triplets=len(tx_packets))

        ber_list.append(bit_error_rate(tx_bits, rx_bits))
        per_list.append(packet_error_rate(tx_bits, rx_bits, packet_size_bits=48))
        triplet_match_list.append(sg_triplet_exact_match_rate(tx_packets, rx_packets))
        field_acc_list.append(sg_field_accuracy(tx_packets, rx_packets))
        latency_list.append(
            communication_latency_sec(
                num_bits=len(tx_bits),
                bandwidth_hz=bandwidth_hz,
                snr_db=snr_db,
            )
        )
        bit_list.append(len(tx_bits))

    return {
        "semantic_type": "sg",
        "channel": channel_type,
        "snr_db": snr_db,
        "n_top": n_top,
        "num_samples": len(ber_list),
        "avg_source_bits": sum(bit_list) / len(bit_list),
        "ber": sum(ber_list) / len(ber_list),
        "packet_error_rate": sum(per_list) / len(per_list),
        "semantic_exact_match": sum(triplet_match_list) / len(triplet_match_list),
        "semantic_field_accuracy": sum(field_acc_list) / len(field_acc_list),
        "bbox_object_accuracy": "",
        "bbox_mean_l1_error": "",
        "t_com_sec": sum(latency_list) / len(latency_list),
    }


def run_bbox_mode(ds, channel_type, snr_db, n_top, max_samples, seed, perfect_csi, bandwidth_hz):
    rows_raw = ds.load_bbox_packets(limit=max_samples)

    ber_list = []
    per_list = []
    object_acc_list = []
    bbox_l1_list = []
    bbox_exact_list = []
    latency_list = []
    bit_list = []

    for sample_idx, row in enumerate(rows_raw):
        bboxes = row["bboxes"][:n_top]

        if not bboxes:
            continue

        tx_packets = [
            BBoxPacket(
                object_id=b["object_id"],
                x1=b["bbox"][0],
                y1=b["bbox"][1],
                x2=b["bbox"][2],
                y2=b["bbox"][3],
            )
            for b in bboxes
        ]

        tx_bits = encode_bboxes(tx_packets)

        rx_bits = transmit_bits(
            bits=tx_bits,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
        )

        rx_packets = decode_bboxes(rx_bits, num_bboxes=len(tx_packets))

        ber_list.append(bit_error_rate(tx_bits, rx_bits))
        per_list.append(packet_error_rate(tx_bits, rx_bits, packet_size_bits=80))
        object_acc_list.append(bbox_object_accuracy(tx_packets, rx_packets))
        bbox_l1_list.append(bbox_mean_l1_error(tx_packets, rx_packets))
        bbox_exact_list.append(bbox_exact_match_rate(tx_packets, rx_packets))
        latency_list.append(
            communication_latency_sec(
                num_bits=len(tx_bits),
                bandwidth_hz=bandwidth_hz,
                snr_db=snr_db,
            )
        )
        bit_list.append(len(tx_bits))

    return {
        "semantic_type": "bbox",
        "channel": channel_type,
        "snr_db": snr_db,
        "n_top": n_top,
        "num_samples": len(ber_list),
        "avg_source_bits": sum(bit_list) / len(bit_list),
        "ber": sum(ber_list) / len(ber_list),
        "packet_error_rate": sum(per_list) / len(per_list),
        "semantic_exact_match": sum(bbox_exact_list) / len(bbox_exact_list),
        "semantic_field_accuracy": "",
        "bbox_object_accuracy": sum(object_acc_list) / len(object_acc_list),
        "bbox_mean_l1_error": sum(bbox_l1_list) / len(bbox_l1_list),
        "t_com_sec": sum(latency_list) / len(latency_list),
    }


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")

    seed = int(cfg["project"]["seed"])
    perfect_csi = bool(cfg["channel"]["perfect_csi"])
    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])

    snr_db = 8.0
    max_samples = 100
    n_top_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

    ds = GQACommSubset(Path(cfg["data"]["root"]))

    out_rows = []

    for channel_type in ["awgn", "rayleigh"]:
        for n_top in n_top_list:
            sg_row = run_sg_mode(
                ds=ds,
                channel_type=channel_type,
                snr_db=snr_db,
                n_top=n_top,
                max_samples=max_samples,
                seed=seed,
                perfect_csi=perfect_csi,
                bandwidth_hz=bandwidth_hz,
            )
            out_rows.append(sg_row)
            print(sg_row)

            bbox_row = run_bbox_mode(
                ds=ds,
                channel_type=channel_type,
                snr_db=snr_db,
                n_top=n_top,
                max_samples=max_samples,
                seed=seed,
                perfect_csi=perfect_csi,
                bandwidth_hz=bandwidth_hz,
            )
            out_rows.append(bbox_row)
            print(bbox_row)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ntop_sweep.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
