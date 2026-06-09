import csv
import json
from pathlib import Path

from src.comm.bpsk import bpsk_demodulate_hard, bpsk_modulate
from src.comm.channel import awgn_channel, rayleigh_channel
from src.comm.latency import communication_latency_sec
from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate, packet_error_rate
from src.eval.semantic_metrics import (
    bbox_exact_match_rate,
    bbox_exact_match_rate_with_drop,
    bbox_mean_l1_error,
    bbox_mean_l1_error_with_drop,
    bbox_object_accuracy,
    bbox_object_accuracy_with_drop,
)
from src.semantic.packet_codec import (
    BBoxPacket,
    decode_bboxes,
    encode_bboxes,
)
from src.semantic.packet_validation import (
    summarize_validation_results,
    validate_bbox_packet,
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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_object_vocab_size(cfg, data_root: Path) -> int:
    vocab_name = cfg["data"].get("object_vocab", "object_vocab.json")
    object_vocab = load_json(data_root / vocab_name)
    return len(object_vocab)


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


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")

    seed = int(cfg["project"]["seed"])
    snr_db_list = cfg["channel"]["snr_db_list"]
    perfect_csi = bool(cfg["channel"]["perfect_csi"])
    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])

    n_top = 9
    max_samples = 100

    data_root = Path(cfg["data"]["root"])
    ds = GQACommSubset(data_root)
    rows_raw = ds.load_bbox_packets(limit=max_samples)

    object_vocab_size = get_object_vocab_size(cfg, data_root)

    out_rows = []

    for channel_type in ["awgn", "rayleigh"]:
        for snr_db in snr_db_list:
            ber_list = []
            per_list = []
            object_acc_list = []
            bbox_l1_list = []
            bbox_exact_list = []
            object_acc_validated_list = []
            bbox_l1_validated_list = []
            bbox_exact_validated_list = []
            latency_list = []
            source_bits_list = []

            validation_results_all = []

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
                    snr_db=float(snr_db),
                    seed=seed + sample_idx,
                    perfect_csi=perfect_csi,
                )

                rx_packets = decode_bboxes(
                    rx_bits,
                    num_bboxes=len(tx_packets),
                )

                validation_results = [
                    validate_bbox_packet(
                        pkt,
                        object_vocab_size=object_vocab_size,
                    )
                    for pkt in rx_packets
                ]
                validation_results_all.extend(validation_results)
                rx_packets_validated = [
                    pkt if status.valid else None
                    for pkt, status in zip(rx_packets, validation_results)
                ]
                
                ber = bit_error_rate(tx_bits, rx_bits)
                per = packet_error_rate(tx_bits, rx_bits, packet_size_bits=80)
                object_acc = bbox_object_accuracy(tx_packets, rx_packets)
                bbox_l1 = bbox_mean_l1_error(tx_packets, rx_packets)
                bbox_exact = bbox_exact_match_rate(tx_packets, rx_packets)

                object_acc_validated = bbox_object_accuracy_with_drop(
                    tx_packets,
                    rx_packets_validated,
                )
                bbox_l1_validated = bbox_mean_l1_error_with_drop(
                    tx_packets,
                    rx_packets_validated,
                )
                bbox_exact_validated = bbox_exact_match_rate_with_drop(
                    tx_packets,
                    rx_packets_validated,
                )                
                
                t_com = communication_latency_sec(
                    num_bits=len(tx_bits),
                    bandwidth_hz=bandwidth_hz,
                    snr_db=float(snr_db),
                )

                ber_list.append(ber)
                per_list.append(per)
                object_acc_list.append(object_acc)
                bbox_l1_list.append(bbox_l1)
                bbox_exact_list.append(bbox_exact)
                object_acc_validated_list.append(object_acc_validated)
                bbox_l1_validated_list.append(bbox_l1_validated)
                bbox_exact_validated_list.append(bbox_exact_validated)
                latency_list.append(t_com)
                source_bits_list.append(len(tx_bits))

            validation_summary = summarize_validation_results(validation_results_all)

            out = {
                "channel": channel_type,
                "snr_db": float(snr_db),
                "n_top": n_top,
                "num_samples": len(ber_list),
                "avg_source_bits": mean_or_zero(source_bits_list),
                "ber": mean_or_zero(ber_list),
                "packet_error_rate": mean_or_zero(per_list),
                "bbox_object_accuracy": mean_or_zero(object_acc_list),
                "bbox_mean_l1_error": mean_or_zero(bbox_l1_list),
                "bbox_exact_match": mean_or_zero(bbox_exact_list),
                "bbox_object_accuracy_validated": mean_or_zero(object_acc_validated_list),
                "bbox_mean_l1_error_validated": mean_or_zero(bbox_l1_validated_list),
                "bbox_exact_match_validated": mean_or_zero(bbox_exact_validated_list),
                "t_com_sec": mean_or_zero(latency_list),
            }
            
            out.update(validation_summary)

            out_rows.append(out)
            print(out)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bbox_packet_sanity.csv"

    write_csv_union_fieldnames(output_path, out_rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()