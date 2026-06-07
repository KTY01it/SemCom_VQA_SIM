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
)
from src.semantic.packet_codec import (
    BBoxPacket,
    decode_bboxes,
    encode_bboxes,
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


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")

    seed = int(cfg["project"]["seed"])
    snr_db_list = cfg["channel"]["snr_db_list"]
    perfect_csi = bool(cfg["channel"]["perfect_csi"])
    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])

    n_top = 9
    max_samples = 100

    ds = GQACommSubset(Path(cfg["data"]["root"]))
    rows_raw = ds.load_bbox_packets(limit=max_samples)

    out_rows = []

    for channel_type in ["awgn", "rayleigh"]:
        for snr_db in snr_db_list:
            ber_list = []
            per_list = []
            object_acc_list = []
            bbox_l1_list = []
            bbox_exact_list = []
            latency_list = []

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

                ber = bit_error_rate(tx_bits, rx_bits)
                per = packet_error_rate(tx_bits, rx_bits, packet_size_bits=80)
                object_acc = bbox_object_accuracy(tx_packets, rx_packets)
                bbox_l1 = bbox_mean_l1_error(tx_packets, rx_packets)
                bbox_exact = bbox_exact_match_rate(tx_packets, rx_packets)
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
                latency_list.append(t_com)

            out = {
                "channel": channel_type,
                "snr_db": float(snr_db),
                "n_top": n_top,
                "num_samples": len(ber_list),
                "avg_source_bits": n_top * 80,
                "ber": sum(ber_list) / len(ber_list),
                "packet_error_rate": sum(per_list) / len(per_list),
                "bbox_object_accuracy": sum(object_acc_list) / len(object_acc_list),
                "bbox_mean_l1_error": sum(bbox_l1_list) / len(bbox_l1_list),
                "bbox_exact_match": sum(bbox_exact_list) / len(bbox_exact_list),
                "t_com_sec": sum(latency_list) / len(latency_list),
            }

            out_rows.append(out)
            print(out)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bbox_packet_sanity.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
