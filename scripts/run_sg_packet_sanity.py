import csv
import json
from pathlib import Path

from src.comm.bpsk import bpsk_demodulate_hard, bpsk_modulate
from src.comm.channel import awgn_channel, rayleigh_channel
from src.comm.latency import communication_latency_sec
from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate, packet_error_rate
from src.eval.semantic_metrics import (
    sg_field_accuracy,
    sg_field_accuracy_with_drop,
    sg_triplet_exact_match_rate,
    sg_triplet_exact_match_rate_with_drop,
)
from src.semantic.packet_codec import (
    SGTripletPacket,
    decode_sg_triplets,
    encode_sg_triplets,
)
from src.semantic.packet_validation import (
    summarize_validation_results,
    validate_sg_packet,
    vocab_to_valid_id_set,
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


def get_vocab_info(cfg, data_root: Path) -> dict:
    object_vocab_name = cfg["data"].get(
        "object_vocab_file",
        cfg["data"].get("object_vocab", "object_vocab.json"),
    )
    relation_vocab_name = cfg["data"].get(
        "relation_vocab_file",
        cfg["data"].get("relation_vocab", "relation_vocab.json"),
    )

    object_vocab = load_json(data_root / object_vocab_name)
    relation_vocab = load_json(data_root / relation_vocab_name)

    valid_object_ids = vocab_to_valid_id_set(object_vocab)
    valid_relation_ids = vocab_to_valid_id_set(relation_vocab)

    return {
        "object_vocab_size": len(object_vocab),
        "relation_vocab_size": len(relation_vocab),
        "valid_object_ids": valid_object_ids,
        "valid_relation_ids": valid_relation_ids,
        "min_object_id": min(valid_object_ids) if valid_object_ids else None,
        "max_object_id": max(valid_object_ids) if valid_object_ids else None,
        "min_relation_id": min(valid_relation_ids) if valid_relation_ids else None,
        "max_relation_id": max(valid_relation_ids) if valid_relation_ids else None,
    }


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
    rows_raw = ds.load_sg_triplets(limit=max_samples)

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

    out_rows = []

    for channel_type in ["awgn", "rayleigh"]:
        for snr_db in snr_db_list:
            ber_list = []
            per_list = []
            triplet_match_list = []
            field_acc_list = []
            triplet_match_validated_list = []
            field_acc_validated_list = []
            latency_list = []
            source_bits_list = []

            validation_results_all = []

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
                    snr_db=float(snr_db),
                    seed=seed + sample_idx,
                    perfect_csi=perfect_csi,
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

                rx_packets_validated = [
                    pkt if status.valid else None
                    for pkt, status in zip(rx_packets, validation_results)
                ]

                ber = bit_error_rate(tx_bits, rx_bits)
                per = packet_error_rate(tx_bits, rx_bits, packet_size_bits=48)
                triplet_match = sg_triplet_exact_match_rate(tx_packets, rx_packets)
                field_acc = sg_field_accuracy(tx_packets, rx_packets)
                triplet_match_validated = sg_triplet_exact_match_rate_with_drop(
                    tx_packets,
                    rx_packets_validated,
                )
                field_acc_validated = sg_field_accuracy_with_drop(
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
                triplet_match_list.append(triplet_match)
                field_acc_list.append(field_acc)
                triplet_match_validated_list.append(triplet_match_validated)
                field_acc_validated_list.append(field_acc_validated)
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
                "sg_triplet_exact_match": mean_or_zero(triplet_match_list),
                "sg_field_accuracy": mean_or_zero(field_acc_list),
                "sg_triplet_exact_match_validated": mean_or_zero(triplet_match_validated_list),
                "sg_field_accuracy_validated": mean_or_zero(field_acc_validated_list),
                "t_com_sec": mean_or_zero(latency_list),
            }
            out.update(validation_summary)

            out_rows.append(out)
            print(out)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sg_packet_sanity.csv"

    write_csv_union_fieldnames(output_path, out_rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()