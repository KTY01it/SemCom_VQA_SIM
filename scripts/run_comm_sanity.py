import csv
from pathlib import Path

import numpy as np

from src.comm.bpsk import bpsk_demodulate_hard, bpsk_modulate
from src.comm.channel import awgn_channel, rayleigh_channel
from src.comm.latency import communication_latency_sec
from src.eval.metrics import bit_error_rate, packet_error_rate
from src.utils.config import load_yaml


def run_once(
    channel_type: str,
    snr_db: float,
    num_bits: int,
    packet_size_bits: int,
    bandwidth_hz: float,
    seed: int,
    perfect_csi: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    tx_bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    symbols = bpsk_modulate(tx_bits)

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

    rx_bits = bpsk_demodulate_hard(rx_symbols)

    ber = bit_error_rate(tx_bits, rx_bits)
    per = packet_error_rate(tx_bits, rx_bits, packet_size_bits)
    t_com = communication_latency_sec(num_bits, bandwidth_hz, snr_db)

    return {
        "channel": channel_type,
        "snr_db": snr_db,
        "num_bits": num_bits,
        "packet_size_bits": packet_size_bits,
        "ber": ber,
        "packet_error_rate": per,
        "t_com_sec": t_com,
    }


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")

    seed = int(cfg["project"]["seed"])
    num_bits = int(cfg["source"]["num_bits"])
    packet_size_bits = int(cfg["source"]["packet_size_bits"])
    snr_db_list = cfg["channel"]["snr_db_list"]
    perfect_csi = bool(cfg["channel"]["perfect_csi"])
    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comm_sanity.csv"

    rows = []
    for channel_type in ["awgn", "rayleigh"]:
        for snr_db in snr_db_list:
            row = run_once(
                channel_type=channel_type,
                snr_db=float(snr_db),
                num_bits=num_bits,
                packet_size_bits=packet_size_bits,
                bandwidth_hz=bandwidth_hz,
                seed=seed,
                perfect_csi=perfect_csi,
            )
            rows.append(row)
            print(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
