import csv
from pathlib import Path

import numpy as np

from src.comm.bpsk import bpsk_demodulate_hard, bpsk_modulate
from src.comm.channel import awgn_channel, rayleigh_channel
from src.comm.ldpc_codec import LDPCConfig, SystematicLDPC
from src.eval.metrics import bit_error_rate


def transmit_uncoded(bits, channel_type, snr_db, seed, perfect_csi=True):
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


def run_once(channel_type, snr_db, num_bits, seed):
    rng = np.random.default_rng(seed)

    tx_bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    codec = SystematicLDPC(
        LDPCConfig(
            k=256,
            m=256,
            col_weight=3,
            max_iter=30,
            seed=123,
        )
    )

    # Uncoded path
    rx_uncoded = transmit_uncoded(
        bits=tx_bits,
        channel_type=channel_type,
        snr_db=snr_db,
        seed=seed,
    )

    uncoded_ber = bit_error_rate(tx_bits, rx_uncoded)

    # LDPC-coded path
    coded_bits, original_len = codec.encode(tx_bits)

    rx_coded_hard = transmit_uncoded(
        bits=coded_bits,
        channel_type=channel_type,
        snr_db=snr_db,
        seed=seed,
    )

    coded_channel_ber = bit_error_rate(coded_bits, rx_coded_hard)

    decoded_bits, block_success_rate = codec.decode(rx_coded_hard, original_len)

    decoded_ber = bit_error_rate(tx_bits, decoded_bits)

    return {
        "channel": channel_type,
        "snr_db": snr_db,
        "num_source_bits": len(tx_bits),
        "num_coded_bits": len(coded_bits),
        "nominal_code_rate": len(tx_bits) / len(coded_bits),
        "uncoded_ber": uncoded_ber,
        "coded_channel_ber": coded_channel_ber,
        "decoded_ber": decoded_ber,
        "ldpc_block_success_rate": block_success_rate,
    }


def main() -> None:
    snr_db_list = [-4, -2, 0, 2, 4, 6, 8, 10, 12]
    channels = ["awgn", "rayleigh"]
    num_bits = 4096
    seed = 0

    rows = []

    for channel_type in channels:
        for snr_db in snr_db_list:
            row = run_once(
                channel_type=channel_type,
                snr_db=float(snr_db),
                num_bits=num_bits,
                seed=seed,
            )
            rows.append(row)
            print(row)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ldpc_sanity.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
