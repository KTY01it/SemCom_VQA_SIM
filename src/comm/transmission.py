from typing import Dict, Tuple

import numpy as np

from src.comm.bpsk import bpsk_demodulate_hard, bpsk_modulate
from src.comm.channel import awgn_channel, rayleigh_channel
from src.comm.ldpc_codec import LDPCConfig, SystematicLDPC
from src.eval.metrics import bit_error_rate


def transmit_uncoded_bits(
    bits: np.ndarray,
    channel_type: str,
    snr_db: float,
    seed: int,
    perfect_csi: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    bits = np.asarray(bits, dtype=np.uint8)

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

    rx_bits = bpsk_demodulate_hard(rx_symbols)

    stats = {
        "source_bits": float(len(bits)),
        "coded_bits": float(len(bits)),
        "code_rate_effective": 1.0,
        "channel_ber": bit_error_rate(bits, rx_bits),
        "decoded_ber": bit_error_rate(bits, rx_bits),
        "ldpc_block_success_rate": 1.0,
    }

    return rx_bits, stats


def transmit_ldpc_bits(
    bits: np.ndarray,
    channel_type: str,
    snr_db: float,
    seed: int,
    perfect_csi: bool = True,
    codec: SystematicLDPC | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    bits = np.asarray(bits, dtype=np.uint8)

    if codec is None:
        codec = SystematicLDPC(
            LDPCConfig(
                k=256,
                m=256,
                col_weight=3,
                max_iter=30,
                seed=123,
            )
        )

    coded_bits, original_len = codec.encode(bits)

    symbols = bpsk_modulate(coded_bits)

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

    rx_coded_bits = bpsk_demodulate_hard(rx_symbols)

    decoded_bits, block_success_rate = codec.decode(
        coded_bits=rx_coded_bits,
        original_len=original_len,
    )

    stats = {
        "source_bits": float(len(bits)),
        "coded_bits": float(len(coded_bits)),
        "code_rate_effective": float(len(bits) / len(coded_bits)),
        "channel_ber": bit_error_rate(coded_bits, rx_coded_bits),
        "decoded_ber": bit_error_rate(bits, decoded_bits),
        "ldpc_block_success_rate": float(block_success_rate),
    }

    return decoded_bits, stats
