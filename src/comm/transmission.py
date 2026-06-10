from typing import Any, Dict, Tuple

import numpy as np

from src.comm.bpsk import (
    bpsk_demodulate_hard,
    bpsk_llr_from_equalized,
    bpsk_modulate,
)
from src.comm.channel import (
    awgn_channel_with_metadata,
    rayleigh_channel_with_metadata,
)
from src.comm.ldpc_codec import LDPCConfig, SystematicLDPC
from src.eval.metrics import bit_error_rate


def _mean_float(x) -> float:
    arr = np.asarray(x, dtype=np.float32)
    return float(np.mean(arr))


def transmit_symbols_with_metadata(
    symbols: np.ndarray,
    channel_type: str,
    snr_db: float,
    seed: int,
    perfect_csi: bool = True,
):
    if channel_type == "awgn":
        return awgn_channel_with_metadata(
            symbols,
            snr_db=snr_db,
            seed=seed,
        )

    if channel_type == "rayleigh":
        return rayleigh_channel_with_metadata(
            symbols,
            snr_db=snr_db,
            seed=seed,
            perfect_csi=perfect_csi,
        )

    raise ValueError(f"Unknown channel_type: {channel_type}")


def channel_metadata_stats(channel_out) -> Dict[str, float]:
    stats = {
        "channel_snr_linear": float(channel_out.snr_linear),
        "channel_signal_power": float(channel_out.signal_power),
        "channel_noise_power": float(channel_out.noise_power),
        "channel_noise_variance_mean": _mean_float(channel_out.noise_variance),
        "soft_llr_available": 1.0,
    }

    if channel_out.channel_gain_power is not None:
        stats["channel_gain_power_mean"] = _mean_float(
            channel_out.channel_gain_power
        )
    else:
        stats["channel_gain_power_mean"] = 1.0

    return stats


def transmit_uncoded_bits(
    bits: np.ndarray,
    channel_type: str,
    snr_db: float,
    seed: int,
    perfect_csi: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    bits = np.asarray(bits, dtype=np.uint8)

    symbols = bpsk_modulate(bits)
    channel_out = transmit_symbols_with_metadata(
        symbols=symbols,
        channel_type=channel_type,
        snr_db=snr_db,
        seed=seed,
        perfect_csi=perfect_csi,
    )

    # Current reported path is still hard-decision demodulation.
    rx_bits = bpsk_demodulate_hard(channel_out.received)

    # LLR is computed and exposed for future soft decoder / BP work.
    llr = bpsk_llr_from_equalized(
        received_equalized=channel_out.received,
        noise_variance=channel_out.noise_variance,
    )

    stats: Dict[str, Any] = {
        "source_bits": float(len(bits)),
        "coded_bits": float(len(bits)),
        "code_rate_effective": 1.0,
        "channel_ber": bit_error_rate(bits, rx_bits),
        "decoded_ber": bit_error_rate(bits, rx_bits),
        "ldpc_block_success_rate": 1.0,
        "demod_type": "hard",
        "llr_mean_abs": float(np.mean(np.abs(llr))) if len(llr) else 0.0,
    }

    stats.update(channel_metadata_stats(channel_out))

    return rx_bits, stats


def transmit_ldpc_bits(
    bits: np.ndarray,
    channel_type: str,
    snr_db: float,
    seed: int,
    perfect_csi: bool = True,
    codec: SystematicLDPC | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    channel_out = transmit_symbols_with_metadata(
        symbols=symbols,
        channel_type=channel_type,
        snr_db=snr_db,
        seed=seed,
        perfect_csi=perfect_csi,
    )

    # Current LDPC-like decoder still uses hard-decision bits.
    rx_coded_bits = bpsk_demodulate_hard(channel_out.received)

    # LLR is computed and exposed for future BP/min-sum work.
    llr = bpsk_llr_from_equalized(
        received_equalized=channel_out.received,
        noise_variance=channel_out.noise_variance,
    )

    decoded_bits, block_success_rate = codec.decode(
        coded_bits=rx_coded_bits,
        original_len=original_len,
    )

    stats: Dict[str, Any] = {
        "source_bits": float(len(bits)),
        "coded_bits": float(len(coded_bits)),
        "code_rate_effective": float(len(bits) / len(coded_bits)),
        "channel_ber": bit_error_rate(coded_bits, rx_coded_bits),
        "decoded_ber": bit_error_rate(bits, decoded_bits),
        "ldpc_block_success_rate": float(block_success_rate),
        "demod_type": "hard",
        "llr_mean_abs": float(np.mean(np.abs(llr))) if len(llr) else 0.0,
    }

    stats.update(channel_metadata_stats(channel_out))

    return decoded_bits, stats