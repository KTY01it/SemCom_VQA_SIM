import numpy as np


def shannon_like_rate_bps(bandwidth_hz: float, snr_db: float) -> float:
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(bandwidth_hz * np.log2(1.0 + snr_linear))


def communication_latency_sec(
    num_bits: int,
    bandwidth_hz: float,
    snr_db: float,
) -> float:
    rate = shannon_like_rate_bps(bandwidth_hz, snr_db)

    if rate <= 0:
        raise ValueError(f"Invalid communication rate: {rate}")

    return float(num_bits / rate)

def communication_latency_from_bits_sec(
    num_bits: int,
    bandwidth_hz: float,
    snr_db: float,
) -> float:
    return communication_latency_sec(
        num_bits=num_bits,
        bandwidth_hz=bandwidth_hz,
        snr_db=snr_db,
    )