import numpy as np


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    BPSK mapping:
        bit 0 -> -1
        bit 1 -> +1
    """
    bits = np.asarray(bits, dtype=np.uint8)

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Input bits must contain only 0 or 1.")

    return (2.0 * bits.astype(np.float32)) - 1.0


def bpsk_demodulate_hard(received: np.ndarray) -> np.ndarray:
    """
    Hard-decision BPSK demodulation.
    """
    received = np.asarray(received)
    return (received >= 0).astype(np.uint8)
