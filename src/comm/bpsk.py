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

    With mapping:
        bit 0 -> -1
        bit 1 -> +1

    Decision:
        received >= 0 -> bit 1
        received < 0  -> bit 0
    """
    received = np.asarray(received)
    return (received >= 0).astype(np.uint8)


def bpsk_llr_from_equalized(
    received_equalized: np.ndarray,
    noise_variance: float | np.ndarray,
) -> np.ndarray:
    """
    Compute BPSK log-likelihood ratio after channel equalization.

    Mapping:
        bit 0 -> -1
        bit 1 -> +1

    LLR definition:
        LLR = log P(bit=1 | y) / P(bit=0 | y)

    For real AWGN:
        y = x + n, n ~ N(0, sigma^2)

        LLR = 2y / sigma^2

    For Rayleigh with perfect CSI, caller should provide equalized symbols and
    the post-equalization real noise variance.

    Positive LLR favors bit 1.
    Negative LLR favors bit 0.
    """
    y = np.asarray(received_equalized, dtype=np.float32)
    var = np.asarray(noise_variance, dtype=np.float32)

    eps = np.float32(1e-12)
    return (2.0 * y / np.maximum(var, eps)).astype(np.float32)


def bpsk_demodulate_llr_hard(llr: np.ndarray) -> np.ndarray:
    """
    Convert BPSK LLR to hard bits.

    Positive LLR -> bit 1.
    Negative LLR -> bit 0.
    """
    llr = np.asarray(llr, dtype=np.float32)
    return (llr >= 0).astype(np.uint8)