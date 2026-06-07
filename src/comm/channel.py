import numpy as np


def awgn_channel(symbols: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """
    Real AWGN channel for BPSK symbols.

    BPSK:
        bit 0 -> -1
        bit 1 -> +1

    Noise variance follows an Eb/N0-style convention:
        sigma^2 = signal_power / (2 * snr_linear)
    """
    rng = np.random.default_rng(seed)

    symbols = np.asarray(symbols, dtype=np.float32)
    snr_linear = 10.0 ** (snr_db / 10.0)

    signal_power = float(np.mean(np.abs(symbols) ** 2))
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power / 2.0),
        size=symbols.shape,
    ).astype(np.float32)

    return symbols + noise


def rayleigh_channel(
    symbols: np.ndarray,
    snr_db: float,
    seed: int = 0,
    perfect_csi: bool = True,
) -> np.ndarray:
    """
    Flat Rayleigh fading channel:
        y = h * x + n

    If perfect_csi=True, equalize by h before hard demodulation.
    """
    rng = np.random.default_rng(seed)

    symbols = np.asarray(symbols, dtype=np.float32)
    x = symbols.astype(np.complex64)

    h = (
        rng.normal(0.0, 1.0 / np.sqrt(2.0), size=x.shape)
        + 1j * rng.normal(0.0, 1.0 / np.sqrt(2.0), size=x.shape)
    ).astype(np.complex64)

    snr_linear = 10.0 ** (snr_db / 10.0)

    signal_power = float(np.mean(np.abs(h * x) ** 2))
    noise_power = signal_power / snr_linear

    noise = (
        rng.normal(0.0, np.sqrt(noise_power / 2.0), size=x.shape)
        + 1j * rng.normal(0.0, np.sqrt(noise_power / 2.0), size=x.shape)
    ).astype(np.complex64)

    y = h * x + noise

    if perfect_csi:
        eps = 1e-12
        y = y / (h + eps)

    return np.real(y).astype(np.float32)
