from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelOutput:
    """
    Channel output with metadata for future soft decoding.

    received:
        Real-valued symbols used by the current hard demodulator.
        For Rayleigh with perfect_csi=True, this is the equalized real signal.

    raw_received:
        Raw channel output before equalization.
        AWGN: real-valued y
        Rayleigh: complex-valued y

    noise_variance:
        Real-domain noise variance corresponding to `received`.
        AWGN:
            scalar sigma^2
        Rayleigh with perfect CSI:
            per-symbol sigma_eq^2 after equalization

    noise_power:
        E[|n|^2] convention used to generate channel noise.

    h:
        Rayleigh fading coefficient.
        None for AWGN.

    channel_gain_power:
        |h|^2 for Rayleigh.
        1 for AWGN.
    """

    received: np.ndarray
    raw_received: np.ndarray
    noise_variance: float | np.ndarray
    noise_power: float
    signal_power: float
    snr_linear: float
    h: np.ndarray | None = None
    channel_gain_power: float | np.ndarray | None = None
    perfect_csi: bool = True


def awgn_channel_with_metadata(
    symbols: np.ndarray,
    snr_db: float,
    seed: int = 0,
) -> ChannelOutput:
    """
    Real AWGN channel for BPSK symbols.

    BPSK:
        bit 0 -> -1
        bit 1 -> +1

    Noise convention:
        signal_power = E[x^2]
        noise_power = signal_power / snr_linear
        real noise variance = noise_power / 2

    This preserves the previous implementation behavior.
    """
    rng = np.random.default_rng(seed)

    symbols = np.asarray(symbols, dtype=np.float32)
    snr_linear = 10.0 ** (snr_db / 10.0)

    signal_power = float(np.mean(np.abs(symbols) ** 2))
    noise_power = signal_power / snr_linear
    noise_variance = noise_power / 2.0

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_variance),
        size=symbols.shape,
    ).astype(np.float32)

    y = symbols + noise

    return ChannelOutput(
        received=y.astype(np.float32),
        raw_received=y.astype(np.float32),
        noise_variance=float(noise_variance),
        noise_power=float(noise_power),
        signal_power=float(signal_power),
        snr_linear=float(snr_linear),
        h=None,
        channel_gain_power=1.0,
        perfect_csi=True,
    )


def awgn_channel(symbols: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """
    Backward-compatible AWGN function.
    """
    return awgn_channel_with_metadata(symbols, snr_db=snr_db, seed=seed).received


def rayleigh_channel_with_metadata(
    symbols: np.ndarray,
    snr_db: float,
    seed: int = 0,
    perfect_csi: bool = True,
) -> ChannelOutput:
    """
    Flat Rayleigh fading channel:
        y = h * x + n

    If perfect_csi=True, equalize by h before demodulation:
        y_eq = y / h = x + n / h

    The returned noise_variance corresponds to the real-valued signal used by
    the hard demodulator:
        perfect_csi=True:
            sigma_eq^2 = noise_power / (2 |h|^2)
        perfect_csi=False:
            sigma^2 = noise_power / 2
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

    eps = 1e-12
    gain_power = np.abs(h) ** 2

    if perfect_csi:
        y_used = y / (h + eps)
        received = np.real(y_used).astype(np.float32)
        noise_variance = (noise_power / (2.0 * np.maximum(gain_power, eps))).astype(
            np.float32
        )
    else:
        received = np.real(y).astype(np.float32)
        noise_variance = float(noise_power / 2.0)

    return ChannelOutput(
        received=received,
        raw_received=y,
        noise_variance=noise_variance,
        noise_power=float(noise_power),
        signal_power=float(signal_power),
        snr_linear=float(snr_linear),
        h=h,
        channel_gain_power=gain_power.astype(np.float32),
        perfect_csi=bool(perfect_csi),
    )


def rayleigh_channel(
    symbols: np.ndarray,
    snr_db: float,
    seed: int = 0,
    perfect_csi: bool = True,
) -> np.ndarray:
    """
    Backward-compatible Rayleigh function.
    """
    return rayleigh_channel_with_metadata(
        symbols,
        snr_db=snr_db,
        seed=seed,
        perfect_csi=perfect_csi,
    ).received