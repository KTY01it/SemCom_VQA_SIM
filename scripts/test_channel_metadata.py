import numpy as np

from src.comm.bpsk import (
    bpsk_demodulate_hard,
    bpsk_demodulate_llr_hard,
    bpsk_llr_from_equalized,
    bpsk_modulate,
)
from src.comm.channel import awgn_channel_with_metadata, rayleigh_channel_with_metadata
from src.comm.transmission import transmit_ldpc_bits, transmit_uncoded_bits


def test_awgn_metadata():
    bits = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    symbols = bpsk_modulate(bits)

    out = awgn_channel_with_metadata(symbols, snr_db=10.0, seed=0)

    assert out.received.shape == symbols.shape
    assert out.raw_received.shape == symbols.shape
    assert out.noise_power > 0
    assert float(np.mean(out.noise_variance)) > 0
    assert out.h is None

    llr = bpsk_llr_from_equalized(out.received, out.noise_variance)
    hard_from_llr = bpsk_demodulate_llr_hard(llr)
    hard_direct = bpsk_demodulate_hard(out.received)

    assert np.array_equal(hard_from_llr, hard_direct)


def test_rayleigh_metadata():
    bits = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    symbols = bpsk_modulate(bits)

    out = rayleigh_channel_with_metadata(
        symbols,
        snr_db=10.0,
        seed=0,
        perfect_csi=True,
    )

    assert out.received.shape == symbols.shape
    assert out.raw_received.shape == symbols.shape
    assert out.h is not None
    assert out.h.shape == symbols.shape
    assert out.noise_power > 0
    assert np.asarray(out.noise_variance).shape == symbols.shape

    llr = bpsk_llr_from_equalized(out.received, out.noise_variance)
    hard_from_llr = bpsk_demodulate_llr_hard(llr)
    hard_direct = bpsk_demodulate_hard(out.received)

    assert np.array_equal(hard_from_llr, hard_direct)


def test_transmission_stats():
    bits = np.random.default_rng(0).integers(0, 2, size=512, dtype=np.uint8)

    rx_uncoded, stats_uncoded = transmit_uncoded_bits(
        bits=bits,
        channel_type="awgn",
        snr_db=8.0,
        seed=0,
        perfect_csi=True,
    )

    assert len(rx_uncoded) == len(bits)
    assert "channel_noise_power" in stats_uncoded
    assert "channel_noise_variance_mean" in stats_uncoded
    assert "channel_gain_power_mean" in stats_uncoded
    assert "llr_mean_abs" in stats_uncoded
    assert stats_uncoded["soft_llr_available"] == 1.0

    rx_ldpc, stats_ldpc = transmit_ldpc_bits(
        bits=bits,
        channel_type="rayleigh",
        snr_db=8.0,
        seed=0,
        perfect_csi=True,
    )

    assert len(rx_ldpc) == len(bits)
    assert "channel_noise_power" in stats_ldpc
    assert "channel_noise_variance_mean" in stats_ldpc
    assert "channel_gain_power_mean" in stats_ldpc
    assert "llr_mean_abs" in stats_ldpc
    assert stats_ldpc["soft_llr_available"] == 1.0


def main():
    test_awgn_metadata()
    test_rayleigh_metadata()
    test_transmission_stats()
    print("channel metadata tests: PASS")


if __name__ == "__main__":
    main()
