import numpy as np


def bit_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    tx_bits = np.asarray(tx_bits, dtype=np.uint8)
    rx_bits = np.asarray(rx_bits, dtype=np.uint8)

    if tx_bits.shape != rx_bits.shape:
        raise ValueError(f"Shape mismatch: {tx_bits.shape} vs {rx_bits.shape}")

    return float(np.mean(tx_bits != rx_bits))


def packet_error_rate(
    tx_bits: np.ndarray,
    rx_bits: np.ndarray,
    packet_size_bits: int,
) -> float:
    tx_bits = np.asarray(tx_bits, dtype=np.uint8)
    rx_bits = np.asarray(rx_bits, dtype=np.uint8)

    if tx_bits.shape != rx_bits.shape:
        raise ValueError(f"Shape mismatch: {tx_bits.shape} vs {rx_bits.shape}")

    if packet_size_bits <= 0:
        raise ValueError("packet_size_bits must be positive.")

    n_packets = len(tx_bits) // packet_size_bits

    if n_packets == 0:
        raise ValueError(
            f"Not enough bits for one packet: num_bits={len(tx_bits)}, "
            f"packet_size_bits={packet_size_bits}"
        )

    usable = n_packets * packet_size_bits

    tx = tx_bits[:usable].reshape(n_packets, packet_size_bits)
    rx = rx_bits[:usable].reshape(n_packets, packet_size_bits)

    packet_error = np.any(tx != rx, axis=1)
    return float(np.mean(packet_error))
