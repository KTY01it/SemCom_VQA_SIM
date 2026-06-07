import numpy as np

from src.comm.ldpc_codec import LDPCConfig, SystematicLDPC


def main() -> None:
    rng = np.random.default_rng(0)

    codec = SystematicLDPC(
        LDPCConfig(
            k=256,
            m=256,
            col_weight=3,
            max_iter=30,
            seed=0,
        )
    )

    tx_bits = rng.integers(0, 2, size=1000, dtype=np.uint8)

    coded_bits, original_len = codec.encode(tx_bits)
    decoded_bits, block_success_rate = codec.decode(coded_bits, original_len)

    print("tx_bits:", len(tx_bits))
    print("coded_bits:", len(coded_bits))
    print("code_rate:", len(tx_bits) / len(coded_bits))
    print("original_len:", original_len)
    print("block_success_rate:", block_success_rate)
    print("exact_match:", bool(np.array_equal(tx_bits, decoded_bits)))

    if not np.array_equal(tx_bits, decoded_bits):
        ber = np.mean(tx_bits != decoded_bits)
        print("decode_ber:", ber)


if __name__ == "__main__":
    main()
