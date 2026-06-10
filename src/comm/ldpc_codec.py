from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LDPCConfig:
    k: int = 256
    m: int = 256
    col_weight: int = 3
    max_iter: int = 30
    seed: int = 0


class SystematicLDPC:
    """
    Simple systematic sparse parity-check code.

    Message:
        u, length k

    Parity:
        p = u @ P mod 2, length m

    Codeword:
        c = [u, p], length n = k + m

    Parity-check:
        H = [P^T | I_m]
        H c^T = 0
    """

    def __init__(self, cfg: LDPCConfig):
        self.cfg = cfg
        self.k = int(cfg.k)
        self.m = int(cfg.m)
        self.n = self.k + self.m
        self.max_iter = int(cfg.max_iter)

        if self.k <= 0 or self.m <= 0:
            raise ValueError("k and m must be positive.")

        if cfg.col_weight <= 0 or cfg.col_weight > self.m:
            raise ValueError("col_weight must be in [1, m].")

        self.P = self._make_sparse_p(
            k=self.k,
            m=self.m,
            col_weight=int(cfg.col_weight),
            seed=int(cfg.seed),
        )

        self.H = self._make_h(self.P)

        self.var_to_checks = [
            np.flatnonzero(self.H[:, j]).astype(np.int32)
            for j in range(self.n)
        ]

    @staticmethod
    def _make_sparse_p(k: int, m: int, col_weight: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        P = np.zeros((k, m), dtype=np.uint8)

        for i in range(k):
            cols = rng.choice(m, size=col_weight, replace=False)
            P[i, cols] = 1

        return P

    @staticmethod
    def _make_h(P: np.ndarray) -> np.ndarray:
        k, m = P.shape
        left = P.T.astype(np.uint8)
        right = np.eye(m, dtype=np.uint8)
        return np.concatenate([left, right], axis=1)

    def encode_block(self, msg_bits: np.ndarray) -> np.ndarray:
        msg_bits = np.asarray(msg_bits, dtype=np.uint8)

        if msg_bits.shape != (self.k,):
            raise ValueError(f"Expected message block shape {(self.k,)}, got {msg_bits.shape}")

        parity = (msg_bits @ self.P) % 2
        codeword = np.concatenate([msg_bits, parity.astype(np.uint8)])

        return codeword.astype(np.uint8)

    def syndrome(self, code_bits: np.ndarray) -> np.ndarray:
        code_bits = np.asarray(code_bits, dtype=np.uint8)

        if code_bits.shape != (self.n,):
            raise ValueError(f"Expected code block shape {(self.n,)}, got {code_bits.shape}")

        return (self.H @ code_bits) % 2

    def is_codeword(self, code_bits: np.ndarray) -> bool:
        return bool(np.all(self.syndrome(code_bits) == 0))

    def metadata(self) -> dict:
        """
        Return coding metadata for reports and CSV outputs.

        This code is LDPC-like because it uses a sparse parity-check
        construction and iterative hard-decision bit flipping. It is not
        a standard LDPC BP/min-sum decoder with soft LLR inputs.
        """
        return {
            "coding_family": "ldpc_like_sparse_systematic",
            "decoder_type": "hard_bit_flipping",
            "soft_llr_decoder": False,
            "standard_ldpc_bp": False,
            "k": self.k,
            "m": self.m,
            "n": self.n,
            "nominal_code_rate": self.k / self.n,
            "col_weight": int(self.cfg.col_weight),
            "max_iter": int(self.cfg.max_iter),
            "seed": int(self.cfg.seed),
        }
        
    def decode_block_bitflip(self, rx_bits: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """
        Hard-decision bit-flipping decoder.

        Returns:
            decoded_message_bits, success, num_iterations
        """
        x = np.asarray(rx_bits, dtype=np.uint8).copy()

        if x.shape != (self.n,):
            raise ValueError(f"Expected rx block shape {(self.n,)}, got {x.shape}")

        for it in range(self.max_iter + 1):
            syn = self.syndrome(x)

            if np.all(syn == 0):
                return x[: self.k].copy(), True, it

            unsatisfied = syn.astype(np.int32)

            scores = np.zeros(self.n, dtype=np.int32)
            for j in range(self.n):
                checks = self.var_to_checks[j]
                if len(checks) > 0:
                    scores[j] = int(np.sum(unsatisfied[checks]))

            max_score = int(scores.max())

            if max_score <= 0:
                break

            candidate_idxs = np.flatnonzero(scores == max_score)

            # Flip all bits that participate in the largest number of unsatisfied checks.
            x[candidate_idxs] ^= 1

        return x[: self.k].copy(), False, self.max_iter

    def encode(self, bits: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Encode arbitrary-length bitstream by zero padding to multiples of k.

        Returns:
            coded_bits, original_length
        """
        bits = np.asarray(bits, dtype=np.uint8)

        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("Input bits must be binary.")

        original_len = len(bits)
        n_blocks = int(np.ceil(original_len / self.k))

        padded_len = n_blocks * self.k
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:original_len] = bits

        coded_blocks = []
        for i in range(n_blocks):
            start = i * self.k
            end = start + self.k
            coded_blocks.append(self.encode_block(padded[start:end]))

        return np.concatenate(coded_blocks).astype(np.uint8), original_len

    def decode(self, coded_bits: np.ndarray, original_len: int) -> Tuple[np.ndarray, float]:
        """
        Decode coded bitstream.

        Returns:
            decoded_source_bits trimmed to original_len,
            block_success_rate
        """
        coded_bits = np.asarray(coded_bits, dtype=np.uint8)

        if len(coded_bits) % self.n != 0:
            raise ValueError(
                f"Coded bit length must be multiple of n={self.n}, got {len(coded_bits)}"
            )

        n_blocks = len(coded_bits) // self.n

        decoded_blocks = []
        success_count = 0

        for i in range(n_blocks):
            start = i * self.n
            end = start + self.n

            msg, success, _ = self.decode_block_bitflip(coded_bits[start:end])

            decoded_blocks.append(msg)
            success_count += int(success)

        decoded = np.concatenate(decoded_blocks).astype(np.uint8)
        decoded = decoded[:original_len]

        return decoded, success_count / n_blocks
