from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


FIELD_BITS = 16


def int_to_bits(x: int, width: int = FIELD_BITS) -> List[int]:
    x = int(x)
    if x < 0:
        x = 0
    if x > 65535:
        x = 65535
    return [(x >> i) & 1 for i in reversed(range(width))]


def bits_to_int(bits: List[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def encode_compressed_sg_triplets(
    selected_triplets: List[Dict[str, Any]],
    keep_masks: List[Dict[str, bool]],
) -> np.ndarray:
    if len(selected_triplets) != len(keep_masks):
        raise ValueError("selected_triplets and keep_masks length mismatch.")

    bits: List[int] = []

    for t, keep in zip(selected_triplets, keep_masks):
        keep_subject = bool(keep["subject"])
        keep_relation = bool(keep["relation"])
        keep_object = bool(keep["object"])

        if not (keep_subject or keep_relation or keep_object):
            keep_object = True

        bits.extend([int(keep_subject), int(keep_relation), int(keep_object)])

        if keep_subject:
            bits.extend(int_to_bits(int(t["subject_id"])))
        if keep_relation:
            bits.extend(int_to_bits(int(t["relation_id"])))
        if keep_object:
            bits.extend(int_to_bits(int(t["object_id"])))

    return np.asarray(bits, dtype=np.uint8)


def decode_compressed_sg_triplets(bits: np.ndarray, num_triplets: int) -> List[Dict[str, int | None]]:
    bits = list(map(int, np.asarray(bits, dtype=np.uint8).tolist()))
    idx = 0
    out: List[Dict[str, int | None]] = []

    for _ in range(num_triplets):
        if idx + 3 > len(bits):
            break

        keep_subject = bool(bits[idx])
        keep_relation = bool(bits[idx + 1])
        keep_object = bool(bits[idx + 2])
        idx += 3

        pkt: Dict[str, int | None] = {
            "subject_id": None,
            "relation_id": None,
            "object_id": None,
        }

        if keep_subject:
            if idx + FIELD_BITS > len(bits):
                break
            pkt["subject_id"] = bits_to_int(bits[idx: idx + FIELD_BITS])
            idx += FIELD_BITS

        if keep_relation:
            if idx + FIELD_BITS > len(bits):
                break
            pkt["relation_id"] = bits_to_int(bits[idx: idx + FIELD_BITS])
            idx += FIELD_BITS

        if keep_object:
            if idx + FIELD_BITS > len(bits):
                break
            pkt["object_id"] = bits_to_int(bits[idx: idx + FIELD_BITS])
            idx += FIELD_BITS

        out.append(pkt)

    return out
