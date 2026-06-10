from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PacketValidationResult:
    valid: bool
    reason: str = "ok"


def _get_field(packet: Any, name: str) -> Any:
    """
    Support both dataclass/object packet and dict packet.
    """
    if isinstance(packet, Mapping):
        return packet.get(name)
    return getattr(packet, name, None)


def vocab_to_valid_id_set(vocab: Any) -> set[int]:
    """
    Convert vocab object to a set of valid integer IDs.

    Supported formats:
      - {"label": id, ...}
      - [id0, id1, ...]
      - ["label0", "label1", ...] fallback: {0, 1, ..., len(vocab)-1}
    """
    if isinstance(vocab, Mapping):
        valid_ids = set()
        for value in vocab.values():
            try:
                valid_ids.add(int(value))
            except Exception:
                continue
        return valid_ids

    if isinstance(vocab, list):
        if all(isinstance(x, int) for x in vocab):
            return {int(x) for x in vocab}
        return set(range(len(vocab)))

    raise TypeError(f"Unsupported vocab type: {type(vocab)}")


def _normalize_valid_ids(valid_ids: Any | None) -> set[int] | None:
    if valid_ids is None:
        return None
    return {int(x) for x in valid_ids}


def _id_is_valid(
    value: int,
    valid_ids: set[int] | None = None,
    vocab_size: int | None = None,
) -> bool:
    """
    Prefer exact vocab ID-set validation.

    vocab_size is kept only for backward compatibility with old scripts/tests.
    """
    if valid_ids is not None:
        return value in valid_ids

    if vocab_size is not None:
        return 0 <= value < vocab_size

    raise ValueError("Either valid_ids or vocab_size must be provided")


def bits_to_bytes_for_crc(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape={bits.shape}")

    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    return np.packbits(bits).tobytes()


def int_to_bits(value: int, n_bits: int) -> np.ndarray:
    if value < 0 or value >= (1 << n_bits):
        raise ValueError(f"value={value} does not fit in {n_bits} bits")

    return np.array(
        [(value >> i) & 1 for i in range(n_bits - 1, -1, -1)],
        dtype=np.uint8,
    )


def bits_to_int(bits: np.ndarray) -> int:
    bits = np.asarray(bits, dtype=np.uint8)
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def add_crc16(bits: np.ndarray) -> np.ndarray:
    """
    Append CRC16 to payload bits.

    CRC is computed from byte-packed payload bits.
    If payload bit length is not byte-aligned, zero padding is used only
    for CRC computation, not appended to payload.
    """
    payload = np.asarray(bits, dtype=np.uint8)
    if payload.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape={payload.shape}")

    crc = zlib.crc32(bits_to_bytes_for_crc(payload)) & 0xFFFF
    crc_bits = int_to_bits(crc, 16)

    return np.concatenate([payload, crc_bits])


def check_crc16(bits_with_crc: np.ndarray) -> tuple[np.ndarray, PacketValidationResult]:
    """
    Return payload bits and validation status.
    """
    bits_with_crc = np.asarray(bits_with_crc, dtype=np.uint8)
    if bits_with_crc.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape={bits_with_crc.shape}")

    if len(bits_with_crc) < 16:
        return bits_with_crc, PacketValidationResult(False, "too_short_for_crc")

    payload = bits_with_crc[:-16]
    rx_crc = bits_to_int(bits_with_crc[-16:])
    calc_crc = zlib.crc32(bits_to_bytes_for_crc(payload)) & 0xFFFF

    if rx_crc != calc_crc:
        return payload, PacketValidationResult(False, "crc_fail")

    return payload, PacketValidationResult(True, "ok")


def validate_sg_packet(
    packet: Any,
    object_vocab_size: int | None = None,
    relation_vocab_size: int | None = None,
    valid_object_ids: set[int] | None = None,
    valid_relation_ids: set[int] | None = None,
) -> PacketValidationResult:
    """
    Validate decoded SG triplet packet.

    Preferred validation:
      subject_id in valid_object_ids
      object_id in valid_object_ids
      relation_id in valid_relation_ids

    object_vocab_size/relation_vocab_size are kept for backward compatibility.
    """
    valid_object_ids = _normalize_valid_ids(valid_object_ids)
    valid_relation_ids = _normalize_valid_ids(valid_relation_ids)

    subject_id = _get_field(packet, "subject_id")
    relation_id = _get_field(packet, "relation_id")
    object_id = _get_field(packet, "object_id")

    if subject_id is None or relation_id is None or object_id is None:
        return PacketValidationResult(False, "missing_sg_field")

    try:
        subject_id = int(subject_id)
        relation_id = int(relation_id)
        object_id = int(object_id)
    except Exception:
        return PacketValidationResult(False, "non_integer_sg_id")

    if not _id_is_valid(
        subject_id,
        valid_ids=valid_object_ids,
        vocab_size=object_vocab_size,
    ):
        return PacketValidationResult(False, "subject_id_oov")

    if not _id_is_valid(
        object_id,
        valid_ids=valid_object_ids,
        vocab_size=object_vocab_size,
    ):
        return PacketValidationResult(False, "object_id_oov")

    if not _id_is_valid(
        relation_id,
        valid_ids=valid_relation_ids,
        vocab_size=relation_vocab_size,
    ):
        return PacketValidationResult(False, "relation_id_oov")

    return PacketValidationResult(True, "ok")


def validate_bbox_packet(
    packet: Any,
    object_vocab_size: int | None = None,
    valid_object_ids: set[int] | None = None,
) -> PacketValidationResult:
    """
    Validate decoded BBox packet.

    Preferred validation:
      object_id in valid_object_ids

    object_vocab_size is kept for backward compatibility.
    """
    valid_object_ids = _normalize_valid_ids(valid_object_ids)

    object_id = _get_field(packet, "object_id")
    x1 = _get_field(packet, "x1")
    y1 = _get_field(packet, "y1")
    x2 = _get_field(packet, "x2")
    y2 = _get_field(packet, "y2")

    if object_id is None:
        return PacketValidationResult(False, "missing_object_id")

    try:
        object_id = int(object_id)
    except Exception:
        return PacketValidationResult(False, "non_integer_object_id")

    if not _id_is_valid(
        object_id,
        valid_ids=valid_object_ids,
        vocab_size=object_vocab_size,
    ):
        return PacketValidationResult(False, "object_id_oov")

    coords = [x1, y1, x2, y2]
    if any(v is None for v in coords):
        return PacketValidationResult(False, "missing_bbox_coord")

    try:
        x1, y1, x2, y2 = [float(v) for v in coords]
    except Exception:
        return PacketValidationResult(False, "non_numeric_bbox_coord")

    if not all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
        return PacketValidationResult(False, "bbox_coord_out_of_range")

    if x1 > x2:
        return PacketValidationResult(False, "bbox_x_order_invalid")

    if y1 > y2:
        return PacketValidationResult(False, "bbox_y_order_invalid")

    return PacketValidationResult(True, "ok")


def summarize_validation_results(results: list[PacketValidationResult]) -> dict[str, float | int]:
    total = len(results)
    valid = sum(1 for r in results if r.valid)
    invalid = total - valid

    summary: dict[str, float | int] = {
        "num_packets": total,
        "num_valid_packets": valid,
        "num_invalid_packets": invalid,
        "valid_packet_rate": valid / total if total else 0.0,
        "invalid_packet_rate": invalid / total if total else 0.0,
    }

    reason_counts: dict[str, int] = {}
    for r in results:
        reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1

    for reason, count in sorted(reason_counts.items()):
        summary[f"reason_{reason}"] = count

    return summary
