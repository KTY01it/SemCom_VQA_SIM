from __future__ import annotations

import numpy as np

from src.semantic.packet_validation import (
    add_crc16,
    check_crc16,
    validate_sg_packet,
    validate_bbox_packet,
)


def test_crc_ok() -> None:
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
    bits_crc = add_crc16(bits)
    payload, status = check_crc16(bits_crc)

    assert status.valid, status
    assert np.array_equal(payload, bits)


def test_crc_fail() -> None:
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
    bits_crc = add_crc16(bits)

    corrupted = bits_crc.copy()
    corrupted[2] ^= 1

    payload, status = check_crc16(corrupted)

    assert not status.valid
    assert status.reason == "crc_fail"

    # Payload is returned as received. Since we flipped a payload bit,
    # it should no longer equal the original clean payload.
    assert not np.array_equal(payload, bits)


def test_sg_vocab_validation() -> None:
    object_vocab_size = 100
    relation_vocab_size = 20

    good = {"subject_id": 1, "relation_id": 2, "object_id": 3}
    bad_obj = {"subject_id": 1, "relation_id": 2, "object_id": 999}
    bad_rel = {"subject_id": 1, "relation_id": 99, "object_id": 3}

    assert validate_sg_packet(good, object_vocab_size, relation_vocab_size).valid

    r1 = validate_sg_packet(bad_obj, object_vocab_size, relation_vocab_size)
    assert not r1.valid
    assert r1.reason == "object_id_oov"

    r2 = validate_sg_packet(bad_rel, object_vocab_size, relation_vocab_size)
    assert not r2.valid
    assert r2.reason == "relation_id_oov"


def test_bbox_validation() -> None:
    object_vocab_size = 100

    good = {"object_id": 1, "x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9}
    bad_obj = {"object_id": 999, "x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9}
    bad_order = {"object_id": 1, "x1": 0.8, "y1": 0.2, "x2": 0.1, "y2": 0.9}
    bad_range = {"object_id": 1, "x1": -0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9}

    assert validate_bbox_packet(good, object_vocab_size).valid

    r1 = validate_bbox_packet(bad_obj, object_vocab_size)
    assert not r1.valid
    assert r1.reason == "object_id_oov"

    r2 = validate_bbox_packet(bad_order, object_vocab_size)
    assert not r2.valid
    assert r2.reason == "bbox_x_order_invalid"

    r3 = validate_bbox_packet(bad_range, object_vocab_size)
    assert not r3.valid
    assert r3.reason == "bbox_coord_out_of_range"


def main() -> None:
    test_crc_ok()
    test_crc_fail()
    test_sg_vocab_validation()
    test_bbox_validation()
    print("packet validation tests: PASS")


if __name__ == "__main__":
    main()
