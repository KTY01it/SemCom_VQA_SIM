import struct
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from src.semantic.packet_validation import add_crc16, check_crc16, PacketValidationResult

def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)

    padded_len = int(np.ceil(len(bits) / 8.0) * 8)
    padded = np.zeros(padded_len, dtype=np.uint8)
    padded[: len(bits)] = bits

    return np.packbits(padded).tobytes()


@dataclass(frozen=True)
class SGTripletPacket:
    subject_id: int
    relation_id: int
    object_id: int

SG_PAYLOAD_BITS = 48
SG_CRC_BITS = 16
SG_TOTAL_BITS_CRC16 = SG_PAYLOAD_BITS + SG_CRC_BITS

BBOX_PAYLOAD_BITS = 80
BBOX_CRC_BITS = 16
BBOX_TOTAL_BITS_CRC16 = BBOX_PAYLOAD_BITS + BBOX_CRC_BITS

def encode_sg_triplet(packet: SGTripletPacket) -> np.ndarray:
    """
    SG triplet binary format:
        subject_id  uint16
        relation_id uint16
        object_id   uint16

    Total:
        6 bytes = 48 bits
    """
    for name, value in [
        ("subject_id", packet.subject_id),
        ("relation_id", packet.relation_id),
        ("object_id", packet.object_id),
    ]:
        if not (0 <= int(value) <= 65535):
            raise ValueError(f"{name} out of uint16 range: {value}")

    data = struct.pack(
        ">HHH",
        int(packet.subject_id),
        int(packet.relation_id),
        int(packet.object_id),
    )
    return bytes_to_bits(data)


def decode_sg_triplet(bits: np.ndarray) -> SGTripletPacket:
    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) < 48:
        raise ValueError(f"SG triplet requires at least 48 bits, got {len(bits)}")

    data = bits_to_bytes(bits[:48])
    subject_id, relation_id, object_id = struct.unpack(">HHH", data[:6])

    return SGTripletPacket(
        subject_id=subject_id,
        relation_id=relation_id,
        object_id=object_id,
    )


def encode_sg_triplets(packets: Iterable[SGTripletPacket]) -> np.ndarray:
    chunks: List[np.ndarray] = []

    for packet in packets:
        chunks.append(encode_sg_triplet(packet))

    if not chunks:
        return np.zeros(0, dtype=np.uint8)

    return np.concatenate(chunks).astype(np.uint8)


def decode_sg_triplets(bits: np.ndarray, num_triplets: int) -> List[SGTripletPacket]:
    bits = np.asarray(bits, dtype=np.uint8)

    expected_bits = num_triplets * 48
    if len(bits) < expected_bits:
        raise ValueError(
            f"Not enough bits: expected {expected_bits}, got {len(bits)}"
        )

    packets = []
    for i in range(num_triplets):
        start = i * 48
        end = start + 48
        packets.append(decode_sg_triplet(bits[start:end]))

    return packets

def encode_sg_triplet_crc16(packet: SGTripletPacket) -> np.ndarray:
    payload_bits = encode_sg_triplet(packet)
    return add_crc16(payload_bits)


def decode_sg_triplet_crc16(bits: np.ndarray) -> tuple[SGTripletPacket, PacketValidationResult]:
    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) < SG_TOTAL_BITS_CRC16:
        raise ValueError(
            f"CRC16 SG triplet requires at least {SG_TOTAL_BITS_CRC16} bits, got {len(bits)}"
        )

    payload_bits, crc_status = check_crc16(bits[:SG_TOTAL_BITS_CRC16])

    # Decode payload even when CRC fails, so downstream arrays stay aligned.
    packet = decode_sg_triplet(payload_bits[:SG_PAYLOAD_BITS])

    return packet, crc_status


def encode_sg_triplets_crc16(packets: Iterable[SGTripletPacket]) -> np.ndarray:
    chunks: List[np.ndarray] = []

    for packet in packets:
        chunks.append(encode_sg_triplet_crc16(packet))

    if not chunks:
        return np.zeros(0, dtype=np.uint8)

    return np.concatenate(chunks).astype(np.uint8)


def decode_sg_triplets_crc16(
    bits: np.ndarray,
    num_triplets: int,
) -> tuple[List[SGTripletPacket], list[PacketValidationResult]]:
    bits = np.asarray(bits, dtype=np.uint8)

    expected_bits = num_triplets * SG_TOTAL_BITS_CRC16
    if len(bits) < expected_bits:
        raise ValueError(
            f"Not enough bits: expected {expected_bits}, got {len(bits)}"
        )

    packets = []
    crc_results = []

    for i in range(num_triplets):
        start = i * SG_TOTAL_BITS_CRC16
        end = start + SG_TOTAL_BITS_CRC16
        packet, crc_status = decode_sg_triplet_crc16(bits[start:end])
        packets.append(packet)
        crc_results.append(crc_status)

    return packets, crc_results

@dataclass(frozen=True)
class BBoxPacket:
    object_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def _quantize_coord(value: float) -> int:
    """
    Quantize normalized coordinate [0, 1] to uint16.
    """
    value = float(value)
    value = min(max(value, 0.0), 1.0)
    return int(round(value * 65535.0))


def _dequantize_coord(value: int) -> float:
    """
    Recover normalized coordinate from uint16.
    """
    return float(value) / 65535.0


def encode_bbox(packet: BBoxPacket) -> np.ndarray:
    """
    BBox binary format:
        object_id uint16
        x1        uint16
        y1        uint16
        x2        uint16
        y2        uint16

    Total:
        10 bytes = 80 bits
    """
    if not (0 <= int(packet.object_id) <= 65535):
        raise ValueError(f"object_id out of uint16 range: {packet.object_id}")

    qx1 = _quantize_coord(packet.x1)
    qy1 = _quantize_coord(packet.y1)
    qx2 = _quantize_coord(packet.x2)
    qy2 = _quantize_coord(packet.y2)

    data = struct.pack(
        ">HHHHH",
        int(packet.object_id),
        qx1,
        qy1,
        qx2,
        qy2,
    )

    return bytes_to_bits(data)


def decode_bbox(bits: np.ndarray) -> BBoxPacket:
    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) < 80:
        raise ValueError(f"BBox packet requires at least 80 bits, got {len(bits)}")

    data = bits_to_bytes(bits[:80])
    object_id, qx1, qy1, qx2, qy2 = struct.unpack(">HHHHH", data[:10])

    return BBoxPacket(
        object_id=object_id,
        x1=_dequantize_coord(qx1),
        y1=_dequantize_coord(qy1),
        x2=_dequantize_coord(qx2),
        y2=_dequantize_coord(qy2),
    )


def encode_bboxes(packets: Iterable[BBoxPacket]) -> np.ndarray:
    chunks: List[np.ndarray] = []

    for packet in packets:
        chunks.append(encode_bbox(packet))

    if not chunks:
        return np.zeros(0, dtype=np.uint8)

    return np.concatenate(chunks).astype(np.uint8)


def decode_bboxes(bits: np.ndarray, num_bboxes: int) -> List[BBoxPacket]:
    bits = np.asarray(bits, dtype=np.uint8)

    expected_bits = num_bboxes * 80
    if len(bits) < expected_bits:
        raise ValueError(
            f"Not enough bits: expected {expected_bits}, got {len(bits)}"
        )

    packets = []
    for i in range(num_bboxes):
        start = i * 80
        end = start + 80
        packets.append(decode_bbox(bits[start:end]))

    return packets


def encode_bbox_crc16(packet: BBoxPacket) -> np.ndarray:
    payload_bits = encode_bbox(packet)
    return add_crc16(payload_bits)


def decode_bbox_crc16(bits: np.ndarray) -> tuple[BBoxPacket, PacketValidationResult]:
    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) < BBOX_TOTAL_BITS_CRC16:
        raise ValueError(
            f"CRC16 BBox packet requires at least {BBOX_TOTAL_BITS_CRC16} bits, got {len(bits)}"
        )

    payload_bits, crc_status = check_crc16(bits[:BBOX_TOTAL_BITS_CRC16])

    # Decode payload even when CRC fails, so downstream arrays stay aligned.
    packet = decode_bbox(payload_bits[:BBOX_PAYLOAD_BITS])

    return packet, crc_status


def encode_bboxes_crc16(packets: Iterable[BBoxPacket]) -> np.ndarray:
    chunks: List[np.ndarray] = []

    for packet in packets:
        chunks.append(encode_bbox_crc16(packet))

    if not chunks:
        return np.zeros(0, dtype=np.uint8)

    return np.concatenate(chunks).astype(np.uint8)


def decode_bboxes_crc16(
    bits: np.ndarray,
    num_bboxes: int,
) -> tuple[List[BBoxPacket], list[PacketValidationResult]]:
    bits = np.asarray(bits, dtype=np.uint8)

    expected_bits = num_bboxes * BBOX_TOTAL_BITS_CRC16
    if len(bits) < expected_bits:
        raise ValueError(
            f"Not enough bits: expected {expected_bits}, got {len(bits)}"
        )

    packets = []
    crc_results = []

    for i in range(num_bboxes):
        start = i * BBOX_TOTAL_BITS_CRC16
        end = start + BBOX_TOTAL_BITS_CRC16
        packet, crc_status = decode_bbox_crc16(bits[start:end])
        packets.append(packet)
        crc_results.append(crc_status)

    return packets, crc_results