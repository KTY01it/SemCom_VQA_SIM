from typing import List

from src.semantic.packet_codec import SGTripletPacket


def sg_triplet_exact_match_rate(
    tx_packets: List[SGTripletPacket],
    rx_packets: List[SGTripletPacket],
) -> float:
    if len(tx_packets) != len(rx_packets):
        raise ValueError(f"Length mismatch: {len(tx_packets)} vs {len(rx_packets)}")

    if len(tx_packets) == 0:
        return 0.0

    correct = sum(int(a == b) for a, b in zip(tx_packets, rx_packets))
    return correct / len(tx_packets)


def sg_field_accuracy(
    tx_packets: List[SGTripletPacket],
    rx_packets: List[SGTripletPacket],
) -> float:
    if len(tx_packets) != len(rx_packets):
        raise ValueError(f"Length mismatch: {len(tx_packets)} vs {len(rx_packets)}")

    if len(tx_packets) == 0:
        return 0.0

    total = 3 * len(tx_packets)
    correct = 0

    for tx, rx in zip(tx_packets, rx_packets):
        correct += int(tx.subject_id == rx.subject_id)
        correct += int(tx.relation_id == rx.relation_id)
        correct += int(tx.object_id == rx.object_id)

    return correct / total

from src.semantic.packet_codec import BBoxPacket


def bbox_object_accuracy(
    tx_packets: List[BBoxPacket],
    rx_packets: List[BBoxPacket],
) -> float:
    if len(tx_packets) != len(rx_packets):
        raise ValueError(f"Length mismatch: {len(tx_packets)} vs {len(rx_packets)}")

    if len(tx_packets) == 0:
        return 0.0

    correct = sum(
        int(tx.object_id == rx.object_id)
        for tx, rx in zip(tx_packets, rx_packets)
    )

    return correct / len(tx_packets)


def bbox_mean_l1_error(
    tx_packets: List[BBoxPacket],
    rx_packets: List[BBoxPacket],
) -> float:
    if len(tx_packets) != len(rx_packets):
        raise ValueError(f"Length mismatch: {len(tx_packets)} vs {len(rx_packets)}")

    if len(tx_packets) == 0:
        return 0.0

    total = 0.0

    for tx, rx in zip(tx_packets, rx_packets):
        total += abs(tx.x1 - rx.x1)
        total += abs(tx.y1 - rx.y1)
        total += abs(tx.x2 - rx.x2)
        total += abs(tx.y2 - rx.y2)

    return total / (4 * len(tx_packets))


def bbox_exact_match_rate(
    tx_packets: List[BBoxPacket],
    rx_packets: List[BBoxPacket],
    coord_tol: float = 1.0 / 65535.0,
) -> float:
    if len(tx_packets) != len(rx_packets):
        raise ValueError(f"Length mismatch: {len(tx_packets)} vs {len(rx_packets)}")

    if len(tx_packets) == 0:
        return 0.0

    correct = 0

    for tx, rx in zip(tx_packets, rx_packets):
        same_object = tx.object_id == rx.object_id
        same_coord = (
            abs(tx.x1 - rx.x1) <= coord_tol
            and abs(tx.y1 - rx.y1) <= coord_tol
            and abs(tx.x2 - rx.x2) <= coord_tol
            and abs(tx.y2 - rx.y2) <= coord_tol
        )

        correct += int(same_object and same_coord)

    return correct / len(tx_packets)