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

# ---------------------------------------------------------------------
# Validated/drop-aware semantic metrics
# ---------------------------------------------------------------------
# These metrics treat invalid decoded packets as dropped packets.
# A dropped packet is counted as wrong, not ignored.


def _get_field(packet, name):
    if packet is None:
        return None
    if isinstance(packet, dict):
        return packet.get(name)
    return getattr(packet, name, None)


def sg_triplet_exact_match_rate_with_drop(tx_packets, rx_packets_or_none):
    """
    Exact match for SG triplets with invalid packets treated as dropped/wrong.
    """
    if not tx_packets:
        return 0.0

    correct = 0
    total = len(tx_packets)

    for tx, rx in zip(tx_packets, rx_packets_or_none):
        if rx is None:
            continue

        if (
            _get_field(tx, "subject_id") == _get_field(rx, "subject_id")
            and _get_field(tx, "relation_id") == _get_field(rx, "relation_id")
            and _get_field(tx, "object_id") == _get_field(rx, "object_id")
        ):
            correct += 1

    return correct / total


def sg_field_accuracy_with_drop(tx_packets, rx_packets_or_none):
    """
    Field-level SG accuracy with invalid packets treated as all fields wrong.
    Each SG triplet has 3 fields: subject_id, relation_id, object_id.
    """
    if not tx_packets:
        return 0.0

    correct = 0
    total = 3 * len(tx_packets)

    for tx, rx in zip(tx_packets, rx_packets_or_none):
        if rx is None:
            continue

        correct += int(_get_field(tx, "subject_id") == _get_field(rx, "subject_id"))
        correct += int(_get_field(tx, "relation_id") == _get_field(rx, "relation_id"))
        correct += int(_get_field(tx, "object_id") == _get_field(rx, "object_id"))

    return correct / total


def bbox_object_accuracy_with_drop(tx_packets, rx_packets_or_none):
    """
    BBox object-id accuracy with invalid packets treated as dropped/wrong.
    """
    if not tx_packets:
        return 0.0

    correct = 0
    total = len(tx_packets)

    for tx, rx in zip(tx_packets, rx_packets_or_none):
        if rx is None:
            continue

        correct += int(_get_field(tx, "object_id") == _get_field(rx, "object_id"))

    return correct / total


def bbox_exact_match_rate_with_drop(tx_packets, rx_packets_or_none, coord_tol=1.0 / 65535.0):

    if not tx_packets:
        return 0.0

    correct = 0
    total = len(tx_packets)

    for tx, rx in zip(tx_packets, rx_packets_or_none):
        if rx is None:
            continue

        if _get_field(tx, "object_id") != _get_field(rx, "object_id"):
            continue

        tx_coords = [
            float(_get_field(tx, "x1")),
            float(_get_field(tx, "y1")),
            float(_get_field(tx, "x2")),
            float(_get_field(tx, "y2")),
        ]
        rx_coords = [
            float(_get_field(rx, "x1")),
            float(_get_field(rx, "y1")),
            float(_get_field(rx, "x2")),
            float(_get_field(rx, "y2")),
        ]

        if all(abs(a - b) <= coord_tol for a, b in zip(tx_coords, rx_coords)):
            correct += 1

    return correct / total


def bbox_mean_l1_error_with_drop(tx_packets, rx_packets_or_none, dropped_error=1.0):
    """
    Mean coordinate L1 error with invalid packets treated as high-error packets.

    For valid packets:
        error = mean absolute coordinate error over x1,y1,x2,y2

    For dropped/invalid packets:
        error = dropped_error

    Default dropped_error=1.0 because normalized coordinates are in [0,1].
    """
    if not tx_packets:
        return 0.0

    errors = []

    for tx, rx in zip(tx_packets, rx_packets_or_none):
        if rx is None:
            errors.append(float(dropped_error))
            continue

        tx_coords = [
            float(_get_field(tx, "x1")),
            float(_get_field(tx, "y1")),
            float(_get_field(tx, "x2")),
            float(_get_field(tx, "y2")),
        ]
        rx_coords = [
            float(_get_field(rx, "x1")),
            float(_get_field(rx, "y1")),
            float(_get_field(rx, "x2")),
            float(_get_field(rx, "y2")),
        ]

        err = sum(abs(a - b) for a, b in zip(tx_coords, rx_coords)) / 4.0
        errors.append(err)

    return sum(errors) / len(errors)