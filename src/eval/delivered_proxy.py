from typing import Any, Dict, List

from src.eval.proxy_metrics import (
    answer_hit_bboxes,
    answer_hit_triplets,
    keyword_hit_rate_bboxes,
    keyword_hit_rate_triplets,
)
from src.semantic.packet_codec import BBoxPacket, SGTripletPacket


def delivered_triplets_by_exact_recovery(
    selected_triplets: List[Dict[str, Any]],
    tx_packets: List[SGTripletPacket],
    rx_packets: List[SGTripletPacket],
) -> List[Dict[str, Any]]:
    if not (len(selected_triplets) == len(tx_packets) == len(rx_packets)):
        raise ValueError("Length mismatch in delivered_triplets_by_exact_recovery.")

    delivered = []

    for raw, tx, rx in zip(selected_triplets, tx_packets, rx_packets):
        if tx == rx:
            delivered.append(raw)

    return delivered


def delivered_bboxes_by_exact_recovery(
    selected_bboxes: List[Dict[str, Any]],
    tx_packets: List[BBoxPacket],
    rx_packets: List[BBoxPacket],
    coord_tol: float = 1.0 / 65535.0,
) -> List[Dict[str, Any]]:
    if not (len(selected_bboxes) == len(tx_packets) == len(rx_packets)):
        raise ValueError("Length mismatch in delivered_bboxes_by_exact_recovery.")

    delivered = []

    for raw, tx, rx in zip(selected_bboxes, tx_packets, rx_packets):
        same_object = tx.object_id == rx.object_id
        same_coord = (
            abs(tx.x1 - rx.x1) <= coord_tol
            and abs(tx.y1 - rx.y1) <= coord_tol
            and abs(tx.x2 - rx.x2) <= coord_tol
            and abs(tx.y2 - rx.y2) <= coord_tol
        )

        if same_object and same_coord:
            delivered.append(raw)

    return delivered


def delivered_bboxes_by_object_recovery(
    selected_bboxes: List[Dict[str, Any]],
    tx_packets: List[BBoxPacket],
    rx_packets: List[BBoxPacket],
) -> List[Dict[str, Any]]:
    if not (len(selected_bboxes) == len(tx_packets) == len(rx_packets)):
        raise ValueError("Length mismatch in delivered_bboxes_by_object_recovery.")

    delivered = []

    for raw, tx, rx in zip(selected_bboxes, tx_packets, rx_packets):
        if tx.object_id == rx.object_id:
            delivered.append(raw)

    return delivered


def delivered_triplet_keyword_hit_rate(
    selected_triplets: List[Dict[str, Any]],
    tx_packets: List[SGTripletPacket],
    rx_packets: List[SGTripletPacket],
    keywords: List[str],
) -> float:
    delivered = delivered_triplets_by_exact_recovery(
        selected_triplets,
        tx_packets,
        rx_packets,
    )
    return keyword_hit_rate_triplets(delivered, keywords)


def delivered_triplet_answer_hit(
    selected_triplets: List[Dict[str, Any]],
    tx_packets: List[SGTripletPacket],
    rx_packets: List[SGTripletPacket],
    answer: str,
) -> float:
    delivered = delivered_triplets_by_exact_recovery(
        selected_triplets,
        tx_packets,
        rx_packets,
    )
    return answer_hit_triplets(delivered, answer)


def delivered_bbox_keyword_hit_rate_object_level(
    selected_bboxes: List[Dict[str, Any]],
    tx_packets: List[BBoxPacket],
    rx_packets: List[BBoxPacket],
    keywords: List[str],
) -> float:
    delivered = delivered_bboxes_by_object_recovery(
        selected_bboxes,
        tx_packets,
        rx_packets,
    )
    return keyword_hit_rate_bboxes(delivered, keywords)


def delivered_bbox_answer_hit_object_level(
    selected_bboxes: List[Dict[str, Any]],
    tx_packets: List[BBoxPacket],
    rx_packets: List[BBoxPacket],
    answer: str,
) -> float:
    delivered = delivered_bboxes_by_object_recovery(
        selected_bboxes,
        tx_packets,
        rx_packets,
    )
    return answer_hit_bboxes(delivered, answer)
