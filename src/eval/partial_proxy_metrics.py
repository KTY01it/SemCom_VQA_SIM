from __future__ import annotations

from typing import Any, Dict, List


def norm(x: Any) -> str:
    return str(x).lower().strip().replace("_", " ")


def id_recovered(t: Dict[str, Any], rx: Dict[str, Any], key: str) -> bool:
    if rx.get(key) is None:
        return False
    try:
        return int(rx.get(key)) == int(t.get(key))
    except Exception:
        return False


def field_texts(t: Dict[str, Any], field: str) -> List[str]:
    keys = {
        "subject": ["subject", "subject_name", "subject_label", "subj", "s"],
        "relation": ["relation", "relation_name", "predicate", "rel", "r"],
        "object": ["object", "object_name", "object_label", "obj", "o"],
    }

    out = []
    for k in keys[field]:
        if k in t and t[k] is not None:
            out.append(norm(t[k]))
    return out


def partial_delivered_answer_hit(
    selected_triplets: List[Dict[str, Any]],
    rx_partial_packets: List[Dict[str, Any]],
    answer: str,
) -> float:
    ans = norm(answer)

    for t, rx in zip(selected_triplets, rx_partial_packets):
        delivered_texts: List[str] = []

        if id_recovered(t, rx, "subject_id"):
            delivered_texts += field_texts(t, "subject")

        if id_recovered(t, rx, "relation_id"):
            delivered_texts += field_texts(t, "relation")

        if id_recovered(t, rx, "object_id"):
            delivered_texts += field_texts(t, "object")

        if ans and ans in delivered_texts:
            return 1.0

    return 0.0


def partial_field_keep_ratio(rx_partial_packets: List[Dict[str, Any]]) -> float:
    if not rx_partial_packets:
        return 0.0

    total = 3 * len(rx_partial_packets)
    kept = 0

    for rx in rx_partial_packets:
        kept += int(rx.get("subject_id") is not None)
        kept += int(rx.get("relation_id") is not None)
        kept += int(rx.get("object_id") is not None)

    return kept / max(1, total)
