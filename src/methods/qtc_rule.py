from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.methods.nst_labeling import infer_keep_label


def qtc_rule_keep_mask(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    answer: str | None,
    question_type: str | None = None,
) -> Dict[str, bool]:
    """
    QTC-rule: Question-guided Token Compression rule.

    This is the non-neural baseline for DBSS-NST.
    It uses the same pseudo-label logic as NST training, but applies it directly
    at inference time without a learned model.
    """
    labels = infer_keep_label(
        triplet=triplet,
        question=question,
        keywords=keywords,
        answer=answer,
        question_type=question_type,
    )

    keep = {
        "subject": bool(labels["subject"]),
        "relation": bool(labels["relation"]),
        "object": bool(labels["object"]),
    }

    if not any(keep.values()):
        keep["object"] = True

    return keep


def tx_field_keep_ratio(keep_masks: List[Dict[str, bool]]) -> float:
    if not keep_masks:
        return 0.0

    total = 3 * len(keep_masks)
    kept = 0

    for m in keep_masks:
        kept += int(bool(m["subject"]))
        kept += int(bool(m["relation"]))
        kept += int(bool(m["object"]))

    return kept / max(1, total)
