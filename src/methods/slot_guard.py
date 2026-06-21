from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.methods.nst_features import (
    field_text,
    norm,
    question_token_set,
    split_tokens,
)


FIELDS = ["subject", "relation", "object"]

SPATIAL_TOKENS = {
    "on", "in", "under", "above", "below", "behind", "front",
    "near", "next", "left", "right", "inside", "outside",
    "beside", "holding", "wearing", "standing", "sitting",
}


def field_tokens(triplet: Dict[str, Any], field: str) -> set[str]:
    return split_tokens(field_text(triplet, field))


def any_overlap(a: set[str], b: set[str]) -> bool:
    return bool(a and b and (a & b))


def slot_guard_fields(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
    mode: str = "slot",
) -> Dict[str, bool]:
    """
    Inference-safe slot guard.

    Modes:
      slot:
        protect both likely answer-side and question-context side.

      slot_relation:
        slot + spatial/relation protection.

      answer_slot:
        protect only the unknown answer-side.

      answer_slot_relation:
        answer_slot + spatial/relation protection.

      full:
        conservative mode; protect overlapped context fields too.
    """
    q_tokens = question_token_set(question, keywords)
    qtype = norm(question_type) if question_type else ""

    subj = field_tokens(triplet, "subject")
    rel = field_tokens(triplet, "relation")
    obj = field_tokens(triplet, "object")

    subj_ov = any_overlap(subj, q_tokens)
    rel_ov = any_overlap(rel, q_tokens)
    obj_ov = any_overlap(obj, q_tokens)

    guard = {
        "subject": False,
        "relation": False,
        "object": False,
    }

    answer_only = mode in {"answer_slot", "answer_slot_relation"}

    # Context is object side; subject is likely the answer side.
    if obj_ov and not subj_ov:
        guard["subject"] = True
        if not answer_only:
            guard["object"] = True

    # Context is subject side; object is likely the answer side.
    if subj_ov and not obj_ov:
        guard["object"] = True
        if not answer_only:
            guard["subject"] = True

    # Relation / verify / logical questions often need relation evidence.
    if qtype in {"relation", "verify", "logical"}:
        guard["relation"] = True

    # If relation itself overlaps question, preserve it.
    if rel_ov:
        guard["relation"] = True

    if mode in {"slot_relation", "answer_slot_relation", "full"}:
        spatial_q = bool(q_tokens & SPATIAL_TOKENS)
        spatial_rel = bool(rel & SPATIAL_TOKENS)

        if spatial_q or spatial_rel:
            guard["relation"] = True

    if mode == "full":
        if subj_ov:
            guard["subject"] = True
        if obj_ov:
            guard["object"] = True

    return guard


def apply_slot_guard(
    keep: Dict[str, bool],
    probs: List[float],
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
    mode: str = "slot",
    min_guard_prob: float = 0.0,
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    out = {
        "subject": bool(keep["subject"]),
        "relation": bool(keep["relation"]),
        "object": bool(keep["object"]),
    }

    guard = slot_guard_fields(
        triplet=triplet,
        question=question,
        keywords=keywords,
        question_type=question_type,
        mode=mode,
    )

    for idx, f in enumerate(FIELDS):
        if guard[f] and float(probs[idx]) >= float(min_guard_prob):
            out[f] = True

    if not any(out.values()):
        best_idx = max(range(3), key=lambda i: float(probs[i]))
        out[FIELDS[best_idx]] = True

    return out, guard


def compressed_bits_for_mask(mask: Dict[str, bool]) -> int:
    n_fields = int(mask["subject"]) + int(mask["relation"]) + int(mask["object"])
    return 3 + 16 * max(1, n_fields)


def total_compressed_bits(masks: List[Dict[str, bool]]) -> int:
    return sum(compressed_bits_for_mask(m) for m in masks)


def repair_masks_to_budget(
    keep_masks: List[Dict[str, bool]],
    prob_lists: List[List[float]],
    guard_masks: List[Dict[str, bool]],
    target_bits: int,
    allow_guard_drop: bool = False,
    guard_drop_penalty: float = 0.0,
) -> List[Dict[str, bool]]:
    if target_bits <= 0:
        return keep_masks

    repaired = [
        {
            "subject": bool(m["subject"]),
            "relation": bool(m["relation"]),
            "object": bool(m["object"]),
        }
        for m in keep_masks
    ]

    while total_compressed_bits(repaired) > target_bits:
        best = None

        for i, mask in enumerate(repaired):
            kept_fields = [f for f in FIELDS if mask[f]]

            if len(kept_fields) <= 1:
                continue

            for field_idx, field in enumerate(FIELDS):
                if not mask[field]:
                    continue

                is_guarded = bool(guard_masks[i].get(field, False))

                if is_guarded and not allow_guard_drop:
                    continue

                drop_score = float(prob_lists[i][field_idx])
                if is_guarded:
                    drop_score += float(guard_drop_penalty)

                if best is None or drop_score < best[0]:
                    best = (drop_score, i, field)

        if best is None:
            break

        _, i, field = best
        repaired[i][field] = False

        if not any(repaired[i].values()):
            best_idx = max(range(3), key=lambda j: float(prob_lists[i][j]))
            repaired[i][FIELDS[best_idx]] = True

    return repaired
