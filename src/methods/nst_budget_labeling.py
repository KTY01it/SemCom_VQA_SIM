from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set, Tuple

from src.methods.nst_features import norm, split_tokens, question_token_set, field_text


VALID_MASKS: List[Tuple[int, int, int]] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
]


FIELDS = ["subject", "relation", "object"]


def field_tokens(triplet: Dict[str, Any], field: str) -> Set[str]:
    return split_tokens(field_text(triplet, field))


def has_answer_overlap(triplet: Dict[str, Any], field: str, answer: str | None) -> bool:
    if not answer:
        return False

    ans_tokens = split_tokens(answer)
    toks = field_tokens(triplet, field)

    if not ans_tokens or not toks:
        return False

    return bool(ans_tokens & toks)


def mask_num_fields(mask: Tuple[int, int, int]) -> int:
    return int(sum(mask))


def mask_bits(mask: Tuple[int, int, int]) -> int:
    return 3 + 16 * mask_num_fields(mask)


def field_importance(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    answer: str | None,
    question_type: str | None,
) -> Dict[str, float]:
    """
    Softer utility. Do not make subject/object almost always mandatory.
    """
    q_tokens = question_token_set(question, keywords)
    qtype = norm(question_type) if question_type else ""

    imp = {f: 0.0 for f in FIELDS}

    # Offline answer supervision.
    # Keep this strong, but not so strong that full triplets dominate.
    for f in FIELDS:
        if has_answer_overlap(triplet, f, answer):
            imp[f] += 3.0

    # Question overlap.
    # If field is already fully known from question, transmitting it is less useful.
    for f in FIELDS:
        toks = field_tokens(triplet, f)
        if not toks:
            continue

        if toks <= q_tokens:
            imp[f] -= 0.8
        elif toks & q_tokens:
            imp[f] += 0.3

    # Question type prior.
    # Make relation useful, but avoid forcing full triplet.
    if qtype in {"verify", "logical"}:
        imp["relation"] += 1.0
        imp["subject"] += 0.5
        imp["object"] += 0.5

    elif qtype == "relation":
        imp["relation"] += 1.0
        imp["subject"] += 0.2
        imp["object"] += 0.2

    elif qtype == "query":
        imp["object"] += 0.6
        imp["subject"] += 0.2

    elif qtype == "choose":
        imp["subject"] += 0.4
        imp["object"] += 0.4

    return imp


def infer_budget_keep_label(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    answer: str | None,
    question_type: str | None,
    utility_lambda: float = 1.0,
) -> Dict[str, int]:
    """
    NST-v3 budget-aware pseudo-label.

    This function may use answer only for offline label generation.
    The inference model does not receive answer.
    """
    imp = field_importance(
        triplet=triplet,
        question=question,
        keywords=keywords,
        answer=answer,
        question_type=question_type,
    )

    qtype = norm(question_type) if question_type else ""

    answer_fields = {
        f for f in FIELDS
        if has_answer_overlap(triplet, f, answer)
    }
    answer_matched = len(answer_fields) > 0

    best_mask = (0, 0, 1)
    best_score = -1e9

    for mask in VALID_MASKS:
        n_fields = mask_num_fields(mask)

        utility = 0.0
        for bit, f in zip(mask, FIELDS):
            if bit:
                utility += imp[f]

        # If answer evidence exists, avoid dropping all answer fields.
        if answer_matched:
            kept_answer = any(
                bit and f in answer_fields
                for bit, f in zip(mask, FIELDS)
            )
            if not kept_answer:
                utility -= 4.0

        # Relation is important for verify/logical/relation questions,
        # but not always mandatory.
        if qtype in {"verify", "logical"} and mask[1] == 0:
            utility -= 1.0

        if qtype == "relation" and mask[1] == 0:
            utility -= 0.7

        # Penalize full triplet unless its utility is clearly needed.
        if n_fields == 3:
            utility -= 0.6

        # Slightly prefer 1-field or 2-field masks.
        if n_fields == 1:
            utility += 0.25
        elif n_fields == 2:
            utility += 0.10

        # Stronger budget penalty than old version.
        # Old: mask_bits / 51.
        # New: direct field-count penalty + bit penalty.
        field_cost = n_fields / 3.0
        bit_cost = mask_bits(mask) / 51.0
        cost = 0.7 * field_cost + 0.3 * bit_cost

        score = utility - utility_lambda * cost

        if score > best_score:
            best_score = score
            best_mask = mask

    return {
        "subject": int(best_mask[0]),
        "relation": int(best_mask[1]),
        "object": int(best_mask[2]),
    }