from __future__ import annotations

from typing import Any, Dict, Iterable, Set


def norm(x: Any) -> str:
    return str(x).lower().strip().replace("_", " ")


def split_tokens(x: Any) -> Set[str]:
    text = norm(x)
    return {t for t in text.replace("-", " ").split() if t}


def question_tokens(question: str | None, keywords: Iterable[str] | None) -> Set[str]:
    stop = {
        "is", "are", "the", "a", "an", "of", "on", "in", "to", "and",
        "or", "what", "where", "who", "how", "many", "does", "do",
        "there", "that", "this", "with", "at", "by", "it", "which",
        "color", "kind", "type", "side",
    }

    out: Set[str] = set()

    if question:
        out |= {t for t in split_tokens(question.replace("?", " ")) if t not in stop}

    if keywords:
        for k in keywords:
            out |= split_tokens(k)

    return out


def field_text(triplet: Dict[str, Any], field: str) -> str:
    keys = {
        "subject": ["subject", "subject_name", "subject_label", "subj", "s"],
        "relation": ["relation", "relation_name", "predicate", "rel", "r"],
        "object": ["object", "object_name", "object_label", "obj", "o"],
    }

    for k in keys[field]:
        if k in triplet and triplet[k] is not None:
            return norm(triplet[k])

    return ""


def infer_keep_label(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    answer: str | None,
    question_type: str | None = None,
) -> Dict[str, int]:
    """
    Pseudo-label for DBSS-NST.

    1 = transmit this field
    0 = omit because question context likely already provides it
    """
    q_tokens = question_tokens(question, keywords)
    ans_tokens = split_tokens(answer) if answer else set()
    qtype = norm(question_type) if question_type else ""
    question_text = norm(question) if question else ""

    labels: Dict[str, int] = {}

    for field in ["subject", "relation", "object"]:
        text = field_text(triplet, field)
        toks = split_tokens(text)

        keep = 1

        if toks and toks <= q_tokens:
            keep = 0

        if toks and ans_tokens and (toks & ans_tokens):
            keep = 1

        labels[field] = keep

    # Yes/no, verify, spatial, relation questions need relational evidence.
    if (
        "verify" in qtype
        or "logical" in qtype
        or question_text.startswith("is ")
        or question_text.startswith("are ")
        or question_text.startswith("does ")
        or question_text.startswith("do ")
    ):
        labels = {"subject": 1, "relation": 1, "object": 1}

    # Never drop all fields.
    if labels["subject"] + labels["relation"] + labels["object"] == 0:
        labels["object"] = 1

    return labels
