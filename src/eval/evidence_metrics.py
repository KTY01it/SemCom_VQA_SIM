from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def _norm(x: Any) -> str:
    return str(x).lower().strip().replace("_", " ")


def _split_tokens(x: Any) -> Set[str]:
    text = _norm(x)
    if not text:
        return set()
    return {t for t in text.replace("-", " ").split() if t}


def unit_tokens(unit: Dict[str, Any]) -> Set[str]:
    toks = set()

    for key in [
        "subject",
        "subject_name",
        "subject_label",
        "subj",
        "s",
        "relation",
        "relation_name",
        "predicate",
        "rel",
        "r",
        "object",
        "object_name",
        "object_label",
        "obj",
        "o",
    ]:
        if key in unit and unit[key] is not None:
            toks |= _split_tokens(unit[key])

    if not toks:
        for key in ["subject_id", "relation_id", "object_id"]:
            if key in unit and unit[key] is not None:
                toks.add(f"{key}:{unit[key]}")

    return toks


def question_concepts(question: str | None, keywords: Iterable[str] | None = None) -> Set[str]:
    if keywords:
        out = set()
        for k in keywords:
            out |= _split_tokens(k)
        if out:
            return out

    if not question:
        return set()

    stop = {
        "is", "are", "the", "a", "an", "of", "on", "in", "to", "and",
        "or", "what", "where", "who", "how", "many", "does", "do",
        "there", "that", "this", "with", "at", "by", "it", "which",
        "color", "kind", "type", "side"
    }

    toks = _split_tokens(question.replace("?", " "))
    return {t for t in toks if t not in stop}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def coverage_ratio(
    selected_units: List[Dict[str, Any]],
    question: str | None,
    keywords: Iterable[str] | None = None,
) -> float:
    concepts = question_concepts(question, keywords)
    if not concepts:
        return 0.0

    covered = set()
    for u in selected_units:
        covered |= unit_tokens(u)

    return len(covered & concepts) / max(1, len(concepts))


def redundancy_ratio(selected_units: List[Dict[str, Any]]) -> float:
    if len(selected_units) <= 1:
        return 0.0

    token_sets = [unit_tokens(u) for u in selected_units]
    vals = []

    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            vals.append(jaccard(token_sets[i], token_sets[j]))

    return sum(vals) / max(1, len(vals))


def unique_concept_count(selected_units: List[Dict[str, Any]]) -> int:
    out = set()
    for u in selected_units:
        out |= unit_tokens(u)
    return len(out)
