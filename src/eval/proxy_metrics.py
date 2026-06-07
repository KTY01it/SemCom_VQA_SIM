from typing import Any, Dict, List, Set


def normalize_text(x: str) -> str:
    return str(x).lower().strip()


def keyword_hit_rate_bboxes(
    selected_bboxes: List[Dict[str, Any]],
    keywords: List[str],
) -> float:
    if not selected_bboxes:
        return 0.0

    kw: Set[str] = {normalize_text(k) for k in keywords}

    hits = 0
    for b in selected_bboxes:
        obj = normalize_text(b.get("object", ""))
        hits += int(obj in kw)

    return hits / len(selected_bboxes)


def keyword_hit_rate_triplets(
    selected_triplets: List[Dict[str, Any]],
    keywords: List[str],
) -> float:
    if not selected_triplets:
        return 0.0

    kw: Set[str] = {normalize_text(k) for k in keywords}

    hits = 0
    for t in selected_triplets:
        subj = normalize_text(t.get("subject", ""))
        rel = normalize_text(t.get("relation", ""))
        obj = normalize_text(t.get("object", ""))

        hits += int(subj in kw or rel in kw or obj in kw)

    return hits / len(selected_triplets)


def answer_hit_bboxes(
    selected_bboxes: List[Dict[str, Any]],
    answer: str,
) -> float:
    ans = normalize_text(answer)

    if not selected_bboxes or not ans:
        return 0.0

    for b in selected_bboxes:
        obj = normalize_text(b.get("object", ""))
        attrs = {normalize_text(a) for a in b.get("attributes", [])}

        if obj == ans or ans in attrs:
            return 1.0

    return 0.0


def answer_hit_triplets(
    selected_triplets: List[Dict[str, Any]],
    answer: str,
) -> float:
    ans = normalize_text(answer)

    if not selected_triplets or not ans:
        return 0.0

    for t in selected_triplets:
        subj = normalize_text(t.get("subject", ""))
        rel = normalize_text(t.get("relation", ""))
        obj = normalize_text(t.get("object", ""))

        if ans in {subj, rel, obj}:
            return 1.0

    return 0.0
