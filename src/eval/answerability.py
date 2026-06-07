from typing import Any, Dict, List, Set


YES_NO_ANSWERS = {"yes", "no"}


def normalize_text(x: str) -> str:
    return str(x).lower().strip()


def _keyword_set(keywords: List[str]) -> Set[str]:
    return {normalize_text(k) for k in keywords if normalize_text(k)}


def _answer_tokens(answer: str) -> Set[str]:
    """
    Conservative answer normalization.

    For now:
    - exact normalized answer
    - simple singular fallback for plural ending with 's'
    """
    ans = normalize_text(answer)

    if not ans:
        return set()

    tokens = {ans}

    if ans.endswith("s") and len(ans) > 3:
        tokens.add(ans[:-1])

    return tokens


def _text_matches_answer(text: str, answer: str) -> bool:
    text = normalize_text(text)
    answers = _answer_tokens(answer)

    if not text or not answers:
        return False

    return text in answers


def _triplet_has_keyword(triplet: Dict[str, Any], keywords: List[str]) -> bool:
    kw = _keyword_set(keywords)

    if not kw:
        return False

    subj = normalize_text(triplet.get("subject", ""))
    rel = normalize_text(triplet.get("relation", ""))
    obj = normalize_text(triplet.get("object", ""))

    return subj in kw or rel in kw or obj in kw


def _triplet_has_answer(triplet: Dict[str, Any], answer: str) -> bool:
    subj = triplet.get("subject", "")
    rel = triplet.get("relation", "")
    obj = triplet.get("object", "")

    return (
        _text_matches_answer(subj, answer)
        or _text_matches_answer(rel, answer)
        or _text_matches_answer(obj, answer)
    )


def sg_answerability_strict(
    triplets: List[Dict[str, Any]],
    answer: str,
    keywords: List[str],
) -> float:
    """
    Strict SG answerability:
    A sample is answerable if at least one triplet contains the answer
    and also overlaps with the question keywords.

    This is useful for relation-centric VQA questions.
    """
    ans = normalize_text(answer)

    if not triplets or not ans:
        return 0.0

    if ans in YES_NO_ANSWERS:
        return 0.0

    for t in triplets:
        if _triplet_has_answer(t, answer) and _triplet_has_keyword(t, keywords):
            return 1.0

    return 0.0


def sg_answerability_loose(
    triplets: List[Dict[str, Any]],
    answer: str,
) -> float:
    """
    Loose SG answerability:
    A sample is answerable if the answer appears anywhere in any selected triplet.
    """
    ans = normalize_text(answer)

    if not triplets or not ans:
        return 0.0

    if ans in YES_NO_ANSWERS:
        return 0.0

    for t in triplets:
        if _triplet_has_answer(t, answer):
            return 1.0

    return 0.0


def bbox_answerability(
    bboxes: List[Dict[str, Any]],
    answer: str,
) -> float:
    """
    BBox answerability:
    A sample is answerable if the answer appears as object label or attribute.
    """
    ans = normalize_text(answer)

    if not bboxes or not ans:
        return 0.0

    if ans in YES_NO_ANSWERS:
        return 0.0

    for b in bboxes:
        obj = normalize_text(b.get("object", ""))
        attrs = {normalize_text(a) for a in b.get("attributes", [])}

        if _text_matches_answer(obj, answer) or ans in attrs:
            return 1.0

    return 0.0


def is_supported_answer(answer: str) -> bool:
    """
    Whether this proxy metric can reasonably evaluate the answer.

    yes/no questions are excluded for now because answering yes/no
    requires logical verification, not just answer-object presence.
    """
    ans = normalize_text(answer)
    return bool(ans) and ans not in YES_NO_ANSWERS
