from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


QUESTION_TYPES = ["relation", "verify", "query", "logical", "choose"]


def norm(x: Any) -> str:
    return str(x).lower().strip().replace("_", " ")


def split_tokens(x: Any) -> Set[str]:
    text = norm(x)
    for ch in ["?", ".", ",", ":", ";", "!", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, " ")
    return {t for t in text.replace("-", " ").split() if t}


def question_token_set(question: str | None, keywords: Iterable[str] | None) -> Set[str]:
    stop = {
        "is", "are", "the", "a", "an", "of", "on", "in", "to", "and",
        "or", "what", "where", "who", "how", "many", "does", "do",
        "there", "that", "this", "with", "at", "by", "it", "which",
        "color", "kind", "type", "side", "one", "ones", "thing",
    }

    out: Set[str] = set()

    if question:
        out |= {t for t in split_tokens(question) if t not in stop}

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


def overlap_features(field: str, triplet: Dict[str, Any], q_tokens: Set[str]) -> List[float]:
    toks = split_tokens(field_text(triplet, field))

    if not toks:
        return [0.0, 0.0, 0.0]

    any_overlap = float(len(toks & q_tokens) > 0)
    all_overlap = float(toks <= q_tokens)
    frac_overlap = float(len(toks & q_tokens) / max(1, len(toks)))

    return [any_overlap, all_overlap, frac_overlap]


def question_type_onehot(question_type: str | None) -> List[float]:
    qtype = norm(question_type) if question_type else ""
    return [float(qtype == t) for t in QUESTION_TYPES]


def build_nst_v2_features(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
    rank_index: int,
    n_top: int,
) -> List[float]:
    """
    Question-conditioned NST-v2 features available at inference time.

    No answer feature is used here, because the answer is unknown before VQA reasoning.
    """
    q_tokens = question_token_set(question, keywords)

    feats: List[float] = []

    # Field-question overlap features.
    feats.extend(overlap_features("subject", triplet, q_tokens))
    feats.extend(overlap_features("relation", triplet, q_tokens))
    feats.extend(overlap_features("object", triplet, q_tokens))

    # Question type.
    feats.extend(question_type_onehot(question_type))

    # Rank/budget position feature.
    denom = max(1, n_top - 1)
    feats.append(float(rank_index / denom))

    # Basic question length signal.
    feats.append(float(min(len(q_tokens), 20) / 20.0))

    return feats


def nst_v2_feature_names() -> List[str]:
    names = []

    for field in ["subject", "relation", "object"]:
        names.extend([
            f"{field}_any_overlap",
            f"{field}_all_overlap",
            f"{field}_frac_overlap",
        ])

    names.extend([f"qtype_{t}" for t in QUESTION_TYPES])
    names.append("rank_norm")
    names.append("question_len_norm")

    return names
