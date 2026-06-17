from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Set

import torch
import torch.nn.functional as F

from src.methods.nst_features import (
    build_nst_v2_features,
    nst_v2_feature_names,
    norm,
    split_tokens,
    question_token_set,
    field_text,
)


FIELDS = ["subject", "relation", "object"]


def triplet_text(triplet: Dict[str, Any]) -> str:
    s = field_text(triplet, "subject")
    r = field_text(triplet, "relation")
    o = field_text(triplet, "object")
    return " ".join(x for x in [s, r, o] if x).strip()


def field_tokens(triplet: Dict[str, Any], field: str) -> Set[str]:
    return split_tokens(field_text(triplet, field))


def answer_tokens(answer: str | None) -> Set[str]:
    if not answer:
        return set()
    return split_tokens(answer)


def has_answer_overlap(triplet: Dict[str, Any], answer: str | None) -> bool:
    ans = answer_tokens(answer)
    if not ans:
        return False

    for f in FIELDS:
        if field_tokens(triplet, f) & ans:
            return True

    return False


def has_keyword_bridge(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
) -> bool:
    q_tokens = question_token_set(question, keywords)

    if not q_tokens:
        return False

    for f in FIELDS:
        if field_tokens(triplet, f) & q_tokens:
            return True

    return False


def question_triplet_jaccard(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
) -> float:
    q_tokens = question_token_set(question, keywords)
    t_tokens = split_tokens(triplet_text(triplet))

    if not q_tokens or not t_tokens:
        return 0.0

    return len(q_tokens & t_tokens) / max(1, len(q_tokens | t_tokens))


def noanswer_proxy_score(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
) -> float:
    """
    Inference-safe proxy. Does not use answer.
    """
    qtype = norm(question_type) if question_type else ""
    q_tokens = question_token_set(question, keywords)

    score = 0.0

    subj = field_tokens(triplet, "subject")
    rel = field_tokens(triplet, "relation")
    obj = field_tokens(triplet, "object")

    if subj & q_tokens:
        score += 0.6
    if rel & q_tokens:
        score += 0.8
    if obj & q_tokens:
        score += 0.8

    if qtype in {"relation", "verify", "logical"}:
        score += 0.5 if rel else 0.0

    if qtype in {"query", "choose"}:
        score += 0.3 if (subj or obj) else 0.0

    return float(score)


def oracle_utility_score(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    answer: str | None,
    question_type: str | None,
) -> float:
    """
    Balanced offline utility for NSP-v5.2.

    Uses answer only for training, not inference.
    """
    qtype = norm(question_type) if question_type else ""

    score = 0.0

    ans_hit = has_answer_overlap(triplet, answer)
    bridge = has_keyword_bridge(triplet, question, keywords)

    if ans_hit:
        score += 7.0

    if ans_hit and bridge:
        score += 4.0

    if bridge and not ans_hit:
        score += 0.7

    rel = field_tokens(triplet, "relation")
    subj = field_tokens(triplet, "subject")
    obj = field_tokens(triplet, "object")

    if qtype in {"relation", "verify", "logical"} and rel:
        score += 1.0 if ans_hit else 0.4

    if qtype in {"query", "choose"} and (subj or obj):
        score += 0.8 if ans_hit else 0.3

    # Slot-aware weak boost.
    slot_feats = slot_aware_features(triplet, question, keywords, question_type)
    subject_as_answer_slot = slot_feats[3]
    object_as_answer_slot = slot_feats[4]
    spatial_match = slot_feats[9]

    if subject_as_answer_slot or object_as_answer_slot:
        score += 0.8

    if spatial_match:
        score += 0.5

    q_tokens = question_token_set(question, keywords)
    t_tokens = split_tokens(triplet_text(triplet))

    if t_tokens and t_tokens <= q_tokens and not ans_hit:
        score -= 1.2

    return float(score)

SPATIAL_REL_TOKENS = {
    "on", "in", "under", "above", "below", "behind", "front",
    "near", "next", "left", "right", "inside", "outside", "beside",
    "holding", "wearing", "standing", "sitting",
}


def overlap_ratio(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))


def slot_aware_features(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
) -> List[float]:
    q_tokens = question_token_set(question, keywords)
    qtype = norm(question_type) if question_type else ""

    subj = field_tokens(triplet, "subject")
    rel = field_tokens(triplet, "relation")
    obj = field_tokens(triplet, "object")

    subj_ov = overlap_ratio(subj, q_tokens)
    rel_ov = overlap_ratio(rel, q_tokens)
    obj_ov = overlap_ratio(obj, q_tokens)

    # Candidate answer-slot proxy.
    # Example: "What is on the table?" -> object/table overlaps question,
    # so subject is likely the answer slot.
    subject_as_answer_slot = 1.0 if obj_ov > 0.0 and subj_ov == 0.0 else 0.0

    # Example: "What is the man holding?" -> subject/man overlaps question,
    # so object is likely the answer slot.
    object_as_answer_slot = 1.0 if subj_ov > 0.0 and obj_ov == 0.0 else 0.0

    relation_as_answer_slot = 1.0 if qtype == "relation" and rel_ov == 0.0 else 0.0

    relation_bridge = 1.0 if rel_ov > 0.0 else 0.0

    spatial_relation = 1.0 if (rel & SPATIAL_REL_TOKENS) else 0.0
    spatial_question = 1.0 if (q_tokens & SPATIAL_REL_TOKENS) else 0.0
    spatial_match = 1.0 if spatial_relation and spatial_question else 0.0

    # Query questions often need the unknown object/subject side.
    query_like = 1.0 if qtype in {"query", "choose"} else 0.0
    verify_like = 1.0 if qtype in {"verify", "logical", "relation"} else 0.0

    return [
        subj_ov,
        rel_ov,
        obj_ov,
        subject_as_answer_slot,
        object_as_answer_slot,
        relation_as_answer_slot,
        relation_bridge,
        spatial_relation,
        spatial_question,
        spatial_match,
        query_like,
        verify_like,
    ]
    
def build_v5_dense_features(
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
    rank_index: int,
    n_candidates: int,
    semantic_cosine: float,
) -> List[float]:
    feats = build_nst_v2_features(
        triplet=triplet,
        question=question,
        keywords=keywords,
        question_type=question_type,
        rank_index=rank_index,
        n_top=max(1, n_candidates),
    )

    t_len = len(split_tokens(triplet_text(triplet)))

    extra = [
        float(semantic_cosine),
        float(question_triplet_jaccard(triplet, question, keywords)),
        float(noanswer_proxy_score(triplet, question, keywords, question_type) / 4.0),
        float(min(t_len, 10) / 10.0),
    ]

    extra.extend(
        slot_aware_features(
            triplet=triplet,
            question=question,
            keywords=keywords,
            question_type=question_type,
        )
    )

    return feats + extra


def nsp_v5_dense_feature_names() -> List[str]:
    return nst_v2_feature_names() + [
        "semantic_cosine",
        "question_triplet_jaccard",
        "noanswer_proxy_score_norm",
        "triplet_len_norm",
        "subj_question_overlap",
        "rel_question_overlap",
        "obj_question_overlap",
        "subject_as_answer_slot",
        "object_as_answer_slot",
        "relation_as_answer_slot",
        "relation_bridge",
        "spatial_relation",
        "spatial_question",
        "spatial_match",
        "query_like",
        "verify_like",
    ]


def cosine_scalar(q_emb, t_emb) -> float:
    if isinstance(q_emb, torch.Tensor):
        q = q_emb.detach().float()
    else:
        q = torch.tensor(q_emb, dtype=torch.float32)

    if isinstance(t_emb, torch.Tensor):
        t = t_emb.detach().float()
    else:
        t = torch.tensor(t_emb, dtype=torch.float32)

    return float(F.cosine_similarity(q[None, :], t[None, :]).item())


def softmax_teacher_probs(scores: List[float], temperature: float) -> List[float]:
    if not scores:
        return []

    x = torch.tensor(scores, dtype=torch.float32)
    x = x / max(1e-6, float(temperature))
    p = torch.softmax(x, dim=0)
    return [float(v) for v in p.tolist()]
