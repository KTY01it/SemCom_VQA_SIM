from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Set


def _norm(x: Any) -> str:
    return str(x).lower().strip().replace("_", " ")


def _split_tokens(x: Any) -> Set[str]:
    text = _norm(x)
    if not text:
        return set()
    return {t for t in text.replace("-", " ").split() if t}


def triplet_tokens(t: Dict[str, Any]) -> Set[str]:
    """
    Robust token extraction for SG triplet dictionaries.
    Supports both id-based and string-based fields.
    """
    toks: Set[str] = set()

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
        if key in t and t[key] is not None:
            toks |= _split_tokens(t[key])

    # Fallback: include IDs as weak tokens only if no text exists.
    if not toks:
        for key in ["subject_id", "relation_id", "object_id"]:
            if key in t and t[key] is not None:
                toks.add(f"{key}:{t[key]}")

    return toks


def question_concepts(question: str | None, keywords: Iterable[str] | None) -> Set[str]:
    """
    Prefer existing extracted keywords. Fallback to simple question tokenization.
    """
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


def relevance_score(t: Dict[str, Any], q_concepts: Set[str]) -> float:
    toks = triplet_tokens(t)
    if not q_concepts:
        return 0.0
    return len(toks & q_concepts) / max(1, len(q_concepts))


def coverage_gain(
    t: Dict[str, Any],
    selected: List[Dict[str, Any]],
    q_concepts: Set[str],
) -> float:
    if not q_concepts:
        return 0.0

    already = set()
    for s in selected:
        already |= triplet_tokens(s)

    new_cover = (triplet_tokens(t) & q_concepts) - already
    return len(new_cover) / max(1, len(q_concepts))


def redundancy_penalty(t: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    if not selected:
        return 0.0

    toks = triplet_tokens(t)
    return max(jaccard(toks, triplet_tokens(s)) for s in selected)


def channel_reliability_proxy(
    packet_bits: int,
    snr_db: float,
    channel_type: str,
) -> float:
    """
    Lightweight ranking proxy.

    Higher SNR => higher reliability.
    Longer packet => lower reliability.
    Rayleigh is penalized compared with AWGN.

    This is only for ranking, not for replacing the actual channel simulator.
    """
    snr_linear = 10 ** (snr_db / 10.0)

    if channel_type == "awgn":
        bit_error_proxy = math.exp(-snr_linear)
    elif channel_type == "rayleigh":
        bit_error_proxy = 1.0 / (2.0 * (1.0 + snr_linear))
    else:
        bit_error_proxy = math.exp(-snr_linear)

    bit_error_proxy = min(max(bit_error_proxy, 0.0), 0.49)
    return (1.0 - bit_error_proxy) ** max(1, packet_bits)


def dbss_select_triplets(
    triplets: List[Dict[str, Any]],
    question: str | None,
    keywords: Iterable[str] | None,
    n_top: int,
    snr_db: float,
    channel_type: str,
    packet_bits: int = 48,
    alpha: float = 1.0,   # relevance
    beta: float = 1.0,    # coverage gain
    gamma: float = 0.25,  # channel reliability
    lamb: float = 0.75,   # redundancy penalty
    mu: float = 0.05,     # cost penalty
) -> List[Dict[str, Any]]:
    """
    Greedy Diverse Budgeted Semantic Selection.

    Current version keeps the same n_top interface for fair comparison.
    The effective budget is n_top * packet_bits.
    """
    if not triplets:
        return []

    q_concepts = question_concepts(question, keywords)

    budget_bits = max(1, n_top * packet_bits)
    selected: List[Dict[str, Any]] = []
    remaining = list(triplets)
    used_bits = 0

    while remaining and len(selected) < n_top:
        best = None
        best_score = -1e18

        for t in remaining:
            cost = packet_bits
            if used_bits + cost > budget_bits:
                continue

            rel = relevance_score(t, q_concepts)
            cov = coverage_gain(t, selected, q_concepts)
            red = redundancy_penalty(t, selected)
            ch = channel_reliability_proxy(cost, snr_db, channel_type)
            cost_penalty = cost / budget_bits

            gain = (
                alpha * rel
                + beta * cov
                + gamma * ch
                - lamb * red
                - mu * cost_penalty
            )

            score = gain / cost

            if score > best_score:
                best_score = score
                best = t

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)
        used_bits += packet_bits

    # Important: return selected first, then rest.
    # Existing pipeline will do ranked[:n_top].
    selected_ids = {id(x) for x in selected}
    return selected + [t for t in triplets if id(t) not in selected_ids]
