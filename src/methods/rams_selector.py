from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.methods.dbss import (
    channel_reliability_proxy,
    coverage_gain,
    question_concepts,
    redundancy_penalty,
    relevance_score,
)


FIELD_NAMES = ["subject", "relation", "object"]


def make_keep_mask(mask_probs: List[float], threshold: float) -> Dict[str, bool]:
    keep = {
        "subject": bool(mask_probs[0] >= threshold),
        "relation": bool(mask_probs[1] >= threshold),
        "object": bool(mask_probs[2] >= threshold),
    }

    if not any(keep.values()):
        # Keep the strongest field if all probabilities are below threshold.
        best_idx = max(range(3), key=lambda i: float(mask_probs[i]))
        keep[FIELD_NAMES[best_idx]] = True

    return keep


def estimate_compressed_sg_bits(keep: Dict[str, bool]) -> int:
    """
    compressed_sg_codec uses:
      3 mask bits + 16 bits per retained field.
    """
    num_fields = int(keep["subject"]) + int(keep["relation"]) + int(keep["object"])
    return 3 + 16 * max(1, num_fields)


def mask_survival_proxy(mask_probs: List[float], keep: Dict[str, bool]) -> float:
    """
    Inference-safe proxy for whether the evidence-bearing field survives compression.
    Since the true answer field is unknown at inference, use the strongest retained
    field probability as a conservative proxy.
    """
    vals = []
    for idx, name in enumerate(FIELD_NAMES):
        if keep[name]:
            vals.append(float(mask_probs[idx]))

    if not vals:
        return 0.0

    return max(vals)


def noisy_or_marginal_gain(current_fail_prob: float, p_success: float) -> float:
    """
    P(S) = 1 - prod_j(1 - p_success_j)
    Delta_i = P(S union i) - P(S) = current_fail_prob * p_success_i
    """
    return float(current_fail_prob) * float(p_success)


def rams_select_triplets(
    triplets: List[Dict[str, Any]],
    evidence_probs: List[float],
    mask_probs: List[List[float]],
    question: str | None,
    keywords: Iterable[str] | None,
    n_top: int,
    snr_db: float,
    channel_type: str,
    mask_threshold: float = 0.2,
    lambda_survival: float = 4.0,
    lambda_relevance: float = 0.5,
    lambda_coverage: float = 0.5,
    lambda_redundancy: float = 0.2,
    lambda_cost: float = 0.01,
) -> Tuple[List[int], List[Dict[str, bool]]]:
    """
    Reliability-Aware MIL/Submodular Semantic Selector.

    Selects a subset by maximizing:
      survival Noisy-OR marginal gain
      + question relevance
      + new question-concept coverage
      - redundancy
      - bit cost

    No answer is used at inference.
    """
    if not triplets:
        return [], []

    q_concepts = question_concepts(question, keywords)

    selected_indices: List[int] = []
    selected_triplets: List[Dict[str, Any]] = []
    selected_masks: List[Dict[str, bool]] = []
    remaining = list(range(len(triplets)))

    current_fail_prob = 1.0

    while remaining and len(selected_indices) < n_top:
        best_idx = None
        best_mask = None
        best_score = -1e18
        best_p_success = 0.0

        for idx in remaining:
            t = triplets[idx]

            keep = make_keep_mask(mask_probs[idx], mask_threshold)
            cost_bits = estimate_compressed_sg_bits(keep)

            p_evi = min(max(float(evidence_probs[idx]), 1e-6), 1.0 - 1e-6)
            p_keep = min(max(mask_survival_proxy(mask_probs[idx], keep), 1e-6), 1.0)
            p_ch = channel_reliability_proxy(
                packet_bits=cost_bits,
                snr_db=snr_db,
                channel_type=channel_type,
            )

            p_success = p_evi * p_keep * p_ch

            survival_gain = noisy_or_marginal_gain(current_fail_prob, p_success)
            rel = relevance_score(t, q_concepts)
            cov = coverage_gain(t, selected_triplets, q_concepts)
            red = redundancy_penalty(t, selected_triplets)

            score = (
                lambda_survival * survival_gain
                + lambda_relevance * rel
                + lambda_coverage * cov
                - lambda_redundancy * red
                - lambda_cost * (cost_bits / 64.0)
            )

            if score > best_score:
                best_score = score
                best_idx = idx
                best_mask = keep
                best_p_success = p_success

        if best_idx is None or best_mask is None:
            break

        selected_indices.append(best_idx)
        selected_triplets.append(triplets[best_idx])
        selected_masks.append(best_mask)
        remaining.remove(best_idx)

        current_fail_prob *= max(1e-6, 1.0 - best_p_success)

    return selected_indices, selected_masks
