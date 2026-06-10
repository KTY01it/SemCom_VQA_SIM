from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ProxyMetricContract:
    metric_family: str
    metric_name: str
    semantic_type: str
    validated: bool
    invalid_packet_policy: str
    supported_answers: str
    excluded_answers: str
    evidence_definition: str
    not_claimed_as: str


def sg_strict_contract() -> ProxyMetricContract:
    return ProxyMetricContract(
        metric_family="validated_answer_related_semantic_coverage",
        metric_name="sg_strict_validated_proxy_answerability",
        semantic_type="scene_graph_triplets",
        validated=True,
        invalid_packet_policy="drop invalid decoded packets before computing coverage",
        supported_answers="non-empty non-yes/no answers",
        excluded_answers="yes/no answers",
        evidence_definition=(
            "A sample is counted as answerable if at least one delivered valid "
            "SG triplet contains the normalized answer token and overlaps with "
            "question keywords through subject, relation, or object text."
        ),
        not_claimed_as="full VQA accuracy or neural answer generation accuracy",
    )


def sg_loose_contract() -> ProxyMetricContract:
    return ProxyMetricContract(
        metric_family="validated_answer_related_semantic_coverage",
        metric_name="sg_loose_validated_proxy_answerability",
        semantic_type="scene_graph_triplets",
        validated=True,
        invalid_packet_policy="drop invalid decoded packets before computing coverage",
        supported_answers="non-empty non-yes/no answers",
        excluded_answers="yes/no answers",
        evidence_definition=(
            "A sample is counted as answerable if at least one delivered valid "
            "SG triplet contains the normalized answer token in subject, relation, "
            "or object text."
        ),
        not_claimed_as="full VQA accuracy or neural answer generation accuracy",
    )


def bbox_contract() -> ProxyMetricContract:
    return ProxyMetricContract(
        metric_family="validated_answer_related_semantic_coverage",
        metric_name="bbox_validated_proxy_answerability",
        semantic_type="object_bounding_boxes",
        validated=True,
        invalid_packet_policy="drop invalid decoded packets before computing coverage",
        supported_answers="non-empty non-yes/no answers",
        excluded_answers="yes/no answers",
        evidence_definition=(
            "A sample is counted as answerable if at least one delivered valid "
            "BBox object/attribute contains the normalized answer token."
        ),
        not_claimed_as="full VQA accuracy or neural answer generation accuracy",
    )


def all_contracts() -> list[dict]:
    return [
        asdict(sg_strict_contract()),
        asdict(sg_loose_contract()),
        asdict(bbox_contract()),
    ]
