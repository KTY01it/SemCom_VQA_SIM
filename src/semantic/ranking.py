from collections import Counter
from typing import Any, Dict, List, Tuple


def normalize_text(x: str) -> str:
    return str(x).lower().strip()


def build_object_frequency(samples: List[Dict[str, Any]]) -> Counter:
    counter = Counter()

    for sample in samples:
        for obj in sample.get("objects", []):
            name = normalize_text(obj.get("name", ""))
            if name:
                counter[name] += 1

    return counter


def build_relation_frequency(samples: List[Dict[str, Any]]) -> Counter:
    counter = Counter()

    for sample in samples:
        for rel in sample.get("relations", []):
            name = normalize_text(rel.get("relation", ""))
            if name:
                counter[name] += 1

    return counter


def rank_bboxes_original(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(bboxes)


def rank_triplets_original(triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(triplets)


def rank_bboxes_do(
    bboxes: List[Dict[str, Any]],
    object_freq: Counter,
) -> List[Dict[str, Any]]:
    def score(b: Dict[str, Any]) -> Tuple[int, str]:
        obj = normalize_text(b.get("object", ""))
        return object_freq[obj], obj

    return sorted(bboxes, key=score, reverse=True)


def rank_triplets_do(
    triplets: List[Dict[str, Any]],
    object_freq: Counter,
    relation_freq: Counter,
) -> List[Dict[str, Any]]:
    def score(t: Dict[str, Any]) -> Tuple[int, int, int, str]:
        subj = normalize_text(t.get("subject", ""))
        rel = normalize_text(t.get("relation", ""))
        obj = normalize_text(t.get("object", ""))

        return (
            object_freq[subj] + object_freq[obj],
            relation_freq[rel],
            object_freq[subj],
            rel,
        )

    return sorted(triplets, key=score, reverse=True)


def rank_bboxes_go(
    bboxes: List[Dict[str, Any]],
    keywords: List[str],
    object_freq: Counter | None = None,
) -> List[Dict[str, Any]]:
    kw = {normalize_text(k) for k in keywords}

    def score(b: Dict[str, Any]) -> Tuple[int, int, str]:
        obj = normalize_text(b.get("object", ""))

        keyword_hit = int(obj in kw)
        freq = object_freq[obj] if object_freq is not None else 0

        return keyword_hit, freq, obj

    return sorted(bboxes, key=score, reverse=True)


def rank_triplets_go(
    triplets: List[Dict[str, Any]],
    keywords: List[str],
    object_freq: Counter | None = None,
    relation_freq: Counter | None = None,
) -> List[Dict[str, Any]]:
    kw = {normalize_text(k) for k in keywords}

    def score(t: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
        subj = normalize_text(t.get("subject", ""))
        rel = normalize_text(t.get("relation", ""))
        obj = normalize_text(t.get("object", ""))

        subj_hit = int(subj in kw)
        rel_hit = int(rel in kw)
        obj_hit = int(obj in kw)

        goal_score = subj_hit + rel_hit + obj_hit

        obj_freq_score = 0
        rel_freq_score = 0

        if object_freq is not None:
            obj_freq_score = object_freq[subj] + object_freq[obj]

        if relation_freq is not None:
            rel_freq_score = relation_freq[rel]

        return goal_score, rel_hit, obj_freq_score, rel_freq_score, rel

    return sorted(triplets, key=score, reverse=True)
