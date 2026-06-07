from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.semantic.ranking import (
    build_object_frequency,
    build_relation_frequency,
    rank_bboxes_do,
    rank_bboxes_go,
    rank_bboxes_original,
    rank_triplets_do,
    rank_triplets_go,
    rank_triplets_original,
)
from src.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    samples = ds.load_samples()
    bbox_rows = ds.load_bbox_packets(limit=5)
    sg_rows = ds.load_sg_triplets(limit=5)

    sample_by_qid = {s["question_id"]: s for s in samples}

    object_freq = build_object_frequency(samples)
    relation_freq = build_relation_frequency(samples)

    for bbox_row, sg_row in zip(bbox_rows, sg_rows):
        qid = bbox_row["question_id"]
        sample = sample_by_qid[qid]
        keywords = sample["keywords"]

        print("=" * 80)
        print("question_id:", qid)
        print("question:", sample["question"])
        print("answer:", sample["answer"])
        print("keywords:", keywords)

        print("\nOriginal-BBox top 6:")
        for b in rank_bboxes_original(bbox_row["bboxes"])[:6]:
            print(b["object"], b["object_id"])

        print("\nDO-BBox top 6:")
        for b in rank_bboxes_do(bbox_row["bboxes"], object_freq)[:6]:
            print(b["object"], b["object_id"])

        print("\nGO-BBox top 6:")
        for b in rank_bboxes_go(bbox_row["bboxes"], keywords, object_freq)[:6]:
            print(b["object"], b["object_id"])

        print("\nOriginal-SG top 6:")
        for t in rank_triplets_original(sg_row["triplets"])[:6]:
            print(t["subject"], "-", t["relation"], "-", t["object"])

        print("\nDO-SG top 6:")
        for t in rank_triplets_do(sg_row["triplets"], object_freq, relation_freq)[:6]:
            print(t["subject"], "-", t["relation"], "-", t["object"])

        print("\nGO-SG top 6:")
        for t in rank_triplets_go(sg_row["triplets"], keywords, object_freq, relation_freq)[:6]:
            print(t["subject"], "-", t["relation"], "-", t["object"])


if __name__ == "__main__":
    main()
