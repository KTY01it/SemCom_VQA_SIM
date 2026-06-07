from collections import Counter
from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    data_root = Path(cfg["data"]["root"])

    ds = GQACommSubset(data_root)

    samples = ds.load_samples()
    bbox_packets = ds.load_bbox_packets(limit=3)
    sg_triplets = ds.load_sg_triplets(limit=3)
    object_vocab = ds.load_object_vocab()
    relation_vocab = ds.load_relation_vocab()
    answer_vocab = ds.load_answer_vocab()
    stats = ds.load_stats()

    print("=== GQA Communication Subset Check ===")
    print(f"data_root: {data_root}")
    print(f"num_samples: {len(samples)}")
    print(f"object_vocab_size: {len(object_vocab)}")
    print(f"relation_vocab_size: {len(relation_vocab)}")
    print(f"answer_vocab_size: {len(answer_vocab)}")

    if samples:
        print("\n=== Sample keys ===")
        print(sorted(samples[0].keys()))

        if "question_type" in samples[0]:
            qtype_counter = Counter(s.get("question_type", "unknown") for s in samples)
            print("\n=== Question type distribution ===")
            for key, value in qtype_counter.most_common():
                print(f"{key}: {value}")

        if "image_id" in samples[0]:
            num_images = len(set(s["image_id"] for s in samples))
            print(f"\nnum_unique_images: {num_images}")

    print("\n=== First sample ===")
    print(samples[0] if samples else None)

    print("\n=== First bbox packets ===")
    for row in bbox_packets:
        print(row)

    print("\n=== First sg triplets ===")
    for row in sg_triplets:
        print(row)

    print("\n=== stats.json ===")
    print(stats)


if __name__ == "__main__":
    main()
