from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    print("=== First 5 samples ===")
    for row in ds.load_samples(limit=5):
        print(row)
        print("---")

    print("\n=== First 5 bbox packets ===")
    for row in ds.load_bbox_packets(limit=5):
        print(row)
        print("---")

    print("\n=== First 5 sg triplets ===")
    for row in ds.load_sg_triplets(limit=5):
        print(row)
        print("---")


if __name__ == "__main__":
    main()
