from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.utils.config import load_yaml


def main():
    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    samples = ds.load_samples(limit=3)
    sg_rows = ds.load_sg_triplets(limit=3)

    for s, r in zip(samples, sg_rows):
        print("=" * 80)
        print("sample keys:", sorted(s.keys()))
        print("question_id:", s.get("question_id"))
        print("question:", s.get("question"))
        print("answer:", s.get("answer"))
        print("keywords:", s.get("keywords"))
        print("question_type:", s.get("question_type"))

        print("sg row keys:", sorted(r.keys()))
        triplets = r.get("triplets", [])
        print("num_triplets:", len(triplets))
        print("first triplet:", triplets[0] if triplets else None)


if __name__ == "__main__":
    main()
