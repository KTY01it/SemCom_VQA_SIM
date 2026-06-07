from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.semantic.packet_codec import (
    SGTripletPacket,
    decode_sg_triplets,
    encode_sg_triplets,
)
from src.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    row = ds.load_sg_triplets(limit=1)[0]
    triplets = row["triplets"][:9]

    packets = [
        SGTripletPacket(
            subject_id=t["subject_id"],
            relation_id=t["relation_id"],
            object_id=t["object_id"],
        )
        for t in triplets
    ]

    bits = encode_sg_triplets(packets)
    decoded = decode_sg_triplets(bits, num_triplets=len(packets))

    print("question_id:", row["question_id"])
    print("image_id:", row["image_id"])
    print("num_triplets:", len(packets))
    print("num_bits:", len(bits))
    print("expected_bits:", 48 * len(packets))
    print("exact_match:", packets == decoded)

    print("\nOriginal packets:")
    for p in packets:
        print(p)

    print("\nDecoded packets:")
    for p in decoded:
        print(p)


if __name__ == "__main__":
    main()
