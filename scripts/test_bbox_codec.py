from pathlib import Path

from src.data.gqa_subset import GQACommSubset
from src.semantic.packet_codec import (
    BBoxPacket,
    decode_bboxes,
    encode_bboxes,
)
from src.eval.semantic_metrics import (
    bbox_exact_match_rate,
    bbox_mean_l1_error,
    bbox_object_accuracy,
)
from src.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    ds = GQACommSubset(Path(cfg["data"]["root"]))

    row = ds.load_bbox_packets(limit=1)[0]
    bboxes = row["bboxes"][:9]

    packets = [
        BBoxPacket(
            object_id=b["object_id"],
            x1=b["bbox"][0],
            y1=b["bbox"][1],
            x2=b["bbox"][2],
            y2=b["bbox"][3],
        )
        for b in bboxes
    ]

    bits = encode_bboxes(packets)
    decoded = decode_bboxes(bits, num_bboxes=len(packets))

    print("question_id:", row["question_id"])
    print("image_id:", row["image_id"])
    print("num_bboxes:", len(packets))
    print("num_bits:", len(bits))
    print("expected_bits:", 80 * len(packets))

    print("object_accuracy:", bbox_object_accuracy(packets, decoded))
    print("bbox_l1_error:", bbox_mean_l1_error(packets, decoded))
    print("bbox_exact_match:", bbox_exact_match_rate(packets, decoded))

    print("\nOriginal packets:")
    for p in packets:
        print(p)

    print("\nDecoded packets:")
    for p in decoded:
        print(p)


if __name__ == "__main__":
    main()
