import csv
from pathlib import Path

import pandas as pd

from src.baselines.image_transmission import (
    compression_ratio,
    raw_float32_image_bits,
)
from src.comm.latency import communication_latency_sec
from src.data.gqa_subset import GQACommSubset
from src.utils.config import load_yaml


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def load_semantic_reference(results_dir: Path):
    """
    Load semantic reference from answerability_sweep.csv if available.

    We use GO rows because GO is the method of interest.
    For SG: use strict validated proxy answerability.
    For BBox: use loose validated proxy answerability.
    """
    path = results_dir / "answerability_sweep.csv"

    if not path.exists():
        return []

    df = pd.read_csv(path)

    refs = []

    required_cols = {
        "semantic_type",
        "ranking_method",
        "channel",
        "snr_db",
        "n_top",
        "avg_source_bits",
        "t_com_sec",
    }

    if not required_cols.issubset(set(df.columns)):
        return []

    df = df[df["ranking_method"] == "go"].copy()

    for _, row in df.iterrows():
        semantic_type = row["semantic_type"]

        if semantic_type == "sg":
            answer_col = (
                "answerability_after_strict_validated"
                if "answerability_after_strict_validated" in df.columns
                else "answerability_after_strict"
            )
        elif semantic_type == "bbox":
            answer_col = (
                "answerability_after_loose_validated"
                if "answerability_after_loose_validated" in df.columns
                else "answerability_after_loose"
            )
        else:
            continue

        refs.append(
            {
                "semantic_type": semantic_type,
                "ranking_method": "go",
                "channel": row["channel"],
                "snr_db": float(row["snr_db"]),
                "n_top": int(row["n_top"]),
                "semantic_bits": float(row["avg_source_bits"]),
                "semantic_t_com_sec": float(row["t_com_sec"]),
                "semantic_answerability_validated": float(row[answer_col]),
            }
        )

    return refs


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg["data"]["root"])
    ds = GQACommSubset(data_root)

    # Count samples/images for reporting only.
    samples = ds.load_samples()
    num_samples = len(samples)
    num_images = len(set(s["image_id"] for s in samples if "image_id" in s))

    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])
    snr_db_list = [float(x) for x in cfg["channel"]["snr_db_list"]]
    channels = ["awgn", "rayleigh"]

    # Paper-style image baseline.
    image_height = int(cfg.get("image", {}).get("height", 480))
    image_width = int(cfg.get("image", {}).get("width", 320))
    image_channels = int(cfg.get("image", {}).get("channels", 3))
    image_bits_per_value = int(cfg.get("image", {}).get("bits_per_value", 32))

    image_bits = raw_float32_image_bits(
        height=image_height,
        width=image_width,
        channels=image_channels,
        bits_per_value=image_bits_per_value,
    )

    semantic_refs = load_semantic_reference(output_dir)

    out_rows = []

    # Pure image baseline rows.
    for channel in channels:
        for snr_db in snr_db_list:
            t_image = communication_latency_sec(
                num_bits=image_bits,
                bandwidth_hz=bandwidth_hz,
                snr_db=snr_db,
            )

            out_rows.append(
                {
                    "row_type": "image_only",
                    "image_mode": "raw_float32",
                    "channel": channel,
                    "snr_db": snr_db,
                    "num_samples": num_samples,
                    "num_images": num_images,
                    "image_height": image_height,
                    "image_width": image_width,
                    "image_channels": image_channels,
                    "image_bits_per_value": image_bits_per_value,
                    "image_bits": image_bits,
                    "image_t_com_sec": t_image,
                    "semantic_type": "",
                    "ranking_method": "",
                    "n_top": "",
                    "semantic_bits": "",
                    "semantic_t_com_sec": "",
                    "semantic_answerability_validated": "",
                    "image_to_semantic_bit_ratio": "",
                    "image_to_semantic_latency_ratio": "",
                }
            )

    # Comparison rows against semantic GO results, if available.
    for ref in semantic_refs:
        t_image = communication_latency_sec(
            num_bits=image_bits,
            bandwidth_hz=bandwidth_hz,
            snr_db=float(ref["snr_db"]),
        )

        semantic_bits = float(ref["semantic_bits"])
        semantic_t = float(ref["semantic_t_com_sec"])

        out_rows.append(
            {
                "row_type": "image_vs_semantic",
                "image_mode": "raw_float32",
                "channel": ref["channel"],
                "snr_db": ref["snr_db"],
                "num_samples": num_samples,
                "num_images": num_images,
                "image_height": image_height,
                "image_width": image_width,
                "image_channels": image_channels,
                "image_bits_per_value": image_bits_per_value,
                "image_bits": image_bits,
                "image_t_com_sec": t_image,
                "semantic_type": ref["semantic_type"],
                "ranking_method": ref["ranking_method"],
                "n_top": ref["n_top"],
                "semantic_bits": semantic_bits,
                "semantic_t_com_sec": semantic_t,
                "semantic_answerability_validated": ref["semantic_answerability_validated"],
                "image_to_semantic_bit_ratio": compression_ratio(image_bits, semantic_bits),
                "image_to_semantic_latency_ratio": t_image / semantic_t if semantic_t > 0 else 0.0,
            }
        )

    output_path = output_dir / "image_baseline.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"image_bits: {image_bits}")
    print(f"num_samples: {num_samples}")
    print(f"num_images: {num_images}")
    print(f"semantic_reference_rows: {len(semantic_refs)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
