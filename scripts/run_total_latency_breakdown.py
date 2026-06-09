import csv
from pathlib import Path

import pandas as pd

from src.comm.total_latency import TotalLatencyConfig, total_latency_breakdown
from src.utils.config import load_yaml


def make_latency_config(cfg) -> TotalLatencyConfig:
    latency_cfg = cfg.get("latency", {})

    return TotalLatencyConfig(
        bandwidth_hz=float(latency_cfg.get("bandwidth_hz", 100_000.0)),
        device_tflops=float(latency_cfg.get("device_tflops", 1.33)),
        bbox_extraction_tflops=float(
            latency_cfg.get("bbox_extraction_tflops", 0.44)
        ),
        sg_generation_tflops=float(latency_cfg.get("sg_generation_tflops", 0.02)),
        question_parser_ms=float(latency_cfg.get("question_parser_ms", 0.0)),
        answer_reasoning_ms=float(latency_cfg.get("answer_reasoning_ms", 0.0)),
        ranking_ms=float(latency_cfg.get("ranking_ms", 0.1)),
        channel_decode_ms=float(latency_cfg.get("channel_decode_ms", 0.0)),
        raw_image_processing_ms=float(
            latency_cfg.get("raw_image_processing_ms", 0.0)
        ),
    )


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def write_csv_union_fieldnames(output_path: Path, rows: list[dict]) -> None:
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_answerability_rows(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "answerability_sweep.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_image_rows(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "image_baseline.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def get_answerability_value(row) -> float:
    semantic_type = row["semantic_type"]

    if semantic_type == "sg":
        preferred = "answerability_after_strict_validated"
        fallback = "answerability_after_strict"
    elif semantic_type == "bbox":
        preferred = "answerability_after_loose_validated"
        fallback = "answerability_after_loose"
    else:
        return 0.0

    if preferred in row and pd.notna(row[preferred]):
        return float(row[preferred])

    if fallback in row and pd.notna(row[fallback]):
        return float(row[fallback])

    return 0.0


def build_semantic_latency_rows(
    answer_df: pd.DataFrame,
    latency_cfg: TotalLatencyConfig,
) -> list[dict]:
    """
    Build total latency rows for GO semantic transmission.

    Uses results/answerability_sweep.csv.
    """
    rows = []

    df = answer_df[
        (answer_df["ranking_method"] == "go")
        & (answer_df["semantic_type"].isin(["sg", "bbox"]))
    ].copy()

    for _, row in df.iterrows():
        semantic_type = row["semantic_type"]
        num_bits = float(row["avg_source_bits"])
        snr_db = float(row["snr_db"])

        breakdown = total_latency_breakdown(
            semantic_type=semantic_type,
            num_bits=num_bits,
            snr_db=snr_db,
            cfg=latency_cfg,
        )

        out = {
            "row_type": "semantic",
            "semantic_type": semantic_type,
            "ranking_method": row["ranking_method"],
            "channel": row["channel"],
            "snr_db": snr_db,
            "n_top": int(float(row["n_top"])),
            "num_bits": num_bits,
            "answerability_validated": get_answerability_value(row),
            "valid_packet_rate": float(row.get("valid_packet_rate", 0.0)),
            "invalid_packet_rate": float(row.get("invalid_packet_rate", 0.0)),
        }
        out.update(breakdown)
        rows.append(out)

    return rows


def build_image_latency_rows(
    image_df: pd.DataFrame,
    latency_cfg: TotalLatencyConfig,
) -> list[dict]:
    """
    Build total latency rows for raw image baseline.

    Uses image_only rows from results/image_baseline.csv.
    """
    rows = []

    df = image_df[image_df["row_type"] == "image_only"].copy()

    for _, row in df.iterrows():
        num_bits = float(row["image_bits"])
        snr_db = float(row["snr_db"])

        breakdown = total_latency_breakdown(
            semantic_type="image",
            num_bits=num_bits,
            snr_db=snr_db,
            cfg=latency_cfg,
        )

        out = {
            "row_type": "image",
            "semantic_type": "image",
            "ranking_method": "raw_float32",
            "channel": row["channel"],
            "snr_db": snr_db,
            "n_top": "",
            "num_bits": num_bits,
            "answerability_validated": "",
            "valid_packet_rate": "",
            "invalid_packet_rate": "",
        }
        out.update(breakdown)
        rows.append(out)

    return rows


def build_comparison_rows(
    latency_rows: list[dict],
) -> list[dict]:
    """
    Compare image total latency against GO-SG / GO-BBox at same channel/SNR.

    Since answerability_sweep.csv currently uses snr_db=8.0 only,
    comparison rows will be generated mainly for SNR=8.
    """
    image_idx = {}
    semantic_rows = []

    for row in latency_rows:
        if row["row_type"] == "image":
            image_idx[(row["channel"], float(row["snr_db"]))] = row
        elif row["row_type"] == "semantic":
            semantic_rows.append(row)

    out_rows = []

    for sem in semantic_rows:
        key = (sem["channel"], float(sem["snr_db"]))
        image = image_idx.get(key)

        if image is None:
            continue

        image_total = float(image["t_total_ms"])
        sem_total = float(sem["t_total_ms"])

        image_com = float(image["t_com_ms"])
        sem_com = float(sem["t_com_ms"])

        out_rows.append(
            {
                "row_type": "image_vs_semantic_total",
                "semantic_type": sem["semantic_type"],
                "ranking_method": "go",
                "channel": sem["channel"],
                "snr_db": sem["snr_db"],
                "n_top": sem["n_top"],
                "image_bits": image["num_bits"],
                "semantic_bits": sem["num_bits"],
                "image_t_com_ms": image_com,
                "semantic_t_com_ms": sem_com,
                "image_t_total_ms": image_total,
                "semantic_t_total_ms": sem_total,
                "image_to_semantic_com_ratio": image_com / sem_com
                if sem_com > 0
                else 0.0,
                "image_to_semantic_total_ratio": image_total / sem_total
                if sem_total > 0
                else 0.0,
                "semantic_answerability_validated": sem["answerability_validated"],
                "semantic_valid_packet_rate": sem["valid_packet_rate"],
            }
        )

    return out_rows


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    results_dir = Path(cfg["output"]["dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    latency_cfg = make_latency_config(cfg)

    answer_df = load_answerability_rows(results_dir)
    image_df = load_image_rows(results_dir)

    semantic_rows = build_semantic_latency_rows(answer_df, latency_cfg)
    image_rows = build_image_latency_rows(image_df, latency_cfg)
    comparison_rows = build_comparison_rows(image_rows + semantic_rows)

    out_rows = image_rows + semantic_rows + comparison_rows

    output_path = results_dir / "latency_breakdown.csv"
    write_csv_union_fieldnames(output_path, out_rows)

    print(f"semantic_rows: {len(semantic_rows)}")
    print(f"image_rows: {len(image_rows)}")
    print(f"comparison_rows: {len(comparison_rows)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
