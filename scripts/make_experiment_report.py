from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
REPORT_PATH = RESULTS_DIR / "experiment_summary.md"


def read_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"_Missing file: `{path}`_"


def csv_brief(path: Path, max_rows: int = 8) -> str:
    if not path.exists():
        return f"_Missing CSV: `{path}`_"

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return f"_Failed to read `{path}`: {exc}_"

    lines = []
    lines.append(f"- path: `{path}`")
    lines.append(f"- rows: {len(df)}")
    lines.append(f"- columns: `{', '.join(df.columns)}`")

    if len(df) > 0:
        preview = df.head(max_rows).to_csv(index=False)
        lines.append("")
        lines.append("```csv")
        lines.append(preview.strip())
        lines.append("```")

    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sections = []

    sections.append("# SemCom VQA Simulation Experiment Summary")
    sections.append("")
    sections.append("This report summarizes the current communication-model simulation results.")
    sections.append("")
    sections.append("> Note: task-level metrics named `answerability` are proxy metrics. They measure whether delivered semantic packets contain answer-related semantic content. They are not full VQA answering accuracy.")
    sections.append("")

    sections.append("## 1. Available Result Files")
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    txt_files = sorted(RESULTS_DIR.glob("*.txt"))
    png_files = sorted((RESULTS_DIR / "figures").glob("*.png")) if (RESULTS_DIR / "figures").exists() else []

    sections.append(f"- CSV files: {len(csv_files)}")
    sections.append(f"- TXT summaries: {len(txt_files)}")
    sections.append(f"- Figure PNGs: {len(png_files)}")
    sections.append("")

    sections.append("## 2. Main Summary Tables")
    sections.append("### 2.1 No-LDPC / Uncoded Summary")
    sections.append(read_text_if_exists(RESULTS_DIR / "summary_tables.txt"))
    sections.append("")

    sections.append("### 2.2 LDPC-like Summary")
    sections.append(read_text_if_exists(RESULTS_DIR / "summary_ldpc_tables.txt"))
    sections.append("")

    sections.append("## 3. CSV Briefs")
    expected_csvs = [
        "comm_sanity.csv",
        "sg_packet_sanity.csv",
        "bbox_packet_sanity.csv",
        "ntop_sweep.csv",
        "ranking_sweep.csv",
        "answerability_sweep.csv",
        "answerability_sweep_ldpc.csv",
        "image_baseline.csv",
        "latency_breakdown.csv",
        "packet_validation_sanity.csv",
    ]

    for name in expected_csvs:
        sections.append(f"### 3.x `{name}`")
        sections.append(csv_brief(RESULTS_DIR / name))
        sections.append("")

    sections.append("## 4. Current Known Limitations")
    sections.append("- `answerability` is a validated proxy metric, not full VQA accuracy.")
    sections.append("- Current coding is LDPC-like sparse systematic block coding, not yet standard LDPC BP/min-sum with soft LLR.")
    sections.append("- Image Transmission baseline is a raw-float32 image-size baseline, not JPEG-compressed image transmission.")
    sections.append("- Total latency uses paper-style FLOPs approximation, not measured runtime profiling.")
    sections.append("- Packet validation currently covers vocab/range/geometry validation; CRC helper exists, but CRC is not yet integrated into all transmission sweeps.")
    sections.append("")

    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()