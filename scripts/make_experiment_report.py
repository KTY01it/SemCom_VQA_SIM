from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "experiment_summary.md"


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return f"_Missing: `{path.relative_to(ROOT)}`_\n"
    return path.read_text(encoding="utf-8", errors="ignore")


def read_csv_preview(path: Path, max_rows: int = 12) -> str:
    if not path.exists():
        return f"_Missing: `{path.relative_to(ROOT)}`_\n"
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return "_Empty CSV_\n"
        return df.head(max_rows).to_markdown(index=False)
    except Exception as exc:
        return f"_Could not read `{path.relative_to(ROOT)}`: {exc}_\n"


def section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body.strip()}\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    parts = []
    parts.append("# Communication-Model Simulation Summary\n")
    parts.append(
        "This report summarizes the current SemCom-VQA communication simulation environment. "
        "Metrics named `proxy_answerability` are task-level proxies, not full VQA accuracy.\n"
    )

    parts.append(section("1. Communication sanity", read_csv_preview(RESULTS / "comm_sanity.csv")))
    parts.append(section("2. LDPC / LDPC-like sanity", read_csv_preview(RESULTS / "ldpc_sanity.csv")))
    parts.append(section("3. Ranking sweep", read_csv_preview(RESULTS / "ranking_sweep.csv")))
    parts.append(section("4. Answerability sweep", read_csv_preview(RESULTS / "answerability_sweep.csv")))
    parts.append(section("5. LDPC answerability sweep", read_csv_preview(RESULTS / "answerability_sweep_ldpc.csv")))
    parts.append(section("6. Image baseline", read_csv_preview(RESULTS / "image_baseline.csv")))
    parts.append(section("7. Latency breakdown", read_csv_preview(RESULTS / "latency_breakdown.csv")))
    parts.append(section("8. Summary tables", read_text_if_exists(RESULTS / "summary_tables.txt")))
    parts.append(section("9. LDPC summary tables", read_text_if_exists(RESULTS / "summary_ldpc_tables.txt")))

    parts.append(
        section(
            "10. Known limitations",
            "\n".join(
                [
                    "- Current task score is proxy answerability, not full VQA answering accuracy.",
                    "- Current channel coding should be treated as LDPC-like unless replaced by standard LDPC BP with soft LLR.",
                    "- Rayleigh fading assumes simplified channel/equalization unless otherwise configured.",
                    "- Image transmission baseline should be included for paper-level comparison.",
                    "- Total latency should include question parser, semantic extraction, communication, decoding, and answer reasoning.",
                ]
            ),
        )
    )

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()