import argparse
from pathlib import Path

import pandas as pd


def normalize_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = df["ranking_method"].map({
        "original": "Original-SG",
        "do": "DO-SG",
        "go": "GO-SG",
        "dbss": "DBSS",
    }).fillna(df["ranking_method"])
    df["answer_per_kbit"] = (
        df["delivered_answer_hit_rate"] / (df["avg_source_bits"] / 1000.0)
    )
    df["tx_field_keep_ratio"] = 1.0
    return df


def normalize_qtc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = "DBSS-QTC-noanswer"
    return df


def normalize_nst(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    df = df.copy()
    df["method"] = method_name
    return df


def load_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/nst")
    parser.add_argument("--out", default="results/nst/summary_eval8000_9999.csv")
    args = parser.parse_args()

    root = Path(args.result_dir)

    baseline = load_optional(root / "baselines_eval8000_9999_awgn_rayleigh_snr.csv")
    qtc_awgn = load_optional(root / "qtc_noanswer_eval8000_9999_awgn_snr.csv")
    qtc_rayleigh = load_optional(root / "qtc_noanswer_eval8000_9999_rayleigh_snr.csv")
    nst_awgn = load_optional(root / "nst_v3_main_eval8000_9999_awgn_snr.csv")
    nst_rayleigh = load_optional(root / "nst_v3_main_eval8000_9999_rayleigh_snr.csv")
    nst_aggr = load_optional(root / "nst_v3_aggressive_eval8000_9999_awgn8.csv")

    frames = []

    if not baseline.empty:
        frames.append(normalize_baseline(baseline))

    if not qtc_awgn.empty:
        frames.append(normalize_qtc(qtc_awgn))

    if not qtc_rayleigh.empty:
        frames.append(normalize_qtc(qtc_rayleigh))

    if not nst_awgn.empty:
        frames.append(normalize_nst(nst_awgn, "DBSS-NST-v3"))

    if not nst_rayleigh.empty:
        frames.append(normalize_nst(nst_rayleigh, "DBSS-NST-v3"))

    if not nst_aggr.empty:
        frames.append(normalize_nst(nst_aggr, "DBSS-NST-v3-aggressive"))

    if not frames:
        raise RuntimeError("No result files were loaded.")

    out = pd.concat(frames, ignore_index=True)

    keep_cols = [
        "method",
        "semantic_type",
        "channel",
        "snr_db",
        "n_top",
        "num_samples",
        "avg_source_bits",
        "delivered_answer_hit_rate",
        "answer_per_kbit",
        "tx_field_keep_ratio",
        "ber",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

    focus = out[
        (out["snr_db"].astype(float) == 8.0)
        & (out["channel"] == "awgn")
    ].copy()

    if not focus.empty:
        focus = focus.sort_values("answer_per_kbit", ascending=False)
        print("\n=== AWGN 8 dB summary ===")
        print(focus[[
            "method",
            "avg_source_bits",
            "delivered_answer_hit_rate",
            "answer_per_kbit",
            "tx_field_keep_ratio",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
