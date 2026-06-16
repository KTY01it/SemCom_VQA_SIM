from pathlib import Path

import pandas as pd


ROOT = Path("results/nst")


def normalize_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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


def normalize_simple(path: Path, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = method_name
    return df


def main():
    frames = []

    baseline_path = ROOT / "ntop_baselines_eval8000_9999_awgn8.csv"
    qtc_path = ROOT / "ntop_qtc_noanswer_eval8000_9999_awgn8.csv"
    nst_path = ROOT / "ntop_nst_v3_main_eval8000_9999_awgn8.csv"
    aggr_path = ROOT / "ntop_nst_v3_aggressive_eval8000_9999_awgn8.csv"

    if baseline_path.exists():
        frames.append(normalize_baseline(baseline_path))
    else:
        print(f"[WARN] missing {baseline_path}")

    if qtc_path.exists():
        frames.append(normalize_simple(qtc_path, "DBSS-QTC-noanswer"))
    else:
        print(f"[WARN] missing {qtc_path}")

    if nst_path.exists():
        frames.append(normalize_simple(nst_path, "DBSS-NST-v3"))
    else:
        print(f"[WARN] missing {nst_path}")

    if aggr_path.exists():
        frames.append(normalize_simple(aggr_path, "DBSS-NST-v3-aggressive"))
    else:
        print(f"[WARN] missing {aggr_path}")

    if not frames:
        raise RuntimeError("No n_top files found.")

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

    out_path = ROOT / "summary_ntop_eval8000_9999_awgn8.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

    view = out.sort_values(["n_top", "answer_per_kbit"], ascending=[True, False])
    print(view[[
        "n_top",
        "method",
        "avg_source_bits",
        "delivered_answer_hit_rate",
        "answer_per_kbit",
        "tx_field_keep_ratio",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
