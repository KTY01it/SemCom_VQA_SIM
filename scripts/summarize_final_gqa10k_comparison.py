from pathlib import Path

import pandas as pd


ROOT = Path("results/final_gqa10k")


def normalize_baselines(path: Path) -> pd.DataFrame:
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

    path = ROOT / "baselines_awgn8_ntop9.csv"
    if path.exists():
        frames.append(normalize_baselines(path))

    mapping = [
        ("qtc_noanswer_awgn8_ntop9.csv", "DBSS-QTC-noanswer"),
        ("nst_v3_main_awgn8_ntop9.csv", "DBSS-NST-v3"),
        ("nst_v3_aggressive_awgn8_ntop9.csv", "DBSS-NST-v3-aggressive"),
        ("nsp_v1_awgn8_ntop9.csv", "NSP-v1"),
        ("nsp_v2_awgn8_ntop9.csv", "NSP-v2"),
    ]

    for fname, name in mapping:
        path = ROOT / fname
        if path.exists():
            frames.append(normalize_simple(path, name))
        else:
            print(f"[WARN] Missing {path}")

    if not frames:
        raise RuntimeError("No final comparison files found.")

    df = pd.concat(frames, ignore_index=True)

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
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    order = [
        "Original-SG",
        "DO-SG",
        "GO-SG",
        "DBSS",
        "DBSS-QTC-noanswer",
        "DBSS-NST-v3",
        "DBSS-NST-v3-aggressive",
        "NSP-v1",
        "NSP-v2",
    ]

    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    df = df.sort_values("method")

    out_csv = ROOT / "final_awgn8_ntop9_summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(df[[
        "method",
        "avg_source_bits",
        "delivered_answer_hit_rate",
        "answer_per_kbit",
        "tx_field_keep_ratio",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
