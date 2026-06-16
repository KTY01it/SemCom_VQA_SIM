from pathlib import Path

import pandas as pd


def load(path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] missing {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def main():
    files = [
        "results/nsp/nsp_v2_mmr_eval8000_9999_awgn8_sweep.csv",
        "results/nsp/nsp_v1_mmr_eval8000_9999_awgn8_sweep.csv",
    ]

    dfs = [load(f) for f in files]
    dfs = [d for d in dfs if not d.empty]

    if not dfs:
        raise RuntimeError("No NSP-MMR sweep files found.")

    df = pd.concat(dfs, ignore_index=True)

    print("\n=== Top by answer_per_kbit ===")
    print(
        df.sort_values("answer_per_kbit", ascending=False)[[
            "method",
            "mask_threshold",
            "diversity_beta",
            "avg_source_bits",
            "delivered_answer_hit_rate",
            "answer_per_kbit",
            "tx_field_keep_ratio",
        ]].head(20).to_string(index=False)
    )

    print("\n=== Top by delivered_answer_hit_rate ===")
    print(
        df.sort_values("delivered_answer_hit_rate", ascending=False)[[
            "method",
            "mask_threshold",
            "diversity_beta",
            "avg_source_bits",
            "delivered_answer_hit_rate",
            "answer_per_kbit",
            "tx_field_keep_ratio",
        ]].head(20).to_string(index=False)
    )

    out = Path("results/nsp/nsp_mmr_sweep_summary.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
