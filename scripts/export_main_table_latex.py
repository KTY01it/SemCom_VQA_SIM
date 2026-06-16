from pathlib import Path

import pandas as pd


def main():
    df = pd.read_csv("results/nst/summary_eval8000_9999.csv")

    sub = df[
        (df["channel"] == "awgn")
        & (df["snr_db"].astype(float) == 8.0)
        & (df["n_top"].astype(int) == 9)
    ].copy()

    order = [
        "Original-SG",
        "DO-SG",
        "GO-SG",
        "DBSS",
        "DBSS-QTC-noanswer",
        "DBSS-NST-v3",
        "DBSS-NST-v3-aggressive",
    ]

    sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
    sub = sub.sort_values("method")

    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "Method": str(r["method"]),
            "Avg. bits $\\downarrow$": f"{r['avg_source_bits']:.2f}",
            "Delivered answer $\\uparrow$": f"{r['delivered_answer_hit_rate']:.4f}",
            "Answer/kbit $\\uparrow$": f"{r['answer_per_kbit']:.4f}",
            "Field keep $\\downarrow$": f"{r['tx_field_keep_ratio']:.3f}",
        })

    out = pd.DataFrame(rows)

    latex = out.to_latex(
        index=False,
        escape=False,
        column_format="lrrrr",
    )

    out_path = Path("results/nst/main_table_awgn8_ntop9.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")

    print(latex)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
