from pathlib import Path

import pandas as pd


def main():
    src = Path("results/final_gqa10k/final_awgn8_ntop9_summary.csv")
    df = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
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

    out_path = Path("results/final_gqa10k/final_awgn8_ntop9_table.tex")
    out_path.write_text(latex, encoding="utf-8")

    print(latex)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
