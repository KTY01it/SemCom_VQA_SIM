from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = [
    "Original-SG",
    "DO-SG",
    "GO-SG",
    "DBSS",
    "DBSS-QTC-noanswer",
    "DBSS-NST-v3",
    "DBSS-NST-v3-aggressive",
]


def plot_metric(df, metric, ylabel, out_path):
    plt.figure(figsize=(7.0, 4.5))

    for method in METHOD_ORDER:
        m = df[df["method"] == method].sort_values("n_top")
        if m.empty:
            continue

        plt.plot(
            m["n_top"],
            m[metric],
            marker="o",
            label=method,
        )

    plt.xlabel("Number of selected semantic units $N_{top}$")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs semantic budget")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    df = pd.read_csv("results/nst/summary_ntop_eval8000_9999_awgn8.csv")
    out_dir = Path("results/nst/figures")

    plot_metric(
        df,
        "delivered_answer_hit_rate",
        "Delivered answer hit rate",
        out_dir / "delivered_answer_vs_ntop_awgn8.png",
    )

    plot_metric(
        df,
        "answer_per_kbit",
        "Answer per kbit",
        out_dir / "answer_per_kbit_vs_ntop_awgn8.png",
    )

    plot_metric(
        df,
        "avg_source_bits",
        "Average source bits",
        out_dir / "avg_source_bits_vs_ntop_awgn8.png",
    )


if __name__ == "__main__":
    main()
