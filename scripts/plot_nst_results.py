import argparse
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
]


def plot_metric(df, channel, metric, ylabel, out_path):
    sub = df[df["channel"] == channel].copy()

    plt.figure(figsize=(7.0, 4.5))

    for method in METHOD_ORDER:
        m = sub[sub["method"] == method].sort_values("snr_db")
        if m.empty:
            continue

        plt.plot(
            m["snr_db"],
            m[metric],
            marker="o",
            label=method,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs SNR ({channel.upper()})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/nst/summary_eval8000_9999.csv")
    parser.add_argument("--out-dir", default="results/nst/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)

    for channel in ["awgn", "rayleigh"]:
        plot_metric(
            df,
            channel,
            "delivered_answer_hit_rate",
            "Delivered answer hit rate",
            out_dir / f"delivered_answer_vs_snr_{channel}.png",
        )

        plot_metric(
            df,
            channel,
            "answer_per_kbit",
            "Answer per kbit",
            out_dir / f"answer_per_kbit_vs_snr_{channel}.png",
        )

    # Bits figure: bits are SNR-independent for most methods, so one bar chart at AWGN 8 dB.
    focus = df[(df["channel"] == "awgn") & (df["snr_db"].astype(float) == 8.0)].copy()
    focus = focus[focus["method"].isin(METHOD_ORDER)]

    focus["method"] = pd.Categorical(
        focus["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    focus = focus.sort_values("method")

    plt.figure(figsize=(7.0, 4.5))
    plt.bar(focus["method"].astype(str), focus["avg_source_bits"])
    plt.ylabel("Average source bits")
    plt.title("Average source bits at AWGN 8 dB")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = out_dir / "avg_source_bits_awgn8.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
