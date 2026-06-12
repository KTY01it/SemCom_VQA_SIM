import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)


def plot_snr():
    df = pd.read_csv("results/benchmark/main_snr_sg_dbss.csv")

    methods = ["random", "original", "do", "go", "dbss"]

    for channel in ["awgn", "rayleigh"]:
        sub = df[df["channel"] == channel]

        plt.figure(figsize=(6, 4))
        for m in methods:
            s = sub[sub["ranking_method"] == m].sort_values("snr_db")
            plt.plot(
                s["snr_db"],
                s["delivered_answer_hit_rate"],
                marker="o",
                label=m.upper() if m != "dbss" else "DBSS",
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Delivered answer hit rate")
        plt.title(f"SNR sweep ({channel.upper()})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"fig_snr_{channel}.pdf")
        plt.savefig(OUT / f"fig_snr_{channel}.png", dpi=300)
        plt.close()


def plot_ntop():
    df = pd.read_csv("results/benchmark/main_ntop_sg_dbss_metrics.csv")

    methods = ["random", "original", "do", "go", "dbss"]

    for channel in ["awgn", "rayleigh"]:
        sub = df[df["channel"] == channel]

        plt.figure(figsize=(6, 4))
        for m in methods:
            s = sub[sub["ranking_method"] == m].sort_values("n_top")
            plt.plot(
                s["n_top"],
                s["delivered_answer_hit_rate"],
                marker="o",
                label=m.upper() if m != "dbss" else "DBSS",
            )

        plt.xlabel("Number of transmitted triplets")
        plt.ylabel("Delivered answer hit rate")
        plt.title(f"Ntop sweep ({channel.upper()}, SNR=8 dB)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"fig_ntop_{channel}.pdf")
        plt.savefig(OUT / f"fig_ntop_{channel}.png", dpi=300)
        plt.close()


def plot_evidence_quality():
    df = pd.read_csv("results/benchmark/main_ntop_sg_dbss_metrics.csv")

    sub = df[
        (df["channel"] == "rayleigh")
        & (df["ranking_method"].isin(["go", "dbss"]))
        & (df["n_top"].isin([3, 6, 9, 12, 30]))
    ]

    for metric, ylabel in [
        ("coverage_ratio", "Coverage ratio"),
        ("redundancy_ratio", "Redundancy ratio"),
        ("unique_concept_count", "Unique concept count"),
    ]:
        plt.figure(figsize=(6, 4))
        for m in ["go", "dbss"]:
            s = sub[sub["ranking_method"] == m].sort_values("n_top")
            plt.plot(
                s["n_top"],
                s[metric],
                marker="o",
                label="GO-SG" if m == "go" else "DBSS",
            )

        plt.xlabel("Number of transmitted triplets")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} (Rayleigh, SNR=8 dB)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"fig_evidence_{metric}.pdf")
        plt.savefig(OUT / f"fig_evidence_{metric}.png", dpi=300)
        plt.close()


def plot_ablation():
    df = pd.read_csv("results/ablation/dbss_ablation_ntop_rayleigh8.csv")

    # Use average over Ntop for a compact ablation figure.
    order = ["go", "dbss_no_coverage", "dbss_no_redundancy", "dbss"]
    labels = ["GO-SG", "w/o coverage", "w/o redundancy", "DBSS"]

    summary = (
        df[df["ranking_method"].isin(order)]
        .groupby("ranking_method")["delivered_answer_hit_rate"]
        .mean()
        .reindex(order)
    )

    plt.figure(figsize=(6, 4))
    plt.bar(labels, summary.values)
    plt.ylabel("Mean delivered answer hit rate")
    plt.title("DBSS ablation (Rayleigh, SNR=8 dB)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT / "fig_ablation_rayleigh8.pdf")
    plt.savefig(OUT / "fig_ablation_rayleigh8.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_snr()
    plot_ntop()
    plot_evidence_quality()
    plot_ablation()
    print(f"Saved figures to {OUT}")
