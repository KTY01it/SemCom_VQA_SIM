from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def plot_sg_strict_answerability_no_ldpc(df: pd.DataFrame) -> None:
    """
    Figure 1:
    GO/Original/DO comparison for SG strict answerability after channel.
    Use no-LDPC answerability_sweep.csv.
    """
    sub = df[
        (df["semantic_type"] == "sg")
        & (df["channel"].isin(["awgn", "rayleigh"]))
    ].copy()

    methods = ["original", "do", "go"]
    channels = ["awgn", "rayleigh"]

    for channel in channels:
        plt.figure(figsize=(7, 4.5))

        for method in methods:
            cur = sub[
                (sub["channel"] == channel)
                & (sub["ranking_method"] == method)
            ].sort_values("n_top")

            plt.plot(
                cur["n_top"],
                cur["answerability_after_strict"],
                marker="o",
                label=method.upper() if method != "original" else "Original",
            )

        plt.xlabel("Ntop")
        plt.ylabel("SG strict answerability after channel")
        plt.title(f"SG answerability after channel ({channel.upper()}, no LDPC)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(f"fig1_sg_answerability_no_ldpc_{channel}.png")


def plot_bbox_loose_answerability_no_ldpc(df: pd.DataFrame) -> None:
    """
    Figure 2:
    BBox loose answerability after channel.
    """
    sub = df[
        (df["semantic_type"] == "bbox")
        & (df["channel"].isin(["awgn", "rayleigh"]))
    ].copy()

    methods = ["original", "do", "go"]
    channels = ["awgn", "rayleigh"]

    for channel in channels:
        plt.figure(figsize=(7, 4.5))

        for method in methods:
            cur = sub[
                (sub["channel"] == channel)
                & (sub["ranking_method"] == method)
            ].sort_values("n_top")

            plt.plot(
                cur["n_top"],
                cur["answerability_after_loose"],
                marker="o",
                label=method.upper() if method != "original" else "Original",
            )

        plt.xlabel("Ntop")
        plt.ylabel("BBox loose answerability after channel")
        plt.title(f"BBox answerability after channel ({channel.upper()}, no LDPC)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(f"fig2_bbox_answerability_no_ldpc_{channel}.png")


def plot_channel_damage_go(df: pd.DataFrame) -> None:
    """
    Figure 3:
    Before vs after channel for GO.
    Shows how Rayleigh damages delivered semantics.
    """
    sub = df[df["ranking_method"] == "go"].copy()

    # SG strict
    sg = sub[sub["semantic_type"] == "sg"].copy()

    for channel in ["awgn", "rayleigh"]:
        cur = sg[sg["channel"] == channel].sort_values("n_top")

        plt.figure(figsize=(7, 4.5))
        plt.plot(
            cur["n_top"],
            cur["answerability_before_strict"],
            marker="o",
            label="Before channel",
        )
        plt.plot(
            cur["n_top"],
            cur["answerability_after_strict"],
            marker="s",
            label="After channel",
        )

        plt.xlabel("Ntop")
        plt.ylabel("SG strict answerability")
        plt.title(f"Channel-induced answerability drop: GO-SG ({channel.upper()})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(f"fig3_channel_damage_go_sg_{channel}.png")

    # BBox loose
    bbox = sub[sub["semantic_type"] == "bbox"].copy()

    for channel in ["awgn", "rayleigh"]:
        cur = bbox[bbox["channel"] == channel].sort_values("n_top")

        plt.figure(figsize=(7, 4.5))
        plt.plot(
            cur["n_top"],
            cur["answerability_before_loose"],
            marker="o",
            label="Before channel",
        )
        plt.plot(
            cur["n_top"],
            cur["answerability_after_loose"],
            marker="s",
            label="After channel",
        )

        plt.xlabel("Ntop")
        plt.ylabel("BBox loose answerability")
        plt.title(f"Channel-induced answerability drop: GO-BBox ({channel.upper()})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(f"fig3_channel_damage_go_bbox_{channel}.png")


def plot_latency_vs_answerability_no_ldpc(df: pd.DataFrame) -> None:
    """
    Figure 4:
    Latency-answerability trade-off for GO.
    """
    sub = df[df["ranking_method"] == "go"].copy()

    for channel in ["awgn", "rayleigh"]:
        plt.figure(figsize=(7, 4.5))

        sg = sub[
            (sub["semantic_type"] == "sg")
            & (sub["channel"] == channel)
        ].sort_values("n_top")

        bbox = sub[
            (sub["semantic_type"] == "bbox")
            & (sub["channel"] == channel)
        ].sort_values("n_top")

        plt.plot(
            sg["t_com_sec"],
            sg["answerability_after_strict"],
            marker="o",
            label="GO-SG strict",
        )

        plt.plot(
            bbox["t_com_sec"],
            bbox["answerability_after_loose"],
            marker="s",
            label="GO-BBox loose",
        )

        for _, r in sg.iterrows():
            plt.annotate(str(int(r["n_top"])), (r["t_com_sec"], r["answerability_after_strict"]))

        for _, r in bbox.iterrows():
            plt.annotate(str(int(r["n_top"])), (r["t_com_sec"], r["answerability_after_loose"]))

        plt.xlabel("Communication latency t_com (sec)")
        plt.ylabel("Answerability after channel")
        plt.title(f"Latency-answerability trade-off ({channel.upper()}, no LDPC)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(f"fig4_latency_answerability_no_ldpc_{channel}.png")


def plot_ldpc_gain_vs_snr(df: pd.DataFrame) -> None:
    """
    Figure 5:
    LDPC vs uncoded answerability after channel over SNR.
    Focus on GO, Rayleigh.
    """
    sub = df[
        (df["ranking_method"] == "go")
        & (df["channel"] == "rayleigh")
        & (df["n_top"].isin([3, 6, 9, 12]))
    ].copy()

    # SG strict
    for n_top in [3, 6, 9, 12]:
        cur = sub[
            (sub["semantic_type"] == "sg")
            & (sub["n_top"] == n_top)
        ].sort_values("snr_db")

        plt.figure(figsize=(7, 4.5))

        for coding_mode in ["uncoded", "ldpc_like"]:
            c = cur[cur["coding_mode"] == coding_mode]
            plt.plot(
                c["snr_db"],
                c["answerability_after_strict"],
                marker="o",
                label=coding_mode,
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("SG strict answerability after channel")
        plt.title(f"LDPC gain on GO-SG over Rayleigh, Ntop={n_top}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(f"fig5_ldpc_gain_go_sg_rayleigh_ntop{n_top}.png")

    # BBox loose
    for n_top in [3, 6, 9, 12]:
        cur = sub[
            (sub["semantic_type"] == "bbox")
            & (sub["n_top"] == n_top)
        ].sort_values("snr_db")

        plt.figure(figsize=(7, 4.5))

        for coding_mode in ["uncoded", "ldpc_like"]:
            c = cur[cur["coding_mode"] == coding_mode]
            plt.plot(
                c["snr_db"],
                c["answerability_after_loose"],
                marker="o",
                label=coding_mode,
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("BBox loose answerability after channel")
        plt.title(f"LDPC gain on GO-BBox over Rayleigh, Ntop={n_top}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(f"fig5_ldpc_gain_go_bbox_rayleigh_ntop{n_top}.png")


def plot_ldpc_latency_answerability_tradeoff(df: pd.DataFrame) -> None:
    """
    Figure 6:
    LDPC latency-answerability tradeoff on Rayleigh.
    """
    sub = df[
        (df["ranking_method"] == "go")
        & (df["channel"] == "rayleigh")
        & (df["snr_db"].isin([8.0, 10.0, 12.0]))
    ].copy()

    for semantic_type in ["sg", "bbox"]:
        for snr_db in [8.0, 10.0, 12.0]:
            cur = sub[
                (sub["semantic_type"] == semantic_type)
                & (sub["snr_db"] == snr_db)
            ].sort_values(["coding_mode", "n_top"])

            plt.figure(figsize=(7, 4.5))

            for coding_mode in ["uncoded", "ldpc_like"]:
                c = cur[cur["coding_mode"] == coding_mode].sort_values("n_top")

                if semantic_type == "sg":
                    y = c["answerability_after_strict"]
                    ylabel = "SG strict answerability after channel"
                else:
                    y = c["answerability_after_loose"]
                    ylabel = "BBox loose answerability after channel"

                plt.plot(
                    c["coded_t_com_sec"],
                    y,
                    marker="o",
                    label=coding_mode,
                )

                for _, r in c.iterrows():
                    y_val = (
                        r["answerability_after_strict"]
                        if semantic_type == "sg"
                        else r["answerability_after_loose"]
                    )
                    plt.annotate(str(int(r["n_top"])), (r["coded_t_com_sec"], y_val))

            plt.xlabel("Coded communication latency (sec)")
            plt.ylabel(ylabel)
            plt.title(f"LDPC latency-answerability trade-off: GO-{semantic_type.upper()}, Rayleigh {snr_db:.0f} dB")
            plt.grid(True, alpha=0.3)
            plt.legend()
            savefig(f"fig6_ldpc_latency_tradeoff_go_{semantic_type}_rayleigh_{int(snr_db)}db.png")


def plot_ber_per_vs_snr_ldpc(df: pd.DataFrame) -> None:
    """
    Figure 7:
    BER/PER reduction by LDPC for GO-SG Rayleigh.
    """
    sub = df[
        (df["semantic_type"] == "sg")
        & (df["ranking_method"] == "go")
        & (df["channel"] == "rayleigh")
        & (df["n_top"] == 12)
    ].copy()

    plt.figure(figsize=(7, 4.5))
    for coding_mode in ["uncoded", "ldpc_like"]:
        cur = sub[sub["coding_mode"] == coding_mode].sort_values("snr_db")
        plt.plot(
            cur["snr_db"],
            cur["decoded_ber"],
            marker="o",
            label=coding_mode,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Decoded BER")
    plt.title("Decoded BER vs SNR: GO-SG over Rayleigh, Ntop=12")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("fig7_decoded_ber_go_sg_rayleigh_ntop12.png")

    plt.figure(figsize=(7, 4.5))
    for coding_mode in ["uncoded", "ldpc_like"]:
        cur = sub[sub["coding_mode"] == coding_mode].sort_values("snr_db")
        plt.plot(
            cur["snr_db"],
            cur["packet_error_rate"],
            marker="o",
            label=coding_mode,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Packet error rate")
    plt.title("PER vs SNR: GO-SG over Rayleigh, Ntop=12")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("fig7_per_go_sg_rayleigh_ntop12.png")


def main() -> None:
    answer = load_csv("results/answerability_sweep.csv")
    ldpc = load_csv("results/answerability_sweep_ldpc.csv")

    plot_sg_strict_answerability_no_ldpc(answer)
    plot_bbox_loose_answerability_no_ldpc(answer)
    plot_channel_damage_go(answer)
    plot_latency_vs_answerability_no_ldpc(answer)

    plot_ldpc_gain_vs_snr(ldpc)
    plot_ldpc_latency_answerability_tradeoff(ldpc)
    plot_ber_per_vs_snr_ldpc(ldpc)


if __name__ == "__main__":
    main()
