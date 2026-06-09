from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    return pd.read_csv(path)


def savefig(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def metric_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    """
    Prefer validated metric if available.
    Fall back to raw metric for backward compatibility.
    """
    return preferred if preferred in df.columns else fallback


def sg_after_col(df: pd.DataFrame) -> str:
    return metric_col(
        df,
        "answerability_after_strict_validated",
        "answerability_after_strict",
    )


def bbox_after_col(df: pd.DataFrame) -> str:
    return metric_col(
        df,
        "answerability_after_loose_validated",
        "answerability_after_loose",
    )


def method_label(method: str) -> str:
    if method == "original":
        return "Original"
    return method.upper()


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_sg_strict_answerability_no_ldpc(df: pd.DataFrame) -> None:
    df = df[df["semantic_type"] == "sg"].copy()
    df = safe_numeric(df, ["n_top", "snr_db", sg_after_col(df)])

    for channel in ["awgn", "rayleigh"]:
        cur_channel = df[df["channel"] == channel].copy()

        if cur_channel.empty:
            continue

        plt.figure()

        for method in ["original", "do", "go"]:
            cur = cur_channel[cur_channel["ranking_method"] == method].copy()
            if cur.empty:
                continue

            cur = cur.sort_values("n_top")
            y_col = sg_after_col(cur)

            plt.plot(
                cur["n_top"],
                cur[y_col],
                marker="o",
                label=method_label(method),
            )

        plt.xlabel("Ntop")
        plt.ylabel("SG strict proxy answerability after channel (validated)")
        plt.title(
            f"SG validated answerability after channel "
            f"({channel.upper()}, no LDPC)"
        )
        plt.grid(True)
        plt.legend()

        savefig(FIG_DIR / f"fig1_sg_answerability_no_ldpc_{channel}.png")


def plot_bbox_loose_answerability_no_ldpc(df: pd.DataFrame) -> None:
    df = df[df["semantic_type"] == "bbox"].copy()
    df = safe_numeric(df, ["n_top", "snr_db", bbox_after_col(df)])

    for channel in ["awgn", "rayleigh"]:
        cur_channel = df[df["channel"] == channel].copy()

        if cur_channel.empty:
            continue

        plt.figure()

        for method in ["original", "do", "go"]:
            cur = cur_channel[cur_channel["ranking_method"] == method].copy()
            if cur.empty:
                continue

            cur = cur.sort_values("n_top")
            y_col = bbox_after_col(cur)

            plt.plot(
                cur["n_top"],
                cur[y_col],
                marker="o",
                label=method_label(method),
            )

        plt.xlabel("Ntop")
        plt.ylabel("BBox loose proxy answerability after channel (validated)")
        plt.title(
            f"BBox validated answerability after channel "
            f"({channel.upper()}, no LDPC)"
        )
        plt.grid(True)
        plt.legend()

        savefig(FIG_DIR / f"fig2_bbox_answerability_no_ldpc_{channel}.png")


def plot_channel_damage_go(df: pd.DataFrame) -> None:
    df = df[df["ranking_method"] == "go"].copy()
    df = safe_numeric(
        df,
        [
            "n_top",
            "answerability_before_strict",
            "answerability_after_strict",
            "answerability_after_strict_validated",
            "answerability_before_loose",
            "answerability_after_loose",
            "answerability_after_loose_validated",
        ],
    )

    # SG
    sg = df[df["semantic_type"] == "sg"].copy()

    for channel in ["awgn", "rayleigh"]:
        cur = sg[sg["channel"] == channel].copy()
        if cur.empty:
            continue

        cur = cur.sort_values("n_top")
        after_col = sg_after_col(cur)

        plt.figure()
        plt.plot(
            cur["n_top"],
            cur["answerability_before_strict"],
            marker="o",
            label="Before channel",
        )
        plt.plot(
            cur["n_top"],
            cur[after_col],
            marker="s",
            label="After channel, validated",
        )

        plt.xlabel("Ntop")
        plt.ylabel("SG strict proxy answerability")
        plt.title(
            f"Channel-induced answerability drop: "
            f"GO-SG ({channel.upper()}, validated)"
        )
        plt.grid(True)
        plt.legend()

        savefig(FIG_DIR / f"fig3_channel_damage_go_sg_{channel}.png")

    # BBox
    bbox = df[df["semantic_type"] == "bbox"].copy()

    for channel in ["awgn", "rayleigh"]:
        cur = bbox[bbox["channel"] == channel].copy()
        if cur.empty:
            continue

        cur = cur.sort_values("n_top")
        after_col = bbox_after_col(cur)

        plt.figure()
        plt.plot(
            cur["n_top"],
            cur["answerability_before_loose"],
            marker="o",
            label="Before channel",
        )
        plt.plot(
            cur["n_top"],
            cur[after_col],
            marker="s",
            label="After channel, validated",
        )

        plt.xlabel("Ntop")
        plt.ylabel("BBox loose proxy answerability")
        plt.title(
            f"Channel-induced answerability drop: "
            f"GO-BBox ({channel.upper()}, validated)"
        )
        plt.grid(True)
        plt.legend()

        savefig(FIG_DIR / f"fig3_channel_damage_go_bbox_{channel}.png")


def plot_latency_vs_answerability_no_ldpc(df: pd.DataFrame) -> None:
    df = df[df["ranking_method"] == "go"].copy()
    df = safe_numeric(
        df,
        [
            "n_top",
            "t_com_sec",
            "answerability_after_strict_validated",
            "answerability_after_strict",
            "answerability_after_loose_validated",
            "answerability_after_loose",
        ],
    )

    for channel in ["awgn", "rayleigh"]:
        cur = df[df["channel"] == channel].copy()

        if cur.empty:
            continue

        sg = cur[cur["semantic_type"] == "sg"].copy()
        bbox = cur[cur["semantic_type"] == "bbox"].copy()

        plt.figure()

        if not sg.empty:
            sg = sg.sort_values("t_com_sec")
            sg_y = sg_after_col(sg)

            plt.plot(
                sg["t_com_sec"],
                sg[sg_y],
                marker="o",
                label="GO-SG strict validated",
            )

            for _, r in sg.iterrows():
                plt.annotate(
                    str(int(r["n_top"])),
                    (r["t_com_sec"], r[sg_y]),
                )

        if not bbox.empty:
            bbox = bbox.sort_values("t_com_sec")
            bbox_y = bbox_after_col(bbox)

            plt.plot(
                bbox["t_com_sec"],
                bbox[bbox_y],
                marker="s",
                label="GO-BBox loose validated",
            )

            for _, r in bbox.iterrows():
                plt.annotate(
                    str(int(r["n_top"])),
                    (r["t_com_sec"], r[bbox_y]),
                )

        plt.xlabel("Communication latency t_com (sec)")
        plt.ylabel("Validated proxy answerability after channel")
        plt.title(
            f"Latency-answerability trade-off "
            f"({channel.upper()}, no LDPC, validated)"
        )
        plt.grid(True)
        plt.legend()

        savefig(FIG_DIR / f"fig4_latency_answerability_no_ldpc_{channel}.png")


def plot_ldpc_gain_vs_snr(df: pd.DataFrame) -> None:
    df = df[
        (df["ranking_method"] == "go")
        & (df["channel"] == "rayleigh")
    ].copy()

    if df.empty:
        return

    df = safe_numeric(
        df,
        [
            "snr_db",
            "n_top",
            "answerability_after_strict_validated",
            "answerability_after_strict",
            "answerability_after_loose_validated",
            "answerability_after_loose",
        ],
    )

    for semantic_type in ["sg", "bbox"]:
        sub = df[df["semantic_type"] == semantic_type].copy()

        if sub.empty:
            continue

        for n_top in sorted(sub["n_top"].dropna().unique()):
            cur = sub[sub["n_top"] == n_top].copy()

            if cur.empty:
                continue

            plt.figure()

            for coding_mode in ["uncoded", "ldpc_like"]:
                c = cur[cur["coding_mode"] == coding_mode].copy()
                if c.empty:
                    continue

                c = c.sort_values("snr_db")

                if semantic_type == "sg":
                    y_col = sg_after_col(c)
                    ylabel = (
                        "SG strict proxy answerability after channel "
                        "(validated)"
                    )
                    title = (
                        f"LDPC-like gain on GO-SG over Rayleigh, "
                        f"Ntop={int(n_top)}, validated"
                    )
                    fname = f"fig5_ldpc_gain_go_sg_rayleigh_ntop{int(n_top)}.png"
                else:
                    y_col = bbox_after_col(c)
                    ylabel = (
                        "BBox loose proxy answerability after channel "
                        "(validated)"
                    )
                    title = (
                        f"LDPC-like gain on GO-BBox over Rayleigh, "
                        f"Ntop={int(n_top)}, validated"
                    )
                    fname = (
                        f"fig5_ldpc_gain_go_bbox_rayleigh_ntop{int(n_top)}.png"
                    )

                plt.plot(
                    c["snr_db"],
                    c[y_col],
                    marker="o",
                    label=coding_mode,
                )

            plt.xlabel("SNR (dB)")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.legend()

            savefig(FIG_DIR / fname)


def plot_ldpc_latency_answerability_tradeoff(df: pd.DataFrame) -> None:
    df = df[
        (df["ranking_method"] == "go")
        & (df["channel"] == "rayleigh")
    ].copy()

    if df.empty:
        return

    df = safe_numeric(
        df,
        [
            "snr_db",
            "n_top",
            "coded_t_com_sec",
            "answerability_after_strict_validated",
            "answerability_after_strict",
            "answerability_after_loose_validated",
            "answerability_after_loose",
        ],
    )

    for semantic_type in ["sg", "bbox"]:
        sub = df[df["semantic_type"] == semantic_type].copy()

        if sub.empty:
            continue

        for snr_db in [8.0, 10.0, 12.0]:
            cur = sub[sub["snr_db"] == snr_db].copy()

            if cur.empty:
                continue

            plt.figure()

            for coding_mode in ["uncoded", "ldpc_like"]:
                c = cur[cur["coding_mode"] == coding_mode].copy()

                if c.empty:
                    continue

                c = c.sort_values("coded_t_com_sec")

                if semantic_type == "sg":
                    y_col = sg_after_col(c)
                    y = c[y_col]
                    ylabel = (
                        "SG strict proxy answerability after channel "
                        "(validated)"
                    )
                else:
                    y_col = bbox_after_col(c)
                    y = c[y_col]
                    ylabel = (
                        "BBox loose proxy answerability after channel "
                        "(validated)"
                    )

                plt.plot(
                    c["coded_t_com_sec"],
                    y,
                    marker="o",
                    label=coding_mode,
                )

                for _, r in c.iterrows():
                    plt.annotate(
                        str(int(r["n_top"])),
                        (r["coded_t_com_sec"], r[y_col]),
                    )

            plt.xlabel("Communication latency using coded bits (sec)")
            plt.ylabel(ylabel)
            plt.title(
                f"LDPC-like latency-answerability trade-off: "
                f"GO-{semantic_type.upper()}, Rayleigh {snr_db:.0f} dB, "
                f"validated"
            )
            plt.grid(True)
            plt.legend()

            savefig(
                FIG_DIR
                / (
                    f"fig6_ldpc_latency_tradeoff_go_{semantic_type}_"
                    f"rayleigh_{int(snr_db)}db.png"
                )
            )


def plot_decoded_ber_and_per(df: pd.DataFrame) -> None:
    df = df[
        (df["semantic_type"] == "sg")
        & (df["ranking_method"] == "go")
        & (df["channel"] == "rayleigh")
        & (df["n_top"].astype(str) == "12")
    ].copy()

    if df.empty:
        return

    df = safe_numeric(
        df,
        ["snr_db", "decoded_ber", "packet_error_rate"],
    )

    # Decoded BER
    plt.figure()

    for coding_mode in ["uncoded", "ldpc_like"]:
        cur = df[df["coding_mode"] == coding_mode].copy()
        if cur.empty:
            continue

        cur = cur.sort_values("snr_db")

        plt.plot(
            cur["snr_db"],
            cur["decoded_ber"],
            marker="o",
            label=coding_mode,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Decoded BER")
    plt.title("Decoded BER: GO-SG, Rayleigh, Ntop=12")
    plt.grid(True)
    plt.legend()

    savefig(FIG_DIR / "fig7_decoded_ber_go_sg_rayleigh_ntop12.png")

    # PER
    plt.figure()

    for coding_mode in ["uncoded", "ldpc_like"]:
        cur = df[df["coding_mode"] == coding_mode].copy()
        if cur.empty:
            continue

        cur = cur.sort_values("snr_db")

        plt.plot(
            cur["snr_db"],
            cur["packet_error_rate"],
            marker="o",
            label=coding_mode,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Packet error rate")
    plt.title("Packet error rate: GO-SG, Rayleigh, Ntop=12")
    plt.grid(True)
    plt.legend()

    savefig(FIG_DIR / "fig7_per_go_sg_rayleigh_ntop12.png")


def plot_image_vs_semantic_latency() -> None:
    """
    Figure 8:
    Raw image transmission latency vs GO semantic transmission latency.

    This uses image_baseline.csv generated by run_image_baseline.py.
    Image baseline is raw-float32 paper-style image size, not JPEG.
    """
    path = Path("results/image_baseline.csv")
    if not path.exists():
        print(f"Skip image baseline plot: missing {path}")
        return

    df = pd.read_csv(path)
    comp = df[df["row_type"] == "image_vs_semantic"].copy()

    if comp.empty:
        print("Skip image baseline plot: no image_vs_semantic rows")
        return

    comp["n_top"] = comp["n_top"].astype(float)
    comp["semantic_t_com_sec"] = comp["semantic_t_com_sec"].astype(float)
    comp["image_t_com_sec"] = comp["image_t_com_sec"].astype(float)
    comp["image_to_semantic_latency_ratio"] = comp[
        "image_to_semantic_latency_ratio"
    ].astype(float)

    for channel in ["awgn", "rayleigh"]:
        cur = comp[comp["channel"] == channel].copy()

        if cur.empty:
            continue

        plt.figure(figsize=(7, 4.5))

        # Image latency is constant for fixed SNR/image size.
        image_rows = cur.sort_values("n_top")
        image_t = image_rows["image_t_com_sec"].iloc[0]

        plt.axhline(
            y=image_t,
            linestyle="--",
            label="Raw Image Transmission",
        )

        for semantic_type, label in [
            ("sg", "GO-SG semantic"),
            ("bbox", "GO-BBox semantic"),
        ]:
            sub = cur[cur["semantic_type"] == semantic_type].sort_values("n_top")

            if sub.empty:
                continue

            plt.plot(
                sub["n_top"],
                sub["semantic_t_com_sec"],
                marker="o",
                label=label,
            )

        plt.yscale("log")
        plt.xlabel("Ntop")
        plt.ylabel("Communication latency t_com (sec, log scale)")
        plt.title(
            f"Raw image vs semantic transmission latency ({channel.upper()}, SNR=8 dB)"
        )
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        savefig(FIG_DIR / f"fig8_image_vs_semantic_latency_{channel}.png")


def plot_image_to_semantic_ratio() -> None:
    """
    Figure 9:
    Raw image to semantic latency ratio.
    """
    path = Path("results/image_baseline.csv")
    if not path.exists():
        print(f"Skip image ratio plot: missing {path}")
        return

    df = pd.read_csv(path)
    comp = df[df["row_type"] == "image_vs_semantic"].copy()

    if comp.empty:
        print("Skip image ratio plot: no image_vs_semantic rows")
        return

    comp["n_top"] = comp["n_top"].astype(float)
    comp["image_to_semantic_latency_ratio"] = comp[
        "image_to_semantic_latency_ratio"
    ].astype(float)

    for channel in ["awgn", "rayleigh"]:
        cur = comp[comp["channel"] == channel].copy()

        if cur.empty:
            continue

        plt.figure(figsize=(7, 4.5))

        for semantic_type, label in [
            ("sg", "Raw Image / GO-SG"),
            ("bbox", "Raw Image / GO-BBox"),
        ]:
            sub = cur[cur["semantic_type"] == semantic_type].sort_values("n_top")

            if sub.empty:
                continue

            plt.plot(
                sub["n_top"],
                sub["image_to_semantic_latency_ratio"],
                marker="o",
                label=label,
            )

        plt.yscale("log")
        plt.xlabel("Ntop")
        plt.ylabel("Latency ratio: raw image / semantic (log scale)")
        plt.title(
            f"Raw image-to-semantic latency ratio ({channel.upper()}, SNR=8 dB)"
        )
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        savefig(FIG_DIR / f"fig9_image_to_semantic_latency_ratio_{channel}.png")
        

def plot_total_latency_breakdown() -> None:
    """
    Figure 10:
    Paper-style total latency comparison between raw image and GO semantic transmission.
    """
    path = RESULTS_DIR / "latency_breakdown.csv"
    if not path.exists():
        print(f"Skip total latency plot: missing {path}")
        return

    df = pd.read_csv(path)
    comp = df[df["row_type"] == "image_vs_semantic_total"].copy()

    if comp.empty:
        print("Skip total latency plot: no image_vs_semantic_total rows")
        return

    comp = safe_numeric(
        comp,
        [
            "n_top",
            "image_t_total_ms",
            "semantic_t_total_ms",
            "image_to_semantic_total_ratio",
            "semantic_answerability_validated",
        ],
    )

    for channel in ["awgn", "rayleigh"]:
        cur = comp[comp["channel"] == channel].copy()

        if cur.empty:
            continue

        plt.figure(figsize=(7, 4.5))

        image_total = cur["image_t_total_ms"].iloc[0]
        plt.axhline(
            y=image_total,
            linestyle="--",
            label="Raw Image total latency",
        )

        for semantic_type, label in [
            ("sg", "GO-SG total latency"),
            ("bbox", "GO-BBox total latency"),
        ]:
            sub = cur[cur["semantic_type"] == semantic_type].sort_values("n_top")

            if sub.empty:
                continue

            plt.plot(
                sub["n_top"],
                sub["semantic_t_total_ms"],
                marker="o",
                label=label,
            )

        plt.yscale("log")
        plt.xlabel("Ntop")
        plt.ylabel("Total latency (ms, log scale)")
        plt.title(
            f"Paper-style total latency: raw image vs semantic ({channel.upper()}, SNR=8 dB)"
        )
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        savefig(FIG_DIR / f"fig10_total_latency_image_vs_semantic_{channel}.png")


def plot_total_latency_ratio() -> None:
    """
    Figure 11:
    Raw image total latency divided by semantic total latency.
    """
    path = RESULTS_DIR / "latency_breakdown.csv"
    if not path.exists():
        print(f"Skip total latency ratio plot: missing {path}")
        return

    df = pd.read_csv(path)
    comp = df[df["row_type"] == "image_vs_semantic_total"].copy()

    if comp.empty:
        print("Skip total latency ratio plot: no image_vs_semantic_total rows")
        return

    comp = safe_numeric(
        comp,
        [
            "n_top",
            "image_to_semantic_total_ratio",
        ],
    )

    for channel in ["awgn", "rayleigh"]:
        cur = comp[comp["channel"] == channel].copy()

        if cur.empty:
            continue

        plt.figure(figsize=(7, 4.5))

        for semantic_type, label in [
            ("sg", "Raw Image / GO-SG total"),
            ("bbox", "Raw Image / GO-BBox total"),
        ]:
            sub = cur[cur["semantic_type"] == semantic_type].sort_values("n_top")

            if sub.empty:
                continue

            plt.plot(
                sub["n_top"],
                sub["image_to_semantic_total_ratio"],
                marker="o",
                label=label,
            )

        plt.xlabel("Ntop")
        plt.ylabel("Total latency ratio")
        plt.title(
            f"Raw image-to-semantic total latency ratio ({channel.upper()}, SNR=8 dB)"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(FIG_DIR / f"fig11_total_latency_ratio_{channel}.png")
        
                
        
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    answerability_path = RESULTS_DIR / "answerability_sweep.csv"
    ldpc_path = RESULTS_DIR / "answerability_sweep_ldpc.csv"
    image_baseline_path = RESULTS_DIR / "image_baseline.csv"

    if answerability_path.exists():
        answer_df = load_csv(answerability_path)
        plot_sg_strict_answerability_no_ldpc(answer_df)
        plot_bbox_loose_answerability_no_ldpc(answer_df)
        plot_channel_damage_go(answer_df)
        plot_latency_vs_answerability_no_ldpc(answer_df)
    else:
        print(f"Skip no-LDPC plots. Missing: {answerability_path}")

    if ldpc_path.exists():
        ldpc_df = load_csv(ldpc_path)
        plot_ldpc_gain_vs_snr(ldpc_df)
        plot_ldpc_latency_answerability_tradeoff(ldpc_df)
        plot_decoded_ber_and_per(ldpc_df)
    else:
        print(f"Skip LDPC plots. Missing: {ldpc_path}")

    if image_baseline_path.exists():
        plot_image_vs_semantic_latency()
        plot_image_to_semantic_ratio()
    else:
        print(f"Skip image baseline plots. Missing: {image_baseline_path}")

    latency_breakdown_path = RESULTS_DIR / "latency_breakdown.csv"
    if latency_breakdown_path.exists():
        plot_total_latency_breakdown()
        plot_total_latency_ratio()
    else:
        print(f"Skip total latency plots. Missing: {latency_breakdown_path}")
        
    image_baseline_path = RESULTS_DIR / "image_baseline.csv"
    latency_breakdown_path = RESULTS_DIR / "latency_breakdown.csv"
        
        
if __name__ == "__main__":
    main()