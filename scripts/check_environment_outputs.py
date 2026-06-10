from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "results/sg_packet_sanity.csv",
    "results/bbox_packet_sanity.csv",
    "results/answerability_sweep.csv",
    "results/answerability_sweep_ldpc.csv",
    "results/image_baseline.csv",
    "results/latency_breakdown.csv",
    "results/summary_tables.txt",
    "results/summary_ldpc_tables.txt",
    "results/experiment_summary.md",
    "results/proxy_metric_contract.json",    
]

REQUIRED_FIGURES = [
    "fig1_sg_answerability_no_ldpc_awgn.png",
    "fig1_sg_answerability_no_ldpc_rayleigh.png",
    "fig2_bbox_answerability_no_ldpc_awgn.png",
    "fig2_bbox_answerability_no_ldpc_rayleigh.png",
    "fig3_channel_damage_go_sg_awgn.png",
    "fig3_channel_damage_go_sg_rayleigh.png",
    "fig3_channel_damage_go_bbox_awgn.png",
    "fig3_channel_damage_go_bbox_rayleigh.png",
    "fig4_latency_answerability_no_ldpc_awgn.png",
    "fig4_latency_answerability_no_ldpc_rayleigh.png",
    "fig5_ldpc_gain_go_sg_rayleigh_ntop3.png",
    "fig5_ldpc_gain_go_sg_rayleigh_ntop6.png",
    "fig5_ldpc_gain_go_sg_rayleigh_ntop9.png",
    "fig5_ldpc_gain_go_sg_rayleigh_ntop12.png",
    "fig5_ldpc_gain_go_bbox_rayleigh_ntop3.png",
    "fig5_ldpc_gain_go_bbox_rayleigh_ntop6.png",
    "fig5_ldpc_gain_go_bbox_rayleigh_ntop9.png",
    "fig5_ldpc_gain_go_bbox_rayleigh_ntop12.png",
    "fig6_ldpc_latency_tradeoff_go_sg_rayleigh_8db.png",
    "fig6_ldpc_latency_tradeoff_go_sg_rayleigh_10db.png",
    "fig6_ldpc_latency_tradeoff_go_sg_rayleigh_12db.png",
    "fig6_ldpc_latency_tradeoff_go_bbox_rayleigh_8db.png",
    "fig6_ldpc_latency_tradeoff_go_bbox_rayleigh_10db.png",
    "fig6_ldpc_latency_tradeoff_go_bbox_rayleigh_12db.png",
    "fig7_decoded_ber_go_sg_rayleigh_ntop12.png",
    "fig7_per_go_sg_rayleigh_ntop12.png",
    "fig8_image_vs_semantic_latency_awgn.png",
    "fig8_image_vs_semantic_latency_rayleigh.png",
    "fig9_image_to_semantic_latency_ratio_awgn.png",
    "fig9_image_to_semantic_latency_ratio_rayleigh.png",
    "fig10_total_latency_image_vs_semantic_awgn.png",
    "fig10_total_latency_image_vs_semantic_rayleigh.png",
    "fig11_total_latency_ratio_awgn.png",
    "fig11_total_latency_ratio_rayleigh.png",
]


def check_file_exists(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"[MISSING] {path}")
        return False

    if p.is_file() and p.stat().st_size == 0:
        print(f"[EMPTY]   {path}")
        return False

    print(f"[OK]      {path}")
    return True


def check_csv_columns(path: str, required_columns: list[str]) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"[MISSING CSV] {path}")
        return False

    df = pd.read_csv(p)
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        print(f"[BAD CSV] {path}")
        print(f"  missing columns: {missing}")
        print(f"  existing columns: {df.columns.tolist()}")
        return False

    print(f"[OK CSV]  {path} rows={len(df)}")
    return True


def check_report_keywords(path: str, keywords: list[str]) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"[MISSING REPORT] {path}")
        return False

    text = p.read_text(encoding="utf-8")
    ok = True

    for kw in keywords:
        if kw not in text:
            print(f"[REPORT MISSING KEYWORD] {kw}")
            ok = False

    if ok:
        print(f"[OK REPORT] {path}")

    return ok


def main() -> None:
    ok = True

    print("============================================================")
    print("Checking required files")
    print("============================================================")
    for path in REQUIRED_FILES:
        ok = check_file_exists(path) and ok

    print("\n============================================================")
    print("Checking required figures")
    print("============================================================")
    fig_dir = Path("results/figures")
    for name in REQUIRED_FIGURES:
        ok = check_file_exists(str(fig_dir / name)) and ok

    print("\n============================================================")
    print("Checking key CSV schemas")
    print("============================================================")

    ok = check_csv_columns(
        "results/answerability_sweep.csv",
        [
            "semantic_type",
            "ranking_method",
            "channel",
            "n_top",
            "answerability_after_strict_validated",
            "answerability_after_loose_validated",
            "valid_packet_rate",
            "invalid_packet_rate",
        ],
    ) and ok

    ok = check_csv_columns(
        "results/answerability_sweep_ldpc.csv",
        [
            "semantic_type",
            "ranking_method",
            "coding_mode",
            "channel",
            "snr_db",
            "n_top",
            "decoded_ber",
            "packet_error_rate",
            "answerability_after_strict_validated",
            "answerability_after_loose_validated",
            "valid_packet_rate",
            "invalid_packet_rate",
            "coding_family",
            "decoder_type",
            "soft_llr_decoder",
            "standard_ldpc_bp",            
        ],
    ) and ok

    ok = check_csv_columns(
        "results/image_baseline.csv",
        [
            "row_type",
            "image_mode",
            "channel",
            "snr_db",
            "image_bits",
            "image_t_com_sec",
            "semantic_type",
            "semantic_bits",
            "semantic_t_com_sec",
            "image_to_semantic_latency_ratio",
        ],
    ) and ok

    ok = check_csv_columns(
        "results/latency_breakdown.csv",
        [
            "row_type",
            "semantic_type",
            "channel",
            "snr_db",
            "t_image_processing_ms",
            "t_com_ms",
            "t_total_ms",
            "image_to_semantic_total_ratio",
        ],
    ) and ok

    print("\n============================================================")
    print("Checking report keywords")
    print("============================================================")
    ok = check_report_keywords(
        "results/experiment_summary.md",
        [
            "Primary metric: *_validated answerability after invalid-packet drop.",
            "Raw Image Transmission",
            "Paper-style total latency",
            "LDPC-like",
            "Proxy Metric Contract",
            "validated_answer_related_semantic_coverage",            
        ],
    ) and ok

    print("\n============================================================")
    if not ok:
        print("ENVIRONMENT CHECK: FAIL")
        raise SystemExit(1)

    print("ENVIRONMENT CHECK: PASS")
    print("============================================================")


if __name__ == "__main__":
    main()
