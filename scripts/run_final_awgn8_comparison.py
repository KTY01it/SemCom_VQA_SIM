import subprocess
from pathlib import Path


ROOT = Path("results/final_gqa10k")
ROOT.mkdir(parents=True, exist_ok=True)


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    start = "8000"
    num = "2000"
    ntop = "9"
    snr = "8"
    channel = "awgn"

    # 1. Baseline ranking methods.
    run([
        "python", "scripts/run_benchmark_snr.py",
        "--semantic-type", "sg",
        "--methods", "original", "do", "go", "dbss",
        "--channels", channel,
        "--snrs", snr,
        "--n-top", ntop,
        "--start-index", start,
        "--num-samples", num,
        "--out", str(ROOT / "baselines_awgn8_ntop9.csv"),
    ])

    # 2. Fair rule compression.
    run([
        "python", "scripts/run_dbss_qtc_smoke.py",
        "--start-index", start,
        "--num-samples", num,
        "--n-top", ntop,
        "--snr-db", snr,
        "--channel", channel,
        "--no-answer",
        "--out", str(ROOT / "qtc_noanswer_awgn8_ntop9.csv"),
    ])

    # 3. DBSS-NST-v3 main.
    run([
        "python", "scripts/run_dbss_nst_v2_smoke.py",
        "--method-name", "dbss_nst_v3",
        "--start-index", start,
        "--num-samples", num,
        "--n-top", ntop,
        "--snr-db", snr,
        "--channel", channel,
        "--threshold", "0.40",
        "--model", "results/nst/nst_v3_train0_5999_ul0.10_lam0.10.pt",
        "--out", str(ROOT / "nst_v3_main_awgn8_ntop9.csv"),
    ])

    # 4. DBSS-NST-v3 aggressive.
    run([
        "python", "scripts/run_dbss_nst_v2_smoke.py",
        "--method-name", "dbss_nst_v3_aggressive",
        "--start-index", start,
        "--num-samples", num,
        "--n-top", ntop,
        "--snr-db", snr,
        "--channel", channel,
        "--threshold", "0.30",
        "--model", "results/nst/nst_v3_train0_5999_ul0.30_lam0.10.pt",
        "--out", str(ROOT / "nst_v3_aggressive_awgn8_ntop9.csv"),
    ])

    # 5. NSP-v1.
    run([
        "python", "scripts/run_nsp_smoke.py",
        "--method-name", "nsp_v1",
        "--start-index", start,
        "--num-samples", num,
        "--n-top", ntop,
        "--max-candidates", "30",
        "--snr-db", snr,
        "--channel", channel,
        "--mask-threshold", "0.30",
        "--model", "results/nsp/nsp_v1_train0_5999_ul0.10.pt",
        "--out", str(ROOT / "nsp_v1_awgn8_ntop9.csv"),
    ])

    # 6. NSP-v2.
    run([
        "python", "scripts/run_nsp_smoke.py",
        "--method-name", "nsp_v2_dbss_distill",
        "--start-index", start,
        "--num-samples", num,
        "--n-top", ntop,
        "--max-candidates", "30",
        "--snr-db", snr,
        "--channel", channel,
        "--mask-threshold", "0.50",
        "--model", "results/nsp/nsp_v2_train0_5999_ul0.10.pt",
        "--out", str(ROOT / "nsp_v2_awgn8_ntop9.csv"),
    ])

    print("\nDone. Results saved under:", ROOT)


if __name__ == "__main__":
    main()
