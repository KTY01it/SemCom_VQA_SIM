#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

echo "============================================================"
echo "[0/9] Compile check"
echo "============================================================"
python -m compileall -q src scripts

echo "============================================================"
echo "[1/9] Packet validation unit test"
echo "============================================================"
python scripts/test_packet_validation.py

echo "============================================================"
echo "[1b/9] Channel metadata unit test"
echo "============================================================"
python scripts/test_channel_metadata.py

echo "============================================================"
echo "[2/9] SG packet sanity"
echo "============================================================"
python scripts/run_sg_packet_sanity.py

echo "============================================================"
echo "[3/9] BBox packet sanity"
echo "============================================================"
python scripts/run_bbox_packet_sanity.py

echo "============================================================"
echo "[4/9] No-LDPC answerability sweep"
echo "============================================================"
python scripts/run_answerability_sweep.py

echo "============================================================"
echo "[5/9] LDPC-like answerability sweep"
echo "============================================================"
python scripts/run_answerability_sweep_ldpc.py

echo "============================================================"
echo "[6/9] Raw image baseline"
echo "============================================================"
python scripts/run_image_baseline.py

echo "============================================================"
echo "[7/9] Paper-style total latency breakdown"
echo "============================================================"
python scripts/run_total_latency_breakdown.py

echo "============================================================"
echo "[7b/9] Proxy metric contract"
echo "============================================================"
python scripts/export_proxy_metric_contract.py

echo "============================================================"
echo "[8/9] Summaries and figures"
echo "============================================================"
python scripts/summarize_results.py > results/summary_tables.txt
python scripts/summarize_ldpc_results.py > results/summary_ldpc_tables.txt
python scripts/plot_results.py
python scripts/make_experiment_report.py

echo "============================================================"
echo "[9/9] Output check"
echo "============================================================"
python scripts/check_environment_outputs.py

echo "============================================================"
echo "DONE: full environment reproduction completed."
echo "Main report: results/experiment_summary.md"
echo "Figures:     results/figures/"
echo "============================================================"
