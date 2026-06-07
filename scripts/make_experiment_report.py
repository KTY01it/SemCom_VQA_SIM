from pathlib import Path


def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p.read_text(encoding="utf-8")


def main() -> None:
    summary_tables = read_text("results/summary_tables.txt")
    ldpc_tables = read_text("results/summary_ldpc_tables.txt")

    report = f"""# SemCom VQA Communication Simulation Report

## 1. Current pipeline

The current simulation pipeline is:

```text
GQA subset
→ semantic packet selection/ranking
→ SG/BBox source encoding
→ optional LDPC-like channel coding
→ BPSK modulation
→ AWGN/Rayleigh channel
→ hard demodulation
→ optional LDPC-like decoding
→ semantic packet recovery
→ delivered relevance / answerability evaluation
→ latency estimation
