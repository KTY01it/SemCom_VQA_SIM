# GQA10k DBSS/NST Benchmark Summary

## Split

- Train: samples 0–5999
- Validation: samples 6000–7999
- Test: samples 8000–9999

## Main setting

- Dataset: GQA communication subset
- Semantic type: scene graph triplets
- Channel: AWGN
- SNR: 8 dB
- n_top: 9
- Modulation: BPSK

## Methods

- Original-SG
- DO-SG
- GO-SG
- DBSS
- DBSS-QTC-noanswer
- DBSS-NST-v3
- DBSS-NST-v3-aggressive
- NSP-v1
- NSP-v2

## Main test result

DBSS gives the highest delivered answer hit rate, while DBSS-NST-v3 gives the best stable answer-per-kbit trade-off. NSP-v1 and NSP-v2 are included as standalone neural planner baselines, but they are weaker than the hybrid DBSS-NST-v3 pipeline.

| Method | Avg bits | Delivered answer | Answer/kbit | Field keep |
|---|---:|---:|---:|---:|
| Original-SG | 427.94 | 0.2621 | 0.6125 | 1.000 |
| DO-SG | 427.94 | 0.1691 | 0.3951 | 1.000 |
| GO-SG | 427.94 | 0.3297 | 0.7703 | 1.000 |
| DBSS | 427.94 | 0.3762 | 0.8791 | 1.000 |
| DBSS-QTC-noanswer | 391.58 | 0.3692 | 0.9428 | 0.853 |
| DBSS-NST-v3 | 293.08 | 0.3512 | 1.1982 | 0.623 |
| DBSS-NST-v3-aggressive | 247.12 | 0.3042 | 1.2308 | 0.516 |
| NSP-v1 | 323.64 | 0.2586 | 0.7991 | 0.694 |
| NSP-v2 | 293.50 | 0.2681 | 0.9136 | 0.624 |

## Validation consistency

| Method | Avg bits | Delivered answer | Answer/kbit | Field keep |
|---|---:|---:|---:|---:|
| DBSS | 428.01 | 0.4064 | 0.9495 | 1.000 |
| DBSS-QTC-noanswer | 390.58 | 0.3949 | 1.0111 | 0.850 |
| DBSS-NST-v3 | 293.79 | 0.3734 | 1.2709 | 0.624 |

## Current conclusion

The current proposed method is DBSS-NST-v3. It is a hybrid method: DBSS performs semantic evidence selection, and NST-v3 performs budget-aware neural field compression. The standalone neural planners NSP-v1 and NSP-v2 are useful diagnostic baselines but are not stronger than the hybrid method.
