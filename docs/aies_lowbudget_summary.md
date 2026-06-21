# AIES Low-Budget Pure-AI Semantic Selection Summary

Repository:

https://github.com/KTY01it/SemCom_VQA_SIM

Recommended branch:

nsp-v8-crossencoder-reranker

## Project Context

This repository implements a semantic communication simulation pipeline for wireless visual question answering on a GQA subset.

Local data path used in experiments:

/home/cislab301b/Dung/Data/GQA/subsets/gqa_comm_v2

Main split:
- train: indices 0-5999
- validation: indices 6000-7999
- test: indices 8000-9999

## Main Methods

### DBSS-NST-v4 Soft Slot-Guard

This is the strongest balanced-budget method.

Pipeline:

DBSS semantic triplet selection
-> NST neural field compression
-> Soft Slot-Guard
-> budget repair
-> compressed SG packet
-> BPSK + AWGN/Rayleigh
-> partial answerability evaluation

Previously observed balanced-budget full-test result:

- bits: about 296.23
- delivered_answer_hit_rate: about 0.35718
- answer_per_kbit: about 1.2057

### AIES: AI Evidence Selector

AIES is the pure-AI ultra-low-budget operating mode.

Pipeline:

Question + SG candidates
-> CrossEncoder evidence scoring
-> top-k SG triplet selection
-> NST field compression
-> compressed SG packet
-> channel simulation
-> partial answerability evaluation

CrossEncoder base model:

cross-encoder/ms-marco-MiniLM-L6-v2

Local trained CrossEncoder model path:

results/nsp/nsp_v8a_crossencoder_answer_hn_miniLM

NST model path:

results/nst/nst_v3_train0_5999_ul0.10_lam0.10.pt

These model/result files are local artifacts and should not be committed.

## Full-Test Low-Budget Results

Evaluation setting:

- start_index: 8000
- num_samples: 2000
- actual samples: 1999
- channel: AWGN
- snr_db: 8
- max_candidates: 30

AIES wins in ultra-low-budget regimes:

| n_top | target_bits | AIES answer | DBSS answer | delta answer | AIES answer/kbit | DBSS answer/kbit | delta answer/kbit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19 | 0.112056 | 0.063032 | +0.049025 | 5.897686 | 3.317448 | +2.580237 |
| 1 | 35 | 0.240620 | 0.199600 | +0.041021 | 6.874866 | 6.288119 | +0.586747 |
| 1 | 51 | 0.286143 | 0.203102 | +0.083042 | 5.610648 | 5.049940 | +0.560709 |
| 3 | 57 | 0.146073 | 0.101051 | +0.045023 | 2.563022 | 1.773050 | +0.789973 |
| 5 | 95 | 0.166083 | 0.122561 | +0.043522 | 1.749016 | 1.291125 | +0.457891 |

DBSS-NST-v4 remains stronger in some mid-budget regimes:

| n_top | target_bits | AIES answer | DBSS answer | delta answer | AIES answer/kbit | DBSS answer/kbit | delta answer/kbit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 105 | 0.273637 | 0.292646 | -0.019010 | 2.606251 | 2.836171 | -0.229920 |
| 5 | 175 | 0.287144 | 0.324162 | -0.037019 | 1.643094 | 1.870513 | -0.227419 |

## Current Interpretation

DBSS-NST-v4 Soft Slot-Guard is the strongest balanced-budget method.

AIES is a pure-AI semantic evidence selector that is stronger in ultra-low-budget regimes.

Recommended claim:

Under extremely constrained semantic budgets, a pure-AI CrossEncoder evidence selector achieves higher answer utility per transmitted bit than the structured DBSS-NST baseline. In balanced-budget regimes, DBSS-NST-v4 remains stronger.

## Important Branches

- dbss-nst-v4-slotguard: main balanced-budget method
- nsp-v7a-evidence-setpolicy: failed pure-AI set-policy ablation
- nsp-v8-crossencoder-reranker: current AIES / CrossEncoder pure-AI low-budget branch

## Important Files

- scripts/build_nsp_v8_crossencoder_dataset.py
- scripts/train_nsp_v8_crossencoder.py
- scripts/run_nsp_v8_crossencoder_smoke.py
- scripts/diagnose_nsp_v8_selection.py
- scripts/diagnose_nsp_v8_metric_gap.py
- scripts/run_dbss_nst_slotguard_smoke.py
- src/methods/slot_guard.py
- docs/aies_lowbudget_summary.md
