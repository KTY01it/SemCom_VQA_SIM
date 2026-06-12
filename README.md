# SemCom VQA Simulation

This repository contains a communication-model simulation environment for semantic communication in wireless visual question answering (VQA).

The project is designed to reproduce the core communication path of a goal-oriented semantic communication system:

```text
semantic packetization
→ packet-level CRC validation
→ uncoded / LDPC-like channel coding
→ BPSK modulation
→ AWGN / Rayleigh channel
→ hard-decision demodulation
→ semantic packet recovery
→ validated proxy answerability
→ communication and total latency analysis
```

The current implementation is a simulation environment, not a full reproduction of the original paper's neural VQA pipeline.

## Current scope

Implemented:

* GQA communication subset loader
* SG triplet semantic packet codec
* BBox semantic packet codec
* Original / DO / GO semantic ranking baselines
* Random semantic selection baseline
* DBSS: Diverse Budgeted Semantic Selection for SG triplets
* DBSS ablation variants:
  * DBSS without coverage gain
  * DBSS without redundancy penalty
  * DBSS without channel reliability term
  * DBSS without cost penalty
* Evidence-quality metrics:
  * question concept coverage
  * semantic redundancy ratio
  * unique concept count
* Uncoded DBSS benchmark runner
* LDPC-coded DBSS benchmark runner
* Packet-level CRC16 wrapper and CRC-fail drop policy
* Actual object/relation vocab ID-set validation
* BPSK modulation and hard-decision demodulation
* AWGN channel simulation
* Flat Rayleigh fading channel simulation with perfect-CSI equalization
* SNR-dependent channel corruption
* Uncoded semantic packet transmission
* LDPC-like sparse systematic block coding
* LDPC-like hard bit-flipping decoder
* Channel metadata export:

  * SNR linear value
  * signal power
  * noise power
  * noise variance
  * Rayleigh gain-power statistics
  * LLR-ready metadata
* Shannon-like communication latency
* Paper-style total latency approximation
* Raw-float32 image transmission baseline
* Semantic recovery metrics
* Validated proxy answerability metrics
* Proxy metric contract
* Automated summary tables, plots, and experiment report
* Full reproduction script: `scripts/run_all.sh`
* Environment output checker: `scripts/check_environment_outputs.py`

## Important terminology

### Validated proxy answerability

The main task-level metric is `validated proxy answerability`, formally described as:

```text
validated_answer_related_semantic_coverage
```

This metric checks whether delivered valid semantic packets contain answer-related semantic evidence after packet validation and invalid-packet drop.

It is not full VQA accuracy.

### CRC16 packet validation

When `packet.crc16_enabled=true`, semantic packets are transmitted with CRC16:

```text
SG packet:   48 payload bits + 16 CRC bits = 64 bits
BBox packet: 80 payload bits + 16 CRC bits = 96 bits
```

CRC failure has priority during packet validation. Packets that fail CRC are dropped before vocab/range/geometry validation.

### LDPC-like coding

The current coding module is:

```text
LDPC-like sparse systematic block coding
```

It uses:

```text
systematic sparse parity construction
hard-decision bit-flipping decoder
```

It is not standard LDPC BP/min-sum decoding with soft LLR.

### DBSS: Diverse Budgeted Semantic Selection

This repository includes a proposed semantic selection method:

```text
DBSS: Diverse Budgeted Semantic Selection
```

DBSS is designed for SG-based goal-oriented semantic communication. Instead of ranking each scene-graph triplet independently and transmitting the top-`K` triplets, DBSS performs set-level semantic evidence selection.

The selected triplets are optimized to be:

* relevant to the question,
* useful for covering question-related concepts,
* less redundant with already selected triplets,
* diverse in semantic content,
* compatible with the semantic transmission budget.

Conceptually, DBSS changes the semantic transmission strategy from:

```text
independent top-K semantic ranking
```

to:

```text
budgeted diverse semantic evidence selection
```

The current DBSS implementation is an AI-inspired combinatorial optimization method based on diverse subset selection and greedy marginal-gain selection. It is not a deep neural network model.

The main DBSS setting in this repository is:

```text
DBSS-SG
```

BBox-based DBSS is not the main setting because the current contribution focuses on structured SG triplet selection for visual reasoning.

### Channel metadata and LLR-ready statistics

The environment exposes channel metadata and LLR-ready statistics for future soft-decoding or channel-aware semantic selection methods.

Current reported decoding remains hard-decision unless a soft decoder is explicitly added.

## Current limitations

* The task metric is validated proxy answerability, not full VQA accuracy.
* The coding module is LDPC-like hard bit-flipping, not standard LDPC BP/min-sum with soft LLR.
* The image transmission baseline is raw-float32 image-size transmission, not JPEG-compressed image transmission.
* The total latency model uses paper-style FLOPs approximation, not measured runtime profiling.
* The Rayleigh model is flat Rayleigh fading with perfect-CSI equalization.
* Dataset files and generated result files are not included by default in this repository.
* DBSS is currently implemented mainly for SG triplets.
* DBSS is a greedy set-level selection method, not a trained neural selector.
* The current DBSS channel/cost terms have limited influence for fixed-length SG packets; the main observed gain comes from redundancy-aware diverse evidence selection.
* Reported DBSS gains are based on validated proxy answerability, not full VQA model accuracy.

## Expected data path

The GQA communication subset should be placed outside the repository, for example:

```text
/home/cislab301b/Dung/Data/GQA/subsets/gqa_comm_v2
```

Expected files:

```text
samples.jsonl
bbox_packets.jsonl
sg_triplets.jsonl
object_vocab.json
relation_vocab.json
answer_vocab.json
stats.json
```

The data root is configured in:

```text
configs/experiment.yaml
```

Example:

```yaml
data:
  root: /home/cislab301b/Dung/Data/GQA/subsets/gqa_comm_v2
```

## Main configuration

Main experiment settings are stored in:

```text
configs/experiment.yaml
```

Important blocks:

```text
project
data
image
source
packet
modulation
channel
experiment
latency
ldpc
output
```

The most important simulation parameters include:

```text
packet.crc16_enabled
channel.snr_db_list
channel.perfect_csi
experiment.channels
experiment.ranking_methods
experiment.n_top_list
experiment.ldpc_snr_db_list
experiment.coding_modes
latency.bandwidth_hz
ldpc.k
ldpc.m
ldpc.col_weight
ldpc.max_iter
```

## Reproduce the environment

From the repository root:

```bash
export PYTHONPATH=.
bash scripts/run_all.sh
```

Expected final output:

```text
ENVIRONMENT CHECK: PASS
DONE: full environment reproduction completed.
```

The main report is generated at:

```text
results/experiment_summary.md
```

Figures are generated under:

```text
results/figures/
```

## Run DBSS benchmarks

Uncoded SNR benchmark:

```bash
PYTHONPATH=. python scripts/run_benchmark_snr.py \
  --semantic-type sg \
  --methods random original do go dbss \
  --channels awgn rayleigh \
  --n-top 9 \
  --num-samples 500 \
  --out results/benchmark/main_snr_sg_dbss.csv
```
LDPC-coded SNR benchmark::
```bash
PYTHONPATH=. python scripts/run_benchmark_ldpc_snr.py \
  --methods random original do go dbss \
  --channels awgn rayleigh \
  --snrs -4 -2 0 2 4 6 8 10 12 14 16 \
  --n-top 9 \
  --num-samples 500 \
  --out results/benchmark_ldpc/main_snr_sg_ldpc_dbss.csv
```

LDPC-coded Ntop benchmark:
```bash
PYTHONPATH=. python scripts/run_benchmark_ldpc_ntop.py \
  --methods random original do go dbss \
  --channels awgn rayleigh \
  --snr-db 8 \
  --num-samples 500 \
  --out results/benchmark_ldpc/main_ntop_sg_ldpc_dbss.csv
```

DBSS ablation:
```bash
PYTHONPATH=. python scripts/run_benchmark_ntop.py \
  --semantic-type sg \
  --methods go dbss dbss_no_coverage dbss_no_redundancy dbss_no_channel dbss_no_cost \
  --channels rayleigh \
  --snr-db 8 \
  --n-tops 3 6 9 12 15 18 21 24 27 30 \
  --num-samples 500 \
  --out results/ablation/dbss_ablation_ntop_rayleigh8.csv
```


## Main outputs

## Main generated outputs

Generated benchmark outputs are written under:

```text
results/
```

This directory is intentionally ignored by Git and should not be committed.

Important generated result groups include:

```text
results/benchmark/
results/benchmark_ldpc/
results/ablation/
results/figures/
results/tables/
```

The most important benchmark files generated locally are:

```text
results/benchmark/main_snr_sg_dbss.csv
results/benchmark/main_ntop_sg_dbss_metrics.csv
results/benchmark_ldpc/main_snr_sg_ldpc_dbss.csv
results/benchmark_ldpc/main_ntop_sg_ldpc_dbss.csv
results/ablation/dbss_ablation_ntop_rayleigh8.csv
results/ablation/dbss_ablation_ntop_awgn8.csv
```

Only summarized results should be reported in this README or in paper tables. Raw benchmark CSV files and generated figures are not tracked by Git.


## DBSS benchmark summary

The main benchmark compares SG-based semantic selection methods:

| Method      | Description                                  | Role                            |
| ----------- | -------------------------------------------- | ------------------------------- |
| Random-SG   | Randomly selects SG triplets                 | Lower-bound semantic baseline   |
| Original-SG | Uses the original SG triplet order           | No-ranking baseline             |
| DO-SG       | Data-oriented frequency-based ranking        | Data-oriented semantic baseline |
| GO-SG       | Goal-oriented SG ranking                     | Core baseline                   |
| DBSS-SG     | Diverse budgeted semantic evidence selection | Proposed method                 |

### Uncoded benchmark

The uncoded benchmark evaluates semantic packets transmitted with BPSK over AWGN and Rayleigh channels.

| Setting                           | Result summary                            |
| --------------------------------- | ----------------------------------------- |
| AWGN SNR sweep                    | DBSS outperforms GO-SG in 9/11 SNR points |
| Rayleigh SNR sweep                | DBSS outperforms GO-SG in 7/11 SNR points |
| Average gain over GO-SG, AWGN     | +0.022955 delivered answer hit rate       |
| Average gain over GO-SG, Rayleigh | +0.010202 delivered answer hit rate       |

### LDPC-coded benchmark

The LDPC-coded benchmark uses the repository's LDPC-like sparse systematic block code with hard-decision bit-flipping decoding.

| Setting                                           | Result summary                              |
| ------------------------------------------------- | ------------------------------------------- |
| LDPC AWGN SNR sweep                               | DBSS outperforms GO-SG in 9/11 SNR points   |
| LDPC Rayleigh SNR sweep                           | DBSS outperforms GO-SG in 8/11 SNR points   |
| LDPC AWGN Ntop sweep                              | DBSS outperforms GO-SG in 10/10 Ntop points |
| LDPC Rayleigh Ntop sweep                          | DBSS outperforms GO-SG in 10/10 Ntop points |
| Average LDPC gain over GO-SG, AWGN Ntop sweep     | +0.026453 delivered answer hit rate         |
| Average LDPC gain over GO-SG, Rayleigh Ntop sweep | +0.033667 delivered answer hit rate         |

### Evidence quality analysis

In the Rayleigh SNR = 8 setting, DBSS improves semantic evidence quality compared with GO-SG.

| Method              | Mean delivered answer hit rate | Coverage ratio | Redundancy ratio | Unique concept count |
| ------------------- | -----------------------------: | -------------: | ---------------: | -------------------: |
| GO-SG               |                         0.1471 |         0.5460 |           0.4329 |                14.29 |
| DBSS w/o redundancy |                         0.1589 |         0.5509 |           0.4260 |                15.23 |
| DBSS                |                         0.1705 |         0.5590 |           0.2983 |                18.53 |

These results indicate that the main improvement of DBSS comes from redundancy-aware diverse evidence selection. DBSS selects a less redundant and more diverse set of SG triplets than GO-SG under the same semantic budget.


## Validation and tests

Compile check:

```bash
PYTHONPATH=. python -m compileall -q src scripts
```

Packet validation test:

```bash
PYTHONPATH=. python scripts/test_packet_validation.py
```

Channel metadata test:

```bash
PYTHONPATH=. python scripts/test_channel_metadata.py
```

Full output check:

```bash
PYTHONPATH=. python scripts/check_environment_outputs.py
```

## Notes for future method development

This environment is suitable for developing new methods in:

* diverse semantic evidence selection
* submodular-style semantic subset selection
* redundancy-aware semantic packet selection
* channel-aware semantic packet selection
* reliability-aware semantic ranking
* adaptive semantic budget / adaptive `Ntop`
* unequal semantic packet protection
* latency-aware semantic transmission
* semantic recovery under AWGN/Rayleigh channels

Recommended claim boundary:

```text
communication-model simulation with validated proxy answerability
```

Avoid claiming:

```text
full VQA reproduction
standard LDPC BP decoding
JPEG image transmission baseline
```

unless those modules are explicitly added.
