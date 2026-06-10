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

## Main outputs

Expected result files include:

```text
results/sg_packet_sanity.csv
results/bbox_packet_sanity.csv
results/answerability_sweep.csv
results/answerability_sweep_ldpc.csv
results/image_baseline.csv
results/latency_breakdown.csv
results/summary_tables.txt
results/summary_ldpc_tables.txt
results/proxy_metric_contract.json
results/experiment_summary.md
```

Expected figure files include:

```text
results/figures/fig1_sg_answerability_no_ldpc_awgn.png
results/figures/fig1_sg_answerability_no_ldpc_rayleigh.png
results/figures/fig2_bbox_answerability_no_ldpc_awgn.png
results/figures/fig2_bbox_answerability_no_ldpc_rayleigh.png
results/figures/fig3_channel_damage_go_sg_awgn.png
results/figures/fig3_channel_damage_go_sg_rayleigh.png
results/figures/fig3_channel_damage_go_bbox_awgn.png
results/figures/fig3_channel_damage_go_bbox_rayleigh.png
results/figures/fig4_latency_answerability_no_ldpc_awgn.png
results/figures/fig4_latency_answerability_no_ldpc_rayleigh.png
results/figures/fig5_ldpc_gain_go_sg_rayleigh_ntop3.png
results/figures/fig5_ldpc_gain_go_sg_rayleigh_ntop6.png
results/figures/fig5_ldpc_gain_go_sg_rayleigh_ntop9.png
results/figures/fig5_ldpc_gain_go_sg_rayleigh_ntop12.png
results/figures/fig5_ldpc_gain_go_bbox_rayleigh_ntop3.png
results/figures/fig5_ldpc_gain_go_bbox_rayleigh_ntop6.png
results/figures/fig5_ldpc_gain_go_bbox_rayleigh_ntop9.png
results/figures/fig5_ldpc_gain_go_bbox_rayleigh_ntop12.png
results/figures/fig6_ldpc_latency_tradeoff_go_sg_rayleigh_8db.png
results/figures/fig6_ldpc_latency_tradeoff_go_sg_rayleigh_10db.png
results/figures/fig6_ldpc_latency_tradeoff_go_sg_rayleigh_12db.png
results/figures/fig6_ldpc_latency_tradeoff_go_bbox_rayleigh_8db.png
results/figures/fig6_ldpc_latency_tradeoff_go_bbox_rayleigh_10db.png
results/figures/fig6_ldpc_latency_tradeoff_go_bbox_rayleigh_12db.png
results/figures/fig7_decoded_ber_go_sg_rayleigh_ntop12.png
results/figures/fig7_per_go_sg_rayleigh_ntop12.png
results/figures/fig8_image_vs_semantic_latency_awgn.png
results/figures/fig8_image_vs_semantic_latency_rayleigh.png
results/figures/fig9_image_to_semantic_latency_ratio_awgn.png
results/figures/fig9_image_to_semantic_latency_ratio_rayleigh.png
results/figures/fig10_total_latency_image_vs_semantic_awgn.png
results/figures/fig10_total_latency_image_vs_semantic_rayleigh.png
results/figures/fig11_total_latency_ratio_awgn.png
results/figures/fig11_total_latency_ratio_rayleigh.png
```

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
