# SemCom VQA Simulation

This repository contains a communication-model simulation baseline for goal-oriented semantic communication in wireless VQA.

## Current scope

Implemented:
- GQA subset loader
- SG/BBox semantic packet codec
- Original/DO/GO semantic ranking
- BPSK modulation/demodulation
- AWGN and Rayleigh channel simulation
- SNR sweep
- Shannon-like communication latency
- Semantic recovery metrics
- Proxy answerability metric
- LDPC-like channel coding

Limitations:
- Current task metric is proxy answerability, not full VQA accuracy.
- Current coding is LDPC-like hard bit-flipping, not standard LDPC BP.
- Image transmission baseline and total latency model are under development.
- Dataset and generated results are not included in this repository.

## Expected data path

The GQA subset should be placed outside the repository, for example:

```text
/home/cislab301b/Dung/Data/GQA/subsets/gqa_comm_v2

