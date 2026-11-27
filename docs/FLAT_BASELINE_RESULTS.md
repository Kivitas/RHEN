# Flat baseline vs RHEN-2 workspace (plan 10.4 item 7)

Question: does the cached-state workspace beat the Laya-style layout (one bidirectional encoder over
`[state ; question ; marker + option tokens]`, option read at its marker) when parameters are matched?
Everything below comes from real runs on a 4-core CPU. Raw JSON and logs are not committed.

## What was compared

- Workspace: `Rhen2Model`, preset `r2-tiny` (1,841,202 parameters), loops sampled in training, `medium` effort at evaluation.
- Flat: `rhen/v2/flat_baseline.py`, `FlatBaselineModel`. Same embedding, numeric channel, `EncoderBlock`, `Readout` heads and loss. One encoder over the whole sequence, RoPE over the concatenated sequence, no loops, no option-set layer, no state cache. The state is repeated per question (own sequence copy), so M questions cost M full passes.
- Parameter matching: `matched_flat_config` gave 4 layers and 1,820,239 parameters (1.1% below). At `r2-50m` it gives 13 layers, 51.80M vs 51.84M (0.08%).
- Same data, step budget, AdamW (lr 2e-3, grad clip 1.0), batch order and seed for both. 200 steps, 16 examples per batch, 3 seeds (0,1,2).
- Data: `rhen.v2.data.synth`, 15 families (choice, noul, score, multi mixed; no range or extract). 384 train examples, 120 test, 120 OOD per seed. Examples that exceed r2-tiny token limits were dropped: train 1/3/6, test 1/2/0, OOD 15/10/10 for seeds 0/1/2 (OOD loses the most).

Command:

    python3 benchmarks/suite/flat_vs_workspace.py --out <tmp>/tiny.json

## Results, parameter-matched, r2-tiny, 3 seeds (mean +- sample std)

| metric | workspace | flat |
| :--- | ---: | ---: |
| test accuracy (choice/noul/score, ~118 questions per seed) | 0.302 +- 0.040 | 0.405 +- 0.060 |
| OOD accuracy | 0.306 +- 0.059 | 0.329 +- 0.050 |
| test ECE (15 bin) | 0.180 +- 0.028 | 0.163 +- 0.059 |
| OOD ECE | 0.148 +- 0.043 | 0.192 +- 0.044 |
| test multi exact match (13 questions per seed) | 0.019 +- 0.032 | 0.024 +- 0.041 |
| option-shuffle flip rate (choice and multi, test split) | 0.000 | 0.582 +- 0.070 |
| final train loss (mean of last 10 steps) | 4.63 +- 0.71 | 3.17 +- 0.41 |
| FLOPs per example (matmul estimate, probe batch) | 2.4e7 | 4.5e7 |
| parameters | 1,841,202 | 1,820,239 |

Paired difference flat minus workspace, same seeds: test accuracy +0.103 +- 0.034 (all three seeds positive: 0.135, 0.106, 0.069 raw gaps), OOD accuracy +0.023 +- 0.033, test ECE -0.018 +- 0.031, OOD ECE +0.044 +- 0.038.

Accuracies of 0.3 to 0.4 on mostly 2 to 5 option questions are far from solved: 200 steps is a very short budget and both models are under-trained.

## Extra runs

FLOP-matched flat (`--match flops`, 2 to 3 layers, 1.72M to 1.77M parameters, about 3.0e7 FLOPs vs 2.4e7), same settings, 3 seeds: test accuracy workspace 0.315 +- 0.03 vs flat 0.334 +- 0.063; OOD accuracy 0.309 +- 0.057 vs 0.391 +- 0.010; flip rate 0 vs 0.560 +- 0.079. Paired test accuracy difference +0.019 +- 0.054, OOD +0.082 +- 0.065. Note the workspace numbers are not bit-identical across the two runs (thread count differed), which gives a feel for run-to-run noise: seed 0 test accuracy was 0.339 vs 0.347.

`r2-50m` smoke run (1 seed, 10 steps, 4 examples per batch, 24 test examples, none dropped): flat (13 layers) test accuracy 0.30 vs workspace 0.20, OOD 0.318 for both, flip rate 0.40 vs 0. This is 10 optimiser steps and 20 evaluated questions: it only shows the code runs at that size and is not evidence about either architecture.

## Structural check (also in the unit tests)

The workspace model's option-shuffle flip rate is exactly 0 (max probability change about 2e-7); the flat model's is 0.54 to 0.66 per seed after training, because options differ by RoPE position and attend to each other. An untrained flat model also changes probabilities (max change 0.30 in the test).

## What this shows and does not show

Tiny CPU models (1.8M parameters, 1.6M of which are embeddings), synthetic data, 3 seeds, 200 steps, no tuning of either model, learning rate shared. Flat uses 1.8x the FLOPs of the workspace model at matched parameters on this probe, and about 2x the training wall time. It says nothing about 50M or larger scale, real data or calibration after post-hoc fitting.

Plain statement: at matched parameters on this setup the flat baseline had higher in-distribution accuracy (0.405 vs 0.302, paired +0.103 +- 0.034, 3 of 3 seeds) and lower training loss, so the workspace did not win there; on OOD accuracy and on ECE the difference is within seed noise; the workspace was exactly option-order invariant (flip rate 0) and the flat baseline was not (about 0.58). At matched FLOPs the accuracy differences were within noise. Per plan 10.4 item 7 the central claim is therefore not supported by this tiny experiment on accuracy, only on the invariance guarantee; it should be retested at 50M with real training budgets before reconsidering the architecture.
