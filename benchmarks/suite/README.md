# RHEN-2 evaluation suite

Implements plan section 10 (tasks T-701 to T-703). Metrics live in `rhen/v2/eval.py`.

## Scripts

- `run_suite.py`: evaluates a checkpoint (`--checkpoint`, a `torch.save` dict `{'cfg','state_dict'}` or
  anything `rhen.v2.io.load_checkpoint` reads) or, without one, an untrained preset, and writes a JSON file
  (`trained: false` for untrained). Splits: synthetic `test` and `test_ood` (from `rhen.v2.data.synth` when
  importable, else the v1 generator; force v1 with `--synthetic v1`) plus `--data NAME=PATH` JSONL files.
  Examples longer than the model's `max_state_tokens` are dropped and counted (`dropped_too_long`).
  Numpy baselines (majority class, TF-IDF nearest centroid over option texts) run on choice questions when a
  train set exists (the synthetic train split, or `--train-data NAME=PATH`).
- `run_ablations.py`: plan 10.4 items 1 to 6, 3 seeds, tiny step budget (`--steps`), JSON plus markdown table.
  Item 7 (flat baseline) is not implemented.
- `../latency/run_latency.py`: cold and warm p50/p95 by state length, K, effort.

```
python3 benchmarks/suite/run_suite.py --preset r2-tiny --n 200 --out benchmarks/suite/results.json
python3 benchmarks/suite/run_ablations.py --steps 60 --out-dir benchmarks/suite/ablations
python3 benchmarks/latency/run_latency.py --preset r2-tiny --out benchmarks/latency/results.json
```

## Reading the numbers

- ECE: 15 equal-mass bins on top-label confidence. Pooled discrete metrics cover choice, noul and score
  questions; a null-labelled question is correct when the argmax is the null column.
- Conformal: per (type, k_bucket) groups, nonconformity `1 - p[label]`, quantiles fitted on the synthetic
  `val` split with the model's own predictions. Without a trained model and a real calibration set the
  coverage column is only a harness check. The `rhen.v2.calibrate` tables are not used yet.
- Noul consistency is `|P(true | p) - P(false | not p)|`.
- Shuffle probes cover choice and multi questions only (score levels are ordered).
- `reference_points` in the result file are vendor-reported figures from plan 10.1, not reproduced here.
- Tiny CPU ablations and untrained runs say nothing about 50M-scale conclusions.
