# Banking77 evaluation

A reproducible measurement of intent classification on the official Banking77 test split,
with numpy baselines and an **untrained** RHEN model. It is the first measured artifact in
this repository for decision quality. It does not train RHEN and does not compare against
any other system on the same split.

## Reproduce

```bash
# Baselines only (numpy). Works without torch.
python3 benchmarks/banking77/run_eval.py --seed 0

# Quick run on a reproducible random subset of the test split.
python3 benchmarks/banking77/run_eval.py --limit 1000 --seed 0

# Tests (no network, no torch needed).
python3 -m pytest tests/test_benchmark_banking77.py -q -p no:cacheprovider
```

The script writes `benchmarks/banking77/results.json` (override with `--output`) and prints a
markdown table. Downloaded splits are cached under `~/.cache/rhen-benchmarks/banking77`
(override with `--cache-dir`). The cache is never written inside the repository.

Options: `--limit N` (random subset of N test examples), `--seed` (default 0), `--rhen-limit N`
(run the RHEN variant on only the first N examples of the evaluated set), `--skip-rhen`.

The RHEN variant is much slower than the baselines. With the default configuration on a 4-core CPU
without a GPU, a single `choice()` call with the 77 labels takes minutes, so the full test split is
not practical for it. Use `--rhen-limit` to bound that cost. The `n` reported for each variant in
`results.json` is the number of examples that variant actually ran on.

## Data

- Dataset: `PolyAI/banking77` on the Hugging Face Hub, official `train` and `test` splits
  (sizes come from the dataset: 10,003 train, 3,080 test, 77 labels).
- The Hub revision and the SHA-256 of each cached split file are recorded in `results.json`.
- Loader: the `datasets` library if it is installed. Otherwise the script pages through the
  datasets-server rows API (`datasets-server.huggingface.co/rows`), which returns the same rows
  as JSON. This avoids a parquet dependency. The `datasets` path has not been exercised in this
  environment.
- Label names are used as given in the dataset (`activate_my_card`, ...).

## Variants

| Variant | What it is | Probabilities | Abstain signal |
|---|---|---|---|
| `majority_class` | Predict the most frequent training label. | Training label prior | none |
| `tfidf_nearest_centroid` | TF-IDF (numpy, vocabulary from train only, min_df=2). Predict the class whose L2-normalised centroid has the highest cosine similarity. | none | none |
| `tfidf_logistic_regression` | Same TF-IDF features. Multinomial logistic regression, full-batch gradient descent, 300 epochs, L2 = 1e-4. The learning rate is chosen from {0.5, 2, 8, 32, 128} on a 10% validation slice of train, then the model is refit on all of train. | Softmax of fitted logits | none |
| `rhen_untrained` | `RhenEngine().choice(text, "Which banking intent?", labels)` with the default configuration, one engine instance, no checkpoint, no training. | Returned `probabilities` | `abstention_verdict.decision != "Answer"` when present |

The RHEN variant needs `torch` and `rhen` to import. If either fails, the variant is recorded as
`skipped` with the import error. An untrained RHEN is expected to sit near chance (1/77, about
1.3%). That result says nothing about a trained RHEN head.

## Metrics

- **Top-1 accuracy** over the evaluated test examples, with a 95% Wilson score interval.
- **ECE (10 equal-width bins)** on the top-label confidence. Reported only for variants that
  output a probability vector.
- **Abstain rate**: fraction of predictions with an abstain signal. `null` when the variant has
  none. `accuracy_on_answered` is reported alongside it.
- **Seconds per example**: baselines time the prediction step over the whole batch and divide by
  n, so their figures are amortised. The RHEN figure is per call to `choice()`, on CPU. The
  latency figures are not comparable across these two modes.

## Results

Commit `4b51f1c`, seed 0, `results.json` in this directory. The baselines ran on the full test
split (3,080 examples). The RHEN variant ran on the first 3 test rows only (see below).

| variant | n | top-1 | 95% Wilson CI | ECE (10 bins) | abstain rate |
|---|---|---|---|---|---|
| majority_class | 3080 | 1.30% | [0.96%, 1.76%] | 0.57% | n/a |
| tfidf_nearest_centroid | 3080 | 80.84% | [79.42%, 82.20%] | n/a | n/a |
| tfidf_logistic_regression | 3080 | 87.89% | [86.69%, 88.99%] | 37.10% | n/a |
| rhen_untrained | 3 | 0.00% | [0.00%, 56.15%] | 1.36% | 100% (3 of 3) |

Notes:

- **RHEN sample.** One `choice()` call with 77 options took 248 to 352 seconds on the 4-core CPU
  used here (torch 2.15.0, CPU only, default configuration). Full-split RHEN would take days, so it
  was limited to 3 rows. Those 3 results check that the pipeline runs and record its timing. They
  do not estimate accuracy: the 95% interval runs from 0% to 56%. All 3 calls returned
  `NeedClarification`, with the reasons "conformal uncalibrated" and "top-2 margin".
- **Logistic regression.** The selected learning rate was 128, the top of the grid. Validation
  accuracy was still rising at that value, so the model is probably under-trained after 300 epochs.
  The grid was not extended further. The ECE of 37% reflects the overconfident, unregularised
  softmax at that learning rate. The probabilities are not calibrated.
- **Timing.** Baseline seconds per example are batch times divided by n. The RHEN figure is per
  call on CPU. The two are not comparable.

## What the numbers do not show

- No trained RHEN head. The RHEN variant is untrained and measures only the untrained model.
- No comparison to JEV, Kev/Open-Jev or Laya on this split. None of those systems is run here,
  and no numbers from them are reproduced in `results.json`.
- No multilingual test. Banking77 is English only.
- No hyperparameter search for the RHEN variant, and no calibration of any variant.
- No statement about out-of-distribution behaviour, conformal coverage or latency on GPU.
- Seeds affect the test subsample and the logistic-regression validation split only. Results on
  the full split are deterministic for the baselines.

## Context from the repository README

The repository `README.md` makes claims about this task that this directory does not verify:
latency of "<15ms on GPU, <50ms CPU" (line 98), and that the `HierarchicalChoiceRouter` "scores
<25% of options without accuracy loss" on the 77 Banking77 intents (line 468). The `rhen-eval`
section (line 682) refers to a `benchmark_decisions.jsonl` file that is not in the repository.
Those claims remain unmeasured until a run on this split or another labelled set supports them.
The one latency number this run produced is far from the README figure: the default-configuration
RHEN took 248 to 352 seconds per call on a 4-core CPU (3 calls, 77 options). The README's "<50ms CPU"
figure is not reproduced here, and this run does not establish which configuration it refers to.
