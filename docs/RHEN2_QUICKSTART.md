# RHEN-2 quickstart

RHEN-2 (`rhen.v2`) is work in progress. Read the status in [RHEN2_ARCHITECTURE_PLAN.md](RHEN2_ARCHITECTURE_PLAN.md) first.

**Status:** nothing is trained. The commands below run the mechanics (data, packing, forward pass, training loop, calibration fit, pipeline, evaluation). Their outputs on the tiny preset are plumbing checks and say nothing about model quality. No accuracy, calibration or latency result is claimed anywhere in this repository for RHEN-2.

**Fail-closed rule.** Without a calibration object, or with one whose model hash does not match, nothing above the `low` risk tier can be `EXECUTE`. The pipeline also never stages an action from a model that has no `training_run_id`. Both rules are checked in step 5.

Run every command from the repository root. Python 3.10 or later is required; the repository is developed on Python 3.13.

## 1. Run the tests

```bash
python3 -m pytest tests/v2 -q
```

The tests need `pytest`, which is in the `dev` extra (`pip install -e ".[dev]"`). The v2 tests cover packing, the layers, calibration, the Checker, the pipeline, the converters, synthetic data, and the trainer.

## 2. Build a model and pack questions

This block was run as written with `PYTHONPATH=.` set to the repository root, or after `pip install -e .`. The model is untrained.

```python
import torch
from rhen.v2.config import preset
from rhen.v2.model import Rhen2Model
from rhen.v2.packing import QuestionSpec, Rhen2Example, Rhen2Packer

torch.manual_seed(0)
cfg = preset("r2-tiny")                 # other presets: r2-50m, r2-300m, r2-1b, r2-8b
model = Rhen2Model(cfg).eval()
packer = Rhen2Packer(cfg)               # uses the repository tokenizer by default

example = Rhen2Example(
    state="Cluster node cpu=99%, disk=full, last restart 3 days ago.",
    questions=[
        QuestionSpec(type="choice", instructions="What should be done?",
                     options=["Throttle", "Restart", "Ignore"]),
        QuestionSpec(type="noul", instructions="Is the disk full?"),
    ],
)
batch = packer.pack([example])          # one state, two questions, one batch
with torch.no_grad():
    out = model(batch)
print(tuple(out.probs.shape))           # (questions, options + null)
```

Question types are `choice`, `noul`, `score`, `multi`, `range` and `extract`. Each type has its own rules in `QuestionSpec.validate`. Questions are packed in one batch with the state, so a state is read once per batch. Options are a set, so their order does not change the answer by construction.

Run a pipeline-level example by packing your own `Rhen2Example` objects. To build many examples from the synthetic families, see step 4.

## 3. Train a smoke run (planned, not run here)

The CLI is in `rhen/v2/cli.py`. Its flags are these:

```bash
# Validate the config and data, print the plan, do not train
rhen2-train --config configs/v2/r2_tiny_cpu.json --dry-run

# Smoke run on CPU (20 steps, r2-tiny). Checkpoints go to runs/v2/r2_tiny_cpu
rhen2-train --config configs/v2/r2_tiny_cpu.json

# Override any config key with a JSON value (dotted keys reach into dicts and lists)
rhen2-train --config configs/v2/r2_tiny_cpu.json --set steps=40 --set model_overrides.d_model=128

# Pretraining smoke run instead of the decision stage
rhen2-train --config configs/v2/r2_tiny_cpu.json --set stage=pretrain

# Ignore an existing checkpoint in the run directory
rhen2-train --config configs/v2/r2_tiny_cpu.json --no-resume
```

These commands have not been run as part of writing this page. They use the flags defined in `cli.py`. The run prints a JSON summary. Its numbers are not evidence about model quality.

The 50M, 300M and 1B configurations are in `configs/v2/r2_*_decision.json` and `configs/v2/r2_*_pretrain.json`. The 8B configuration is an adapter config, `configs/v2/r2_8b_adapter.json`. Do not start a large run from the smoke config.

## 4. Calibrate

Calibration is a Python API. There is no command-line calibration step yet. It fits temperatures, isotonic maps, group conformal sets and OOD thresholds from labelled records. The records come from the model's outputs on a labelled split, not from the training data.

```python
import torch
from rhen.v2.calibrate import collect_records, fit_calibration
from rhen.v2.config import preset
from rhen.v2.model import Rhen2Model
from rhen.v2.data.synth import generate_split
from rhen.v2.packing import Rhen2Packer, make_batches
from rhen.v2.pipeline import weights_hash

torch.manual_seed(0)
cfg = preset("r2-tiny")
model = Rhen2Model(cfg).eval()
packer = Rhen2Packer(cfg)

# Keep examples that fit the tiny preset (max_state_tokens=256); the suite drops longer ones too.
examples = []
for ex in generate_split("val", 1200, seed=1):
    try:
        packer.pack([ex])
        examples.append(ex)
    except ValueError:
        pass
batches = [packer.pack(chunk) for chunk in make_batches(examples, max_questions=64)]

records = collect_records(model, batches)
cal = fit_calibration(records, model_hash=weights_hash(model))
import os
CAL_PATH = "runs/demo/calibration.json"   # runs/ is git-ignored; never commit a demo calibration
os.makedirs("runs/demo", exist_ok=True)
cal.save(CAL_PATH)
print("calibrated:", cal.calibrated)
```

The `calibrated` flag becomes true only when the fitted tables are complete and at least one group has `MIN_CONFORMAL_N` (100) records. Calibrating an untrained model, as this example does, gives a mechanics check only. The model hash must match the model you later load, or `Calibration.load(path, expected_model_hash=...)` refuses it.

## 5. Run the decision pipeline

```python
from rhen.v2.calibrate import Calibration
from rhen.v2.packing import QuestionSpec
from rhen.v2.pipeline import DecisionPipeline

pipe = DecisionPipeline(model, packer)                      # no calibration
res = pipe.decide(
    "Cluster node cpu=99%, disk=full, last restart 3 days ago.",
    [QuestionSpec(type="choice", instructions="What should be done?",
                  options=["Throttle", "Restart", "Ignore"])],
    effort="low", risk_tier="high",
)
print(res.status.value, res.calibrated)                      # ABSTAIN False: fail closed above low
```

The pipeline runs these stages in order: ingest, plan, encode, deliberate, calibrate, check, policy, stage, audit and feedback. The result has one `QuestionDecision` per question, the `DecisionTrace` of every stage, and a status: `EXECUTE`, `NEEDS_APPROVAL`, `ABSTAIN` or `BLOCKED`. `risk_tier` selects the cost policy. `effort` controls how much of the Checker runs.

With a calibration object, pass it to the constructor:

```python
pipe = DecisionPipeline(model, packer, calibration=Calibration.load(CAL_PATH))
```

An untrained model never stages and `execute` refuses its decisions, whatever the calibration. Load a trained checkpoint and set `training_run_id` before any action can be staged.

## 6. Evaluate

The evaluation command, `rhen2-eval`, loads a checkpoint written by the trainer:

```bash
rhen2-eval --checkpoint runs/v2/r2_tiny_cpu --split val -n 64 --effort medium --out runs/v2/r2_tiny_cpu/eval.json
```

`--checkpoint` takes a model file, a step directory or a run directory. `--weights ema` is the default; `--weights raw` loads the raw model. Use `--jsonl PATH` to evaluate on a labelled decision file instead of the synthetic split. This command has not been run as part of writing this page.

For the full evaluation suite and the ablations, see [`benchmarks/suite/README.md`](../benchmarks/suite/README.md):

```bash
python3 benchmarks/suite/run_suite.py --preset r2-tiny --n 200 --out benchmarks/suite/results.json
python3 benchmarks/suite/run_ablations.py --steps 60 --out-dir benchmarks/suite/ablations
python3 benchmarks/latency/run_latency.py --preset r2-tiny --out benchmarks/latency/results.json
```

Ablations run three seeds with a small step budget. Item 7 of plan section 10.4 (the flat-encoder baseline, which tests the central design claim) is not implemented. The suite marks results as untrained when no checkpoint is given. Do not read tiny-preset or untrained numbers as evidence about any scale.
