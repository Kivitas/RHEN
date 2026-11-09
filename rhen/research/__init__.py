"""Research tooling for EOS and Rhen: devices, data, training, evaluation and sweeps.

Modules
-------
- ``device``: device/precision selection for CPU, CUDA/ROCm, Ascend NPU, Intel XPU and MPS.
- ``distributed``: torchrun/SLURM process groups, DDP and FSDP.
- ``data``: memory-mapped token corpora and the decision-example format.
- ``prepare``: tokenize text into a corpus (``python -m rhen.research.prepare``).
- ``tasks``: the synthetic decision benchmark (``python -m rhen.research.tasks``).
- ``train``: the trainer (``python -m rhen.research.train``).
- ``evaluate``: perplexity and decision metrics (``python -m rhen.research.evaluate``).
- ``metrics`` / ``calibration``: decision metrics, temperature scaling, conformal sets.
- ``sweep``: grid and seed sweeps (``python -m rhen.research.sweep``).
- ``checkpointing`` / ``manifest`` / ``presets``: exact resume, run provenance, model sizes.
"""

__all__ = [
    "calibration",
    "checkpointing",
    "data",
    "device",
    "distributed",
    "evaluate",
    "manifest",
    "metrics",
    "prepare",
    "presets",
    "sweep",
    "tasks",
    "train",
]
