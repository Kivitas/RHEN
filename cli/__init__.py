"""RHEN Command Line Interface (CLI) tools.

Entry points for:
- rhen-decide: CLI decision evaluation on states and typed questions
- rhen-train: CLI decision training loop
- rhen-eval: CLI evaluation of accuracy, ECE calibration, Brier score, and OOD AUROC
- rhen-tasks: CLI synthetic decision benchmark generation
"""

from __future__ import annotations

__all__ = ["decide", "train", "eval", "tasks"]
