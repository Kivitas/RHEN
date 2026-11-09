"""Backward-compatibility aliases must resolve to the canonical modules, not copies.

Old import paths kept alive after the duplicated trees were removed:
  * eos_research.*  -> rhen.research.*
  * optim.*         -> rhen.optim.*
  * root shims      -> rhen.* (model.py, tokenizer.py, ...)

Identity (``is``) is the contract: a second module object would mean duplicated
classes, so ``isinstance`` checks and monkeypatching would silently diverge.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESEARCH_SUBMODULES = [
    "calibration", "checkpointing", "data", "device", "distributed", "evaluate",
    "manifest", "metrics", "prepare", "presets", "sweep", "tasks", "train",
]
OPTIM_SUBMODULES = [
    "advanced_loops", "checkpoint", "fused_swiglu", "grouped_gemm", "mup",
    "quantized_kv", "rhen_mask", "ring_mla",
]
ROOT_SHIMS = [
    "adapter", "agent_tools", "config", "context_builder", "eos_eval", "eos_loop",
    "meta_cognitive_monitor", "model", "optimizers", "rhen_chunker", "rhen_cognition",
    "rhen_credit", "rhen_loss", "rhen_packer", "rhen_pillars", "rhen_verify",
    "tokenizer", "unified_orchestrator",
]


class EosResearchAliasTests(unittest.TestCase):
    def test_package_is_canonical_package(self):
        import eos_research
        import rhen.research

        self.assertIs(eos_research, rhen.research)

    def test_submodules_are_canonical_modules(self):
        for name in RESEARCH_SUBMODULES:
            with self.subTest(name=name):
                old = importlib.import_module(f"eos_research.{name}")
                new = importlib.import_module(f"rhen.research.{name}")
                self.assertIs(old, new)

    def test_from_import_returns_canonical_object(self):
        from eos_research.sweep import main as old_main
        from rhen.research.sweep import main as new_main

        self.assertIs(old_main, new_main)

    def test_sweep_entry_point_is_still_reachable(self):
        self.assertTrue(callable(importlib.import_module("eos_research.sweep").main))


class OptimAliasTests(unittest.TestCase):
    def test_package_is_canonical_package(self):
        import optim
        import rhen.optim

        self.assertIs(optim, rhen.optim)

    def test_submodules_are_canonical_modules(self):
        for name in OPTIM_SUBMODULES:
            with self.subTest(name=name):
                old = importlib.import_module(f"optim.{name}")
                new = importlib.import_module(f"rhen.optim.{name}")
                self.assertIs(old, new)

    def test_from_import_returns_canonical_object(self):
        from optim.advanced_loops import create_loop_bypass_router as old_fn
        from rhen.optim.advanced_loops import create_loop_bypass_router as new_fn

        self.assertIs(old_fn, new_fn)

    def test_fresh_interpreter_old_path_first_gives_identity(self):
        # Import order matters for module aliasing: check the old path first in a
        # clean interpreter, where neither package has been imported yet.
        code = (
            "import sys\n"
            "from optim.checkpoint import CheckpointedRecurrentCore as Old\n"
            "import rhen.optim.checkpoint as new\n"
            "assert sys.modules['optim.checkpoint'] is new\n"
            "assert Old is new.CheckpointedRecurrentCore\n"
            "print('ALIAS_OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ALIAS_OK", proc.stdout)


class RootShimTests(unittest.TestCase):
    def test_root_shims_import_and_forward_to_rhen(self):
        for name in ROOT_SHIMS:
            with self.subTest(name=name):
                shim = importlib.import_module(name)
                canonical = importlib.import_module(f"rhen.{name}")
                self.assertIs(shim, canonical)


if __name__ == "__main__":
    unittest.main()
