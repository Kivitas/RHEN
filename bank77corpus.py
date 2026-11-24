#!/usr/bin/env python3
"""Root-level entrypoint and re-export for the Banking77 RHEN corpus builder.

Delegates to ``benchmarks/banking77/bank77corpus.py`` so callers can run::

    python bank77corpus.py --out-dir data/banking77 --mode mixed
    python bank77corpus.py --synthetic --out-dir data/banking77_smoke
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent / "benchmarks" / "banking77" / "bank77corpus.py"


def _load_impl():
    module_name = "_rhen_benchmarks_banking77_bank77corpus"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _IMPL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Banking77 corpus implementation from {_IMPL_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load_impl()

DEFAULT_CACHE_DIR = _impl.DEFAULT_CACHE_DIR
DEFAULT_OUT_DIR = _impl.DEFAULT_OUT_DIR
HF_DATASET = _impl.HF_DATASET
INTENT_LABELS = _impl.INTENT_LABELS
MODES = _impl.MODES
OOD_QUERIES = _impl.OOD_QUERIES
RHEN_QUESTION = _impl.RHEN_QUESTION
HardNegativeMiner = _impl.HardNegativeMiner
TfidfVectorizer = _impl.TfidfVectorizer
build_corpus = _impl.build_corpus
build_parser = _impl.build_parser
generate_synthetic_banking77 = _impl.generate_synthetic_banking77
load_banking77 = _impl.load_banking77
main = _impl.main
stratified_train_val_split = _impl.stratified_train_val_split

__all__ = list(_impl.__all__)


if __name__ == "__main__":
    sys.exit(main())
