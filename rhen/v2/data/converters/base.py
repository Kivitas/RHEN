"""Converter registry: one small function per public dataset, rows in, ``Rhen2Example`` out.

A converter never touches the network. It receives one row dict exactly as the Hugging Face
``datasets`` library yields it and returns a ``Rhen2Example`` (or ``None`` to skip the row,
for example when the label is missing). Loading rows is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from rhen.v2.packing import Rhen2Example


@dataclass
class DatasetConverter:
    name: str                                   # registry key, lower snake case, e.g. "boolq"
    hf_path: str                                # e.g. "google/boolq"
    hf_config: Optional[str]                    # e.g. "default" or None
    licence: str                                # licence string as published by the dataset
    question_types: Tuple[str, ...]             # subset of choice/noul/score/multi/range/extract
    convert: Callable[[Dict[str, Any]], Optional[Rhen2Example]]
    splits: Dict[str, str] = field(default_factory=lambda: {"train": "train", "val": "validation", "test": "test"})
    benchmark_test_splits: Tuple[str, ...] = ("test",)   # HF split names that must never be trained on
    notes: str = ""


REGISTRY: Dict[str, DatasetConverter] = {}


def register(converter: DatasetConverter) -> DatasetConverter:
    if converter.name in REGISTRY:
        raise ValueError(f"converter {converter.name!r} is already registered")
    if not converter.licence:
        raise ValueError(f"converter {converter.name!r} must record a licence string")
    REGISTRY[converter.name] = converter
    return converter


def convert_rows(name: str, rows: Iterable[Dict[str, Any]]) -> List[Rhen2Example]:
    converter = REGISTRY[name]
    out: List[Rhen2Example] = []
    for row in rows:
        ex = converter.convert(row)
        if ex is not None:
            out.append(ex)
    return out
