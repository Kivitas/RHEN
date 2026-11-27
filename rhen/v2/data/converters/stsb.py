"""STS-B (glue/stsb): two sentences as state, similarity 0..5 binned to six ordered levels."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: sentence1, sentence2, label (float 0..5; -1 or None on unlabelled rows), idx

LEVELS = ["0", "1", "2", "3", "4", "5"]


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    s1, s2, label = row.get("sentence1"), row.get("sentence2"), row.get("label")
    if not isinstance(s1, str) or not s1.strip() or not isinstance(s2, str) or not s2.strip():
        return None
    if label is None or isinstance(label, bool):
        return None
    try:
        value = float(label)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0.0 or value > 5.0:     # -1 marks an unlabelled row
        return None
    level = min(5, int(math.floor(value + 0.5)))                   # round half up
    state = {"sentence1": s1, "sentence2": s2}
    spec = QuestionSpec(
        "score",
        "How similar in meaning are the two sentences in the state? 0 is unrelated, 5 is identical in meaning.",
        options=list(LEVELS), label=level,
    )
    return Rhen2Example(state, [spec])


register(DatasetConverter(
    name="stsb",
    hf_path="glue",
    hf_config="stsb",
    licence="sts-benchmark terms (VERIFY on the dataset card)",
    question_types=("score",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="Test rows carry label -1 and are skipped. Ratings are rounded to whole levels.",
))
