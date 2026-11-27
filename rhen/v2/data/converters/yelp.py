"""Yelp review stars (yelp_review_full): a five-level score over the review text."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: text, label (0..4 meaning 1..5 stars)

LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    text, label = row.get("text"), row.get("label")
    if not isinstance(text, str) or not text.strip():
        return None
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(LEVELS):
        return None
    spec = QuestionSpec("score", "How many stars did the reviewer give?", options=list(LEVELS), label=label)
    return Rhen2Example(text, [spec])


register(DatasetConverter(
    name="yelp",
    hf_path="yelp_review_full",
    hf_config=None,
    licence="yelp-dataset-terms-of-use (VERIFY on the dataset card)",
    question_types=("score",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("test",),
))
