"""Amazon product reviews as a five-level star score.

The HF dataset id and column names here are a best guess (see notes); the converter accepts
either a 0..4 ``label`` or a 1..5 ``stars`` column, and either ``review_body`` or ``text``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: review_body | text (the review), label (0..4 = 1..5 stars) or stars (1..5)

LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def _stars_index(row: Dict[str, Any]) -> Optional[int]:
    label = row.get("label")
    if isinstance(label, bool):
        return None
    if isinstance(label, int) and 0 <= label < len(LEVELS):
        return label
    stars = row.get("stars")
    if isinstance(stars, bool):
        return None
    if isinstance(stars, int) and 1 <= stars <= len(LEVELS):
        return stars - 1
    return None


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    text = row.get("review_body")
    if not isinstance(text, str) or not text.strip():
        text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    label = _stars_index(row)
    if label is None:
        return None
    spec = QuestionSpec("score", "How many stars did the customer give this product?",
                        options=list(LEVELS), label=label)
    return Rhen2Example(text, [spec])


register(DatasetConverter(
    name="amazon_reviews",
    hf_path="mteb/amazon_reviews_multi",     # VERIFY the HF id; the rating column name is unconfirmed
    hf_config="en",
    licence="amazon-reviews-multi terms (VERIFY on the dataset card)",
    question_types=("score",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="Column names unverified: accepts label (0..4) or stars (1..5), review_body or text.",
))
