"""GoEmotions (google-research-datasets/go_emotions, config 'simplified'): multi-label emotions."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: text, labels (list of int ids 0..27), id

EMOTIONS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    text, labels = row.get("text"), row.get("labels")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(labels, (list, tuple)):
        return None
    ids = set()
    for x in labels:
        if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < len(EMOTIONS):
            return None
        ids.add(x)
    # an empty list is a valid answer: no emotion selected
    spec = QuestionSpec(
        "multi", "Which emotions does the text express? Select all that apply, or none.",
        options=list(EMOTIONS), label=sorted(ids),
    )
    return Rhen2Example(text, [spec])


register(DatasetConverter(
    name="goemotions",
    hf_path="google-research-datasets/go_emotions",
    hf_config="simplified",
    licence="apache-2.0 (VERIFY on the dataset card)",
    question_types=("multi",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
))
