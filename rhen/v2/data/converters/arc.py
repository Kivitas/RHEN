"""AI2 ARC (allenai/ai2_arc): ARC-Challenge and ARC-Easy as two registered converters."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: id, question, choices {text: [...], label: ['A'..'D'] or ['1'..'4']}, answerKey
# answerKey is matched against choices.label as strings, so digit and letter labels both work.


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    question, choices, key = row.get("question"), row.get("choices"), row.get("answerKey")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(choices, dict):
        return None
    texts, labels = choices.get("text"), choices.get("label")
    if not isinstance(texts, list) or not isinstance(labels, list) or len(texts) != len(labels) or not texts:
        return None
    labels = [str(x).strip() for x in labels]
    if not isinstance(key, str) or key.strip() not in labels:
        return None
    options = [str(t).strip() for t in texts]
    if not all(options):
        return None
    spec = QuestionSpec(
        "choice", "Which choice answers the question in the state?",
        options=options, label=labels.index(key.strip()),
    )
    return Rhen2Example(question.strip(), [spec])


_SPLITS = {"train": "train", "val": "validation", "test": "test"}

register(DatasetConverter(
    name="arc_challenge",
    hf_path="allenai/ai2_arc",
    hf_config="ARC-Challenge",
    licence="cc-by-sa-4.0 (VERIFY on the dataset card)",
    question_types=("choice",),
    convert=convert,
    splits=dict(_SPLITS),
    benchmark_test_splits=("validation", "test"),
))

register(DatasetConverter(
    name="arc_easy",
    hf_path="allenai/ai2_arc",
    hf_config="ARC-Easy",
    licence="cc-by-sa-4.0 (VERIFY on the dataset card)",
    question_types=("choice",),
    convert=convert,
    splits=dict(_SPLITS),
    benchmark_test_splits=("validation", "test"),
))
