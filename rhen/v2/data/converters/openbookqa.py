"""OpenBookQA (allenai/openbookqa, config 'main'): question stem as state, four choices."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: id, question_stem, choices {text: [...], label: ['A'..'D']}, answerKey
# The open-book fact is not in the row, so only the stem is used.


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    stem, choices, key = row.get("question_stem"), row.get("choices"), row.get("answerKey")
    if not isinstance(stem, str) or not stem.strip():
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
    return Rhen2Example(stem.strip(), [spec])


register(DatasetConverter(
    name="openbookqa",
    hf_path="allenai/openbookqa",
    hf_config="main",
    licence="apache-2.0 (VERIFY on the dataset card)",
    question_types=("choice",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="Uses the question stem only; the core fact is not part of the row.",
))
