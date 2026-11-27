"""CommonsenseQA (tau/commonsense_qa): the question is the state, five choices."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: question, choices {label: ['A'..'E'], text: [...]}, answerKey ('' on the test split)


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
        return None        # test split has no answerKey, and -1/missing answers are skipped
    options = [str(t).strip() for t in texts]
    if not all(options):
        return None
    spec = QuestionSpec(
        "choice", "Which choice is the best answer to the question in the state?",
        options=options, label=labels.index(key.strip()),
    )
    return Rhen2Example(question.strip(), [spec])


register(DatasetConverter(
    name="commonsenseqa",
    hf_path="tau/commonsense_qa",
    hf_config=None,
    licence="mit (VERIFY on the dataset card)",
    question_types=("choice",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="The test split has no answers and is therefore always skipped for training.",
))
