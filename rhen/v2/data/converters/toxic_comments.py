"""Jigsaw toxic comment classification: six binary columns as a multi-label question."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate (each 0/1; -1 = unlabelled)

KINDS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    text = row.get("comment_text")
    if not isinstance(text, str) or not text.strip():
        return None
    selected = []
    for i, name in enumerate(KINDS):
        value = row.get(name)
        if isinstance(value, bool):
            value = int(value)
        if value not in (0, 1):          # -1, None, missing or other values skip the row
            return None
        if value == 1:
            selected.append(i)
    spec = QuestionSpec(
        "multi", "Which kinds of toxicity does the comment contain? Select all that apply, or none.",
        options=list(KINDS), label=selected,
    )
    return Rhen2Example(text, [spec])


register(DatasetConverter(
    name="toxic_comments",
    hf_path="google/jigsaw_toxicity_pred",   # VERIFY the HF id; column names are the Kaggle ones
    hf_config=None,
    licence="jigsaw-toxic-comment competition terms (VERIFY on the dataset card)",
    question_types=("multi",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="Rows with -1 in any label column (unlabelled test rows) are skipped.",
))
