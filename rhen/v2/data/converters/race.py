"""RACE (ehovy/race, config 'all'): reading comprehension over an article, four options."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: article (state), problem (the question), options (list of 4 strings), answer ('A'..'D'), example_id

LETTERS = "ABCD"


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    article, question = row.get("article"), row.get("problem")
    options, answer = row.get("options"), row.get("answer")
    if not isinstance(article, str) or not article.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(options, (list, tuple)) or len(options) != 4:
        return None
    if not all(isinstance(o, str) and o.strip() for o in options):
        return None
    if not isinstance(answer, str) or answer.strip().upper() not in LETTERS:
        return None
    label = LETTERS.index(answer.strip().upper())
    spec = QuestionSpec(
        "choice", question.strip(), options=[o.strip() for o in options], label=label,
    )
    return Rhen2Example(article.strip(), [spec])


register(DatasetConverter(
    name="race",
    hf_path="ehovy/race",
    hf_config="all",
    licence="RACE research-only terms (VERIFY on the dataset card)",
    question_types=("choice",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="Long articles may exceed the state limit; the packer raises on those, so filter upstream if needed.",
))
