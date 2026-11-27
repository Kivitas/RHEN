"""google/boolq: yes/no questions about a Wikipedia passage.

The passage is the state. The question becomes a yes/no claim (a noul proposition).
The HF ``test`` split has no answers, so only ``train`` and ``validation`` are used.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example


def _yes_no(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


# schema: passage (str), question (str), answer (bool; the HF test split has no answers)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    passage, question = row.get("passage"), row.get("question")
    if not isinstance(passage, str) or not passage.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    answer = _yes_no(row.get("answer"))
    if answer is None:
        return None
    claim = question.strip().rstrip("?").strip()
    spec = QuestionSpec("noul", f"Is this claim true according to the passage: {claim}?", label=answer)
    return Rhen2Example(passage.strip(), [spec])


register(DatasetConverter(
    name="boolq",
    hf_path="google/boolq",
    hf_config="default",
    licence="cc-by-sa-3.0 (VERIFY on the dataset card)",
    question_types=("noul",),
    convert=convert,
    splits={"train": "train", "val": "validation"},
    benchmark_test_splits=("validation",),
    notes="Question rendered as a yes/no claim. The HF test split is unlabelled and not registered.",
))
