"""SNLI (stanfordnlp/snli) as a yes/no entailment proposition over the premise."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: premise, hypothesis, label (ClassLabel: 0 entailment, 1 neutral, 2 contradiction; -1 = no gold label)
# Entailment is the only True answer; neutral and contradiction both mean "not entailed".


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    premise, hypothesis, label = row.get("premise"), row.get("hypothesis"), row.get("label")
    if not isinstance(premise, str) or not premise.strip():
        return None
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None
    if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1, 2):
        return None
    spec = QuestionSpec(
        "noul",
        f"Does the premise entail this statement? Statement: {hypothesis.strip()}",
        label=(label == 0),
    )
    return Rhen2Example(premise.strip(), [spec])


register(DatasetConverter(
    name="snli",
    hf_path="stanfordnlp/snli",
    hf_config=None,
    licence="cc-by-sa-4.0 (VERIFY on the dataset card)",
    question_types=("noul",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="Binary: entailment -> true, neutral/contradiction -> false. Label -1 rows are skipped.",
))
