"""ANLI (facebook/anli) as a yes/no entailment proposition, one converter for all rounds."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: premise, hypothesis, label (0 entailment, 1 neutral, 2 contradiction; -1 = no gold label), reason, uid
# Only entailment is True; neutral and contradiction are both False.


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
    name="anli",
    hf_path="facebook/anli",
    hf_config=None,
    licence="cc-by-nc-4.0 (VERIFY on the dataset card)",
    question_types=("noul",),
    convert=convert,
    # ANLI ships rounds as separate splits; the round-1 splits are used here.
    splits={"train": "train_r1", "val": "dev_r1", "test": "test_r1"},
    benchmark_test_splits=("dev_r1", "dev_r2", "dev_r3", "test_r1", "test_r2", "test_r3"),
    notes="Rounds r2/r3 are reached by building a second DatasetConverter with other split names. "
          "Label -1 rows are skipped if present.",
))
