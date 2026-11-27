"""MultiNLI through GLUE (nyu-mll/glue, config ``mnli``).

The premise is the state; the hypothesis becomes a noul proposition: "does the premise entail
this statement". Label 0 (entailment) is true. Labels 1 (neutral) and 2 (contradiction) are
both false, so the noul head answers "is it entailed?" and not "which of three relations?".
Label -1 (unlabelled test rows) is skipped.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

ENTAILMENT = 0
NEUTRAL = 1          # mapped to false, see module docstring
CONTRADICTION = 2    # mapped to false, see module docstring


# schema: premise (str), hypothesis (str), label (int: 0 entailment, 1 neutral, 2 contradiction, -1 test), idx (int)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    premise, hypothesis, label = row.get("premise"), row.get("hypothesis"), row.get("label")
    if not isinstance(premise, str) or not premise.strip():
        return None
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None
    if not isinstance(label, int) or isinstance(label, bool) or label not in (ENTAILMENT, NEUTRAL, CONTRADICTION):
        return None
    spec = QuestionSpec(
        "noul",
        f"Does the premise entail the following statement: {hypothesis.strip()}",
        label=(label == ENTAILMENT),
    )
    return Rhen2Example(premise.strip(), [spec])


register(DatasetConverter(
    name="mnli",
    hf_path="nyu-mll/glue",
    hf_config="mnli",
    licence="mixed (GLUE/MNLI: source-dependent licences, VERIFY on the dataset card)",
    question_types=("noul",),
    convert=convert,
    splits={
        "train": "train",
        "val": "validation_matched",
        "val_mismatched": "validation_mismatched",
        "test": "test_matched",
        "test_mismatched": "test_mismatched",
    },
    benchmark_test_splits=("validation_matched", "validation_mismatched", "test_matched", "test_mismatched"),
    notes="Neutral and contradiction are both labelled false (entailment-only noul). Test labels are -1 and are skipped.",
))
