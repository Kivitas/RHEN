"""QNLI through GLUE (nyu-mll/glue, config ``qnli``).

The candidate sentence is the state. The question becomes a noul proposition: does the sentence
answer it. GLUE label 0 is entailment (the sentence answers the question) and is true; label 1
(not_entailment) is false; -1 (unlabelled test rows) is skipped.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example


# schema: question (str), sentence (str), label (int: 0 entailment, 1 not_entailment, -1 test), idx (int)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    question, sentence, label = row.get("question"), row.get("sentence"), row.get("label")
    if not isinstance(sentence, str) or not sentence.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
        return None
    spec = QuestionSpec(
        "noul",
        f"Does the sentence answer the question: {question.strip().rstrip('?')}?",
        label=(label == 0),
    )
    return Rhen2Example(sentence.strip(), [spec])


register(DatasetConverter(
    name="qnli",
    hf_path="nyu-mll/glue",
    hf_config="qnli",
    licence="cc-by-sa-4.0 (VERIFY on the dataset card; derived from SQuAD, itself cc-by-sa-4.0)",
    question_types=("noul",),
    convert=convert,
    splits={"train": "train", "val": "validation"},
    benchmark_test_splits=("validation", "test"),
    notes="Sentence is the state; question becomes a yes/no noul. Test labels are -1 and are skipped.",
))
