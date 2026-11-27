"""SQuAD 2.0 (rajpurkar/squad_v2): context as state, question as an extract question."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer

# schema: context, question, answers {text: [...], answer_start: [...]} (empty lists = unanswerable), id, title

INSTRUCTION = "Extract the answer to the question from the state. Answer with absent if the state does not answer it."


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    context, question, answers = row.get("context"), row.get("question"), row.get("answers")
    if not isinstance(context, str) or not context.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    # offsets are taken on the serialised state; if sanitising would change the text, skip the row
    if Rhen2Packer.serialise_state(context) != context:
        return None
    if not isinstance(answers, dict):
        return None
    texts, starts = answers.get("text"), answers.get("answer_start")
    if not isinstance(texts, list) or not isinstance(starts, list):
        return None
    if not texts:
        label = NULL_LABEL                       # unanswerable
    else:
        text, start = texts[0], starts[0] if starts else None
        if not isinstance(text, str) or not text:
            return None
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            return None
        end = start + len(text)
        if end > len(context) or context[start:end] != text:
            return None                          # misaligned offsets
        label = (start, end)
    spec = QuestionSpec("extract", f"{INSTRUCTION} Question: {question.strip()}", label=label)
    return Rhen2Example(context, [spec])


register(DatasetConverter(
    name="squad2",
    hf_path="rajpurkar/squad_v2",
    hf_config=None,
    licence="cc-by-sa-4.0 (VERIFY on the dataset card)",
    question_types=("extract",),
    convert=convert,
    splits={"train": "train", "val": "validation"},
    benchmark_test_splits=("validation",),
    notes="There is no public test split; validation is the evaluation set. Only the first answer is used.",
))
