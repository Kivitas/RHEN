"""Record format and converters for RHEN-2 decision examples.

A record is one JSON object per line::

    {"state": ..., "questions": [{"type": "choice", "instructions": "...", "options": [...],
      "criteria": [...], "lo": null, "hi": null, "label": ..., "soft": [...]}], "meta": {...}}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Sequence

from rhen.v2.packing import QuestionSpec, Rhen2Example

_V1_TYPE = {"choice": "choice", "noul": "noul", "score": "score", "multichoice": "multi"}


def question_to_record(q: QuestionSpec) -> Dict[str, Any]:
    return {"type": q.type, "instructions": q.instructions, "options": list(q.options),
            "criteria": list(q.criteria), "lo": q.lo, "hi": q.hi, "label": q.label, "soft": q.soft}


def question_from_record(r: Dict[str, Any]) -> QuestionSpec:
    return QuestionSpec(
        type=r["type"], instructions=r["instructions"], options=list(r.get("options") or []),
        criteria=list(r.get("criteria") or []), lo=r.get("lo"), hi=r.get("hi"),
        label=r.get("label"), soft=r.get("soft"),
    )


def example_to_record(ex: Rhen2Example, meta: Dict[str, Any] = None) -> Dict[str, Any]:
    return {"state": ex.state, "questions": [question_to_record(q) for q in ex.questions], "meta": meta or {}}


def example_from_record(r: Dict[str, Any]) -> Rhen2Example:
    return Rhen2Example(r["state"], [question_from_record(q) for q in r["questions"]])


def write_jsonl(examples: Sequence[Rhen2Example], path: str, meta: Dict[str, Any] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(example_to_record(ex, meta), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[Rhen2Example]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield example_from_record(json.loads(line))


def from_v1_example(ex) -> Rhen2Example:
    """Convert a v1 ``DecisionExample`` (``rhen.research.data``) to a RHEN-2 example."""
    questions: List[QuestionSpec] = []
    for q in ex.questions:
        kind = _V1_TYPE.get(q.primitive)
        if kind is None:
            raise ValueError(f"v1 primitive {q.primitive!r} has no RHEN-2 equivalent in the converter")
        if kind == "noul":
            questions.append(QuestionSpec("noul", q.question, label=bool(q.label)))
        elif kind == "score":
            questions.append(QuestionSpec("score", q.question, list(q.options), label=int(round(q.label))))
        elif kind == "multi":
            questions.append(QuestionSpec("multi", q.question, list(q.options), label=[int(i) for i in q.label]))
        else:
            questions.append(QuestionSpec("choice", q.question, list(q.options), label=int(q.label)))
    return Rhen2Example(ex.state, questions)
