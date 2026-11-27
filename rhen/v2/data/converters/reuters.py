"""Reuters-21578 style topics: multi-label question over the topics listed in the row.

Design caveat: the options are the row's own topics, so every option is a label and the
gold selection is all of them. This follows the brief literally; a fixed topic vocabulary
with distractors would give a real signal and is the better design once the vocabulary is known.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# schema: text (or body / title as fallback), topics (list of str). Column names unverified.


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    text = None
    for key in ("text", "body", "title"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            text = value
            break
    if text is None:
        return None
    topics = row.get("topics")
    if not isinstance(topics, (list, tuple)):
        return None
    options = []
    for t in topics:
        if not isinstance(t, str) or not t.strip():
            return None
        if t.strip() not in options:
            options.append(t.strip())
    if not options:
        return None        # no topics: nothing to ask
    spec = QuestionSpec(
        "multi", "Which topics does the article belong to? Select all that apply.",
        options=options, label=list(range(len(options))),
    )
    return Rhen2Example(text, [spec])


register(DatasetConverter(
    name="reuters",
    hf_path="yangwang825/reuters-21578",     # VERIFY the HF id and the topics column
    hf_config=None,
    licence="reuters-21578 terms (VERIFY on the dataset card)",
    question_types=("multi",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="Options are the row's own topics (see module docstring). Rows without topics are skipped.",
))
