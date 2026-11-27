"""CoNLL-2003 NER as extract questions over the re-joined sentence.

The sentence is ``' '.join(tokens)``; each entity's character span is tracked from the token
offsets. Per sentence, up to three questions are asked: present entity types first (PER, ORG,
LOC, MISC order), then absent types, which get NULL_LABEL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer

# schema: tokens (list of str), ner_tags (list of ClassLabel ints, or strings), id, pos_tags, chunk_tags
# Tag order follows the HF conll2003 ClassLabel names.

NER_NAMES = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC")
TYPES = ("PER", "ORG", "LOC", "MISC")
MAX_QUESTIONS = 3


def _tag_name(tag: Any) -> Optional[str]:
    if isinstance(tag, bool):
        return None
    if isinstance(tag, int) and 0 <= tag < len(NER_NAMES):
        return NER_NAMES[tag]
    if isinstance(tag, str) and tag in NER_NAMES:
        return tag
    return None


def _spans(names: List[str]) -> List[Tuple[str, int, int]]:
    """(type, first_token, last_token) spans; an I- tag that does not continue a span starts one."""
    out: List[Tuple[str, int, int]] = []
    cur: Optional[List[Any]] = None
    for i, name in enumerate(names):
        if name == "O":
            if cur is not None:
                out.append(tuple(cur))
            cur = None
            continue
        prefix, ty = name.split("-", 1)
        if prefix == "B" or cur is None or cur[0] != ty:
            if cur is not None:
                out.append(tuple(cur))
            cur = [ty, i, i]
        else:
            cur[2] = i
    if cur is not None:
        out.append(tuple(cur))
    return out


def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    tokens, tags = row.get("tokens"), row.get("ner_tags")
    if not isinstance(tokens, list) or not tokens or not isinstance(tags, list) or len(tags) != len(tokens):
        return None
    if not all(isinstance(t, str) and t for t in tokens):
        return None
    names = []
    for tag in tags:
        name = _tag_name(tag)
        if name is None:
            return None
        names.append(name)

    # rebuild the sentence and the character span of every token
    starts, ends, pos = [], [], 0
    for i, tok in enumerate(tokens):
        if i:
            pos += 1                             # the single joining space
        starts.append(pos)
        ends.append(pos + len(tok))
        pos += len(tok)
    text = " ".join(tokens)
    if Rhen2Packer.serialise_state(text) != text:
        return None

    first: Dict[str, Tuple[int, int]] = {}
    for ty, s, e in _spans(names):
        if ty not in first:
            first[ty] = (starts[s], ends[e])

    chosen: List[str] = [ty for ty in TYPES if ty in first][:MAX_QUESTIONS]
    for ty in TYPES:
        if len(chosen) >= MAX_QUESTIONS:
            break
        if ty not in first:
            chosen.append(ty)

    questions = [
        QuestionSpec(
            "extract", f"Extract the first {ty} entity in the state, or answer absent if there is none.",
            label=first.get(ty, NULL_LABEL),
        )
        for ty in chosen
    ]
    return Rhen2Example(text, questions)


register(DatasetConverter(
    name="conll_ner",
    hf_path="eriktks/conll2003",              # older id: conll2003; the column names are the same
    hf_config=None,
    licence="conll2003 / Reuters corpus terms (VERIFY on the dataset card)",
    question_types=("extract",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="Up to three extract questions per sentence. Only the first entity of each type is asked for.",
))
