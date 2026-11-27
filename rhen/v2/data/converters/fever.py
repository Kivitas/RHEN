"""FEVER fact verification (fever/fever, config ``v1.0``).

Claim and evidence become the state; the verdict is a three-way choice. ``NOT ENOUGH INFO`` is a
real option (index 2), not a skip. The HF rows of fever/fever carry evidence as wiki URLs and
sentence ids, not text, so the evidence text is used only when a row supplies it under the key
``evidence``; otherwise the state is the claim alone. A converter fed a dump with evidence text
gets the full claim-plus-evidence state.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

UNVERIFIED_FALLBACK = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


OPTIONS = list(UNVERIFIED_FALLBACK)   # active names; replaced by set_label_names()
_EXPECTED_NAMES = 3
_STATE = {"verified": False, "warned": False}


def set_label_names(names):
    """Replace the label names with the dataset's own names, in label-id order.

    Raises ValueError unless the count matches. Load them from the dataset, e.g.
    ``set_label_names(load_dataset(...)["train"].features["label"].names)``, or use apply_hf_features.
    """
    global OPTIONS
    names = [str(name) for name in names]
    if len(names) != _EXPECTED_NAMES:
        raise ValueError(f"fever: expected {_EXPECTED_NAMES} label names, got {len(names)}")
    OPTIONS = names
    _STATE["verified"] = True


def apply_hf_features(features):
    """Take the names from a Hugging Face ``features`` mapping, e.g. ``ds["train"].features``."""
    set_label_names(features["label"].names)


def _active_names():
    if not _STATE["verified"] and not _STATE["warned"]:
        _STATE["warned"] = True
        warnings.warn(
            "fever: label names are UNVERIFIED (written from memory); call apply_hf_features(features) "
            "or set_label_names(names) with the dataset's own names before training",
            UserWarning,
            stacklevel=3,
        )
    return OPTIONS



def _verdict(value: Any, options) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value < len(options) else None
    if isinstance(value, str) and value.strip().upper() in options:
        return options.index(value.strip().upper())
    return None


# schema: claim (str), label (str: SUPPORTS / REFUTES / NOT ENOUGH INFO, or int index into that order),
#         id (int), evidence_wiki_url, evidence_sentence_id (not used; evidence text is optional)
#         evidence (str, optional; not a standard fever/fever column)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    claim = row.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        return None
    names = _active_names()
    label = _verdict(row.get("label"), names)
    if label is None:
        return None
    evidence = row.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        state = f"Claim: {claim.strip()}\nEvidence: {evidence.strip()}"
    else:
        state = claim.strip()
    spec = QuestionSpec(
        "choice",
        "Given the claim and the evidence, is the claim supported, refuted, or is there not enough information?",
        list(names),
        label=label,
    )
    return Rhen2Example(state, [spec])


register(DatasetConverter(
    name="fever",
    hf_path="fever/fever",
    hf_config="v1.0",
    licence="cc-by-sa-3.0 (VERIFY on the dataset card; Wikipedia-derived)",
    question_types=("choice",),
    convert=convert,
    splits={"train": "train", "val": "validation"},
    benchmark_test_splits=("validation", "paper_dev", "paper_test"),
    notes="label names UNVERIFIED until set_label_names is called. Label names are used as strings or as ClassLabel ints in SUPPORTS/REFUTES/NOT ENOUGH INFO order. "
          "Evidence text is not in the standard HF columns; state falls back to the claim alone.",
))
