"""AG News topic classification (fancyzhx/ag_news).

The news text is the state; the answer is one of four topics. Label ids follow the HF ClassLabel
order: 0 World, 1 Sports, 2 Business, 3 Sci/Tech.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

UNVERIFIED_FALLBACK = ["World", "Sports", "Business", "Sci/Tech"]


OPTIONS = list(UNVERIFIED_FALLBACK)   # active names; replaced by set_label_names()
_EXPECTED_NAMES = 4
_STATE = {"verified": False, "warned": False}


def set_label_names(names):
    """Replace the label names with the dataset's own names, in label-id order.

    Raises ValueError unless the count matches. Load them from the dataset, e.g.
    ``set_label_names(load_dataset(...)["train"].features["label"].names)``, or use apply_hf_features.
    """
    global OPTIONS
    names = [str(name) for name in names]
    if len(names) != _EXPECTED_NAMES:
        raise ValueError(f"agnews: expected {_EXPECTED_NAMES} label names, got {len(names)}")
    OPTIONS = names
    _STATE["verified"] = True


def apply_hf_features(features):
    """Take the names from a Hugging Face ``features`` mapping, e.g. ``ds["train"].features``."""
    set_label_names(features["label"].names)


def _active_names():
    if not _STATE["verified"] and not _STATE["warned"]:
        _STATE["warned"] = True
        warnings.warn(
            "agnews: label names are UNVERIFIED (written from memory); call apply_hf_features(features) "
            "or set_label_names(names) with the dataset's own names before training",
            UserWarning,
            stacklevel=3,
        )
    return OPTIONS



# schema: text (str), label (int 0-3)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    names = _active_names()
    text, label = row.get("text"), row.get("label")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(names):
        return None
    spec = QuestionSpec("choice", "Which topic is this news article about?", names, label=label)
    return Rhen2Example(text.strip(), [spec])


register(DatasetConverter(
    name="agnews",
    hf_path="fancyzhx/ag_news",
    hf_config="default",
    licence="unknown (VERIFY on the dataset card; AG News corpus is widely redistributed for research)",
    question_types=("choice",),
    convert=convert,
    splits={"train": "train", "test": "test"},
    benchmark_test_splits=("test",),
    notes="label names UNVERIFIED until set_label_names is called. Four-way topic choice over the article text.",
))
