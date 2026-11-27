"""DBpedia-14 ontology classification (fancyzhx/dbpedia_14, config ``dbpedia_14``).

The title and content of the Wikipedia article are the state; the answer is one of 14 ontology
classes in HF ClassLabel order.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

UNVERIFIED_FALLBACK = [
    "Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation",
    "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
]


OPTIONS = list(UNVERIFIED_FALLBACK)   # active names; replaced by set_label_names()
_EXPECTED_NAMES = 14
_STATE = {"verified": False, "warned": False}


def set_label_names(names):
    """Replace the label names with the dataset's own names, in label-id order.

    Raises ValueError unless the count matches. Load them from the dataset, e.g.
    ``set_label_names(load_dataset(...)["train"].features["label"].names)``, or use apply_hf_features.
    """
    global OPTIONS
    names = [str(name) for name in names]
    if len(names) != _EXPECTED_NAMES:
        raise ValueError(f"dbpedia: expected {_EXPECTED_NAMES} label names, got {len(names)}")
    OPTIONS = names
    _STATE["verified"] = True


def apply_hf_features(features):
    """Take the names from a Hugging Face ``features`` mapping, e.g. ``ds["train"].features``."""
    set_label_names(features["label"].names)


def _active_names():
    if not _STATE["verified"] and not _STATE["warned"]:
        _STATE["warned"] = True
        warnings.warn(
            "dbpedia: label names are UNVERIFIED (written from memory); call apply_hf_features(features) "
            "or set_label_names(names) with the dataset's own names before training",
            UserWarning,
            stacklevel=3,
        )
    return OPTIONS



# schema: label (int 0-13), title (str), content (str)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    names = _active_names()
    label = row.get("label")
    if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(names):
        return None
    title = row.get("title") if isinstance(row.get("title"), str) else ""
    content = row.get("content") if isinstance(row.get("content"), str) else ""
    state = " ".join(part.strip() for part in (title, content) if part and part.strip())
    if not state:
        return None
    spec = QuestionSpec("choice", "Which ontology class does this Wikipedia article belong to?", names, label=label)
    return Rhen2Example(state, [spec])


register(DatasetConverter(
    name="dbpedia",
    hf_path="fancyzhx/dbpedia_14",
    hf_config="dbpedia_14",
    licence="cc-by-sa-3.0 (VERIFY on the dataset card; Wikipedia-derived)",
    question_types=("choice",),
    convert=convert,
    splits={"train": "train", "test": "test"},
    benchmark_test_splits=("test",),
    notes="label names UNVERIFIED until set_label_names is called. Fourteen-way ontology choice. State is title plus content.",
))
