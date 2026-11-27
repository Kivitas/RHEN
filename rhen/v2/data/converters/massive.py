"""MASSIVE intent classification (AmazonScience/massive, config ``en-US``).

The user utterance is the state; the answer is one of the MASSIVE intents. In the HF dataset the
``intent`` column is a ClassLabel (int); names are in ClassLabel (alphabetical) order. A string
intent is also accepted and mapped by name.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# VERIFY: written from memory; check against
# datasets.load_dataset("AmazonScience/massive", "en-US")["train"].features["intent"].names.
UNVERIFIED_FALLBACK = [
    "alarm_query", "alarm_remove", "alarm_set", "audio_volume_down", "audio_volume_mute",
    "audio_volume_other", "audio_volume_up", "calendar_query", "calendar_remove", "calendar_set",
    "cooking_query", "cooking_recipe", "datetime_convert", "datetime_query", "email_addcontact",
    "email_query", "email_querycontact", "email_sendemail", "general_greet", "general_joke",
    "general_quirky", "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff",
    "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on", "lists_createoradd",
    "lists_query", "lists_remove", "music_dislikeness", "music_likeness", "music_query",
    "music_settings", "news_query", "play_audiobook", "play_game", "play_music", "play_podcasts",
    "play_radio", "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
    "recommendation_events", "recommendation_locations", "recommendation_movies", "social_post",
    "social_query", "takeaway_order", "takeaway_query", "transport_query", "transport_taxi",
    "transport_ticket", "transport_traffic", "weather_query",
]


INTENT_NAMES = list(UNVERIFIED_FALLBACK)   # active names; replaced by set_label_names()
_EXPECTED_NAMES = 60
_STATE = {"verified": False, "warned": False}


def set_label_names(names):
    """Replace the label names with the dataset's own names, in label-id order.

    Raises ValueError unless the count matches. Load them from the dataset, e.g.
    ``set_label_names(load_dataset(...)["train"].features["intent"].names)``, or use apply_hf_features.
    """
    global INTENT_NAMES
    names = [str(name) for name in names]
    if len(names) != _EXPECTED_NAMES:
        raise ValueError(f"massive: expected {_EXPECTED_NAMES} label names, got {len(names)}")
    INTENT_NAMES = names
    _STATE["verified"] = True


def apply_hf_features(features):
    """Take the names from a Hugging Face ``features`` mapping, e.g. ``ds["train"].features``."""
    set_label_names(features["intent"].names)


def _active_names():
    if not _STATE["verified"] and not _STATE["warned"]:
        _STATE["warned"] = True
        warnings.warn(
            "massive: label names are UNVERIFIED (written from memory); call apply_hf_features(features) "
            "or set_label_names(names) with the dataset's own names before training",
            UserWarning,
            stacklevel=3,
        )
    return INTENT_NAMES



def _intent_index(value: Any, names) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value < len(names) else None
    if isinstance(value, str) and value.strip() in names:
        return names.index(value.strip())
    return None



# schema: utt (str), intent (int ClassLabel or str), id, locale, partition, scenario (str), annot_utt (str)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    names = _active_names()
    utt = row.get("utt")
    if not isinstance(utt, str) or not utt.strip():
        return None
    names = _active_names()
    label = _intent_index(row.get("intent"), names)
    if label is None:
        return None
    spec = QuestionSpec("choice", "Which intent does the user utterance express?", list(names), label=label)
    return Rhen2Example(utt.strip(), [spec])


register(DatasetConverter(
    name="massive",
    hf_path="AmazonScience/massive",
    hf_config="en-US",
    licence="cc-by-4.0 (VERIFY on the dataset card)",
    question_types=("choice",),
    convert=convert,
    splits={"train": "train", "val": "validation", "test": "test"},
    benchmark_test_splits=("validation", "test"),
    notes="label names UNVERIFIED until set_label_names is called. Choice over MASSIVE intents (60). Intent names need verification.",
))
