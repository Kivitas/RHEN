"""CLINC150 intent classification (clinc_oos, config ``plus``).

The utterance is the state; the answer is one of the 150 in-scope intents. HF label 150 is the
out-of-scope class and becomes ``NULL_LABEL`` on the same 150-option choice, so the model learns
to answer "none of these" instead of guessing an intent.

VERIFY: ``INTENT_NAMES`` was written from memory. Before training, check it against
``datasets.load_dataset("clinc_oos", "plus")["train"].features["intent"].names`` (the first 150
entries, in label order). The label id mapping is the dataset's, so a wrong name is a wrong option
text, and this must be fixed before the data is used.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import NULL_LABEL, QuestionSpec, Rhen2Example

OOS_LABEL = 150

UNVERIFIED_FALLBACK = [
    "accept_reservations", "account_blocked", "alarm", "application_status", "apr", "are_you_a_bot",
    "balance", "bill_balance", "bill_due", "book_flight", "book_hotel", "calculator", "calendar",
    "calendar_update", "calories", "cancel", "cancel_reservation", "car_rental", "card_declined",
    "carry_on", "change_accent", "change_ai_name", "change_language", "change_speed",
    "change_user_name", "change_volume", "confirm_reservation", "cook_time", "credit_limit",
    "credit_limit_change", "credit_score", "current_location", "damaged_card", "date", "definition",
    "direct_deposit", "directions", "distance", "do_you_have_pets", "exchange_rate",
    "expiration_date", "find_phone", "flight_status", "flip_coin", "food_last", "freeze_account",
    "gas", "gas_type", "goodbye", "greeting", "how_busy", "how_old_are_you", "improve_credit_score",
    "income", "ingredient_substitution", "ingredients_list", "insurance", "insurance_change",
    "interest_rate", "international_fees", "international_visa", "jump_start", "last_maintenance",
    "lists_create", "lists_remove", "lists_show", "lost_luggage", "make_call",
    "meal_suggestion", "meaning_of_life", "measurement_conversion", "meeting_schedule",
    "min_payment", "mpg", "new_card", "next_holiday", "next_song", "nutrition_info",
    "oil_change_how", "oil_change_when", "order", "order_checks", "order_status", "pay_bill",
    "payday", "pin_change", "play_music", "plug_type", "pto_balance", "pto_request",
    "pto_request_status", "pto_used", "recipe", "redeem_rewards", "reminder", "reminder_update",
    "repeat", "replacement_card_duration", "report_fraud", "report_lost_card", "reset_settings",
    "restaurant_reservation", "restaurant_reviews", "restaurant_suggestion", "rewards_balance",
    "roll_dice", "rollover_401k", "routing", "schedule_maintenance", "schedule_meeting",
    "share_location", "shopping_list", "shopping_list_update", "smart_home", "spelling",
    "spending_history", "sync_device", "taxes", "tell_joke", "time", "timer", "timezone",
    "tire_change", "tire_pressure", "todo_list", "todo_list_update", "traffic", "transactions",
    "transfer", "translate", "travel_alert", "travel_notification", "travel_suggestion", "uber",
    "update_playlist", "user_name", "vaccines", "w2", "weather", "what_are_your_hobbies",
    "what_can_i_ask_you", "what_is_your_name", "what_song", "where_are_you_from", "whisper_mode",
    "who_am_i", "who_built_you", "who_do_you_work_for", "who_made_you", "yes",
]


INTENT_NAMES = list(UNVERIFIED_FALLBACK)   # active names; replaced by set_label_names()
_EXPECTED_NAMES = 150
_STATE = {"verified": False, "warned": False}


def set_label_names(names):
    """Replace the label names with the dataset's own names, in label-id order.

    Raises ValueError unless the count matches. Load them from the dataset, e.g.
    ``set_label_names(load_dataset(...)["train"].features["intent"].names)``, or use apply_hf_features.
    """
    global INTENT_NAMES
    names = [str(name) for name in names]
    if len(names) == 151 and names[-1] == "oos":   # HF plus config keeps oos as label 150
        names = names[:150]
    if len(names) != _EXPECTED_NAMES:
        raise ValueError(f"clinc150: expected {_EXPECTED_NAMES} label names, got {len(names)}")
    INTENT_NAMES = names
    _STATE["verified"] = True


def apply_hf_features(features):
    """Take the names from a Hugging Face ``features`` mapping, e.g. ``ds["train"].features``."""
    set_label_names(features["intent"].names)


def _active_names():
    if not _STATE["verified"] and not _STATE["warned"]:
        _STATE["warned"] = True
        warnings.warn(
            "clinc150: label names are UNVERIFIED (written from memory); call apply_hf_features(features) "
            "or set_label_names(names) with the dataset's own names before training",
            UserWarning,
            stacklevel=3,
        )
    return INTENT_NAMES



# schema: text (str), intent (int 0-149 in scope, 150 = out of scope)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    names = _active_names()
    text, label = row.get("text"), row.get("intent")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label <= OOS_LABEL:
        return None
    if label == OOS_LABEL:
        answer = NULL_LABEL
    else:
        answer = label
    spec = QuestionSpec("choice", "Which intent does the user request express?", list(names), label=answer)
    return Rhen2Example(text.strip(), [spec])


register(DatasetConverter(
    name="clinc150",
    hf_path="clinc_oos",
    hf_config="plus",
    licence="cc-by-3.0 (VERIFY on the dataset card)",
    question_types=("choice",),
    convert=convert,
    splits={
        "train": "train",
        "val": "validation",
        "test": "test",
        "oos_train": "oos_train",
        "oos_val": "oos_validation",
        "oos_test": "oos_test",
    },
    benchmark_test_splits=("validation", "test", "oos_validation", "oos_test"),
    notes="label names UNVERIFIED until set_label_names is called. 150-way in-scope choice; out-of-scope (label 150) is NULL_LABEL. Intent names need verification.",
))
