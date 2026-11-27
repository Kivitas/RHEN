"""Banking77 intent classification (PolyAI/banking77).

The customer message is the state; the answer is one of 77 intents. Option names are the dataset's
own intent names in ClassLabel order (index = HF label id).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from rhen.v2.data.converters.base import DatasetConverter, register
from rhen.v2.packing import QuestionSpec, Rhen2Example

# VERIFY: written from memory of the dataset card; check against
# datasets.load_dataset("PolyAI/banking77")["train"].features["label"].names before training.
UNVERIFIED_FALLBACK = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance",
    "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
    "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


INTENT_NAMES = list(UNVERIFIED_FALLBACK)   # active names; replaced by set_label_names()
_EXPECTED_NAMES = 77
_STATE = {"verified": False, "warned": False}


def set_label_names(names):
    """Replace the label names with the dataset's own names, in label-id order.

    Raises ValueError unless the count matches. Load them from the dataset, e.g.
    ``set_label_names(load_dataset(...)["train"].features["label"].names)``, or use apply_hf_features.
    """
    global INTENT_NAMES
    names = [str(name) for name in names]
    if len(names) != _EXPECTED_NAMES:
        raise ValueError(f"banking77: expected {_EXPECTED_NAMES} label names, got {len(names)}")
    INTENT_NAMES = names
    _STATE["verified"] = True


def apply_hf_features(features):
    """Take the names from a Hugging Face ``features`` mapping, e.g. ``ds["train"].features``."""
    set_label_names(features["label"].names)


def _active_names():
    if not _STATE["verified"] and not _STATE["warned"]:
        _STATE["warned"] = True
        warnings.warn(
            "banking77: label names are UNVERIFIED (written from memory); call apply_hf_features(features) "
            "or set_label_names(names) with the dataset's own names before training",
            UserWarning,
            stacklevel=3,
        )
    return INTENT_NAMES



# schema: text (str), label (int 0-76)
def convert(row: Dict[str, Any]) -> Optional[Rhen2Example]:
    names = _active_names()
    text, label = row.get("text"), row.get("label")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(names):
        return None
    spec = QuestionSpec("choice", "Which banking intent does the customer message express?", list(names), label=label)
    return Rhen2Example(text.strip(), [spec])


register(DatasetConverter(
    name="banking77",
    hf_path="PolyAI/banking77",
    hf_config=None,
    licence="cc-by-4.0 (VERIFY on the dataset card)",
    question_types=("choice",),
    convert=convert,
    splits={"train": "train", "test": "test"},
    benchmark_test_splits=("test",),
    notes="label names UNVERIFIED until set_label_names is called. 77-way intent choice (more than 64 options; packs with K bucket 128). Intent names need verification.",
))
