"""A synthetic decision benchmark with known ground truth.

Each task family is a small, fully specified rule over a structured state, so decision
accuracy, calibration and robustness can be measured exactly before any real data exists.
The splits are:

- ``train``, ``val``, ``test``: in-distribution (ID) states and question phrasings.
- ``test_ood``: the same rules under distribution shift: magnitudes outside the training
  range, categories never seen in training and question phrasings never seen in training.
  Accuracy should degrade gracefully here, and a calibrated model should report lower
  confidence (a higher free energy) than on ``test``.

Option order is shuffled per example (the label follows the shuffle), so a model cannot
score well by preferring a position. ``rhen.research.evaluate`` additionally measures how
much predictions change when the same options are presented in another order.
"""

from __future__ import annotations

import json
import os
import random
from typing import Callable, Dict, List, Sequence

from rhen.research.data import DecisionExample, DecisionQuestion, write_decision_jsonl

SPLITS = ("train", "val", "test", "test_ood")

_ID_REGIONS = ["north", "south", "east", "west"]
_OOD_REGIONS = ["orbital", "arctic", "offshore"]

_PHRASINGS = {
    "lookup": (["Which city is in the record?", "City on file"],
               ["Name the location stored for this person"]),
    "routing": (["Which queue should handle this transaction?", "Route this transaction to"],
                ["Assign the handling queue for this case"]),
    "approval": (["Approve the refund?", "Should the refund be approved?"],
                 ["Is this refund within policy and safe to pay out?"]),
    "risk": (["Rate the risk level of this account", "Risk level"],
             ["How exposed is this account on a five-step scale?"]),
    "compare": (["Which reading is larger?", "Pick the larger value"],
                ["Select the sensor with the higher measurement"]),
    "checks": (["Which readings exceed their limits?", "Flag every reading over its limit"],
               ["List the out-of-tolerance measurements"]),
}


def _phrase(rng: random.Random, family: str, ood: bool) -> str:
    seen, unseen = _PHRASINGS[family]
    return rng.choice(unseen if ood else seen)


def _shuffled(rng: random.Random, options: Sequence[str]) -> List[str]:
    out = list(options)
    rng.shuffle(out)
    return out


def _routing(rng: random.Random, ood: bool) -> DecisionExample:
    amount = rng.randint(20_000, 400_000) if ood and rng.random() < 0.5 else rng.randint(100, 19_999)
    flagged = rng.random() < 0.25
    region = rng.choice(_OOD_REGIONS if ood else _ID_REGIONS)
    canonical = ["Auto-approve", "Senior approver", "Manual review"]
    if flagged:
        answer = "Manual review"
    elif amount >= 10_000:
        answer = "Senior approver"
    else:
        answer = "Auto-approve"
    options = _shuffled(rng, canonical)
    state = {"amount": amount, "flagged": flagged, "region": region}
    q = DecisionQuestion(_phrase(rng, "routing", ood), options, "choice", options.index(answer))
    return DecisionExample(state, [q], task="routing")


def _approval(rng: random.Random, ood: bool) -> DecisionExample:
    limit = rng.choice([500, 1_000, 2_500])
    amount = rng.randint(3_000, 60_000) if ood and rng.random() < 0.5 else rng.randint(10, 2 * limit)
    flagged = rng.random() < 0.2
    state = {"refund_amount": amount, "policy_limit": limit, "fraud_flag": flagged,
             "region": rng.choice(_OOD_REGIONS if ood else _ID_REGIONS)}
    q = DecisionQuestion(_phrase(rng, "approval", ood), ["Yes", "No"], "noul", bool(amount <= limit and not flagged))
    return DecisionExample(state, [q], task="approval")


def _risk(rng: random.Random, ood: bool) -> DecisionExample:
    # ID exposures span the scale; OOD exposures are far above it (the true level is 4).
    exposure = rng.randint(20_000, 250_000) if ood and rng.random() < 0.5 else rng.randint(0, 12_499)
    level = min(4, exposure // 2_500)
    levels = ["very low", "low", "medium", "high", "very high"]
    state = {"exposure": exposure, "currency": "EUR" if ood else "USD"}
    q = DecisionQuestion(_phrase(rng, "risk", ood), levels, "score", float(level))
    return DecisionExample(state, [q], task="risk")


def _compare(rng: random.Random, ood: bool) -> DecisionExample:
    scale = 1_000_000 if ood else 1_000
    a, b = rng.randint(0, scale), rng.randint(0, scale)
    while a == b:
        b = rng.randint(0, scale)
    names = ["sensor_a", "sensor_b"]
    options = _shuffled(rng, names)
    answer = "sensor_a" if a > b else "sensor_b"
    q = DecisionQuestion(_phrase(rng, "compare", ood), options, "choice", options.index(answer))
    return DecisionExample({"sensor_a": a, "sensor_b": b}, [q], task="compare")


def _checks(rng: random.Random, ood: bool) -> DecisionExample:
    limits = {"temperature": 80, "pressure": 120, "vibration": 5}
    spread = 4.0 if ood else 1.5
    readings = {k: round(rng.uniform(0, v * spread), 1) for k, v in limits.items()}
    options = _shuffled(rng, list(limits))
    label = [i for i, name in enumerate(options) if readings[name] > limits[name]]
    state = {"readings": readings, "limits": limits}
    q = DecisionQuestion(_phrase(rng, "checks", ood), options, "multichoice", label)
    return DecisionExample(state, [q], task="checks")


_ID_CITIES = ["Lisbon", "Oslo", "Lagos", "Lima", "Hanoi", "Perth", "Quito", "Riga", "Doha", "Kyoto"]
_OOD_CITIES = ["Ushuaia", "Nuuk", "Thimphu", "Funafuti", "Longyearbyen"]
_NAMES = ["Ada", "Bo", "Chen", "Dee", "Eli", "Fatima", "Goro", "Hana"]


def _lookup(rng: random.Random, ood: bool) -> DecisionExample:
    """Copy a value from the state: no arithmetic, only grounding options in the state.

    This is the floor of the benchmark. A model that cannot learn it has a plumbing or
    optimisation problem rather than a capacity problem.
    """
    pool = _OOD_CITIES if ood else _ID_CITIES
    city = rng.choice(pool)
    distractors = rng.sample([c for c in pool if c != city], 2)
    options = _shuffled(rng, [city] + distractors)
    state = {"name": rng.choice(_NAMES), "city": city, "account": rng.randint(1000, 9999)}
    q = DecisionQuestion(_phrase(rng, "lookup", ood), options, "choice", options.index(city))
    return DecisionExample(state, [q], task="lookup")


FAMILIES: Dict[str, Callable[[random.Random, bool], DecisionExample]] = {
    "lookup": _lookup,
    "routing": _routing,
    "approval": _approval,
    "risk": _risk,
    "compare": _compare,
    "checks": _checks,
}

RULES = {
    "lookup": "the option equal to the state's city",
    "routing": "flagged -> Manual review; else amount >= 10000 -> Senior approver; else Auto-approve",
    "approval": "Yes iff refund_amount <= policy_limit and not fraud_flag",
    "risk": "level = min(4, exposure // 2500) on a 5-level scale",
    "compare": "the sensor with the larger reading",
    "checks": "every reading strictly above its limit",
}


def generate_split(split: str, n: int, seed: int, families: Sequence[str] = tuple(FAMILIES)) -> List[DecisionExample]:
    """``n`` examples of ``split``, cycling through ``families``; fixed by (seed, split)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown task families: {sorted(unknown)}")
    rng = random.Random(f"eos-decision-bench:{seed}:{split}")
    ood = split == "test_ood"
    examples = []
    for i in range(n):
        ex = FAMILIES[families[i % len(families)]](rng, ood)
        ex.split = split
        examples.append(ex)
    return examples


def write_benchmark(
    out_dir: str,
    n_train: int = 20_000,
    n_val: int = 1_000,
    n_test: int = 2_000,
    n_ood: int = 2_000,
    seed: int = 0,
    families: Sequence[str] = tuple(FAMILIES),
) -> Dict[str, str]:
    """Write ``<split>.jsonl`` for every split plus a ``benchmark.json`` description."""
    sizes = {"train": n_train, "val": n_val, "test": n_test, "test_ood": n_ood}
    paths = {}
    for split, n in sizes.items():
        path = os.path.join(out_dir, f"{split}.jsonl")
        write_decision_jsonl(generate_split(split, n, seed, families), path)
        paths[split] = path
    with open(os.path.join(out_dir, "benchmark.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "sizes": sizes, "families": list(families), "rules": RULES}, fh, indent=2)
    return paths


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write the synthetic EOS decision benchmark.")
    parser.add_argument("out_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=20_000)
    parser.add_argument("--val", type=int, default=1_000)
    parser.add_argument("--test", type=int, default=2_000)
    parser.add_argument("--ood", type=int, default=2_000)
    parser.add_argument("--families", default=",".join(FAMILIES))
    args = parser.parse_args(argv)
    paths = write_benchmark(
        args.out_dir, args.train, args.val, args.test, args.ood, args.seed, args.families.split(",")
    )
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
