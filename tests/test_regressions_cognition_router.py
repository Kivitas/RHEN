"""
Regression tests for HierarchicalChoiceRouter bucket assignment (rhen/rhen_cognition.py).

The original router split options into contiguous, positional buckets and labelled each
bucket with its first three members only. A target outside those three was scored only
through a label that did not mention it, so it could be lost. The option set below puts
the target at index 9, which is position 9 of the first positional bucket of 77 options,
and spreads its related options across later buckets, so a positional partition fails.
"""

import math

from rhen.rhen_cognition import HierarchicalChoiceRouter

QUERY = "my card top up was declined"
QUERY_TOKENS = set(QUERY.split())
TARGET = "card top up declined"
RELATED = [
    "card top up pending",
    "card top up failed",
    "card top up reversed",
    "card top up limit",
    "card top up fee",
]
RELATED_INDICES = [20, 31, 42, 53, 64]


def build_options():
    options = [
        "bank transfer daily limit",
        "standing order cancel",
        "refund status check",
        "open savings account",
        "update postal address",
        "close credit account",
        "loan repayment holiday",
        "change overdraft limit",
        "international wire fee",
        TARGET,  # index 9
    ]
    related_slots = dict(zip(RELATED_INDICES, RELATED))
    filler = (f"service query topic {i:02d}" for i in range(100))
    for idx in range(10, 77):
        options.append(related_slots[idx] if idx in related_slots else next(filler))
    assert len(options) == 77
    assert options[9] == TARGET
    return options


def relevance(labels):
    """Count of query words present in each label: a stand-in for a model's relevance score."""
    return [
        float(len(set(label.lower().replace("|", " ").split()) & QUERY_TOKENS))
        for label in labels
    ]


def test_target_survives_positional_bucket_position_nine():
    options = build_options()
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    result = router.route(options, score_fn=relevance)
    assert result["decomposed"] is True
    assert result["chosen_option"] == TARGET
    assert result["chosen_index"] == 9


def test_related_options_share_a_group():
    options = build_options()
    clusters = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20).build_clusters(options)
    group = next(c for c in clusters if 9 in c)
    for idx in RELATED_INDICES:
        assert idx in group, f"related option {options[idx]!r} (index {idx}) not grouped with target"


def test_group_label_is_a_single_medoid_option():
    options = build_options()
    seen = []

    def scorer(labels):
        seen.append(list(labels))
        return relevance(labels)

    HierarchicalChoiceRouter(num_groups=8, flat_threshold=20).route(options, score_fn=scorer)
    group_labels = seen[0]
    assert len(group_labels) == 8
    for label in group_labels:
        assert label in options, f"group label {label!r} is not one option"
        assert "|" not in label


def test_partition_is_complete_balanced_and_deterministic():
    options = build_options()
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    clusters = router.build_clusters(options)
    assert len(clusters) == 8
    assert all(clusters), "no cluster may be empty"
    assert sorted(i for c in clusters for i in c) == list(range(77))
    assert max(len(c) for c in clusters) <= math.ceil(77 / 8)
    assert clusters == router.build_clusters(options)


def test_router_still_scores_fewer_than_flat_set():
    options = build_options()
    result = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20).route(
        options, score_fn=relevance
    )
    assert result["scored_options"] < result["total_options"] == 77
