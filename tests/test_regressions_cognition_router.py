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


# ---------------------------------------------------------------------------
# Partition caching (F9a) and group-member expansion (F9b)
# ---------------------------------------------------------------------------


def test_second_route_reuses_cached_partition(monkeypatch):
    import rhen.rhen_cognition as rc

    options = build_options()
    calls = {"n": 0}
    real = rc._balanced_clusters

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(rc, "_balanced_clusters", counting)
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    first = router.route(options, score_fn=relevance)
    assert calls["n"] == 1, "first route() must cluster once"
    second = router.route(options, score_fn=relevance)
    router.build_clusters(options)
    assert calls["n"] == 1, "same options must not re-run the clustering helper"
    assert second["chosen_index"] == first["chosen_index"] == 9
    assert second["clusters"] == first["clusters"]


def test_cached_partition_is_a_copy():
    options = build_options()
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    clusters = router.build_clusters(options)
    clusters[0].clear()
    clusters.pop()
    fresh = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20).build_clusters(options)
    assert router.build_clusters(options) == fresh


def test_partition_cache_is_bounded():
    import rhen.rhen_cognition as rc

    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    for n in range(30, 30 + rc._PARTITION_CACHE_SIZE + 4):
        router.build_clusters([f"option {i}" for i in range(n)])
    assert len(router._partition_cache) == rc._PARTITION_CACHE_SIZE


def test_target_in_expanded_group_is_chosen():
    """
    Synthetic scorer: group 0 has the best medoid score (1.0) and group 1 scores 0.95,
    within the 10% margin. The old router scored only group 0, so a target in group 1
    was lost. The target's label gets the top member score.
    """
    options = build_options()
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    clusters = router.build_clusters(options)
    target = clusters[1][0]
    assert target not in clusters[0]
    calls = []

    def scorer(labels):
        calls.append(list(labels))
        if len(calls) == 1:  # level 1: one score per group, by group index
            return [1.0 if g == 0 else 0.95 if g == 1 else 0.0 for g in range(len(labels))]
        return [1.0 if label == options[target] else 0.0 for label in labels]

    result = router.route(options, score_fn=scorer)
    assert result["expanded_clusters"] == [0, 1]
    assert result["chosen_index"] == target
    assert set(result["scored_indices"]) == set(clusters[0]) | set(clusters[1])


def test_expansion_is_bounded_when_all_groups_tie():
    options = build_options()
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    calls = []

    def scorer(labels):
        calls.append(list(labels))
        return [1.0] * len(labels) if len(calls) == 1 else relevance(labels)

    result = router.route(options, score_fn=scorer)
    cap = HierarchicalChoiceRouter.MAX_EXPANDED_GROUPS
    assert len(result["expanded_clusters"]) == cap
    assert len(result["scored_indices"]) <= cap * math.ceil(77 / 8)
    assert result["scored_options"] <= 8 + cap * math.ceil(77 / 8)  # documented bound: 28


def test_group_outside_margin_is_not_expanded():
    options = build_options()
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    clusters = router.build_clusters(options)
    calls = []

    def scorer(labels):
        calls.append(list(labels))
        if len(calls) == 1:
            return [1.0 if g == 0 else 0.5 for g in range(len(labels))]
        return relevance(labels)

    result = router.route(options, score_fn=scorer)
    assert result["expanded_clusters"] == [0]
    assert result["scored_indices"] == sorted(clusters[0])
    assert result["scored_options"] == 8 + len(clusters[0])


def test_margin_is_relative_to_negative_best_score():
    options = build_options()
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)
    calls = []

    def scorer(labels):
        calls.append(list(labels))
        if len(calls) == 1:
            # best = -1.0, threshold = -1.1: group 1 at -1.05 is inside the margin.
            return [-1.0 if g == 0 else -1.05 if g == 1 else -2.0 for g in range(len(labels))]
        return relevance(labels)

    result = router.route(options, score_fn=scorer)
    assert result["expanded_clusters"] == [0, 1]
