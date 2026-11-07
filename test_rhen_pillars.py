"""
Rhen-1.0 integration test suite.

Covers the three new modules (``rhen_verify``, ``rhen_credit``, ``eos_loop``),
the measurement module (``eos_eval``), the pillar wiring
(``rhen_pillars``) and the model/orchestrator integration points.

Every test asserts the *specific empirical claim* drawn from its source paper
where that claim is checkable here (e.g. that a quarter step rescues an
overshooting update, that schema-contrastive advantage recovers signal from a
collapsed batch, that a paid-call verification signal is rejected when it does
not earn its cost), rather than merely asserting that a function runs.
"""

import json
import os
import socket
import sys
import tempfile
import threading
import time

import torch

from rhen_verify import (
    AuditSinkError,
    BranchSparsityTracker,
    CheapVerifier,
    FileAuditSink,
    GeometrySelectiveClassifier,
    OutOfBandAuditLedger,
    SocketAuditSink,
    serve_audit_socket,
    verify_sealed_records,
)
from rhen_credit import (
    CascadeSearch,
    HillSampler,
    SchemaContrastiveAdvantage,
    SegmentAdvantage,
    TrajectoryGraph,
    fit_boptr,
    search_vs_rl,
)
from eos_loop import (
    DepthControlProtocol,
    FlashLoopUpdater,
    LoopCurriculum,
    StepSizeController,
    WorkingSetAttention,
)
from eos_eval import CostLedger, PreRegistration, ReferenceRelativeThreshold
from rhen_pillars import PillarStack, RhenPillarConfig

AUDIT_KEY = b"32-bytes-of-real-entropy-here!!"
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------------------
# 1. Audit integrity (arXiv:2609.30266)
# ---------------------------------------------------------------------------


@test
def test_audit_detects_tampering():
    ledger = OutOfBandAuditLedger(key=AUDIT_KEY)
    entry = ledger.record_decision("Primary action", "Failover", 0.87, ["Failover", "Scale"])
    assert ledger.verify(entry)["valid"], "a freshly sealed entry must verify"

    tampered = entry.to_dict()
    tampered["payload"]["chosen"] = "Scale"
    result = ledger.verify(tampered)
    assert not result["valid"], "tampered payload must not verify"
    assert not result["mac_valid"], "HMAC must catch a modified payload"


@test
def test_audit_chain_linkage_and_gap_detection():
    ledger = OutOfBandAuditLedger(key=AUDIT_KEY)
    for i in range(4):
        ledger.record("step", {"i": i})
    assert ledger.verify_chain()["all_valid"], "intact chain must verify"

    # A missing sequence number is the observable signature of an agent that
    # stopped emitting. `verify` checks an individual record against the local
    # chain, so a *gap* is detected by comparing sequence numbers, not by
    # re-verifying a survivor (whose own hash is still perfectly valid).
    full = [e.to_dict() for e in ledger.entries]
    assert ledger.detect_gaps(full) == [], "a complete chain has no gaps"

    partial = OutOfBandAuditLedger(key=AUDIT_KEY)
    partial.record("step", {"i": 0})
    partial.record("step", {"i": 1})
    assert partial.detect_gaps(full) == [3, 4], (
        "sequences present externally but missing locally must be reported"
    )

    # A record whose predecessor hash was rewritten must fail chain validation.
    forged = dict(full[2])
    forged["prev_hash"] = "0" * 64
    assert not ledger.verify(forged)["chain_valid"], "a rewritten link must be caught"


@test
def test_audit_rejects_short_key():
    try:
        OutOfBandAuditLedger(key=b"tooshort")
    except ValueError:
        return
    raise AssertionError("a weak sealing key must be rejected")


@test
def test_audit_fails_closed_when_sink_unavailable():
    # A socket sink is remote-sealing, so the agent-side ledger holds no key.
    dead = SocketAuditSink("127.0.0.1:1", reconnect_attempts=1)
    ledger = OutOfBandAuditLedger(sink=dead)
    try:
        ledger.record("decision", {"q": "primary"})
    except AuditSinkError:
        assert ledger.audit_failures == 1
        return
    raise AssertionError("an unauditable record must not be reported as audited")


@test
def test_audit_records_reach_a_separate_process_store():
    """The actual security property: records leave the agent's process."""
    tmp = socket.socket()
    tmp.bind(("127.0.0.1", 0))
    port = tmp.getsockname()[1]
    tmp.close()
    address = f"127.0.0.1:{port}"
    workdir = tempfile.mkdtemp()
    store = os.path.join(workdir, "audit.jsonl")

    stop = threading.Event()
    # The receiver holds the sealing key; the agent-side ledger holds none.
    server = threading.Thread(
        target=serve_audit_socket,
        args=(address, store, stop, 3),
        kwargs={"key": AUDIT_KEY},
        daemon=True,
    )
    server.start()
    time.sleep(0.4)
    try:
        ledger = OutOfBandAuditLedger(sink=SocketAuditSink(address))
        for i in range(3):
            ledger.record("decision", {"i": i})
        time.sleep(0.6)
        ledger.close()
    finally:
        stop.set()
        server.join(timeout=2)

    with open(store, "r", encoding="utf-8") as fh:
        external = [json.loads(line) for line in fh if line.strip()]
    assert len(external) == 3, f"expected 3 records out-of-process, got {len(external)}"
    check = verify_sealed_records(external, AUDIT_KEY)
    assert check["all_valid"] is True, f"external records must verify independently: {check}"
    assert check["gaps"] == [], "a complete stream has no gaps"


@test
def test_file_sink_persists_records():
    path = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
    ledger = OutOfBandAuditLedger(key=AUDIT_KEY, sink=FileAuditSink(path))
    ledger.record("a", {})
    ledger.record("b", {})
    ledger.close()
    with open(path, "r", encoding="utf-8") as fh:
        assert len([line for line in fh if line.strip()]) == 2


# ---------------------------------------------------------------------------
# 2. Selectivity, cheap verification, branch sparsity
# ---------------------------------------------------------------------------


@test
def test_geometry_selectivity_rejects_outliers():
    g = torch.Generator().manual_seed(0)
    tight = torch.randn(20, 4, generator=g) * 0.2
    other = torch.randn(6, 4, generator=g) * 0.2 + 8.0
    far = torch.randn(2, 4, generator=g) * 0.2 + 30.0
    embeddings = torch.cat([tight, other, far])
    # The far cluster is spatially isolated but carries the *wrong* label, so
    # the nearest-centroid rule misclassifies it and rejecting it must pay.
    labels = ["a"] * 20 + ["b"] * 6 + ["a"] * 2

    clf = GeometrySelectiveClassifier(max_radius=1.0, min_accepted=3)
    clf.fit(torch.cat([tight, other]), labels[:26])
    report = clf.predict(embeddings, labels)

    assert not report["accept"][-1], "a far outlier must be rejected"
    assert not report["accept"][-2], "both far outliers must be rejected"
    assert report["n_rejected"] == 2, f"expected 2 rejections, got {report['n_rejected']}"
    sel = report["selectivity"]
    assert sel["accuracy_on_accepted"] == 1.0, "accepted rows should all be correct"
    assert sel["accuracy_overall"] < 1.0, "the rejected outliers must actually be wrong"
    assert sel["utility"] > 0.0, f"rejection should pay for itself, got {sel['utility']}"
    assert sel["feasible"] is True


@test
def test_selectivity_refuses_when_rejection_does_not_pay():
    """Utility must be able to be zero/negative -- a pre-deployment check."""
    points = torch.zeros(6, 2)
    labels = ["a"] * 6
    clf = GeometrySelectiveClassifier(max_radius=1.0, min_accepted=3)
    clf.fit(points, labels)
    report = clf.predict(points, labels)
    assert report["selectivity"]["utility"] == 0.0
    assert report["selectivity"]["feasible"] is False
    assert "does not improve" in report["selectivity"]["reason"]


@test
def test_cheap_verifier_rejects_signal_that_does_not_earn_its_calls():
    """Reproduces arXiv:2609.28879: top-1 probability beats sampled entropy."""
    probs = [
        [0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.8, 0.1, 0.1],
        [0.34, 0.33, 0.33], [0.7, 0.2, 0.1], [0.2, 0.6, 0.2],
    ]
    labels = [1, 0, 1, 0, 1, 0]
    # Sampled groups deliberately carry less signal than the single pass.
    samples = [
        [[0.9, 0.05, 0.05]] * 3, [[0.2, 0.5, 0.3]] * 3, [[0.5, 0.4, 0.1]] * 3,
        [[0.3, 0.3, 0.4]] * 3, [[0.6, 0.3, 0.1]] * 3, [[0.2, 0.2, 0.6]] * 3,
    ]
    out = CheapVerifier(samples=3).signals(probs, labels, samples)

    assert out["signals"]["top1_probability"]["extra_calls"] == 0
    assert out["signals"]["discrete_semantic_entropy"]["extra_calls"] == 3
    assert "discrete_semantic_entropy" in out["rejected_signals"], (
        "a paid signal that does not beat the free one must be rejected"
    )


@test
def test_auroc_matches_known_value():
    assert CheapVerifier.auroc([0.9, 0.8, 0.7, 0.1], [1, 1, 1, 0]) == 1.0
    assert CheapVerifier.auroc([0.1, 0.2, 0.3, 0.9], [1, 1, 1, 0]) == 0.0
    assert CheapVerifier.auroc([0.5, 0.5], [1, 0]) == 0.5


@test
def test_branch_sparsity_identifies_frozen_branches():
    tracker = BranchSparsityTracker(threshold=0.05, min_loops=2)
    for loop in range(1, 6):
        tracker.observe(
            loop,
            {0: torch.zeros(4), 1: torch.ones(4) * 0.1 * loop, 2: torch.zeros(4)},
        )
    assert tracker.frozen_branches() == [0, 2]
    assert tracker.active_branches() == [1]
    savings = tracker.estimated_savings(6)
    assert savings["frozen_count"] == 2
    assert 0.0 < savings["estimated_saved_fraction"] < 1.0


# ---------------------------------------------------------------------------
# 3. Credit assignment
# ---------------------------------------------------------------------------


@test
def test_trajectory_graph_credits_steps_inside_failed_rollouts():
    """A step in a failed trajectory can still deserve positive credit."""
    graph = TrajectoryGraph(gamma=0.9, iterations=200, tol=1e-9)
    graph.add_rollout(["s0", "s1", "good"], [0.1, 0.1], terminal_reward=1.0)
    graph.add_rollout(["s0", "s1", "bad"], [-0.1, 0.0], terminal_reward=0.0)
    graph.add_rollout(["s0", "alt", "win"], [0.05, 0.05], terminal_reward=0.6)

    values = graph.solve()
    assert values["good"] > values["bad"], "a good terminal must outrank a bad one"

    credits = {(c["tail"], c["head"]): c["credit"] for c in graph.edge_credit()}
    assert credits[("s1", "good")] > 0, "step toward success must earn credit"
    assert credits[("s1", "bad")] < 0, "step toward failure must be penalised"
    assert credits[("s0", "alt")] > credits[("s0", "s1")], (
        "the better branch from the same state must earn more credit"
    )


@test
def test_trajectory_graph_value_iteration_converges():
    """Guards against the accumulating-backup divergence bug."""
    graph = TrajectoryGraph(gamma=0.95, iterations=500, tol=1e-10)
    for _ in range(3):
        graph.add_rollout(["a", "b", "c"], [0.1, 0.1], terminal_reward=1.0)
    values = graph.solve()
    assert all(abs(v) < 100.0 for v in values.values()), f"values must stay bounded: {values}"
    assert values["c"] == max(values.values())


@test
def test_segment_advantage_uses_intermediate_credit():
    adv = SegmentAdvantage(gamma=0.9)
    # out[i] is the return *from* segment i onwards, so the final segment sees
    # the reward undiscounted while earlier segments see it discounted by
    # gamma ** (n - 1 - i). Ordering must therefore be non-decreasing in i.
    returns = adv.segment_returns([0.0, 0.0, 1.0], terminal_bonus=1.0)
    assert returns[2] > returns[1] > returns[0], (
        f"returns must grow toward the rewarding segment, got {returns}"
    )
    # The earlier segment's return must still *include* the later reward,
    # discounted -- this is what distinguishes it from sequence-level credit.
    assert returns[0] > 0.0, "an early segment must still receive downstream credit"
    assert abs(returns[0] - 0.9 * returns[1]) < 1e-6, "the gap must be a gamma factor"

    normalised = adv.advantage([0.2, 0.5, -0.1])
    assert abs(sum(normalised)) < 1e-5, "normalised advantages must centre on zero"


@test
def test_schema_contrastive_rescues_a_collapsed_batch():
    """EAGER's core claim: a global baseline yields no signal; contrastive does."""
    estimator = SchemaContrastiveAdvantage()

    diag = estimator.diagnose([-1.0, -1.0, -1.0, -1.0], ["noul", "noul", "choice:3", "noul"])
    assert diag["collapsed"] is True
    assert diag["naive_global_std"] == 0.0

    # A rare correct answer inside an all-wrong batch must still get signal.
    advantages = estimator.advantage(
        [-1.0, -1.0, -1.0, 1.0], ["noul", "noul", "noul", "choice:3"]
    )
    assert advantages[:3] == [0.0, 0.0, 0.0]
    assert advantages[3] > 0.0, "the rare correct decision must receive positive advantage"


@test
def test_schema_contrastive_handles_mismatched_lengths():
    try:
        SchemaContrastiveAdvantage().advantage([1.0, 0.0], ["noul"])
    except ValueError:
        return
    raise AssertionError("mismatched rewards/schemas must raise")


@test
def test_hill_sampler_refuses_unverified_high_scorer():
    """The mechanism that separates this from naive best-of-N."""
    sampler = HillSampler(seed=0)

    def propose(champion):
        # The 'cheat' candidate scores highest but fails verification.
        return [(champion + "-cheat", 0.99), (champion + "-ok", 0.5)]

    result = sampler.run(
        propose, verify_fn=lambda c: "cheat" not in c, initial="A", rounds=3
    )
    assert "cheat" not in result["champion"], "an unverified candidate was accepted"
    assert result["rejected_unverified"] == 3


@test
def test_cascade_search_beam_finds_best_path():
    result = CascadeSearch().beam(
        lambda node: [(node + "a", 0.3), (node + "b", 0.7)], root="S", width=2, depth=3
    )
    # Each step appends a branch label, so the optimal 3-step path is S -> Sb ->
    # Sbb -> Sbbb and the terminal node is "Sbbb".
    assert result["path"] == ["S", "Sb", "Sbb", "Sbbb"], f"got {result['path']}"
    assert result["best_item"] == "Sbbb"
    assert abs(result["best_score"] - 0.7 * 3) < 1e-6


@test
def test_search_vs_rl_attributes_gain_to_sampling():
    report = search_vs_rl([0.50, 0.70, 0.78, 0.80], [0.50, 0.72, 0.81, 0.82])
    assert report.best_search == 0.80
    assert report.unexplained_by_search is not None
    assert report.unexplained_by_search < 0.05, "most of this gain is search, not training"
    assert "sampling efficiency" in report.interpretation


@test
def test_boptr_refuses_to_fit_two_points():
    fit = fit_boptr([0.5], [0.6])
    assert fit["fitted"] is False, "a two-point fit must be refused, not guessed"


# ---------------------------------------------------------------------------
# 4. Recurrent-loop control (arXiv:2609.16665)
# ---------------------------------------------------------------------------


def _overshooting_step(state, target, gain=3.0):
    """A step whose direction is helpful but whose scale overshoots."""
    return target + gain * (target - state)


@test
def test_quarter_step_rescues_an_overshooting_loop():
    """The headline claim of arXiv:2609.16665, in miniature.

    With a deliberately overshooting proposal, a full update diverges while the
    paper's quarter step converges.
    """
    target = torch.ones(8)

    def run(controller):
        state = torch.zeros(8)
        controller.reset()
        for i in range(1, 7):
            state, _ = controller.apply(state, _overshooting_step(state, target), i)
        return float((state - target).pow(2).sum())

    legacy = run(StepSizeController(enabled=False))
    quarter = run(
        StepSizeController(base_alpha=0.25, anneal=False, overshoot_rejection=False)
    )
    assert legacy > 1e4, f"the overshooting full step should diverge, got {legacy}"
    assert quarter < 1e-6, f"the quarter step should converge, got {quarter}"


@test
def test_overshoot_rejection_fires_and_repairs():
    """The repair must trigger even when alpha is 1.0 (vacuous-compare guard)."""
    target = torch.ones(8)
    controller = StepSizeController(
        base_alpha=1.0,
        anneal=False,
        overshoot_rejection=True,
        utility_fn=lambda s: -float((s - target).pow(2).sum()),
    )
    controller.set_budget(4)
    state = torch.zeros(8)
    repairs = 0
    for i in range(1, 5):
        state, stats = controller.apply(state, _overshooting_step(state, target), i)
        repairs += int(stats.overshoot)
    assert repairs > 0, "overshoot rejection never fired"
    assert float((state - target).pow(2).sum()) < 1e-6, "rejection did not rescue the state"


@test
def test_step_controller_disabled_is_a_passthrough():
    controller = StepSizeController(enabled=False)
    current, proposed = torch.randn(2, 4), torch.randn(2, 4)
    out, stats = controller.apply(current, proposed, 1)
    assert torch.equal(out, proposed)
    assert stats.alpha == 1.0


@test
def test_step_alpha_anneals_downward():
    controller = StepSizeController(base_alpha=0.5, anneal=True, anneal_to=0.5)
    controller.set_budget(5)
    current = torch.zeros(2, 4)
    alphas = [
        controller.apply(current, torch.ones(2, 4), i)[1].alpha for i in range(1, 6)
    ]
    assert alphas[0] > alphas[-1], f"alpha should shrink across the budget: {alphas}"
    assert all(0.0 < a <= 0.5 for a in alphas)



# ---------------------------------------------------------------------------
# 5. Depth control, sparsity, curriculum, working set
# ---------------------------------------------------------------------------


@test
def test_depth_protocol_separates_real_depth_from_readout_drift():
    exploited = DepthControlProtocol(total_loops=12)
    exploited.record("truncated_standard", 1.0, loss=1.0, n_block_applications=12, n_distinct_iterations=12)
    exploited.record("full_budget_single_iteration", 1.0, loss=3.0, n_block_applications=12, n_distinct_iterations=1)
    assert exploited.conclude()["verdict"] == "depth_exploited"

    confounded = DepthControlProtocol(total_loops=12)
    confounded.record("truncated_standard", 1.0, loss=1.0, n_block_applications=12, n_distinct_iterations=12)
    confounded.record("full_budget_single_iteration", 1.0, loss=1.0, n_block_applications=12, n_distinct_iterations=1)
    assert confounded.conclude()["verdict"] == "depth_not_exploited"


@test
def test_depth_protocol_refuses_to_conclude_without_controls():
    protocol = DepthControlProtocol(total_loops=12)
    protocol.record("truncated_standard", 1.0, loss=1.0, n_block_applications=12, n_distinct_iterations=12)
    assert protocol.conclude()["verdict"] == "insufficient_data"
    plan = protocol.plan()
    assert any(p["condition"] == "dense_negative_control" for p in plan)
    assert any(p.get("n_distinct_iterations") == 1 for p in plan), (
        "the linchpin control must be present"
    )


@test
def test_flash_loop_holds_frozen_tokens():
    updater = FlashLoopUpdater(threshold=0.01, warmup_loops=1)
    base = torch.randn(1, 10, 8)
    updater.observe(base)
    moved = base.clone()
    moved[0, 0, :] += 5.0
    moved[0, 1, :] += 3.0
    mask = updater.observe(moved)

    assert mask is not None, "sparsity mask never unfroze after warmup"
    assert int(mask.sum()) == 2, f"expected 2 active tokens, got {int(mask.sum())}"

    proposal = torch.randn(1, 10, 8)
    out = updater.apply(moved, proposal)
    # A frozen token keeps its current value; an active token takes the proposal.
    assert torch.allclose(out[0, 5], moved[0, 5]), "a frozen token was overwritten"
    assert torch.allclose(out[0, 0], proposal[0, 0]), "an active token was not updated"
    assert not torch.allclose(out[0, 0], moved[0, 0]), "an active token was frozen"


@test
def test_loop_curriculum_ramps_and_concentrates():
    curriculum = LoopCurriculum(min_loops=2, max_loops=12, warmup_steps=100)
    assert curriculum.cap(0) == 2
    assert curriculum.cap(200) == 12
    caps = [curriculum.cap(s) for s in range(0, 201, 10)]
    assert all(caps[i] <= caps[i + 1] for i in range(len(caps) - 1)), "cap must be monotone"

    gen = torch.Generator().manual_seed(0)
    samples = [curriculum.sample(200, gen) for _ in range(2000)]
    assert all(2 <= s <= 12 for s in samples)
    assert samples.count(12) > samples.count(2), "sampling should concentrate near the cap"


@test
def test_working_set_freezes_only_when_stable():
    working = WorkingSetAttention(warmup_loops=2, stability_threshold=0.5, min_fraction=0.25)
    attn = torch.eye(8).unsqueeze(0)
    frozen = None
    for _ in range(4):
        frozen = working.observe(attn)
    assert frozen is not None, "stable routing never froze a working set"
    assert bool(frozen.diagonal().all()), "a query must always attend to itself"





# ---------------------------------------------------------------------------
# 6. Measurement infrastructure (arXiv:2609.20538)
# ---------------------------------------------------------------------------


@test
def test_prereg_distinguishes_refusal_from_failure():
    # A protocol that could not be run must not read as a failed claim.
    refused = PreRegistration("c", "val_loss", 0.0, direction="decrease")
    refused.record("c", protocol_executed=False, value=-0.1)
    assert refused.summary()["verdict"] == "refused"

    # Supported: the metric moved in the hypothesised direction (loss fell).
    supported = PreRegistration("c", "val_loss", 3.0, direction="decrease", min_examples=2)
    supported.record("c", True, 2.0)
    supported.record("c", True, 2.5)
    assert supported.summary()["verdict"] == "supported"

    # Not supported: the metric moved the wrong way, or not far enough.
    failed = PreRegistration("c", "val_loss", 1.0, direction="decrease", min_examples=2)
    failed.record("c", True, 2.0)
    failed.record("c", True, 2.5)
    assert failed.summary()["verdict"] == "not_supported"

    # Too few observations to conclude either way.
    thin = PreRegistration("c", "m", 0.0, direction="increase", min_examples=5)
    thin.record("c", True, 0.1)
    assert thin.summary()["verdict"] == "inconclusive"


@test
def test_prereg_refresh_discards_prior_observations():
    reg = PreRegistration("c", "m", 0.0, direction="increase")
    reg.record("c", True, 0.1)
    outcome = reg.refresh("distribution shift detected")
    assert outcome["refreshed"] is True
    assert reg.summary()["n"] == 0, "refresh must invalidate the prior reference"
    assert reg.refresh_count == 1


@test
def test_reference_relative_threshold_exposes_global_threshold_bias():
    thresholds = ReferenceRelativeThreshold(quantile=0.95, min_reference=8)
    quiet = [0.1 * i for i in range(15)]
    busy = [5.0 + 0.1 * i for i in range(15)]
    thresholds.calibrate("quiet", quiet)
    thresholds.calibrate("busy", busy)

    assert abs(thresholds.threshold("busy") - thresholds.threshold("quiet")) > 1.0, (
        "the two regimes must not share a threshold"
    )
    report = thresholds.cross_regime_false_alarm_rate({"quiet": quiet, "busy": busy})
    assert set(report["per_regime"]) == {"quiet", "busy"}
    for row in report["per_regime"].values():
        assert row["relative_threshold"] is not None
        assert row["relative_false_alarm_rate"] is not None


@test
def test_reference_relative_threshold_abstains_when_uncalibrated():
    verdict = ReferenceRelativeThreshold(min_reference=8).is_anomalous(99.0, "never-calibrated")
    assert verdict["admitted"] is False
    assert verdict["verdict"] == "unadmitted", (
        "an uncalibrated regime must abstain, not guess"
    )


@test
def test_cost_ledger_separates_unmeasured_from_unprofitable():
    ledger = CostLedger()
    ledger.register("profitable", measured_gain=0.05)
    ledger.register("unprofitable", measured_gain=-0.02)
    ledger.register("never_measured")

    with ledger.time("profitable"):
        pass
    try:
        with ledger.time("unprofitable"):
            raise ValueError("boom")
    except ValueError:
        pass

    report = ledger.report()
    assert report["n_pillars"] == 3
    assert report["n_errors"] == 1, "the ledger must observe errors without swallowing them"
    assert report["unprofitable_pillars"] == ["unprofitable"]
    assert report["unmeasured_pillars"] == ["never_measured"]
    # Unprofitable is listed before merely-unmeasured.
    assert ledger.prune_candidates() == ["unprofitable", "never_measured"]


# ---------------------------------------------------------------------------
# 7. Pillar wiring and backward compatibility
# ---------------------------------------------------------------------------


@test
def test_pillars_are_inert_by_default():
    """Backward compatibility: an unconfigured stack must change nothing."""
    stack = PillarStack()
    assert stack.route_calibration("Der Server ist down")["applied"] is False
    assert stack.route_choice([f"i{i}" for i in range(77)])["applied"] is False
    assert stack.check_regression(["a"], ["a"]).allowed is True
    assert stack.plan_depth(torch.randn(2, 8))["applied"] is False
    assert stack.audit is None
    assert stack.search_baseline(lambda i: (i, 0.0))["applied"] is False


@test
def test_pillar_language_calibration_routes_german():
    stack = PillarStack(RhenPillarConfig(enable_language_calibration=True))
    record = stack.route_calibration("Der Server ist down und die Datenbank ist nicht erreichbar")
    assert record["applied"] is True
    assert record["language"] == "de"


@test
def test_pillar_hierarchical_choice_reduces_scored_options():
    stack = PillarStack(
        RhenPillarConfig(
            enable_hierarchical_choice=True, choice_num_groups=8, choice_flat_threshold=20
        )
    )
    result = stack.route_choice([f"intent_{i}" for i in range(77)])
    assert result["decomposed"] is True
    assert result["scored_options"] < result["total_options"]


@test
def test_pillar_ood_guard_blocks_regression():
    stack = PillarStack(RhenPillarConfig(enable_ood_guard=True))
    assert stack.check_regression(["a"], ["a"]).detail["verdict"] == "baseline_pending"
    stack.establish_baseline(["a", "b", "c", "d"], ["a", "b", "c", "d"])
    assert stack.check_regression(["a", "b", "c", "d"], ["a", "b", "c", "d"]).allowed is True
    assert stack.check_regression(["w", "x", "y", "z"], ["a", "b", "c", "d"]).allowed is False


@test
def test_pillar_audit_records_and_verifies():
    path = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
    stack = PillarStack(
        RhenPillarConfig(
            enable_audit=True, audit_key=AUDIT_KEY, audit_sink="file", audit_path=path
        )
    )
    entry = stack.authorize_action("Primary action", "Failover", 0.87)
    assert entry.sequence == 1
    assert stack.audit.verify_chain()["all_valid"] is True
    stack.close()


@test
def test_orchestrator_exposes_pillar_api_and_stays_compatible():
    from config import EOSConfig
    from unified_orchestrator import UnifiedAIOrchestrator

    config = EOSConfig(
        hidden_size=64, intermediate_size=128, vocab_size=128,
        num_hidden_layers=1, num_recurrent_layers=1,
        num_pre_layers=1, num_post_layers=1, max_recurrent_loops=2,
    )
    orchestrator = UnifiedAIOrchestrator(config)
    assert orchestrator.verify_audit_chain() == {"enabled": False}
    assert orchestrator.audit_decision("q", "a", 0.5) is None
    assert orchestrator.calibration_route("Der Server ist down")["applied"] is False
    assert orchestrator.pillar_cost_report()["n_pillars"] > 0


@test
def test_model_loop_controls_do_not_pollute_checkpoints():
    """New loop controllers must not appear in the state_dict."""
    from config import EOSConfig
    from model import EOSModel

    def build(**kwargs):
        base = dict(
            hidden_size=64, intermediate_size=128, vocab_size=128,
            num_hidden_layers=1, num_recurrent_layers=1,
            num_pre_layers=1, num_post_layers=1, max_recurrent_loops=3,
            enable_mtp=False, use_moe=False,
        )
        base.update(kwargs)
        return EOSModel(EOSConfig(**base))

    torch.manual_seed(1234)
    legacy = build()
    torch.manual_seed(1234)
    enhanced = build(enable_step_size_control=True, enable_flash_loop_sparsity=True)

    assert set(legacy.state_dict()) == set(enhanced.state_dict()), (
        "loop controls leaked into the checkpoint"
    )
    assert sum(p.numel() for p in legacy.parameters()) == sum(
        p.numel() for p in enhanced.parameters()
    ), "loop controls must not add trainable parameters"

    assert legacy.loop_diagnostics()["step_size_control"] is False
    assert enhanced.loop_diagnostics()["step_size_control"] is True


@test
def test_model_step_control_is_bitwise_neutral_when_disabled():
    """A default model must reproduce exactly, and toggling must have effect."""
    from config import EOSConfig
    from model import EOSModel

    def build(**kwargs):
        base = dict(
            hidden_size=64, intermediate_size=128, vocab_size=128,
            num_hidden_layers=1, num_recurrent_layers=1,
            num_pre_layers=1, num_post_layers=1, max_recurrent_loops=3,
            enable_mtp=False, use_moe=False, loop_sampling="fixed",
        )
        base.update(kwargs)
        torch.manual_seed(1234)  # identical init for a fair comparison
        return EOSModel(EOSConfig(**base)).eval()

    torch.manual_seed(0)
    ids = torch.randint(0, 128, (1, 16))

    baseline = build()
    with torch.no_grad():
        reference = baseline(ids)[0]

    twin = build()
    with torch.no_grad():
        # Not bitwise: fused/compiled kernels are not bit-reproducible run to
        # run, so the bar is numerical equality within float tolerance.
        assert torch.allclose(reference, twin(ids)[0], atol=1e-5), (
            "the default path must be numerically reproducible"
        )

    controlled = build(enable_step_size_control=True, loop_step_alpha=0.25)
    with torch.no_grad():
        changed = controlled(ids)[0]
    assert not torch.allclose(reference, changed, atol=1e-4), (
        "step control must actually alter the computation"
    )

    controlled.loop_step_controller.enabled = False
    with torch.no_grad():
        assert torch.allclose(reference, controlled(ids)[0], atol=1e-5), (
            "disabling the controller must reproduce the legacy update"
        )


@test
def test_model_forward_runs_with_loop_controls_enabled():
    from config import EOSConfig
    from model import EOSModel

    torch.manual_seed(0)
    config = EOSConfig(
        hidden_size=64, intermediate_size=128, vocab_size=128,
        num_hidden_layers=1, num_recurrent_layers=1,
        num_pre_layers=1, num_post_layers=1, max_recurrent_loops=4,
        enable_mtp=False, use_moe=False,
        enable_step_size_control=True, loop_step_alpha=0.25,
        enable_flash_loop_sparsity=True, flash_loop_threshold=1e-3,
        flash_loop_warmup_loops=1,
    )
    model = EOSModel(config)
    ids = torch.randint(0, 128, (2, 16))
    _, loss, stats = model(ids, labels=ids.clone(), output_loop_stats=True)
    assert torch.isfinite(loss), "loss went non-finite with loop controls enabled"
    assert stats["avg_loops_per_token"] >= 1.0
    assert model.loop_diagnostics()["step"]["loops"] > 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    passed, failed = 0, []
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - the harness reports everything
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            failed.append(name)
    print(f"\n{passed}/{len(TESTS)} Rhen-1.0 pillar tests passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. The pillar tests register through the ``@test`` decorator and
# run via ``main()``; its exit code is the battery verdict.
import unittest as _unittest


class TestRhenPillars(_unittest.TestCase):
    """test_rhen_pillars verification suite (master battery)."""

    def test_rhen_pillars_battery(self):
        # Running all registered pillar tests through the harness; a return
        # code of 0 means every pillar test passed (the harness already
        # reports per-test failures and never raises past this point).
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    sys.exit(main())

