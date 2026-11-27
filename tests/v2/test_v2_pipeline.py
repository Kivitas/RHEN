"""Decision pipeline, policy, feedback and routing (plan section 9)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings

import numpy as np
import torch

from rhen.agent_tools import ActionStagingEngine
from rhen.decision_pipeline import DecisionGateError, DecisionPolicy
from rhen.rhen_verify import OutOfBandAuditLedger
from rhen.v2.calibrate import Calibration, OnlineRecalibrator
from rhen.v2.checker import CheckedDecision, Checker
from rhen.v2.config import preset
from rhen.v2.feedback import DecisionEntry, OutcomeLedger
from rhen.v2.model import Rhen2Model
from rhen.v2.packing import QuestionSpec, Rhen2Packer
from rhen.v2.pipeline import STAGES, DecisionPipeline, weights_hash
from rhen.v2.policy import (
    NULL, CostSpec, DecisionPolicy2, LegacyDecisionPolicy, PolicyStatus, cascade_confidence, from_v1_policy,
)
from rhen.v2.routing import kmeans, route_large_option_set

SECRET_STATE = "ZEBRA-SECRET-9137 account balance 4412 units"
CHOICE = QuestionSpec("choice", "which is it?", ["a", "b", "c"])
LENIENT = CostSpec(cost_wrong=1.0, cost_abstain=0.999)


def make_cal(model_hash: str, n: int = 600) -> Calibration:
    return Calibration(
        model_hash=model_hash, data_hash="d", created="now",
        temperature_table={"choice": {"2": 1.0, "4": 1.0, "8": 1.0, "all": 1.0}},
        isotonic_table={"choice": {}},
        conformal_table={"choice": {"2": 0.5, "4": 0.05, "8": 0.5}},
        tau_act_table={"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0},
        ood={"energy": {"choice": {"all": 1e9}}},
        n_calibration={"choice": {"2": n, "4": n, "8": n}},
        extras={"alpha": 0.1},
    )


class StubChecker:
    """Returns a fixed confirmed decision for every question; counts calls."""

    def __init__(self, confidence=0.99, verdict="CONFIRM", calibrated=True, boom=None):
        self.conf, self.verdict, self.calibrated, self.boom = confidence, verdict, calibrated, boom
        self.log = []                  # shared with the pipeline's shallow copy
        self.cal = None

    @property
    def calls(self):
        return len(self.log)

    def check(self, batch, effort="medium", risk_tier="low"):
        self.log.append(1)
        if self.boom is not None:
            raise self.boom
        out = []
        for b in range(batch.batch_size):
            for m in range(batch.max_questions):
                if not bool(batch.q_valid[b, m]):
                    continue
                spec = batch.specs[b][m]
                labels = spec.options or ["true", "false"]
                p = {labels[0]: self.conf}
                for lab in labels[1:]:
                    p[lab] = (1 - self.conf) / len(labels)
                p[NULL] = max(0.0, 1.0 - sum(p.values()))
                out.append(CheckedDecision(labels[0], p, self.conf, [labels[0]], self.verdict, [], [], 2, effort,
                                           self.calibrated))
        return out


class TestPolicy(unittest.TestCase):
    def test_shorthand_equals_threshold_rule(self):
        pol = DecisionPolicy2()
        rng = np.random.RandomState(0)
        n = 0
        for _ in range(400):
            cw = float(rng.uniform(0.5, 5.0))
            ca = float(rng.uniform(0.01, 0.99)) * cw
            p = float(rng.uniform(0.34, 0.99))
            thr = 1.0 - ca / cw
            if abs(p - thr) < 1e-6:
                continue
            probs = {"a": p, "b": (1 - p) * 0.6, NULL: (1 - p) * 0.4}
            out = pol.evaluate(probs, costs=CostSpec(cost_wrong=cw, cost_abstain=ca), risk_tier="low")
            self.assertEqual(out.status is PolicyStatus.EXECUTE, p > thr, (p, thr))
            # the same through p_correct anchoring
            out2 = pol.evaluate({"a": 0.5, "b": 0.3, NULL: 0.2}, p_correct=p, top_label="a",
                                costs=CostSpec(cost_wrong=cw, cost_abstain=ca))
            self.assertEqual(out2.status is PolicyStatus.EXECUTE, p > thr)
            n += 1
        self.assertGreater(n, 300)

    def test_default_tier_costs_use_plan_tolerances(self):
        pol = DecisionPolicy2(approval_required_tiers=())
        for tier, tol in (("low", 0.10), ("medium", 0.05), ("high", 0.01), ("critical", 0.001)):
            above = pol.evaluate({"a": 1 - tol / 2, NULL: tol / 2}, risk_tier=tier)
            below = pol.evaluate({"a": 1 - tol * 2, NULL: tol * 2}, risk_tier=tier)
            self.assertIs(above.status, PolicyStatus.EXECUTE, tier)
            self.assertIs(below.status, PolicyStatus.ABSTAIN, tier)

    def test_reject_option_and_full_matrix(self):
        pol = DecisionPolicy2()
        probs = {"a": 0.55, "b": 0.45, NULL: 0.0}
        # abstaining is cheaper than the expected loss of either action
        out = pol.evaluate(probs, costs=CostSpec(cost_wrong=1.0, cost_abstain=0.3))
        self.assertIs(out.status, PolicyStatus.ABSTAIN)
        # asymmetric matrix: wrongly taking "a" is very costly, so "b" wins despite lower probability
        m = {"a": {"a": 0.0, "b": 10.0, NULL: 10.0}, "b": {"a": 1.0, "b": 0.0, NULL: 1.0}}
        out = pol.evaluate(probs, costs=CostSpec(matrix=m, cost_abstain=5.0))
        self.assertIs(out.status, PolicyStatus.EXECUTE)
        self.assertEqual(out.option, "b")
        self.assertAlmostEqual(out.risk_act, 0.55)

    def test_constraints_and_blocked(self):
        pol = DecisionPolicy2()
        probs = {"a": 0.9, "b": 0.08, NULL: 0.02}
        out = pol.evaluate(probs, costs=LENIENT, constraints=[("no_a", lambda o, f: o != "a")])
        self.assertEqual(out.option, "b")
        self.assertTrue(any("no_a" in r for r in out.reasons))
        out = pol.evaluate(probs, costs=LENIENT, constraints=[("none", lambda o, f: False)])
        self.assertIs(out.status, PolicyStatus.BLOCKED)
        self.assertIsNone(out.option)

    def test_tier_approval_and_escalate_and_uncalibrated(self):
        pol = DecisionPolicy2()
        probs = {"a": 0.9999, NULL: 0.0001}
        self.assertIs(pol.evaluate(probs, risk_tier="low").status, PolicyStatus.EXECUTE)
        self.assertIs(pol.evaluate(probs, risk_tier="medium").status, PolicyStatus.EXECUTE)
        self.assertIs(pol.evaluate(probs, risk_tier="high").status, PolicyStatus.NEEDS_APPROVAL)
        self.assertIs(pol.evaluate(probs, risk_tier="medium", approval=True).status, PolicyStatus.NEEDS_APPROVAL)
        self.assertIs(pol.evaluate(probs, verdict="ESCALATE").status, PolicyStatus.ABSTAIN)
        self.assertIs(pol.evaluate(probs, verdict="REVISE").status, PolicyStatus.ABSTAIN)
        self.assertIs(pol.evaluate(probs, risk_tier="medium", calibrated=False).status, PolicyStatus.ABSTAIN)
        self.assertIs(pol.evaluate(probs, risk_tier="low", calibrated=False).status, PolicyStatus.EXECUTE)

    def test_cascade_label(self):
        r = cascade_confidence([0.9, 0.8, 0.5])
        self.assertAlmostEqual(r.value, 0.36)
        self.assertEqual(r.label, "uncalibrated_joint")
        r = cascade_confidence([0.9, 0.8], whole_workflow_calibration=lambda p: p * 0.5)
        self.assertEqual(r.label, "calibrated_joint")
        self.assertAlmostEqual(r.value, 0.36)

    def test_v1_shim_warns_and_converts(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old = LegacyDecisionPolicy(utility=lambda o, f: 1.0, min_confidence=0.8)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) and "DecisionPolicy2" in str(x.message)
                            for x in w))
        self.assertIsInstance(old, DecisionPolicy)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            new, spec = from_v1_policy(old)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertAlmostEqual(spec.threshold(), 0.8)
        self.assertIs(new.evaluate({"a": 0.85, NULL: 0.15}, costs=spec).status, PolicyStatus.EXECUTE)
        self.assertIs(new.evaluate({"a": 0.75, NULL: 0.25}, costs=spec).status, PolicyStatus.ABSTAIN)


class TestRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = preset("r2-tiny")
        cls.model = Rhen2Model(cls.cfg).eval()
        cls.pk = Rhen2Packer(cls.cfg)
        cls.options = [f"opt{i}" for i in range(300)]

    def test_kmeans_deterministic_and_bounded(self):
        x = np.random.RandomState(1).randn(300, 8)
        a, b = kmeans(x, 20, seed=3), kmeans(x, 20, seed=3)
        self.assertTrue(np.array_equal(a, b))
        self.assertLessEqual(a.max() + 1, 20)
        self.assertEqual(len(np.unique(a)), a.max() + 1)

    def test_route_k300(self):
        q = QuestionSpec("choice", "pick one", [])
        r1 = route_large_option_set(self.model, self.pk, "alpha 1 beta 2", q, self.options)
        r2 = route_large_option_set(self.model, self.pk, "alpha 1 beta 2", q, self.options)
        self.assertLessEqual(r1.report.n_groups, 255)
        self.assertGreaterEqual(r1.report.n_groups, 2)
        self.assertEqual(sum(r1.report.group_sizes), 300)
        self.assertEqual(r1.report.option_pass_calls, 2)        # 300 options need two passes of at most 255
        self.assertEqual(r1.report.model_calls, 1 + 2 + 1 + r1.report.inner_choice_calls)
        self.assertEqual(r1.report.inner_choice_calls, 0 if r1.p_null_inside == 0.0 and r1.abstained else 1)
        self.assertEqual((r1.option, r1.index, r1.group), (r2.option, r2.index, r2.group))
        self.assertEqual(r1.joint_label, "uncalibrated_joint")
        if not r1.abstained:
            self.assertIn(r1.option, self.options)
            self.assertEqual(self.options[r1.index], r1.option)

    def test_flat_set_rejected(self):
        with self.assertRaises(ValueError):
            route_large_option_set(self.model, self.pk, "x", "q", ["a", "b"])


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = preset("r2-tiny")
        cls.model = Rhen2Model(cls.cfg).eval()
        cls.pk = Rhen2Packer(cls.cfg)
        cls.mh = weights_hash(cls.model)

    def pipe(self, **kw):
        kw.setdefault("training_run_id", "run-1")
        return DecisionPipeline(self.model, self.pk, **kw)

    # ---- stages and trace ---------------------------------------------------------------
    def test_stage_order_and_trace(self):
        res = self.pipe().decide(SECRET_STATE, [CHOICE])
        self.assertEqual(res.trace.stage_names(), list(STAGES))
        for rec in res.trace.records:
            self.assertGreaterEqual(rec.duration_s, 0.0)
            self.assertIsInstance(rec.summary, dict)
        self.assertEqual(len(res.trace.get("INGEST").summary["state_hash"]), 64)
        self.assertEqual(res.trace.get("PLAN").summary["effort"], "medium")
        self.assertEqual(res.trace.get("CHECK").summary["passes"], 1)
        self.assertEqual(res.risk_tier, "low")
        self.assertFalse(res.calibrated)

    def test_high_tier_forces_high_effort(self):
        res = self.pipe().decide(SECRET_STATE, [CHOICE], effort="low", risk_tier="critical")
        self.assertEqual(res.effort, "high")
        self.assertEqual(res.trace.get("PLAN").summary["effort"], "high")

    def test_batch_variant(self):
        rs = self.pipe().decide_batch([(SECRET_STATE, [CHOICE]), ("other state 5", [CHOICE, CHOICE])])
        self.assertEqual([len(r.decisions) for r in rs], [1, 2])
        self.assertNotEqual(rs[0].trace.trace_id, rs[1].trace.trace_id)
        self.assertNotEqual(rs[0].state_hash, rs[1].state_hash)

    def test_invalid_spec_raises(self):
        with self.assertRaises(ValueError):
            self.pipe().decide("s", [QuestionSpec("choice", "q", [])])
        with self.assertRaises(ValueError):
            self.pipe().decide("s", [CHOICE], risk_tier="extreme")

    def test_cache_fallback_and_hits(self):
        p = self.pipe(state_cache=object())
        p.decide(SECRET_STATE, [CHOICE])
        r = p.decide(SECRET_STATE, [CHOICE])
        self.assertEqual(r.trace.get("ENCODE").summary["cache"], "fallback_lru")
        self.assertGreaterEqual(p._fallback.hits, 1)

    # ---- fail closed ----------------------------------------------------------------------
    def test_no_calibration_blocks_execute_above_low(self):
        p = self.pipe(checker=StubChecker())
        low = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="low")
        self.assertIs(low.status, PolicyStatus.EXECUTE)
        self.assertFalse(low.decision.calibrated)
        for tier in ("medium", "high", "critical"):
            r = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier=tier)
            self.assertIs(r.status, PolicyStatus.ABSTAIN, tier)

    def test_hash_mismatch_and_ok(self):
        stub = StubChecker()
        bad = self.pipe(checker=stub, calibration=make_cal("deadbeef"))
        r = bad.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="medium")
        self.assertIs(r.status, PolicyStatus.ABSTAIN)
        self.assertFalse(r.calibrated)
        self.assertTrue(any("model_hash_mismatch" in x for x in r.decision.outcome.reasons))
        empty = self.pipe(checker=stub, calibration=make_cal(""))
        self.assertIs(empty.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="medium").status,
                      PolicyStatus.ABSTAIN)
        good = self.pipe(checker=stub, calibration=make_cal(self.mh))
        r = good.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="medium")
        self.assertTrue(r.calibrated)
        self.assertIs(r.status, PolicyStatus.EXECUTE)
        self.assertIs(good.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="high").status,
                      PolicyStatus.NEEDS_APPROVAL)

    def test_exception_gives_abstain_with_error_recorded(self):
        stub = StubChecker(boom=RuntimeError("kaboom " + SECRET_STATE))
        led = OutOfBandAuditLedger(key=b"k" * 16)
        p = self.pipe(checker=stub, audit=led)
        r = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT)
        self.assertIs(r.status, PolicyStatus.ABSTAIN)
        self.assertEqual(stub.calls, 1)                      # no second scorer, no retry
        self.assertIn("kaboom", r.trace.get("CHECK").summary["error"])
        self.assertEqual(r.trace.get("CHECK").summary["error_type"], "RuntimeError")
        self.assertTrue(r.trace.get("POLICY").summary["statuses"] == ["ABSTAIN"])
        self.assertEqual(r.trace.stage_names(), list(STAGES))
        self.assertTrue(r.decision.error.startswith("CHECK"))
        blob = json.dumps(led.entries[0].payload)
        self.assertNotIn("ZEBRA", blob)
        self.assertNotIn("kaboom", blob)

    def test_encode_failure_abstains(self):
        class Bad:
            def memory_for(self, model, batch):
                raise RuntimeError("cache exploded")

        r = self.pipe(state_cache=Bad(), checker=StubChecker()).decide(SECRET_STATE, [CHOICE], costs=LENIENT)
        self.assertIs(r.status, PolicyStatus.ABSTAIN)
        self.assertIn("cache exploded", r.trace.get("ENCODE").summary["error"])
        self.assertTrue(r.trace.get("DELIBERATE").summary["skipped"])

    # ---- staging / execute -----------------------------------------------------------------
    def test_untrained_never_stages_or_executes(self):
        eng = ActionStagingEngine(ephemeral_key=True)
        p = self.pipe(checker=StubChecker(), staging=eng, training_run_id=None)
        r = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT)
        self.assertIs(r.status, PolicyStatus.EXECUTE)
        self.assertEqual(eng.get_staged_cards(), [])
        self.assertEqual(r.trace.get("STAGE").summary["skipped_reason"], "untrained model")
        with self.assertRaises(DecisionGateError):
            p.execute(r, lambda o, d: "ran")

    def test_stage_and_execute_gates(self):
        eng = ActionStagingEngine(ephemeral_key=True)
        p = self.pipe(checker=StubChecker(), staging=eng, calibration=make_cal(self.mh))
        ran = []
        # EXECUTE at low tier stages a card and runs once
        r = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT)
        self.assertIsNotNone(r.decision.card_id)
        self.assertEqual(len(eng.get_staged_cards()), 1)
        self.assertEqual(p.execute(r, lambda o, d: ran.append(o) or "done"), "done")
        self.assertEqual(ran, ["a"])
        with self.assertRaises(DecisionGateError):
            p.execute(r, lambda o, d: ran.append(o))                  # at most once
        # NEEDS_APPROVAL needs approve()
        r2 = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="high")
        self.assertIs(r2.status, PolicyStatus.NEEDS_APPROVAL)
        with self.assertRaises(DecisionGateError):
            p.execute(r2, lambda o, d: ran.append(o))
        with self.assertRaises(DecisionGateError):
            p.approve(r.decision.decision_id, "bob")                  # EXECUTE decisions are not approvable
        p.approve(r2.decision.decision_id, "alice")
        self.assertEqual(p.execute(r2, lambda o, d: "ok2"), "ok2")
        # ABSTAIN is refused, and editing the status on the object does not help
        r3 = DecisionPipeline(self.model, self.pk, checker=StubChecker(verdict="ESCALATE"), training_run_id="r",
                              staging=eng).decide(SECRET_STATE, [CHOICE], costs=LENIENT)
        self.assertIs(r3.status, PolicyStatus.ABSTAIN)
        self.assertIsNone(r3.decision.card_id)
        with self.assertRaises(DecisionGateError):
            p.execute(r3, lambda o, d: ran.append(o))
        # forging: a decision object the pipeline did not make, or a mutated status
        r4 = p.decide(SECRET_STATE, [CHOICE], costs=CostSpec(cost_wrong=1.0, cost_abstain=1e-9))
        self.assertIs(r4.status, PolicyStatus.ABSTAIN)
        r4.decision.outcome.status = PolicyStatus.EXECUTE
        with self.assertRaises(DecisionGateError):
            p.execute(r4, lambda o, d: ran.append(o))
        self.assertEqual(ran, ["a"])

    # ---- audit ------------------------------------------------------------------------------
    def test_audit_has_hashes_not_raw_state(self):
        led = OutOfBandAuditLedger(key=b"k" * 16)
        p = self.pipe(checker=StubChecker(), audit=led, calibration=make_cal(self.mh))
        r = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT)
        self.assertEqual(len(led.entries), 1)
        payload = led.entries[0].payload
        blob = json.dumps(payload)
        self.assertNotIn("ZEBRA", blob)
        self.assertNotIn("4412", blob)
        self.assertEqual(payload["state_hash"], r.state_hash)
        self.assertEqual(payload["model_hash"], self.mh)
        self.assertEqual(payload["calibration_version"], p.calibration_version())
        self.assertEqual([s["stage"] for s in payload["trace"]["stages"]], list(STAGES[:8]))
        self.assertTrue(r.trace.get("AUDIT").summary["recorded"])
        self.assertTrue(led.verify(led.entries[0])["valid"])

    def test_audit_failure_withholds_authorisation(self):
        class Broken:
            def record(self, kind, payload):
                raise RuntimeError("sink down")

        eng = ActionStagingEngine(ephemeral_key=True)
        p = self.pipe(checker=StubChecker(), audit=Broken(), staging=eng)
        r = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT)
        self.assertIs(r.status, PolicyStatus.ABSTAIN)
        self.assertEqual(eng.get_staged_cards(), [])
        with self.assertRaises(DecisionGateError):
            p.execute(r, lambda o, d: None)

    # ---- revise -------------------------------------------------------------------------------
    def test_revise_at_most_once(self):
        calls = []

        class Reviser(Checker):
            def _pass(self, batch, S, effort, tier, can_revise):
                calls.append((effort, can_revise))
                drafts = super()._pass(batch, S, effort, tier, can_revise)
                if can_revise:
                    for d in drafts:
                        if d is not None:
                            d.verdict = "REVISE"
                return drafts

        p = self.pipe(checker=Reviser(self.model))
        r = p.decide(SECRET_STATE, [CHOICE])
        self.assertEqual(calls, [("medium", True), ("high", False)])
        self.assertEqual(r.trace.get("CHECK").summary["passes"], 2)
        self.assertEqual(r.trace.get("CHECK").summary["revised"], 1)

    # ---- routing through the pipeline ---------------------------------------------------------
    def test_large_option_set_is_routed_and_uncalibrated(self):
        big = QuestionSpec("choice", "pick one", [f"opt{i}" for i in range(300)])
        p = self.pipe(calibration=make_cal(self.mh))
        r = p.decide(SECRET_STATE, [big], costs=LENIENT, risk_tier="medium")
        d = r.decision
        self.assertTrue(d.routed)
        self.assertFalse(d.calibrated)
        self.assertIn("ROUTED", d.checked.flags)
        self.assertEqual(d.checked.extras["joint_label"], "uncalibrated_joint")
        self.assertIs(r.status, PolicyStatus.ABSTAIN)
        self.assertLessEqual(d.checked.extras["routing"]["n_groups"], 255)

    # ---- feedback and drift -------------------------------------------------------------------
    def test_feedback_alarm_downgrades_group(self):
        cal = make_cal(self.mh)
        ledger = OutcomeLedger(OnlineRecalibrator(cal))
        p = self.pipe(checker=StubChecker(), calibration=cal, feedback=ledger)
        first = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="medium")
        self.assertIs(first.status, PolicyStatus.EXECUTE)
        self.assertEqual(len(ledger), 1)
        alarms = []
        ids = [first.decision.decision_id] + [
            p.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="medium").decision.decision_id for _ in range(2)]
        for did in ids:
            e = ledger.get(did)
            e.logits = [6.0, 0.0, 0.0, 0.0]         # confident in option 0
            alarms += p.record_outcome(did, decision_correct=False, label_idx=1)
        self.assertIn("miss_streak", alarms)
        self.assertFalse(ledger.group_calibrated("choice", 3))
        after = p.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="medium")
        self.assertFalse(after.decision.calibrated)
        self.assertIs(after.status, PolicyStatus.ABSTAIN)
        self.assertTrue(any("drift alarm" in x for x in after.decision.outcome.reasons))
        # low tier may still act
        self.assertIs(p.decide(SECRET_STATE, [CHOICE], costs=LENIENT, risk_tier="low").status, PolicyStatus.EXECUTE)


class TestOutcomeLedger(unittest.TestCase):
    def entry(self, i, **kw):
        return DecisionEntry(decision_id=f"d{i}", qtype="choice", k=3, status="EXECUTE", confidence=0.9,
                             predicted_idx=0, logits=[8.0, 0.0, 0.0, 0.0], **kw)

    def test_bounded_ring_buffer(self):
        led = OutcomeLedger(max_entries=5)
        for i in range(12):
            led.record_decision(self.entry(i))
        self.assertEqual(len(led), 5)
        self.assertEqual(led.evicted, 7)
        self.assertEqual([e.decision_id for e in led.entries], [f"d{i}" for i in range(7, 12)])
        with self.assertRaises(KeyError):
            led.get("d0")
        with self.assertRaises(KeyError):
            led.record_outcome("d0", True)
        with self.assertRaises(ValueError):
            led.record_decision(self.entry(11))

    def test_outcomes_and_jsonl_round_trip(self):
        led = OutcomeLedger(max_entries=10)
        for i in range(4):
            led.record_decision(self.entry(i))
        led.record_outcome("d0", True)
        led.record_outcome("d1", False, label_idx=2)
        led.record_outcome("d2", None)
        with self.assertRaises(ValueError):
            led.record_outcome("d0", True)
        self.assertEqual((led.n_outcomes, led.n_correct), (2, 1))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "o.jsonl")
            led.to_jsonl(path)
            back = OutcomeLedger.from_jsonl(path, max_entries=10)
            self.assertEqual([e.to_dict() for e in back.entries], [e.to_dict() for e in led.entries])
            self.assertEqual(back.get("d1").decision_correct, False)
            self.assertIsNone(back.get("d2").decision_correct)
            self.assertEqual((back.n_outcomes, back.n_correct), (2, 1))
            small = OutcomeLedger.from_jsonl(path, max_entries=2)
            self.assertEqual(len(small), 2)

    def test_recalibrator_bridge_and_unobserved(self):
        cal = make_cal("h")
        rc = OnlineRecalibrator(cal)
        led = OutcomeLedger(rc)
        for i in range(4):
            led.record_decision(self.entry(i))
        self.assertEqual(led.record_outcome("d0", True), [])          # correct: label is the predicted index
        self.assertEqual(led.record_outcome("d1", False), [])         # wrong with no label: not observable
        self.assertEqual(led.unobserved, 1)
        self.assertTrue(led.group_calibrated("choice", 3))
        out = []
        for i in (2, 3):
            out += led.record_outcome(f"d{i}", False, label_idx=1)
        self.assertEqual(out, [])                                      # streak of 2 (d0 was a hit)
        led.record_decision(self.entry(4))
        self.assertIn("miss_streak", led.record_outcome("d4", False, label_idx=1))
        self.assertFalse(led.group_calibrated("choice", 3))
        self.assertTrue(led.group_calibrated("choice", 2))
        self.assertEqual(len(led.alarm_log), 1)


if __name__ == "__main__":
    unittest.main()
