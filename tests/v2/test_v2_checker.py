"""Checker (plan section 6): verdict logic, stages and integration on a briefly trained r2-tiny."""

from __future__ import annotations

import json
import unittest

import torch

from rhen.research.tasks import generate_split
from rhen.v2.checker import (
    CheckedDecision, Checker, CorrectnessHead, RuleChecker, correctness_features, decide_verdict,
    decide_verdict_ex, decode_keys, depth_trace, evidence_spans, fit_correctness_head,
)
from rhen.v2.config import preset
from rhen.v2.data.schema import from_v1_example
from rhen.v2.model import Rhen2Model
from rhen.v2.packing import QuestionSpec, Rhen2Example, Rhen2Packer
from rhen.v2.workspace import Interventions, LoopPlan


class FakeCal:
    """A calibration object with fixed thresholds, standing in for rhen/v2/calibrate.py."""

    def __init__(self, calibrated=True, tau=None, tau_review=None, delta_ev=0.0, conf_q=None, ood=None, temp=1.0):
        self.calibrated = calibrated
        self._tau = tau if tau is not None else {"low": 0.5, "medium": 0.7, "high": 0.9, "critical": 0.99}
        self._tau_review, self._delta, self._q, self._ood, self._temp = tau_review, delta_ev, conf_q, ood or [], temp

    tau_review = property(lambda s: s._tau_review)
    delta_ev = property(lambda s: s._delta)
    halt_threshold = property(lambda s: None)

    def temperature(self, qtype, k):
        return self._temp

    def conformal_q(self, qtype, k):
        return self._q

    def tau_act(self, tier):
        return self._tau.get(tier)

    def isotonic(self, qtype, p):
        return p

    def ood_flags(self, qtype, k, energy, p_null, q_vec):
        return list(self._ood)


class TestVerdictLogic(unittest.TestCase):
    cal = FakeCal()

    def v(self, flags=(), p=0.95, null=False, empty=False, tier="low", cal="default", revise=True):
        cal = self.cal if isinstance(cal, str) else cal
        return decide_verdict(list(flags), p, null, empty, tier, cal, revise)

    def test_rule1_conflict_confirms_even_with_low_p_and_null(self):
        self.assertEqual(self.v(["RULE_CONFLICT"], p=0.0, null=True, empty=True), "CONFIRM")
        self.assertEqual(self.v(["RULE_CONFLICT", "UNGROUNDED", "UNSTABLE"]), "CONFIRM")

    def test_rule2_null_or_empty_conformal_escalates(self):
        self.assertEqual(self.v(null=True), "ESCALATE")
        self.assertEqual(self.v(empty=True), "ESCALATE")
        self.assertEqual(self.v(["UNSTABLE"], null=True), "ESCALATE")      # rule 2 before rule 4

    def test_rule3_ungrounded_prior_driven_escalate_before_revise(self):
        for f in ("UNGROUNDED", "PRIOR_DRIVEN"):
            self.assertEqual(self.v([f]), "ESCALATE")
            self.assertEqual(self.v([f, "UNSTABLE"], revise=True), "ESCALATE")
            self.assertEqual(decide_verdict_ex([f, "UNSTABLE"], 0.99, False, False, "low", self.cal, True)[1], 3)

    def test_ood_escalates_above_low_only(self):
        self.assertEqual(self.v(["OOD"], tier="medium"), "ESCALATE")
        self.assertEqual(self.v(["OOD"], tier="low"), "CONFIRM")

    def test_rule4_revise_then_escalate(self):
        for f in ("UNSTABLE", "MARGINAL"):
            self.assertEqual(self.v([f], revise=True), "REVISE")
            self.assertEqual(self.v([f], revise=False), "ESCALATE")
        self.assertEqual(self.v(["UNSTABLE", "MARGINAL"], p=0.0), "REVISE")   # before the p threshold

    def test_rule5_threshold_per_tier(self):
        self.assertEqual(self.v(p=0.5, tier="low"), "CONFIRM")
        self.assertEqual(self.v(p=0.49, tier="low"), "ESCALATE")
        self.assertEqual(self.v(p=0.8, tier="medium"), "CONFIRM")
        self.assertEqual(self.v(p=0.8, tier="high"), "ESCALATE")
        self.assertEqual(self.v(p=0.995, tier="critical"), "CONFIRM")

    def test_missing_tau_escalates_when_calibrated(self):
        cal = FakeCal(tau={"low": None})
        self.assertEqual(self.v(p=1.0, tier="low", cal=cal), "ESCALATE")

    def test_uncalibrated_never_confirms_above_low(self):
        for cal in (None, FakeCal(calibrated=False)):
            for tier in ("medium", "high", "critical"):
                self.assertEqual(self.v(p=1.0, tier=tier, cal=cal), "ESCALATE")
                self.assertEqual(self.v(["RULE_CONFLICT"], tier=tier, cal=cal), "ESCALATE")
            self.assertEqual(self.v(p=1.0, tier="low", cal=cal), "CONFIRM")
            self.assertEqual(self.v(["UNGROUNDED"], tier="low", cal=cal), "ESCALATE")

    def test_rule_violation_escalates(self):
        self.assertEqual(self.v(["RULE_VIOLATION"]), "ESCALATE")

    def test_unknown_tier_rejected(self):
        with self.assertRaises(ValueError):
            self.v(tier="extreme")


class TestRuleChecker(unittest.TestCase):
    def spec(self, text, options=("sensor_a", "sensor_b")):
        return QuestionSpec("choice", text, list(options))

    def test_larger_and_smaller(self):
        st = json.dumps({"sensor_a": 3, "sensor_b": 9.5})
        self.assertEqual(RuleChecker.numeric_comparison("Which reading is larger?", ["sensor_a", "sensor_b"], st), "sensor_b")
        self.assertEqual(RuleChecker.numeric_comparison("Pick the lower value", ["sensor_a", "sensor_b"], st), "sensor_a")
        self.assertEqual(RuleChecker.numeric_comparison("Which has more?", ["sensor_a", "sensor_b"], st), "sensor_b")

    def test_not_applicable(self):
        st = json.dumps({"sensor_a": 3, "sensor_b": 3})
        o = ["sensor_a", "sensor_b"]
        self.assertIsNone(RuleChecker.numeric_comparison("Which reading is larger?", o, st))           # tie
        self.assertIsNone(RuleChecker.numeric_comparison("Which is larger or smaller?", o, json.dumps({"sensor_a": 1, "sensor_b": 2})))
        self.assertIsNone(RuleChecker.numeric_comparison("Which is larger?", ["x", "y"], st))          # not keys
        self.assertIsNone(RuleChecker.numeric_comparison("Which is larger?", o, json.dumps({"sensor_a": "1", "sensor_b": 2})))
        self.assertIsNone(RuleChecker.numeric_comparison("Which is larger?", o, "not json"))

    def test_check_conflict_and_agreement(self):
        st = json.dumps({"sensor_a": 3, "sensor_b": 9})
        sp = self.spec("Pick the larger value")
        r = RuleChecker.check(sp, st, "sensor_a")
        self.assertEqual((r.flag, r.answer), ("RULE_CONFLICT", "sensor_b"))
        self.assertIsNone(RuleChecker.check(sp, st, "sensor_b").flag)
        self.assertEqual(RuleChecker.check(sp, st, None).flag, "RULE_CONFLICT")       # null answer is also overridden

    def test_extract_substring(self):
        sp = QuestionSpec("extract", "Find the city")
        self.assertIsNone(RuleChecker.check(sp, "city Lima", "Lima").flag)
        self.assertEqual(RuleChecker.check(sp, "city Lima", "Oslo").flag, "RULE_VIOLATION")


class TestEvidenceSpans(unittest.TestCase):
    def test_merge_and_limit(self):
        ls = 40
        ev = torch.zeros(ls)
        ev[[2, 3, 4]] = 0.2
        ev[8] = 0.3
        ev[10] = 0.15
        mask = torch.ones(ls, dtype=torch.bool)
        offs = [(0, 0)] + [(3 * i, 3 * i + 2) for i in range(1, ls)]
        chars, toks = evidence_spans(ev, mask, offs)
        self.assertEqual(toks[0], (2, 4))                       # heaviest first (0.6)
        self.assertEqual(chars[0][:2], (6, 14))
        self.assertAlmostEqual(chars[0][2], 0.6, places=5)
        self.assertIn((8, 8), toks)
        self.assertLessEqual(len(chars), 5)

    def test_flat_evidence_falls_back_to_top_token(self):
        ev = torch.full((6,), 0.1)
        ev[3] = 0.11
        chars, toks = evidence_spans(ev, torch.ones(6, dtype=torch.bool), [(0, 0)] + [(i, i + 1) for i in range(1, 6)])
        self.assertEqual(toks, [(3, 3)])


# ------------------------------------------------------------------------------ integration


def _train(cfg, steps=150):
    torch.manual_seed(0)
    model, pk = Rhen2Model(cfg), Rhen2Packer(cfg)
    ex = [from_v1_example(e) for e in generate_split("train", 64, seed=0, families=("lookup", "compare"))]
    batch = pk.pack(ex)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        out = model(batch)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model, pk


class TestCheckerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = preset("r2-tiny")
        cls.model, cls.pk = _train(cls.cfg)
        cls.examples = [from_v1_example(e) for e in generate_split("train", 16, seed=0, families=("lookup", "compare"))]
        cls.batch = cls.pk.pack(cls.examples)

    def test_all_efforts_and_tiers(self):
        for effort in ("low", "medium", "high"):
            res = Checker(self.model).check(self.batch, effort=effort)
            self.assertEqual(len(res), 16)
            for r in res:
                self.assertIsInstance(r, CheckedDecision)
                self.assertIn(r.verdict, ("CONFIRM", "ESCALATE"))
                self.assertIn("__null__", r.probabilities)
                self.assertAlmostEqual(sum(r.probabilities.values()), 1.0, places=4)
                self.assertFalse(r.calibrated)
                self.assertGreaterEqual(r.loops_run, 1)
        res = Checker(self.model).check(self.batch, effort="low", risk_tier="high")
        self.assertTrue(all(r.effort == "high" for r in res))

    def test_uncalibrated_never_confirms_above_low(self):
        for tier in ("medium", "high", "critical"):
            res = Checker(self.model).check(self.batch, risk_tier=tier)
            self.assertTrue(all(r.verdict == "ESCALATE" for r in res), tier)
        res = Checker(self.model).check(self.batch, risk_tier="low")
        self.assertTrue(any(r.verdict == "CONFIRM" for r in res))

    def test_evidence_and_spans_inside_state_text(self):
        S = self.model.encode_state(self.batch.state_ids, self.batch.state_mask, self.batch.state_num)
        with torch.no_grad():
            out = self.model.deliberate(S, self.batch, LoopPlan.effort(self.cfg, "high"))
        ev = out.evidence
        real = self.batch.state_mask.repeat_interleave(self.batch.max_questions, 0)
        self.assertTrue(torch.allclose(ev.sum(-1), torch.ones(ev.shape[0]), atol=1e-4))
        self.assertLess(float(ev.masked_fill(real, 0.0).max()), 1e-6)       # nothing on padding
        for r, ex_i in zip(Checker(self.model).check(self.batch, effort="high"), range(16)):
            text = self.batch.state_texts[ex_i]
            self.assertTrue(1 <= len(r.evidence_spans) <= 5)
            for a, z, w in r.evidence_spans:
                self.assertTrue(0 <= a < z <= len(text))
                self.assertGreater(w, 0.0)

    def test_answers_match_question_types(self):
        res = Checker(self.model).check(self.batch, effort="high")
        for r, ex in zip(res, self.examples):
            labels = ex.questions[0].options
            self.assertTrue(r.answer is None or r.answer in labels)
            self.assertEqual(set(r.probabilities) - {"__null__"}, set(labels))

    def test_depth_trace_and_features(self):
        S = self.model.encode_state(self.batch.state_ids, self.batch.state_mask, self.batch.state_num)
        tr = depth_trace(self.model, S, self.batch, 3)
        n = 16
        self.assertEqual((len(tr.keys), len(tr.keys[0]), tuple(tr.q_vec.shape[:2])), (3, n, (3, n)))
        with torch.no_grad():
            out = self.model.deliberate(S, self.batch, LoopPlan(n_loops=3, min_loops=3))
        feats = correctness_features(self.model, self.batch, out, tr)
        head = CorrectnessHead(self.cfg.d_model)
        p = head(feats)
        self.assertEqual(tuple(p.shape), (n,))
        self.assertTrue(bool(((p >= 0) & (p <= 1)).all()))
        self.assertEqual(decode_keys(out, self.batch, self.cfg.max_span), tr.keys[-1])

    def test_head_fits_and_is_used(self):
        S = self.model.encode_state(self.batch.state_ids, self.batch.state_mask, self.batch.state_num)
        out = self.model.deliberate(S, self.batch, LoopPlan.effort(self.cfg, "medium"))
        feats = correctness_features(self.model, self.batch, out, depth_trace(self.model, S, self.batch, 3))
        labels = (feats[:, 2 * self.cfg.d_model] > feats[:, 2 * self.cfg.d_model].median()).float()   # top-1 prob split
        head = CorrectnessHead(self.cfg.d_model)
        torch.manual_seed(1)
        losses = fit_correctness_head(head, feats, labels, epochs=150, lr=3e-3)
        self.assertLess(losses[-1], losses[0])
        res = Checker(self.model, head=head).check(self.batch, effort="medium")
        self.assertEqual(len(res), 16)
        self.assertTrue(all(0.0 <= r.confidence <= 1.0 for r in res))

    def test_c3a_evidence_masking_lowers_chosen_probability(self):
        """Honest measurement on the briefly trained tiny model (see module report)."""
        S = self.model.encode_state(self.batch.state_ids, self.batch.state_mask, self.batch.state_num)
        plan = LoopPlan.effort(self.cfg, "high")
        with torch.no_grad():
            out = self.model.deliberate(S, self.batch, plan)
            keys = decode_keys(out, self.batch, self.cfg.max_span)
            mask = torch.zeros(out.probs.shape[0], S.shape[1], dtype=torch.bool)
            for i in range(mask.shape[0]):
                _, toks = evidence_spans(out.evidence[i], self.batch.state_mask[i], self.batch.state_offsets[i])
                for a, z in toks:
                    mask[i, a:z + 1] = True
            masked = self.model.deliberate(S, self.batch, plan, Interventions(memory_mask=mask))
        drops = []
        for i, k in enumerate(keys):
            if k >= 0:
                drops.append(float(out.probs[i, k] - masked.probs[i, k]))
        self.assertGreater(len(drops), 8)
        mean_drop = sum(drops) / len(drops)
        print(f"\n[checker] C3a mean chosen-probability drop on tiny model: {mean_drop:.3f}, "
              f"fraction dropping: {sum(d > 0 for d in drops) / len(drops):.2f}")
        self.assertGreater(mean_drop, 0.0)

    def test_c3_flags_run_and_are_well_formed(self):
        res = Checker(self.model).check(self.batch, effort="high")
        known = {"UNGROUNDED", "PRIOR_DRIVEN", "UNSTABLE", "MARGINAL", "RULE_CONFLICT", "OOD", "RULE_VIOLATION"}
        for r in res:
            self.assertTrue(set(r.flags) <= known)
            self.assertEqual(len(r.flags), len(set(r.flags)))

    def test_rule_checker_overrides_wrong_model_answer(self):
        torch.manual_seed(3)
        weak = Rhen2Model(self.cfg).eval()                      # untrained: wrong about half the time
        exs = [from_v1_example(e) for e in generate_split("train", 24, seed=5, families=("compare",))]
        batch = self.pk.pack(exs)
        res = Checker(weak).check(batch, effort="low", risk_tier="low")
        conflicts = 0
        for r, e in zip(res, generate_split("train", 24, seed=5, families=("compare",))):
            truth = e.questions[0].options[e.questions[0].label]
            self.assertEqual(r.answer, truth)
            self.assertEqual(r.verdict, "CONFIRM")
            if "RULE_CONFLICT" in r.flags:
                conflicts += 1
                self.assertNotEqual(r.extras["model_answer"], truth)
        self.assertGreater(conflicts, 0)

    def test_extract_range_multi_noul_score_decode(self):
        state = {"name": "Ada", "city": "Lima", "amount": 42.0}
        qs = [
            QuestionSpec("extract", "What is the city?"),
            QuestionSpec("range", "What is the amount?", lo=0.0, hi=100.0),
            QuestionSpec("multi", "Which fields are present?", ["name", "city", "zip"]),
            QuestionSpec("noul", "Is the amount above 10?"),
            QuestionSpec("score", "Rate the risk", ["low", "mid", "high"]),
        ]
        batch = self.pk.pack([Rhen2Example(state, qs)])
        res = Checker(Rhen2Model(self.cfg).eval()).check(batch, effort="high")
        self.assertEqual(len(res), 5)
        ext, rng, mul, nou, sco = res
        self.assertTrue(ext.answer is None or ext.answer in batch.state_texts[0])
        self.assertTrue(0.0 <= rng.answer <= 100.0)
        lo, hi = rng.extras["interval"]
        self.assertLessEqual(lo, hi)
        self.assertIsInstance(mul.answer, list)
        self.assertTrue(set(mul.answer) <= {"name", "city", "zip"})
        self.assertIn(nou.answer, (True, False, None))
        self.assertIn(sco.answer, ("low", "mid", "high", None))
        self.assertTrue(0.0 <= sco.extras["expected_level"] <= 2.0)
        self.assertEqual(len(Checker(Rhen2Model(self.cfg).eval()).check(batch, effort="low")), 5)

    def test_invalid_questions_skipped(self):
        two = self.pk.pack([
            Rhen2Example({"a": 1}, [QuestionSpec("noul", "q1"), QuestionSpec("noul", "q2")]),
            Rhen2Example({"a": 2}, [QuestionSpec("noul", "q1")]),
        ])
        self.assertEqual(len(Checker(self.model).check(two, effort="low")), 3)


class TestCalibratedPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = preset("r2-tiny")
        torch.manual_seed(0)
        cls.model = Rhen2Model(cls.cfg).eval()
        cls.pk = Rhen2Packer(cls.cfg)
        cls.batch = cls.pk.pack([from_v1_example(e) for e in generate_split("train", 8, seed=1, families=("lookup",))])

    def test_calibrated_flag_and_confirm_at_high_tier(self):
        res = Checker(self.model, FakeCal(tau={"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0}, conf_q=1.0)).check(
            self.batch, risk_tier="high")
        self.assertTrue(all(r.calibrated for r in res))
        for r in res:
            if r.answer is not None and not r.flags:
                self.assertEqual(r.verdict, "CONFIRM")
            self.assertEqual(len(r.conformal_set), len(r.probabilities))      # q=1.0 keeps every label

    def test_empty_conformal_set_escalates(self):
        res = Checker(self.model, FakeCal(conf_q=-1.0, tau={"low": 0.0})).check(self.batch)
        self.assertTrue(all(r.conformal_set == [] and r.verdict == "ESCALATE" for r in res))

    def test_unreachable_tau_escalates(self):
        res = Checker(self.model, FakeCal(tau={"low": 2.0})).check(self.batch)
        self.assertTrue(all(r.verdict == "ESCALATE" for r in res))

    def test_ood_flag_escalates_above_low(self):
        cal = FakeCal(ood=["OOD"], tau={"low": 0.0, "medium": 0.0})
        self.assertTrue(all("OOD" in r.flags and r.verdict == "ESCALATE"
                            for r in Checker(self.model, cal).check(self.batch, risk_tier="medium")))
        low = Checker(self.model, cal).check(self.batch, risk_tier="low")
        self.assertTrue(all("OOD" in r.flags for r in low))

    def test_temperature_changes_probabilities(self):
        a = Checker(self.model, FakeCal(temp=1.0)).check(self.batch, effort="low")
        b = Checker(self.model, FakeCal(temp=5.0)).check(self.batch, effort="low")
        self.assertLess(max(b[0].probabilities.values()), max(a[0].probabilities.values()) + 1e-9)
        self.assertNotAlmostEqual(max(a[0].probabilities.values()), max(b[0].probabilities.values()), places=4)

    def test_review_threshold_gates_c3_in_medium(self):
        # tau_review = 0 -> p_correct is never below it -> no C3 flags at medium; high always runs C3
        cal = FakeCal(tau_review=0.0, tau={"low": 0.0})
        res = Checker(self.model, cal).check(self.batch, effort="medium")
        self.assertTrue(all(not ({"UNGROUNDED", "PRIOR_DRIVEN", "UNSTABLE", "MARGINAL"} & set(r.flags)) for r in res))

    def test_revise_reruns_once_at_next_effort(self):
        cal = FakeCal(tau={"low": 0.0})
        res = Checker(self.model, cal).check(self.batch, effort="low")
        for r in res:
            self.assertIn(r.effort, ("low", "medium"))
            if r.effort == "medium":
                self.assertEqual(r.extras["revised_from"], "low")


if __name__ == "__main__":
    unittest.main()
