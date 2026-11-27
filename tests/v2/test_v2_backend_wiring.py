"""backend='v2' wiring of RhenEngine, UnifiedAIOrchestrator and ForecastAwareDecider (plan task T-603, 9.5)."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import unittest

import torch

from rhen.decision_pipeline import DecisionPolicy, ForecastAwareDecider, PolicyStatus
from rhen.forecasting import NaiveQuantileForecaster
from rhen.research.tasks import generate_split
from rhen.rhen import (
    AbstentionVerdict, Cascade, CascadeResult, Choice, ChoiceResult, Extract, ExtractResult, MultiChoice,
    MultiChoiceResult, Noul, NoulResult, Range, RangeResult, RhenEngine, Score, ScoreResult, decide,
)
from rhen.unified_orchestrator import UnifiedAIOrchestrator
from rhen.v2.calibrate import Calibration
from rhen.v2.checker import CheckedDecision
from rhen.v2.compat import V2Backend
from rhen.v2.config import preset
from rhen.v2.data.schema import from_v1_example
from rhen.v2.model import Rhen2Model
from rhen.v2.packing import Rhen2Packer
from rhen.v2.pipeline import DecisionPipeline, weights_hash
from rhen.v2.policy import NULL, CostSpec

STATE = {"name": "Ada", "city": "Lima", "account": 4412, "reading": 17.5}
PRIMITIVES = [
    Noul("The account number is above 4000"),
    Choice("Which city is on file?", ["Lima", "Oslo", "Cairo"]),
    Score("Rate the account", ["l0", "l1", "l2", "l3", "l4"], 0.0, 10.0),
    MultiChoice("Which fields are present?", ["name", "city", "fax"]),
    Range("Estimate the reading", 0.0, 100.0),
    Extract("city", "city on file"),
]
RESULT_TYPES = [NoulResult, ChoiceResult, ScoreResult, MultiChoiceResult, RangeResult, ExtractResult]


def tiny_untrained_backend(**kw) -> V2Backend:
    torch.manual_seed(0)
    return V2Backend.from_model(Rhen2Model(preset("r2-tiny")), None, **kw)


class FixedChecker:
    """Returns a fixed confirmed answer with confidence ``conf`` for every question; marked calibrated."""

    def __init__(self, conf: float) -> None:
        self.conf = conf
        self.cal = None

    def check(self, batch, effort="medium", risk_tier="low"):
        out = []
        for b in range(batch.batch_size):
            for m in range(batch.max_questions):
                if not bool(batch.q_valid[b, m]):
                    continue
                labels = batch.specs[b][m].options
                p = {labels[0]: self.conf}
                for lab in labels[1:]:
                    p[lab] = (1 - self.conf) / len(labels)
                p[NULL] = max(0.0, 1.0 - sum(p.values()))
                out.append(CheckedDecision(labels[0], p, self.conf, [labels[0]], "CONFIRM", [], [], 2, effort, True))
        return out


def calibration_for(model_hash: str, n: int = 600) -> Calibration:
    return Calibration(
        model_hash=model_hash, data_hash="d", created="now",
        temperature_table={"choice": {"2": 1.0, "4": 1.0, "8": 1.0, "all": 1.0}},
        isotonic_table={"choice": {}},
        conformal_table={"choice": {"2": 0.5, "4": 0.05, "8": 0.5}},
        tau_act_table={"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0},
        ood={"energy": {"choice": {"all": 1e9}}},
        n_calibration={"choice": {"2": n, "4": n, "8": n}}, extras={"alpha": 0.1},
    )


class TestDefaultsStayV1(unittest.TestCase):
    def test_default_backend_is_v1(self):
        for fn in (RhenEngine.__init__, UnifiedAIOrchestrator.__init__, ForecastAwareDecider.__init__):
            self.assertEqual(inspect.signature(fn).parameters["backend"].default, "v1")
            self.assertIs(inspect.signature(fn).parameters["backend"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(inspect.signature(decide).parameters["backend"].default)

    def test_v1_engine_is_untouched(self):
        from rhen.config import EOSConfig

        cfg = EOSConfig(hidden_size=64, intermediate_size=128, num_pre_layers=1, num_recurrent_layers=1,
                        num_post_layers=1, max_recurrent_loops=1, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=512)
        eng = RhenEngine(config=cfg)
        self.assertEqual(eng.backend, "v1")
        self.assertIsNone(eng._v2)
        res = eng.choice("The city is Lima.", "Which city?", ["Lima", "Oslo"])
        self.assertIsInstance(res, ChoiceResult)
        self.assertNotIn("backend", res.meta_reasoning_context or {})
        with self.assertRaises(ValueError):
            RhenEngine(config=cfg, v2_checkpoint="x.pt")
        with self.assertRaises(ValueError):
            RhenEngine(config=cfg, backend="v3")

    def test_import_rhen_does_not_import_v2(self):
        code = ("import sys, rhen\n"
                "assert not [m for m in sys.modules if m == 'rhen.v2' or m.startswith('rhen.v2.')], 'v2 imported'\n"
                "assert 'V2Backend' in dir(rhen)\n"
                "from rhen import V2Backend, DecisionPipeline, Rhen2Model, Rhen2Config\n"
                "assert 'rhen.v2.compat' in sys.modules and V2Backend.__name__ == 'V2Backend'\n"
                "from rhen.v2 import Rhen2Config as C2\n"
                "assert C2 is Rhen2Config\n")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        with self.assertRaises(AttributeError):
            getattr(__import__("rhen"), "NoSuchThing")


class TestUntrainedV2(unittest.TestCase):
    def test_all_six_primitives_return_v1_types_and_abstain_above_low(self):
        backend = tiny_untrained_backend()
        eng = RhenEngine(backend="v2", v2_backend=backend)
        self.assertEqual(eng.backend, "v2")
        for tier in ("medium", "high", "critical"):
            results = eng.evaluate(STATE, PRIMITIVES, risk_tier=tier)
            self.assertEqual([type(r) for r in results], RESULT_TYPES)
            for r in results:
                self.assertIsInstance(r.abstention_verdict, AbstentionVerdict)
                self.assertTrue(r.should_abstain, (tier, type(r).__name__))
                self.assertFalse(r.should_answer)
                self.assertEqual(r.meta_reasoning_context["backend"], "rhen2")
                self.assertTrue(r.meta_reasoning_context["untrained"])
                self.assertFalse(r.meta_reasoning_context["calibrated"])
                self.assertTrue(r.abstention_verdict.reasons)
                r.to_json()                                   # serialisable like a v1 result

    def test_single_primitive_methods_and_fields(self):
        eng = RhenEngine(backend="v2", v2_backend=tiny_untrained_backend(risk_tier="medium"), effort="low")
        self.assertEqual(eng.effort, "low")
        n = eng.noul(STATE, "The city is Lima")
        self.assertIsInstance(n, NoulResult)
        self.assertAlmostEqual(n.p_true + n.meta_reasoning_context["p_false"]
                               + n.meta_reasoning_context["null_probability"], 1.0, places=4)
        c = eng.choice(STATE, "Which city?", ["Lima", "Oslo", "Cairo"])
        self.assertEqual(set(c.probabilities), {"Lima", "Oslo", "Cairo"})
        self.assertEqual(c.choice_label, c.probabilities and max(c.probabilities, key=c.probabilities.get))
        s = eng.score(STATE, "Rate", ["a", "b", "c"], 0.0, 5.0)
        self.assertTrue(0.0 <= s.score <= 5.0)
        self.assertEqual(len(s.level_probabilities), 3)
        s2 = eng.score(STATE, "Rate", [f"l{i}" for i in range(80)], 0.0, 5.0)    # more levels than the model has
        self.assertEqual(s2.meta_reasoning_context["score_via"], "range")
        r = eng.range(STATE, "How much", 10.0, 20.0)
        self.assertTrue(10.0 <= r.interval_low <= r.expected_value <= r.interval_high <= 20.0)
        e = eng.extract(STATE, "city", "city on file")
        self.assertIsInstance(e.extracted_value, str)
        m = eng.multichoice(STATE, "fields", ["name", "city"], threshold=0.9)
        self.assertEqual(m.selected_options, [o for o, p in m.probabilities.items() if p > 0.9])

    def test_unsupported_features_fail_loudly(self):
        eng = RhenEngine(backend="v2", v2_backend=tiny_untrained_backend())
        with self.assertRaises(NotImplementedError):
            eng.evaluate(STATE, [PRIMITIVES[1]], delta_state="x")
        with self.assertRaises(NotImplementedError):
            eng.evaluate(STATE, [PRIMITIVES[1]], latent_tensors={"a": torch.zeros(1)})
        with self.assertRaises(NotImplementedError):
            eng.evaluate(STATE, [PRIMITIVES[1]], num_loops=2)
        with self.assertRaises(TypeError):
            eng.evaluate(STATE, [object()])

    def test_cascade_confidence_is_uncalibrated_joint(self):
        eng = RhenEngine(backend="v2", v2_backend=tiny_untrained_backend())
        res = eng.cascade(STATE, Cascade(stages=[PRIMITIVES[0], PRIMITIVES[1]], id="c"), )
        self.assertIsInstance(res, CascadeResult)
        self.assertEqual(len(res), 2)
        prod = res.stages[0].confidence * res.stages[1].confidence
        self.assertAlmostEqual(res.confidence, prod, places=6)
        self.assertEqual(res.meta_reasoning_context["confidence_label"], "uncalibrated_joint")
        out = eng.cascade(STATE, [lambda ctx: PRIMITIVES[0], lambda ctx: Choice("Pick", ["x", "y"])])
        self.assertEqual(len(out), 2)

    def test_module_level_decide_with_backend_v2(self):
        eng = RhenEngine(backend="v2", v2_backend=tiny_untrained_backend())
        res = decide(STATE, Choice("Which city?", ["Lima", "Oslo"]), engine=eng, backend="v2", risk_tier="high")
        self.assertIsInstance(res, ChoiceResult)
        self.assertTrue(res.should_abstain)
        verdict = decide(STATE, Choice("Which city?", ["Lima", "Oslo"]), engine=eng, on_abstain="verdict",
                         risk_tier="high")
        self.assertIsInstance(verdict, AbstentionVerdict)
        with self.assertRaises(ValueError):
            decide(STATE, Noul("x"), engine=object.__new__(RhenEngine), backend="v2")
        with self.assertRaises(ValueError):
            decide(STATE, Noul("x"), backend="v9")

    def test_orchestrator_v2_keeps_staging_and_audit_behaviour(self):
        orch = UnifiedAIOrchestrator(None, backend="v2", v2_backend=tiny_untrained_backend())
        self.assertEqual(orch.backend, "v2")
        out = orch.unified_decide_sync(STATE, [PRIMITIVES[1]], risk_tier="medium", auto_stage_actions=True)
        self.assertEqual(out["backend"], "v2")
        self.assertIsInstance(out["results"][0], ChoiceResult)
        self.assertTrue(out["results"][0].should_abstain)
        self.assertEqual(len(out["staged_actions"]), 1)             # v1 staging path unchanged
        self.assertIn("meta_cognitive", out)
        self.assertEqual(len(orch.get_staged_actions()), 1)
        with self.assertRaises(NotImplementedError):
            asyncio.run(orch.unified_decide(STATE, [PRIMITIVES[1]], speculative_branches={0: [PRIMITIVES[0]]}))
        with self.assertRaises(ValueError):
            UnifiedAIOrchestrator(None, backend="v7")


class TestTrainedV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = preset("r2-tiny")
        cls.model, pk = Rhen2Model(cfg), Rhen2Packer(cfg)
        cls.examples = generate_split("train", 24, seed=0, families=("lookup",))
        batch = pk.pack([from_v1_example(e) for e in cls.examples])
        opt = torch.optim.AdamW(cls.model.parameters(), lr=3e-3, weight_decay=0.0)
        cls.model.train()
        for _ in range(100):
            out = cls.model(batch)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(cls.model.parameters(), 1.0)
            opt.step()
        cls.model.eval()

    def test_lookup_answered_correctly_through_rhen_engine(self):
        backend = V2Backend.from_model(self.model, None, training_run_id="wiring-test", effort="high")
        eng = RhenEngine(backend="v2", v2_backend=backend)
        right = 0
        for ex in self.examples[:8]:
            q = ex.questions[0]
            res = eng.choice(ex.state, q.question, q.options)
            self.assertIsInstance(res, ChoiceResult)
            right += res.choice_label == q.options[int(q.label)]
        self.assertGreaterEqual(right, 7, right)

    def test_checkpoint_round_trip_builds_the_same_backend(self):
        import os
        import tempfile

        from rhen.v2.io import save_checkpoint

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            save_checkpoint(path, self.model, self.model.cfg, {"training_run_id": "run-x"})
            eng = RhenEngine(backend="v2", v2_checkpoint=path)
            self.assertEqual(eng._v2.pipeline.training_run_id, "run-x")
            self.assertEqual(eng._v2.pipeline.model_hash, weights_hash(self.model))
            ex = self.examples[0]
            q = ex.questions[0]
            self.assertEqual(eng.choice(ex.state, q.question, q.options).choice_label,
                             q.options[int(q.label)])
            eng2 = RhenEngine.from_effort("low", backend="v2", v2_checkpoint=path)
            self.assertEqual(eng2.backend, "v2")


class TestForecastDeciderV2(unittest.TestCase):
    HIST = [10, 10.4, 9.8, 10.2, 10.1, 9.9, 10.3, 10.0, 10.2, 9.7, 10.1, 10.0]

    def decider(self, conf: float, **kw):
        torch.manual_seed(0)
        model = Rhen2Model(preset("r2-tiny"))
        mh = weights_hash(model)
        pipe = DecisionPipeline(model, Rhen2Packer(model.cfg), calibration=calibration_for(mh),
                                checker=FixedChecker(conf), training_run_id="run-1", model_hash=mh)
        return ForecastAwareDecider(None, NaiveQuantileForecaster(), backend="v2", pipeline=pipe, **kw)

    def run_one(self, decider, threshold):
        return decider.decide("Stock check.", "Act?", ["act", "hold"], {"load": self.HIST}, 3,
                              thresholds={"load": threshold})

    def test_cost_wrong_rises_with_breach_upper_bound_and_decisions_follow(self):
        d = self.decider(0.97)
        low, mid, high = (self.run_one(d, t) for t in (13.0, 10.6, 10.4))
        ups = [x.features["breach_prob_any_step_upper"] for x in (low, mid, high)]
        self.assertEqual(ups[0], 0.0)
        self.assertLess(ups[0], ups[1])
        self.assertLess(ups[1], ups[2])
        costs = [x.features["v2"]["costs"]["cost_wrong"] for x in (low, mid, high)]
        self.assertAlmostEqual(costs[0], 1.0)
        for c, u in zip(costs, ups):
            self.assertAlmostEqual(c, 1.0 + u)
        self.assertLess(costs[0], costs[1])
        self.assertLess(costs[1], costs[2])
        self.assertEqual([x.features["v2"]["risk_tier"] for x in (low, mid, high)], ["low", "high", "critical"])
        self.assertIs(low.outcome.status, PolicyStatus.EXECUTE)
        self.assertIs(mid.outcome.status, PolicyStatus.ABSTAIN)      # costlier mistakes need more confidence
        self.assertIs(high.outcome.status, PolicyStatus.ABSTAIN)
        self.assertIn("Forecast summary", low.augmented_state)
        self.assertIsInstance(low.choice, ChoiceResult)

    def test_needs_approval_when_confident_enough_for_the_scaled_cost(self):
        d = self.decider(0.9995)
        low, mid, high = (self.run_one(d, t) for t in (13.0, 10.6, 10.4))
        self.assertIs(low.outcome.status, PolicyStatus.EXECUTE)
        self.assertIs(mid.outcome.status, PolicyStatus.NEEDS_APPROVAL)
        self.assertIs(high.outcome.status, PolicyStatus.NEEDS_APPROVAL)
        with self.assertRaises(Exception):
            d.execute(mid, lambda option, dec: option)                 # gate still needs a human approval
        d.approve(mid.decision_id, "ops")
        self.assertEqual(d.execute(mid, lambda option, dec: option), mid.outcome.option)

    def test_custom_cost_fn_and_hard_gate_replaced(self):
        seen = []

        def cost_fn(features):
            seen.append(features["breach_prob_any_step_upper"])
            return CostSpec(cost_wrong=5.0)

        policy = DecisionPolicy(utility=lambda o, f: 1.0, max_forecast_uncertainty=0.0)   # v1 gate would abstain
        d = self.decider(0.999, cost_fn=cost_fn, policy=policy)
        res = self.run_one(d, 13.0)
        self.assertEqual(seen, [0.0])
        self.assertEqual(res.features["v2"]["costs"]["cost_wrong"], 5.0)
        self.assertIs(res.outcome.status, PolicyStatus.EXECUTE)

    def test_v1_decider_unchanged_and_arguments_checked(self):
        with self.assertRaises(ValueError):
            ForecastAwareDecider(object(), NaiveQuantileForecaster(), cost_fn=lambda f: None)
        with self.assertRaises(ValueError):
            ForecastAwareDecider(object(), NaiveQuantileForecaster(), backend="v2")
        v1 = ForecastAwareDecider(object(), NaiveQuantileForecaster())
        self.assertEqual(v1.backend, "v1")


if __name__ == "__main__":
    unittest.main()
