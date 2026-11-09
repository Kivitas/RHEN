"""
Regression tests for the Rhen SDK cleanup (rhen.py, unified_orchestrator.py).

Covers:
- user callbacks that raise are counted and logged, never silently swallowed
- seeds are sha256-based, not salted by hash()
- extract without a model value returns an empty value with zero confidence
- the schema mapping argmaxes over all eight mutation types
- model errors propagate (no retry with altered questions)
- the convex-hull interior test is exact
- uncalibrated conformal sets are recorded but do not fire abstention
- energy comes only from explicit scores or logits
- schema mutations fail loudly or report no_mutation instead of inventing choices
- the orchestrator runs a single decision layer and rejects generation requests
- removed layer config keys are not mentioned by rhen.py or unified_orchestrator.py
"""

import hashlib
import importlib.util
import itertools
import math
import os
import types
import unittest
from unittest import mock

import torch

from config import EOSConfig
from meta_cognitive_monitor import EnergyOODDetector
from rhen import (
    ABSTAIN,
    ANSWER,
    AdvancedRhenEngine,
    Choice,
    ChoiceResult,
    ExtractResult,
    GeometricAbstentionGate,
    LatentRolloutResult,
    LatentSearchEngine,
    LatentSearchResult,
    Noul,
    NoulResult,
    RangeResult,
    RhenEngine,
    SCHEMA_MUTATION_TYPES,
    SchemaMutation,
)
from tokenizer import get_tokenizer
from unified_orchestrator import UnifiedAIOrchestrator

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "tests" else _HERE

#: Rhen-layer config keys removed in RHEN 1.0.0 (the three-layer pipeline and
#: meta-reasoning enhancer were deleted). The SDK modules must not name them.
REMOVED_LAYER_CONFIG_KEYS = (
    "rhen_enable_deep_reasoning",
    "rhen_confidence_threshold",
    "rhen_enable_deepseek_reasoning",
    "rhen_enable_qwen_reasoning",
    "rhen_meta_reasoning_cache_size",
    "rhen_enable_three_layer_pipeline",
    "rhen_semantic_layer_weight",
    "rhen_reasoning_layer_weight",
    "rhen_generation_layer_weight",
    "rhen_enable_confidence_routing",
    "rhen_min_semantic_confidence",
    "rhen_max_reasoning_depth",
)


class _FakeRhenModel(torch.nn.Module):
    """Stand-in for EOSModel that returns canned branch results or raises a given error."""

    def __init__(self, raw_results=None, error=None):
        super().__init__()
        self._raw_results = list(raw_results or [])
        self._error = error
        self.calls = 0

    def evaluate_rhen(self, packed_or_state=None, questions_and_options=None, tokenizer=None, **kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return list(self._raw_results)


def _brute_force_hull_distance(points):
    """Distance from the origin to conv(points), by enumerating every subset (reference)."""
    pts = torch.as_tensor(points, dtype=torch.float64)
    best = math.inf
    k = pts.shape[0]
    for size in range(1, k + 1):
        for idx in itertools.combinations(range(k), size):
            q = pts[list(idx)]
            m = q.shape[0]
            kkt = torch.zeros(m + 1, m + 1, dtype=torch.float64)
            kkt[:m, :m] = q @ q.T
            kkt[:m, m] = 1.0
            kkt[m, :m] = 1.0
            rhs = torch.zeros(m + 1, dtype=torch.float64)
            rhs[m] = 1.0
            try:
                alpha = torch.linalg.solve(kkt, rhs)[:m]
            except RuntimeError:
                continue
            if bool((alpha >= -1e-9).all()):
                closest = (alpha.unsqueeze(1) * q).sum(dim=0)
                best = min(best, float(torch.linalg.vector_norm(closest)))
    return best


class TestCallbackFallbacksAreLoud(unittest.TestCase):
    def setUp(self):
        self.engine = LatentSearchEngine()
        self.z = torch.linspace(-1.0, 1.0, 16)
        self.options = ["a", "b", "c"]

    def test_default_search_reports_heuristic_and_no_fallbacks(self):
        res = self.engine.search(self.z, self.options, beam_width=2, max_depth=2)
        self.assertIsInstance(res, LatentSearchResult)
        self.assertEqual(res.source, "heuristic")
        self.assertEqual(res.fallbacks, 0)

    def test_raising_callbacks_are_counted_and_logged(self):
        def boom(*args, **kwargs):
            raise RuntimeError("callback exploded")

        with self.assertLogs(level="WARNING") as logs:
            res = self.engine.search(
                self.z,
                self.options,
                transition_operator=boom,
                prospective_value_head=boom,
                prm_scorer=boom,
                beam_width=2,
                max_depth=2,
            )
        self.assertGreater(res.fallbacks, 0)
        self.assertEqual(res.fallbacks, len(logs.records))
        self.assertTrue(all("callback exploded" in r.getMessage() for r in logs.records))
        self.assertEqual(res.source, "heuristic")

    def test_working_callbacks_report_callable_source(self):
        res = self.engine.search(
            self.z,
            self.options,
            transition_operator=lambda z, a: z * 1.0,
            prospective_value_head=lambda z: (0.25, 0.01),
            prm_scorer=lambda z_curr, a, z_next: 0.5,
            beam_width=2,
            max_depth=2,
        )
        self.assertEqual(res.source, "callable")
        self.assertEqual(res.fallbacks, 0)

    def test_partly_failing_callback_reports_mixed_source(self):
        calls = {"n": 0}

        def flaky_value(z):
            calls["n"] += 1
            if calls["n"] % 2 == 1:
                raise ValueError("odd calls fail")
            return (0.25, 0.01)

        res = self.engine.search(self.z, self.options, prospective_value_head=flaky_value, beam_width=2, max_depth=2)
        self.assertGreater(res.fallbacks, 0)
        self.assertEqual(res.source, "mixed")

    def test_search_requires_a_state_and_candidates(self):
        with self.assertRaises(ValueError):
            self.engine.search(None, self.options)
        with self.assertRaises(ValueError):
            self.engine.search(self.z, [])

    def test_result_defaults(self):
        res = LatentSearchResult(
            best_trajectory=[], best_action_sequence=[], prospective_value=0.0, step_rewards=[], search_tree={}
        )
        self.assertEqual(res.fallbacks, 0)
        self.assertEqual(res.source, "heuristic")
        roll = LatentRolloutResult(
            state_summary="", candidate_options=[], horizon=1, prospective_values={}, best_option="", best_value=0.0
        )
        self.assertEqual(roll.source, "calibrated_choice_projection")

    def test_latent_search_on_text_state_is_heuristic_and_repeatable(self):
        engine = RhenEngine(model=_FakeRhenModel(), tokenizer=get_tokenizer())
        first = engine.latent_search("degraded node", "Which action?", ["Reboot", "Drain"], beam_width=2, max_depth=2)
        second = engine.latent_search("degraded node", "Which action?", ["Reboot", "Drain"], beam_width=2, max_depth=2)
        self.assertEqual(first.source, "heuristic")
        self.assertEqual(first.fallbacks, 0)
        self.assertEqual(first.best_action_sequence, second.best_action_sequence)
        self.assertAlmostEqual(first.prospective_value, second.prospective_value, places=9)


class TestStableSeeds(unittest.TestCase):
    def test_action_embedding_uses_a_sha256_seed(self):
        engine = LatentSearchEngine()
        got = engine._get_action_tensor("left", 16, torch.device("cpu"))
        payload = "\x1f".join(["action", "left"]).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)
        expected = torch.randn(16, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
        expected = expected / (torch.linalg.vector_norm(expected) + 1e-8)
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))


class TestExtractWithoutAModelValue(unittest.TestCase):
    def test_text_key_value_is_not_parsed_when_model_gives_no_value(self):
        model = _FakeRhenModel(raw_results=[{"confidence": 0.9, "probabilities": [0.5, 0.5]}])
        engine = RhenEngine(model=model, tokenizer=get_tokenizer())
        res = engine.extract("Deployment: region: us-east-2 failed", slot_name="region")
        self.assertIsInstance(res, ExtractResult)
        self.assertEqual(res.extracted_value, "")
        self.assertEqual(res.confidence, 0.0)
        self.assertEqual((res.start_idx, res.end_idx), (0, 0))

    def test_dict_entity_is_not_read_when_model_gives_no_value(self):
        model = _FakeRhenModel(raw_results=[{"confidence": 0.9, "probabilities": [0.5, 0.5]}])
        engine = RhenEngine(model=model, tokenizer=get_tokenizer())
        res = engine.extract({"region": "us-east-2"}, slot_name="region")
        self.assertEqual(res.extracted_value, "")
        self.assertEqual(res.confidence, 0.0)

    def test_range_free_energy_comes_only_from_the_model(self):
        model = _FakeRhenModel(
            raw_results=[{"probabilities": [0.2, 0.8], "expected_value": 30.0, "confidence": 0.7, "scores": [30.0]}]
        )
        engine = RhenEngine(model=model, tokenizer=get_tokenizer())
        res = engine.range("Recovery minutes?", "Estimated recovery", min_val=0.0, max_val=60.0)
        self.assertIsInstance(res, RangeResult)
        self.assertIsNone(res.free_energy)
        self.assertIsNone(res.conformal_set)


class TestModelErrorsPropagate(unittest.TestCase):
    def test_value_error_from_the_model_is_not_retried_or_swallowed(self):
        model = _FakeRhenModel(error=ValueError("packing failed"))
        engine = RhenEngine(model=model, tokenizer=get_tokenizer())
        with self.assertRaises(ValueError):
            engine.evaluate("state text", [Choice("Pick one", ["a", "b"])])
        self.assertEqual(model.calls, 1)

    def test_runtime_error_from_the_model_propagates(self):
        model = _FakeRhenModel(error=RuntimeError("model failed"))
        engine = RhenEngine(model=model, tokenizer=get_tokenizer())
        with self.assertRaises(RuntimeError):
            engine.evaluate("state text", [Noul("Is it up?")])
        self.assertEqual(model.calls, 1)

    def test_non_dict_speculative_branch_result_is_rejected(self):
        model = _FakeRhenModel(raw_results=[42, 43])
        engine = RhenEngine(model=model, tokenizer=get_tokenizer())
        with self.assertRaises(TypeError):
            engine.speculative_cascade(
                "state text",
                Choice("Root?", ["a", "b"]),
                {0: [Noul("child a?")], 1: [Noul("child b?")]},
            )


class TestSchemaMapping(unittest.TestCase):
    def setUp(self):
        self.engine = RhenEngine.__new__(RhenEngine)

    def test_schema_types_are_the_eight_declared_names(self):
        self.assertEqual(
            SCHEMA_MUTATION_TYPES,
            ("spawn_child", "switch_primitive", "terminate", "route_tool", "refine", "verify", "escalate", "noop"),
        )

    def test_argmax_over_all_eight_logits_maps_every_type(self):
        for idx, name in enumerate(SCHEMA_MUTATION_TYPES):
            logits = [0.0] * len(SCHEMA_MUTATION_TYPES)
            logits[idx] = 4.0
            with self.subTest(mutation=name):
                sm = self.engine._extract_schema_mutation({"schema_logits": logits}, "q?")
                self.assertEqual(sm.mutation_type, name)
                self.assertGreater(sm.confidence, 0.0)

    def test_tensor_logits_are_accepted(self):
        logits = torch.zeros(1, len(SCHEMA_MUTATION_TYPES))
        logits[0, 7] = 3.0
        sm = self.engine._extract_schema_mutation({"schema_logits": logits}, "q?")
        self.assertEqual(sm.mutation_type, "noop")

    def test_wrong_width_logits_raise_instead_of_collapsing(self):
        with self.assertRaises(ValueError):
            self.engine._extract_schema_mutation({"schema_logits": [0.1, 0.9, 0.0, 0.0]}, "q?")


class TestConvexHullInterior(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.gate = GeometricAbstentionGate()

    def test_interior_queries_stay_inside_for_every_configuration(self):
        for k in range(2, 9):
            for d in (2, 3, 8, 16):
                for scale in (1.0, 50.0):
                    opts = torch.randn(k, d, dtype=torch.float64) * scale
                    weights = torch.distributions.Dirichlet(torch.ones(k, dtype=torch.float64)).sample()
                    q = (weights.unsqueeze(1) * opts).sum(dim=0)
                    with self.subTest(k=k, d=d, scale=scale):
                        self.assertLessEqual(self.gate.convex_hull_distance(opts.tolist(), q.tolist()), 0.0)
                        verdict = self.gate.evaluate(
                            probabilities=[0.9, 0.05, 0.05],
                            free_energy=1.0,
                            conformal_set={"set": ["a"]},
                            option_latents=opts.tolist(),
                            query_latent=q.tolist(),
                        )
                        self.assertEqual(verdict.decision, ANSWER)
                        self.assertFalse(any("convex hull" in r for r in verdict.reasons))

    def test_vertex_is_inside_and_exterior_point_abstains(self):
        opts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        self.assertLessEqual(self.gate.convex_hull_distance(opts, opts[1]), 0.0)
        self.assertGreater(self.gate.convex_hull_distance(opts, [2.0, 2.0]), 0.0)
        verdict = self.gate.evaluate(
            probabilities=[0.9, 0.05, 0.05], free_energy=1.0, option_latents=opts, query_latent=[2.0, 2.0]
        )
        self.assertEqual(verdict.decision, ABSTAIN)

    def test_distance_matches_brute_force_reference(self):
        for trial in range(60):
            k = 2 + trial % 4
            d = 1 + trial % 6
            opts = torch.randn(k, d, dtype=torch.float64) * 3.0
            q = torch.randn(d, dtype=torch.float64) * 3.0
            reference = _brute_force_hull_distance(opts - q)
            got = self.gate.convex_hull_distance(opts.tolist(), q.tolist())
            with self.subTest(trial=trial, k=k, d=d):
                self.assertAlmostEqual(abs(got), reference, places=6)


class TestUncalibratedConformalSets(unittest.TestCase):
    def setUp(self):
        self.gate = GeometricAbstentionGate()
        self.opts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    def test_uncalibrated_empty_set_is_recorded_but_does_not_abstain(self):
        verdict = self.gate.evaluate(
            probabilities=[0.9, 0.05, 0.05],
            free_energy=1.0,
            conformal_set={"prediction_set": [], "calibrated": False},
            option_latents=self.opts,
            query_latent=[0.3, 0.3],
        )
        self.assertEqual(verdict.decision, ANSWER)
        self.assertIn("conformal uncalibrated", verdict.reasons)
        self.assertIsNone(verdict.conformal_size)

    def test_calibrated_empty_set_still_abstains(self):
        verdict = self.gate.evaluate(
            probabilities=[0.9, 0.05, 0.05],
            free_energy=1.0,
            conformal_set={"prediction_set": [], "calibrated": True},
            option_latents=self.opts,
            query_latent=[0.3, 0.3],
        )
        self.assertEqual(verdict.decision, ABSTAIN)
        self.assertEqual(verdict.conformal_size, 0)


class TestEnergyFromExplicitScoresOnly(unittest.TestCase):
    def _bare_engine(self):
        engine = RhenEngine.__new__(RhenEngine)
        engine.conformal_engine = None
        engine.ood_detector = EnergyOODDetector(ood_threshold=4.5)
        return engine

    def test_probabilities_alone_give_no_energy(self):
        engine = self._bare_engine()
        _, energy = engine._extract_conformal_and_energy({}, probabilities={"A": 0.9, "B": 0.1})
        self.assertIsNone(energy)

    def test_explicit_scores_give_the_detector_energy(self):
        engine = self._bare_engine()
        scores = [6.0, 1.0, 0.5, -2.0]
        _, energy = engine._extract_conformal_and_energy({"scores": scores}, probabilities={"A": 0.9, "B": 0.1})
        self.assertAlmostEqual(energy, engine.ood_detector.compute_energy(scores, temperature=1.0), places=6)


class TestAdvancedEngineSurface(unittest.TestCase):
    def test_deep_reasoning_methods_and_result_class_are_gone(self):
        for name in ("enhanced_evaluate", "enhanced_evaluate_sync", "_apply_meta_reasoning"):
            self.assertFalse(hasattr(AdvancedRhenEngine, name), name)
        import rhen

        self.assertFalse(hasattr(rhen, "EnhancedCascadeResult"))

    def test_simulate_latent_outcomes_projects_the_choice_without_model_rollout(self):
        class _RolloutSpy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rollout_calls = 0

            def simulate_latent_rollout(self, *args, **kwargs):
                self.rollout_calls += 1
                raise AssertionError("the model rollout must not be used")

        engine = RhenEngine.__new__(RhenEngine)
        engine.model = _RolloutSpy()
        engine.choice = lambda state, question, options: ChoiceResult(
            question=question,
            choice_index=0,
            choice_label=options[0],
            probabilities={"Reboot": 0.7, "Drain": 0.3},
            raw_probabilities=[0.7, 0.3],
            confidence=0.7,
            temperature=1.0,
        )
        res = engine.simulate_latent_outcomes("state", "Action?", ["Reboot", "Drain"], horizon=2)
        self.assertIsInstance(res, LatentRolloutResult)
        self.assertEqual(res.source, "calibrated_choice_projection")
        self.assertEqual(res.best_option, "Reboot")
        self.assertEqual(engine.model.rollout_calls, 0)


class TestDynamicSchemaMutation(unittest.TestCase):
    def setUp(self):
        self.engine = AdvancedRhenEngine.__new__(AdvancedRhenEngine)
        self.question = Choice("Node action", ["Reboot", "Drain"])

    def test_typed_result_without_a_schema_reports_no_mutation(self):
        res = NoulResult(proposition="Node healthy?", p_true=0.9, decision=True, confidence=0.8, temperature=1.0)
        out = self.engine.dynamic_schema_mutation(state_or_result=res)
        self.assertEqual(out["action"], "none")
        self.assertEqual(out["status"], "no_mutation")
        self.assertIsNone(out["mutation"])

    def test_spawn_child_requires_explicit_speculative_branches(self):
        mutation = SchemaMutation(
            parent_question="Node action", mutation_type="spawn_child", mutation_logits=[1.0] + [0.0] * 7, confidence=0.9
        )
        with self.assertRaises(ValueError):
            self.engine.dynamic_schema_mutation(state="Node-04 degraded", question=self.question, mutation=mutation)

    def test_spawn_child_on_a_string_question_requires_root_options(self):
        mutation = SchemaMutation(
            parent_question="Node action", mutation_type="spawn_child", mutation_logits=[1.0] + [0.0] * 7, confidence=0.9
        )
        with self.assertRaises(ValueError):
            self.engine.dynamic_schema_mutation(
                state="Node-04 degraded",
                question="Node action?",
                mutation=mutation,
                speculative_branches={0: [Noul("Recovers?")]},
            )

    def test_terminate_noop_and_pending_types_are_reported_without_invention(self):
        cases = {
            "terminate": ("terminate", "terminated"),
            "noop": ("none", "noop"),
            "refine": ("refine", "not_executed"),
            "verify": ("verify", "not_executed"),
            "escalate": ("escalate", "not_executed"),
        }
        for mut_type, (action, status) in cases.items():
            with self.subTest(mutation=mut_type):
                mutation = SchemaMutation(
                    parent_question="Node action", mutation_type=mut_type, mutation_logits=[0.0] * 8, confidence=0.5
                )
                out = self.engine.dynamic_schema_mutation(state="Node-04", question=self.question, mutation=mutation)
                self.assertEqual(out["action"], action)
                self.assertEqual(out["status"], status)

    def test_unknown_mutation_type_raises(self):
        mutation = SchemaMutation(parent_question="q", mutation_type="bogus", mutation_logits=[], confidence=0.5)
        with self.assertRaises(ValueError):
            self.engine.dynamic_schema_mutation(state="Node-04", question=self.question, mutation=mutation)


class TestRemovedConfigKeysAreNotMentioned(unittest.TestCase):
    def test_rhen_and_orchestrator_do_not_mention_removed_layer_keys(self):
        for filename in ("rhen.py", "unified_orchestrator.py"):
            with open(os.path.join(_REPO_ROOT, filename), encoding="utf-8") as handle:
                source = handle.read()
            for key in REMOVED_LAYER_CONFIG_KEYS:
                with self.subTest(file=filename, key=key):
                    self.assertNotIn(key, source)

    def test_three_layer_pipeline_module_is_removed(self):
        self.assertIsNone(importlib.util.find_spec("three_layer_pipeline"))


class TestOrchestratorSingleDecisionLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = EOSConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, max_recurrent_loops=2)
        cls.orchestrator = UnifiedAIOrchestrator(cls.config)

    def test_generation_request_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "generation layer removed"):
            self.orchestrator.unified_decide_sync(
                state="Telemetry: node_2 cpu=98%.", generation_request="Summarize the incident"
            )

    def test_missing_questions_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.orchestrator.unified_decide_sync(state="Telemetry: node_2 cpu=98%.")

    def test_decision_mode_routes_through_the_single_layer(self):
        result = self.orchestrator.unified_decide_sync(
            state="Telemetry: node_2 cpu=98%, network stable.",
            questions=[Choice("Mitigation", ["Restart", "Drain", "Ignore"])],
        )
        self.assertEqual(result["mode"], "decision")
        self.assertEqual(result["routing"], "single_decision_layer")
        self.assertEqual(len(result["final_answers"]), 1)
        self.assertEqual(result["final_answers"][0]["question_index"], 0)
        self.assertEqual(result["final_answers"][0]["question_type"], "Choice")
        self.assertIsInstance(result["final_answers"][0]["final_answer"], ChoiceResult)
        self.assertIn("meta_cognitive", result)

    def test_use_three_layer_flag_is_ignored(self):
        result = self.orchestrator.unified_decide_sync(
            state="Telemetry: node_2 cpu=98%, network stable.",
            questions=[Noul("Is node_2 healthy?")],
            use_three_layer=True,
        )
        self.assertEqual(result["routing"], "single_decision_layer")
        self.assertIsInstance(result["results"][0], NoulResult)

    def test_auto_staging_passes_no_risk_tier_so_the_engine_infers_it(self):
        with mock.patch.object(
            self.orchestrator, "stage_action", wraps=self.orchestrator.stage_action
        ) as spy:
            result = self.orchestrator.unified_decide_sync(
                state="Telemetry: node_2 cpu=98%, network stable.",
                questions=[Choice("Mitigation", ["Restart", "Drain", "Ignore"])],
                auto_stage_actions=True,
            )
        self.assertEqual(len(result["staged_actions"]), 1)
        self.assertIsNone(spy.call_args.kwargs["risk_tier"])

    def test_action_signing_key_is_passed_to_the_staging_engine(self):
        key = b"orchestrator-test-signing-key-0123456789"
        orchestrator = UnifiedAIOrchestrator(self.config, action_signing_key=key)
        self.assertEqual(orchestrator.action_staging.secret_key, key)

    def test_orchestrator_does_not_record_execute_twice(self):
        orchestrator = UnifiedAIOrchestrator(self.config)
        card = types.SimpleNamespace(id="act-regression", metadata={})

        class _RecordingLedger:
            def __init__(self):
                self.events = []

            def record_event(self, recorded_card, event):
                self.events.append((recorded_card, event))
                return {"leaf_index": 7, "leaf_hash": "ab" * 32, "merkle_root": "cd" * 32}

        ledger = _RecordingLedger()
        orchestrator.staging_engine.merkle_ledger = ledger
        orchestrator.action_staging.execute_action = lambda **kwargs: card

        out = orchestrator.execute_action("act-regression")

        self.assertIs(out, card)
        # The engine records the execute event itself; the orchestrator must not record it again.
        self.assertEqual(ledger.events, [])
        self.assertEqual(card.metadata, {})

    def test_execute_leaves_one_execute_event_on_the_real_ledger(self):
        from agent_tools import ActionStagingEngine

        engine = ActionStagingEngine(secret_key=b"single-record-test-key-0123456789")
        card = engine.stage_action(
            {"confidence": 0.9}, action_type="execute", payload={"op": "x"}, risk_tier="low"
        )
        engine.preflight_verification(card.id)
        engine.approve_action(card.id, approver="ops")
        engine.execute_action(card.id, executor_fn=lambda p: {"ok": True})
        self.assertEqual(engine.merkle_ledger.events, ["stage", "approve", "execute"])
        self.assertIn("merkle_execution_leaf_index", card.metadata)


if __name__ == "__main__":
    unittest.main()
