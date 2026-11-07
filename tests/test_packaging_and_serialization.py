"""
Unit tests for Packaging, Engine Facade, Serialization/Deserialization,
Ergonomic API (decide/evaluate), Adapter Pooling, and MockEncoder.
"""

import json
import os
import pytest
import numpy as np
import torch

from rhen import (
    Noul, NoulResult,
    Choice, ChoiceResult,
    Score, ScoreResult,
    MultiChoice, MultiChoiceResult,
    Range, RangeResult,
    Extract, ExtractResult,
    Cascade, CascadeResult,
    SchemaMutation,
    AbstentionVerdict,
    ANSWER, ABSTAIN, NEED_CLARIFICATION,
    RhenAbstentionError,
    decide, evaluate,
    get_default_engine, reset_default_engine,
    ContextBuilder,
    ActionCard,
    MockEncoder,
    HuggingFaceEncoderAdapter,
)
import rhen.engine as rhen_engine


class TestPackagingAndFacade:
    def test_py_typed_exists(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        py_typed = os.path.join(root, "rhen", "py.typed")
        assert os.path.isfile(py_typed), f"Missing {py_typed}"

    def test_engine_facade_exports(self):
        assert hasattr(rhen_engine, "RhenEngine")
        assert hasattr(rhen_engine, "AdvancedRhenEngine")
        assert hasattr(rhen_engine, "LatentSearchEngine")
        assert hasattr(rhen_engine, "GeometricAbstentionGate")
        assert hasattr(rhen_engine, "ANSWER")
        assert hasattr(rhen_engine, "ABSTAIN")
        assert hasattr(rhen_engine, "NEED_CLARIFICATION")
        assert hasattr(rhen_engine, "decide")
        assert hasattr(rhen_engine, "evaluate")
        assert hasattr(rhen_engine, "Noul")
        assert hasattr(rhen_engine, "Choice")
        assert hasattr(rhen_engine, "Score")
        assert hasattr(rhen_engine, "MultiChoice")
        assert hasattr(rhen_engine, "Range")
        assert hasattr(rhen_engine, "Extract")
        assert hasattr(rhen_engine, "Cascade")


class TestSerializationAndDeserialization:
    def test_schema_mutation_roundtrip(self):
        sm = SchemaMutation(
            parent_question="test?",
            mutation_type="route_tool",
            mutation_logits=[0.1, 0.9],
            confidence=0.85,
            metadata={"tool": "ActionCardStaging"},
        )
        d = sm.to_dict()
        assert d["mutation_type"] == "route_tool"
        j = sm.to_json()
        assert json.loads(j)["mutation_type"] == "route_tool"
        sm2 = SchemaMutation.from_json(j)
        assert sm2.parent_question == sm.parent_question
        assert sm2.mutation_type == sm.mutation_type
        assert sm2.confidence == sm.confidence
        assert sm2.metadata == sm.metadata

    def test_noul_result_serialization(self):
        verdict = AbstentionVerdict(decision=ANSWER, reasons=[])
        sm = SchemaMutation(parent_question="prop", mutation_type="noop", mutation_logits=[0.0], confidence=0.9)
        res = NoulResult(
            proposition="The sky is blue",
            p_true=0.98,
            decision=True,
            confidence=0.96,
            temperature=1.0,
            free_energy=1.2,
            schema_mutation=sm,
            abstention_verdict=verdict,
        )
        assert res.should_answer is True
        assert res.should_abstain is False

        d = res.to_dict()
        assert d["proposition"] == "The sky is blue"
        assert d["decision"] is True
        j = res.to_json()
        assert "The sky is blue" in j

        res2 = NoulResult.from_json(j)
        assert res2.proposition == res.proposition
        assert res2.decision == res.decision
        assert res2.p_true == res.p_true
        assert res2.schema_mutation.mutation_type == "noop"
        assert res2.abstention_verdict.decision == ANSWER
        assert res2.should_answer is True

    def test_choice_result_serialization(self):
        verdict = AbstentionVerdict(decision=ABSTAIN, reasons=["free_energy 5.2 > 4.5"])
        res = ChoiceResult(
            question="Select color",
            choice_index=1,
            choice_label="green",
            probabilities={"red": 0.2, "green": 0.8},
            raw_probabilities=[0.2, 0.8],
            confidence=0.6,
            temperature=1.0,
            abstention_verdict=verdict,
        )
        assert res.should_abstain is True
        assert res.should_answer is False

        d = res.to_dict()
        assert d["choice_label"] == "green"
        res2 = ChoiceResult.from_dict(d)
        assert res2.choice_label == "green"
        assert res2.should_abstain is True

    def test_score_result_serialization(self):
        res = ScoreResult(
            question="Rate severity",
            score=7.5,
            min_score=0.0,
            max_score=10.0,
            level_probabilities=[0.1, 0.2, 0.7],
            confidence=0.8,
            temperature=1.0,
            abstention_verdict=AbstentionVerdict(decision=ANSWER),
        )
        j = res.to_json()
        res2 = ScoreResult.from_json(j)
        assert res2.score == 7.5
        assert res2.should_answer is True

    def test_multichoice_result_serialization(self):
        res = MultiChoiceResult(
            question="Pick tags",
            selected_indices=[0, 2],
            selected_options=["AI", "ML"],
            probabilities={"AI": 0.9, "Crypto": 0.1, "ML": 0.85},
            raw_probabilities=[0.9, 0.1, 0.85],
            confidence=0.9,
            temperature=1.0,
            threshold=0.5,
            abstention_verdict=AbstentionVerdict(decision=ANSWER),
        )
        j = res.to_json()
        res2 = MultiChoiceResult.from_json(j)
        assert res2.selected_options == ["AI", "ML"]
        assert res2.should_answer is True

    def test_range_result_serialization(self):
        res = RangeResult(
            question="Estimate latency",
            expected_value=42.5,
            interval_low=38.0,
            interval_high=47.0,
            confidence_interval=0.95,
            min_val=0.0,
            max_val=100.0,
            confidence=0.9,
            temperature=1.0,
            abstention_verdict=AbstentionVerdict(decision=ANSWER),
        )
        j = res.to_json()
        res2 = RangeResult.from_json(j)
        assert res2.expected_value == 42.5
        assert res2.interval_low == 38.0
        assert res2.should_answer is True

    def test_extract_result_serialization(self):
        res = ExtractResult(
            slot_name="order_id",
            extracted_value="ORD-9912",
            start_idx=10,
            end_idx=18,
            confidence=0.95,
            temperature=1.0,
            abstention_verdict=AbstentionVerdict(decision=ANSWER),
        )
        j = res.to_json()
        res2 = ExtractResult.from_json(j)
        assert res2.extracted_value == "ORD-9912"
        assert res2.start_idx == 10
        assert res2.should_answer is True

    def test_cascade_result_serialization(self):
        c1 = ChoiceResult(
            question="q1",
            choice_index=0,
            choice_label="A",
            probabilities={"A": 0.9},
            raw_probabilities=[0.9],
            confidence=0.9,
            temperature=1.0,
            abstention_verdict=AbstentionVerdict(decision=ANSWER),
        )
        n1 = NoulResult(
            proposition="prop",
            p_true=0.8,
            decision=True,
            confidence=0.6,
            temperature=1.0,
            abstention_verdict=AbstentionVerdict(decision=ANSWER),
        )
        cas = CascadeResult(
            id="cascade_test",
            stages=[c1, n1],
            decisions={"q1": "A", "prop": True},
            confidence=0.85,
            temperature=1.0,
            abstention_verdict=AbstentionVerdict(decision=ANSWER),
        )
        d = cas.to_dict()
        assert len(d["stages"]) == 2
        j = cas.to_json()
        cas2 = CascadeResult.from_json(j)
        assert len(cas2.stages) == 2
        assert isinstance(cas2.stages[0], ChoiceResult)
        assert isinstance(cas2.stages[1], NoulResult)
        assert cas2.stages[0].choice_label == "A"
        assert cas2.stages[1].decision is True

    def test_robust_encoder_types(self):
        # Test handling of torch.Tensor, numpy scalar/arrays, sets
        card = ActionCard(
            id="act-123",
            title="Deploy model",
            risk_tier="low",
            action_type="execute",
            payload={
                "tensor_scalar": torch.tensor(3.14),
                "tensor_vec": torch.tensor([1.0, 2.0, 3.0]),
                "np_float": np.float64(2.718),
                "np_int": np.int32(42),
                "np_arr": np.array([10, 20]),
                "set_val": {"beta", "alpha"},
            },
            required_permissions=["query", "read"],
            confidence=0.95,
        )
        d = card.to_dict()
        assert isinstance(d["payload"]["tensor_scalar"], float)
        assert d["payload"]["tensor_vec"] == [1.0, 2.0, 3.0]
        assert isinstance(d["payload"]["np_float"], float)
        assert isinstance(d["payload"]["np_int"], int)
        assert d["payload"]["np_arr"] == [10, 20]
        assert d["payload"]["set_val"] == ["alpha", "beta"]

        j = card.to_json()
        card2 = ActionCard.from_json(j)
        assert card2.id == "act-123"
        assert card2.title == "Deploy model"
        assert card2.payload["np_int"] == 42


class TestErgonomicApi:
    def test_empty_options_validation(self):
        with pytest.raises(ValueError, match="Choice requires at least one option"):
            Choice(question="Empty?", options=[])

        with pytest.raises(ValueError, match="Choice requires at least one option"):
            MultiChoice(question="Empty?", options=[])

        with pytest.raises(ValueError, match="Choice requires at least one option"):
            decide({"status": "ok"}, Choice(question="Test?", options=[]))

    def test_decide_single_and_list(self):
        state = {"user": "Alice", "balance": 150.0}
        q_single = Noul("Is balance positive?")
        res_single = decide(state, q_single)
        assert isinstance(res_single, NoulResult)
        assert hasattr(res_single, "decision")

        q_list = [
            Noul("Is balance positive?"),
            Choice("Select tier", ["standard", "premium"]),
        ]
        res_list = decide(state, q_list)
        assert isinstance(res_list, list)
        assert len(res_list) == 2
        assert isinstance(res_list[0], NoulResult)
        assert isinstance(res_list[1], ChoiceResult)

    def test_evaluate_and_all_primitives(self):
        state = {"user": "Alice", "balance": 150.0, "latency": 45.2}
        # Test evaluate with single primitive
        q_score = Score("Performance rating", levels=["Poor", "OK", "Good"])
        res_score = evaluate(state, q_score)
        assert isinstance(res_score, ScoreResult)

        # Test evaluate with Range
        q_range = Range("Latency bound", min_val=0.0, max_val=100.0)
        res_range = evaluate(state, q_range)
        assert isinstance(res_range, RangeResult)

        # Test evaluate with Extract
        q_extract = Extract("Extract username")
        res_extract = evaluate(state, q_extract)
        assert isinstance(res_extract, ExtractResult)

        # Test evaluate with Cascade
        q_cascade = Cascade(stages=[Noul("Is active?"), Choice("Tier", ["A", "B"])])
        res_cascade = evaluate(state, q_cascade)
        assert isinstance(res_cascade, CascadeResult)

        # Test get_default_engine and reset_default_engine
        eng = get_default_engine()
        assert eng is not None
        reset_default_engine()
        eng2 = get_default_engine()
        assert eng2 is not None

        # Test NEED_CLARIFICATION constant
        assert NEED_CLARIFICATION == "NeedClarification"

    def test_context_builder_from_data(self):
        # Raw dict
        b1 = ContextBuilder.from_data({"a": 1, "b": 2})
        assert "a" in b1.build_state_string()

        # Tabular list of dicts
        b2 = ContextBuilder.from_data([{"name": "Alice", "role": "admin"}])
        assert "Alice" in b2.build_state_string()

        # JSON string
        b3 = ContextBuilder.from_data('{"server": "prod-1"}')
        assert "prod-1" in b3.build_state_string()

    def test_on_abstain_handling(self):
        # Mock an abstained result by setting threshold or gate
        state = "Some normal state"
        res = decide(state, Noul("Is it sunny?"), on_abstain="return")
        assert hasattr(res, "should_abstain")

        # Create a mock engine that always abstains
        class MockAbstainingEngine:
            def evaluate(self, state, questions, **kwargs):
                return [
                    NoulResult(
                        proposition="test",
                        p_true=0.5,
                        decision=True,
                        confidence=0.0,
                        temperature=1.0,
                        abstention_verdict=AbstentionVerdict(decision=ABSTAIN, reasons=["OOD test"]),
                    )
                ]
            def abstention_verdict(self, **kwargs):
                return AbstentionVerdict(decision=ABSTAIN, reasons=["OOD test"])

        mock_eng = MockAbstainingEngine()

        # on_abstain="return" -> returns result
        res_ret = decide(state, Noul("test"), engine=mock_eng, on_abstain="return")
        assert isinstance(res_ret, NoulResult)
        assert res_ret.should_abstain is True

        # on_abstain="raise" -> raises RhenAbstentionError
        with pytest.raises(RhenAbstentionError, match="Decision abstained"):
            decide(state, Noul("test"), engine=mock_eng, on_abstain="raise")

        # on_abstain="verdict" -> returns AbstentionVerdict
        res_v = decide(state, Noul("test"), engine=mock_eng, on_abstain="verdict")
        assert isinstance(res_v, AbstentionVerdict)
        assert res_v.decision == ABSTAIN


class TestBackboneAdapterAndMockEncoder:
    def test_mock_encoder_forward(self):
        encoder = MockEncoder(hidden_size=128, vocab_size=500)
        input_ids = torch.randint(0, 500, (2, 16))
        out = encoder(input_ids)
        assert hasattr(out, "last_hidden_state")
        assert out.last_hidden_state.shape == (2, 16, 128)
        assert out[0].shape == (2, 16, 128)

    def test_adapter_with_mock_encoder_and_pooling(self):
        encoder = MockEncoder(hidden_size=64, vocab_size=1000)
        adapter = HuggingFaceEncoderAdapter(
            encoder=encoder,
            hidden_size=64,
            num_heads=2,
            rank=32,
            pooling_mode="mean",
        )
        assert adapter.pooling_mode == "mean"

        state = "Context: Patient exhibits symptoms A and B."
        questions = [("Diagnose condition", ["Cond A", "Cond B", "Cond C"])]

        # Test mean pooling
        res_mean = adapter.evaluate_rhen(state, questions, pooling_mode="mean")
        assert len(res_mean) == 1
        assert "probabilities" in res_mean[0]

        # Test last pooling
        res_last = adapter.evaluate_rhen(state, questions, pooling_mode="last")
        assert len(res_last) == 1

        # Test cls pooling
        res_cls = adapter.evaluate_rhen(state, questions, pooling_mode="cls")
        assert len(res_cls) == 1

        # Test invalid pooling
        with pytest.raises(ValueError, match="Invalid pooling_mode"):
            adapter.evaluate_rhen(state, questions, pooling_mode="invalid")

    def test_padding_mask_extraction(self):
        encoder = MockEncoder(hidden_size=64, vocab_size=1000)
        adapter = HuggingFaceEncoderAdapter(encoder=encoder, hidden_size=64)

        # 4D mask [B, 1, S, S]
        mask_4d = torch.zeros(1, 1, 10, 10)
        mask_4d[0, 0, :5, :5] = 1.0
        # 3D mask [B, S, S]
        mask_3d = torch.zeros(1, 10, 10)
        mask_3d[0, :5, :5] = 1.0

        class DummyPacked:
            def __init__(self, mask):
                self.input_ids = torch.randint(0, 100, (1, 10))
                self.attention_mask = mask
                self.branches = []

        # Run forward with 4D mask
        out_4d = adapter.forward_rhen(DummyPacked(mask_4d))
        assert isinstance(out_4d, list)

        # Run forward with 3D mask
        out_3d = adapter.forward_rhen(DummyPacked(mask_3d))
        assert isinstance(out_3d, list)


class TestRootShims:
    def test_root_shims_importable(self):
        import adapter
        import agent_tools
        import config
        import context_builder
        import model
        import rhen

        assert hasattr(adapter, "HuggingFaceEncoderAdapter")
        assert hasattr(agent_tools, "ActionCard")
        assert hasattr(config, "EOSConfig")
        assert hasattr(context_builder, "ContextBuilder")
        assert hasattr(model, "EOSModel")
        assert hasattr(rhen, "RhenEngine")
        assert hasattr(rhen, "decide")
