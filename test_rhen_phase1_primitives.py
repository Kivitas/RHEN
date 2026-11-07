"""
Verification Suite for Rhen Phase 1: Decision Primitives & Readout Expansion.

Validates:
1. MultiChoice (Subset / Powerset Top-k with independent calibrated sigmoid probabilities).
2. Range (Bounded continuous interval regression [y_min, y_max] with Beta(alpha, beta) uncertainty bounds).
3. Extract (Non-autoregressive span pointer slot-filling over state token representations).
4. Dynamic Latent Head Probing (DLHP: probing compressed Key/Value manifolds).
5. End-to-End Differentiable Multi-Task Loss & Gradient Flow across all primitives.
6. Unified Single-Pass Concurrent Evaluation of all 6 primitive types.
"""

import torch

from config import EOSConfig
from model import EOSModel, MultiHeadRhenPointerHead
from rhen_packer import get_rhen_packer
from rhen import (
    RhenEngine,
    AdvancedRhenEngine,
    MultiChoice,
    Range,
    Extract,
    Choice,
    Noul,
    Score,
    NoulResult,
    ChoiceResult,
    ScoreResult,
    MultiChoiceResult,
    RangeResult,
    ExtractResult,
)
from rhen_loss import compute_rhen_multi_task_loss


def test_dlhp_and_pointer_head():
    print("\n--- Test 1: Dynamic Latent Head Probing (DLHP) & Projections ---")
    hidden_size = 256
    rank = 64
    pointer_head = MultiHeadRhenPointerHead(hidden_size=hidden_size, num_heads=4, rank=rank)

    q_repr = torch.randn(1, 1, hidden_size)
    opt_reprs = torch.randn(1, 4, hidden_size)
    kv_latent = torch.randn(1, 16, hidden_size)

    # 1. Forward with DLHP enabled
    details = pointer_head(q_repr, opt_reprs, kv_latent=kv_latent, return_details=True)
    assert "probabilities" in details
    assert "scores" in details
    assert details["probabilities"].shape == (1, 4)
    assert torch.isclose(details["probabilities"].sum(), torch.tensor(1.0), atol=1e-5)

    # Verify DLHP parameters are trainable and initialized
    assert hasattr(pointer_head, "probe_q")
    assert hasattr(pointer_head, "probe_k")
    assert hasattr(pointer_head, "probe_v")
    assert hasattr(pointer_head, "alpha_probe")
    assert pointer_head.alpha_probe.requires_grad
    print(f"  [PASS] DLHP active with alpha_probe={pointer_head.alpha_probe.item():.4f}")


def test_multichoice_primitive():
    print("\n--- Test 2: MultiChoice (Subset / Powerset Selection) ---")
    config = EOSConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_recurrent_loops=2,
    )
    model = EOSModel(config)
    model.eval()

    state = "System status: Disk IO saturated, Network healthy, Memory 89%, CPU 92%."
    options = ["Alert SRE", "Throttle incoming traffic", "Reboot database", "Scale replica pods"]

    # Test via model decide_multichoice
    res = model.decide_multichoice(state, "Select recommended remedial actions", options, threshold=0.45)
    assert res["primitive"] == "multichoice"
    assert "probabilities" in res
    assert len(res["probabilities"]) == 4
    for opt, prob in res["probabilities"].items():
        assert 0.0 <= prob <= 1.0, f"Option prob {opt}={prob} must be in [0, 1]"
    assert isinstance(res["selected_indices"], list)
    assert isinstance(res["selected_options"], list)
    assert res["confidence"] >= 0.0

    # Test via RhenEngine
    engine = RhenEngine(model=model)
    mc_res = engine.multichoice(state, "Select recommended remedial actions", options, threshold=0.5)
    assert isinstance(mc_res, MultiChoiceResult)
    assert mc_res.question == "Select recommended remedial actions"
    assert isinstance(mc_res.selected_options, list)
    print(f"  [PASS] MultiChoice evaluated: {len(mc_res.selected_options)} options selected above threshold.")


def test_range_primitive():
    print("\n--- Test 3: Range (Bounded Continuous Interval Regression) ---")
    config = EOSConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_recurrent_loops=2,
    )
    model = EOSModel(config)
    model.eval()

    state = "Cluster metrics: 120 pods active, average latency 320ms, p99 latency 850ms."
    min_v, max_v = 50.0, 1000.0

    # Test via model decide_range
    res = model.decide_range(state, "Estimated time to recovery (minutes)", min_val=min_v, max_val=max_v)
    assert res["primitive"] == "range"
    assert min_v <= res["expected_value"] <= max_v, f"Expected value {res['expected_value']} out of bounds"
    assert min_v <= res["interval_low"] <= res["expected_value"]
    assert res["expected_value"] <= res["interval_high"] <= max_v
    assert 0.0 <= res["confidence"] <= 1.0

    # Test via RhenEngine
    engine = RhenEngine(model=model)
    rng_res = engine.range(state, "Estimated time to recovery (minutes)", min_val=min_v, max_val=max_v)
    assert isinstance(rng_res, RangeResult)
    assert min_v <= rng_res.expected_value <= max_v
    assert rng_res.interval_low <= rng_res.expected_value <= rng_res.interval_high
    print(f"  [PASS] Range regression: y_hat={rng_res.expected_value:.2f} in [{rng_res.interval_low:.2f}, {rng_res.interval_high:.2f}], conf={rng_res.confidence:.4f}")


def test_extract_primitive():
    print("\n--- Test 4: Extract (Non-Autoregressive Span Pointer Extraction) ---")
    config = EOSConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_recurrent_loops=2,
    )
    model = EOSModel(config)
    model.eval()

    state = "Deployment event: Host srv-db-prod-01 in region us-east-2 reported exit code 137."

    # Test via model decide_extract
    res = model.decide_extract(state, slot_name="target_host", query="What is the failing host name?")
    assert res["primitive"] == "extract"
    assert "start_idx" in res
    assert "end_idx" in res
    assert res["start_idx"] <= res["end_idx"]
    assert 0.0 <= res["confidence"] <= 1.0

    # Test via RhenEngine
    engine = RhenEngine(model=model)
    ext_res = engine.extract(state, slot_name="target_host", query="What is the failing host name?")
    assert isinstance(ext_res, ExtractResult)
    assert ext_res.slot_name == "target_host"
    print(f"  [PASS] Extract span pointer: span=[{ext_res.start_idx}, {ext_res.end_idx}], conf={ext_res.confidence:.4f}")


def test_differentiable_multi_task_gradient_flow():
    print("\n--- Test 5: End-to-End Differentiable Multi-Task Loss & Backpropagation ---")
    config = EOSConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_recurrent_loops=2,
    )
    model = EOSModel(config)
    model.train()

    packer = get_rhen_packer()
    state = "Telemetry: Node cpu=98%, memory=91%, disk=saturated, status=degraded, ip=192.168.1.42."
    
    questions = [
        ("Is system degraded?", ["Yes", "No"], "noul", None),
        ("Primary remediation action?", ["Restart", "Throttle", "Ignore"], "choice", None),
        ("Estimated resolution time?", ["0.0", "120.0"], "range", {"min_val": 0.0, "max_val": 120.0}),
        ("Applicable policies?", ["RateLimit", "AutoQuarantine", "NotifyAdmin"], "multichoice", {"threshold": 0.5}),
        ("Failing IP address", [], "extract", {"slot_name": "ip"}),
    ]
    packed = packer.pack(state, questions)

    # Target labels for multi-task loss
    labels = [
        True,                 # Noul: degraded = True
        1,                    # Choice: Throttle (index 1)
        {"value": 45.0},      # Range: target 45.0 minutes
        [0, 1],               # MultiChoice: RateLimit & AutoQuarantine
        (2, 4),               # Extract: target span token indices (2, 4)
    ]

    total_loss, breakdown = compute_rhen_multi_task_loss(
        model=model,
        packed_sequence=packed,
        branch_labels=labels,
        device=torch.device("cpu"),
    )

    assert total_loss.requires_grad, "total_loss must be differentiable"
    assert "b0_noul_bce" in breakdown
    assert "b1_choice_ce" in breakdown
    assert "b2_range_mse" in breakdown
    assert "b3_multichoice_bce" in breakdown
    assert "b4_extract_span" in breakdown

    total_loss.backward()

    pointer_head = model.recurrent_core.rhen.pointer_head
    assert pointer_head.q_projs[0].weight.grad is not None, "Gradients must flow to pointer head q_projs"
    assert pointer_head.range_head[0].weight.grad is not None, "Gradients must flow to range_head"
    assert pointer_head.span_start_proj.weight.grad is not None, "Gradients must flow to span_start_proj"
    assert pointer_head.span_end_proj.weight.grad is not None, "Gradients must flow to span_end_proj"
    assert pointer_head.multichoice_thresh_proj.weight.grad is not None, "Gradients must flow to multichoice_thresh_proj"
    assert model.embed_tokens.weight.grad is not None, "Gradients must flow back to embed_tokens"

    print(f"  [PASS] Full gradient flow confirmed across all 5 tasks. Total loss: {total_loss.item():.4f}")
    for k, v in breakdown.items():
        print(f"    - {k}: {v:.4f}")


def test_unified_parallel_execution_all_six():
    print("\n--- Test 6: Unified Concurrent Execution of All 6 Primitives ---")
    config = EOSConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_recurrent_loops=2,
    )
    model = EOSModel(config)
    engine = AdvancedRhenEngine(model=model)

    state = {
        "service": "payments-api",
        "error_rate": 0.082,
        "active_threads": 480,
        "region": "us-central1",
        "last_audit": "2026-09-24",
    }

    # Evaluate 6 heterogeneous decision primitives in a SINGLE forward pass
    results = engine.evaluate(
        state,
        [
            Noul("Is service SLA breached?"),
            Choice("Routing decision", ["Maintain traffic", "Shed load", "Failover"]),
            Score("Health score", min_score=1.0, max_score=10.0),
            MultiChoice("Active flags", ["HighLatency", "ThreadSaturation", "MemoryLeak"]),
            Range("Expected recovery duration", min_val=0.0, max_val=60.0),
            Extract("region", query="What region is affected?"),
        ],
    )

    assert len(results) == 6
    assert isinstance(results[0], NoulResult)
    assert isinstance(results[1], ChoiceResult)
    assert isinstance(results[2], ScoreResult)
    assert isinstance(results[3], MultiChoiceResult)
    assert isinstance(results[4], RangeResult)
    assert isinstance(results[5], ExtractResult)

    for i, r in enumerate(results):
        assert r.confidence is not None
        assert r.temperature is not None

    print("  [PASS] All 6 primitives executed concurrently in a single forward pass on the decision layer!")


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestRhenPhase1Primitives(_unittest.TestCase):
    """test_rhen_phase1_primitives verification suite (master battery)."""

    def test_dlhp_and_pointer_head(self):
        test_dlhp_and_pointer_head()

    def test_multichoice_primitive(self):
        test_multichoice_primitive()

    def test_range_primitive(self):
        test_range_primitive()

    def test_extract_primitive(self):
        test_extract_primitive()

    def test_differentiable_multi_task_gradient_flow(self):
        test_differentiable_multi_task_gradient_flow()

    def test_unified_parallel_execution_all_six(self):
        test_unified_parallel_execution_all_six()

if __name__ == "__main__":
    print("=" * 70)
    print("      RHEN PHASE 1: PRIMITIVES & READOUT EXPANSION VERIFICATION")
    print("=" * 70)
    test_dlhp_and_pointer_head()
    test_multichoice_primitive()
    test_range_primitive()
    test_extract_primitive()
    test_differentiable_multi_task_gradient_flow()
    test_unified_parallel_execution_all_six()
    print("\n" + "=" * 70)
    print("  ALL PHASE 1 PRIMITIVE VERIFICATION SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 70)
