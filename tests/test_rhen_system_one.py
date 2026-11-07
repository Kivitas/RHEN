"""
Rhen System One Decision Engine - Comprehensive Verification Suite.

Validates:
1. 2D Block-Causal Branch Mask Engine (hybrid state prefix + intra-branch causality + cross-branch isolation)
2. Delimiter Packaging Engine (state serialization + coordinate extraction for h_q and h_o)
3. Multi-Head Bilinear Factorized Pointer Head (H=4 specialized heads + adaptive temperature)
4. Typed Decision Primitives (Noul, Choice, Score, Cascade)
5. Multi-Task Losses (Choice CE, Noul BCE, Score RPS, Slot Orthogonality)
"""

import torch

from config import EOSConfig
from model import EOSModel, MultiHeadRhenPointerHead
from optim.rhen_mask import Rhen2DBranchMask
from rhen_packer import get_rhen_packer
from rhen import RhenEngine, Noul, Choice, Score, NoulResult, ChoiceResult, ScoreResult
from rhen_loss import compute_rhen_multi_task_loss


def test_2d_branch_mask():
    print("\n--- Test 1: 2D Block-Causal Branch Mask Engine ---")
    state_len = 4
    branch_lens = [3, 2, 4]
    mask = Rhen2DBranchMask.create_mask(state_len, branch_lens).squeeze()
    total_len = state_len + sum(branch_lens)
    assert mask.shape == (total_len, total_len), f"Expected shape ({total_len}, {total_len}), got {mask.shape}"

    # 1. State attends to itself bidirectionally
    assert torch.all(mask[:state_len, :state_len] == 0.0), "State self-attention must be 0.0"

    # 2. Branch tokens attend to full state prefix
    curr = state_len
    for b_idx, b_len in enumerate(branch_lens):
        b_end = curr + b_len
        assert torch.all(mask[curr:b_end, :state_len] == 0.0), f"Branch {b_idx} must attend to state prefix"
        
        # Intra-branch causal attention
        branch_block = mask[curr:b_end, curr:b_end]
        tril_idx = torch.tril_indices(b_len, b_len, offset=0)
        triu_idx = torch.triu_indices(b_len, b_len, offset=1)
        assert torch.all(branch_block[tril_idx[0], tril_idx[1]] == 0.0), f"Branch {b_idx} lower triangle must be 0.0"
        assert torch.all(branch_block[triu_idx[0], triu_idx[1]] == float("-inf")), f"Branch {b_idx} upper triangle must be -inf"
        
        # Cross-branch isolation
        other_start = state_len
        for other_idx, other_len in enumerate(branch_lens):
            other_end = other_start + other_len
            if other_idx != b_idx:
                assert torch.all(mask[curr:b_end, other_start:other_end] == float("-inf")), \
                    f"Branch {b_idx} must be strictly isolated (-inf) from Branch {other_idx}"
            other_start = other_end
            
        curr = b_end

    print("  [PASS] 2D Block-Causal Branch Mask strictly isolates branches with O(1) state prefix compute.")


def test_packer():
    print("\n--- Test 2: Delimiter Packaging & Coordinate Extraction ---")
    packer = get_rhen_packer()
    state = {
        "user_id": 1042,
        "subscription": "pro_tier",
        "last_login_hours": 3,
        "failed_attempts": 0,
    }
    questions = [
        ("Is user verified?", ["Yes", "No"]),
        ("Upgrade path?", ["Enterprise", "Team", "None"]),
        ("Risk severity?", ["Level 0", "Level 1", "Level 2", "Level 3"]),
    ]
    packed = packer.pack(state, questions)

    assert packed.state_length > 0, "State length must be > 0"
    assert len(packed.branches) == 3, "Must have 3 branches"
    assert packed.input_ids.shape[1] == packed.state_length + sum(packed.branch_lengths)
    
    for b_idx, b_meta in enumerate(packed.branches):
        assert b_meta.decide_idx > 0, f"Branch {b_idx} decide_idx must be valid"
        assert len(b_meta.option_end_indices) == len(questions[b_idx][1]), \
            f"Branch {b_idx} option count mismatch"
        for opt_idx in b_meta.option_end_indices:
            assert opt_idx < b_meta.decide_idx, f"Option index {opt_idx} must precede decide token {b_meta.decide_idx}"

    print("  [PASS] Delimiter Packaging accurately coordinates <|decide|> and <|/opt|> indices.")


def test_multi_head_pointer():
    print("\n--- Test 3: Multi-Head Bilinear Factorized Pointer Head ---")
    hidden_size = 560
    num_heads = 4
    rank = 128
    pointer_head = MultiHeadRhenPointerHead(hidden_size=hidden_size, num_heads=num_heads, rank=rank)

    q_repr = torch.randn(1, 1, hidden_size)
    opt_reprs = torch.randn(1, 5, hidden_size)

    details = pointer_head(q_repr, opt_reprs, return_details=True)
    assert details["probabilities"].shape == (1, 5), "Probabilities shape mismatch"
    assert torch.allclose(details["probabilities"].sum(dim=-1), torch.ones(1), atol=1e-5), \
        "Probabilities must sum to 1.0"
    assert details["head_scores"].shape == (1, 5, num_heads), "Head scores shape mismatch"
    assert details["temperature"].item() > 0.0, "Temperature must be positive"

    print(f"  [PASS] Multi-Head Pointer Head: H={num_heads} heads, rank={rank}, tau={details['temperature'].item():.3f}")


def test_typed_primitives():
    print("\n--- Test 4: Typed Decision Primitives (Noul, Choice, Score, Cascade) ---")
    config = EOSConfig(
        vocab_size=50048,
        hidden_size=560,
        max_recurrent_loops=4,
        enable_rhen=True,
    )
    model = EOSModel(config)
    engine = RhenEngine(model=model)

    state = {
        "engine_rpm": 3200,
        "oil_pressure_psi": 45.2,
        "coolant_temp_c": 92.0,
        "warning_lights": ["check_engine"],
    }

    # 1. Parallel Evaluation of mixed primitives
    results = engine.evaluate(
        state=state,
        questions=[
            Noul("Is coolant temperature within normal operational bounds?"),
            Choice("Recommended action", ["Continue monitoring", "Schedule maintenance", "Emergency shutdown"]),
            Score("Fault severity level", min_score=0.0, max_score=10.0),
        ]
    )

    noul_res, choice_res, score_res = results
    assert isinstance(noul_res, NoulResult)
    assert 0.0 <= noul_res.p_true <= 1.0
    assert isinstance(noul_res.decision, bool)

    assert isinstance(choice_res, ChoiceResult)
    assert choice_res.choice_label in ["Continue monitoring", "Schedule maintenance", "Emergency shutdown"]
    assert len(choice_res.probabilities) == 3
    assert abs(sum(choice_res.probabilities.values()) - 1.0) < 1e-4

    assert isinstance(score_res, ScoreResult)
    assert 0.0 <= score_res.score <= 10.0
    assert len(score_res.level_probabilities) == 5

    # 2. Decision Cascade
    stages = [
        lambda ctx: Noul("Is maintenance required immediately?"),
        lambda ctx: Choice("Dispatch crew type", ["Mechanical", "Electrical", "Standard"]) if ctx.get("Is maintenance required immediately?") else Choice("Log priority", ["Normal", "Low"]),
    ]
    cascade_res = engine.cascade(state, stages)
    assert len(cascade_res) == 2

    print(f"  [PASS] Noul: P(True)={noul_res.p_true:.4f}, Decision={noul_res.decision}")
    print(f"  [PASS] Choice: Chosen='{choice_res.choice_label}', Probabilities={choice_res.probabilities}")
    print(f"  [PASS] Score: Expected Score={score_res.score:.2f}/10.0")
    print(f"  [PASS] Cascade: Completed {len(cascade_res)} stages successfully.")


def test_multi_task_losses():
    print("\n--- Test 5: Multi-Task Losses (Choice CE, Noul BCE, Score RPS, Ortho) ---")
    config = EOSConfig(vocab_size=50048, hidden_size=560, max_recurrent_loops=3, enable_rhen=True)
    model = EOSModel(config)
    device = torch.device("cpu")

    packer = get_rhen_packer()
    packed = packer.pack(
        state={"transaction_amount": 12500, "flagged": True},
        questions_and_options=[
            ("Is fraudulent?", ["Yes", "No"]),
            ("Routing tier", ["Tier 1", "Tier 2", "Tier 3"]),
            ("Risk scale", ["L0", "L1", "L2", "L3", "L4"]),
        ]
    )

    labels = [True, 1, 3.5]  # [Noul True, Choice 1, Score 3.5]
    total_loss, breakdown = compute_rhen_multi_task_loss(model, packed, labels, device)

    assert total_loss.item() > 0.0, "Total loss must be positive"
    assert "b0_noul_bce" in breakdown, "Noul BCE missing"
    assert "b1_choice_ce" in breakdown, "Choice CE missing"
    assert "b2_score_rps" in breakdown, "Score RPS missing"
    assert "rhen_ortho" in breakdown, "Rhen Ortho missing"

    print(f"  [PASS] Multi-Task Losses: Total={total_loss.item():.4f}")
    for k, v in breakdown.items():
        print(f"    - {k}: {v:.4f}")


def run_all_tests():
    print("=" * 70)
    print("      RHEN SYSTEM ONE DECISION ENGINE - MASTER VERIFICATION")
    print("=" * 70)

    test_2d_branch_mask()
    test_packer()
    test_multi_head_pointer()
    test_typed_primitives()
    test_multi_task_losses()

    print("\n" + "=" * 70)
    print("  ALL 5 MASTER VERIFICATION TEST SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 70)


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestRhenSystemOne(_unittest.TestCase):
    """test_rhen_system_one verification suite (master battery)."""

    def test_2d_branch_mask(self):
        test_2d_branch_mask()

    def test_packer(self):
        test_packer()

    def test_multi_head_pointer(self):
        test_multi_head_pointer()

    def test_typed_primitives(self):
        test_typed_primitives()

    def test_multi_task_losses(self):
        test_multi_task_losses()

if __name__ == "__main__":
    run_all_tests()
