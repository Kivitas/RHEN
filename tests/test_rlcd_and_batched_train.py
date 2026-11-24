"""
Tests for RLCDOptimizer mathematical fixes, Range Beta variance calibration loss,
and PackedRhenSequence batch collation / micro-batch decision accumulation.
"""

from __future__ import annotations

import math
import unittest

import torch

from rhen.config import EOSConfig
from rhen.model import EOSModel
from rhen.research.data import DecisionExample, DecisionQuestion, collate_packed_sequences
from rhen.research.presets import preset_config
from rhen.research.train import TrainingModule
from rhen.rhen_cognition import RLCDOptimizer
from rhen.rhen_loss import compute_rhen_multi_task_loss
from rhen.rhen_packer import PackedBranchMeta, PackedRhenSequence, get_rhen_packer
from rhen.tokenizer import get_tokenizer


class RLCDMathTests(unittest.TestCase):
    def test_compute_reward_positive_margin_for_top1_vs_runner_up(self):
        opt = RLCDOptimizer(margin_weight=0.5, calibration_weight=1.0, seed=42)

        # Top-1 correct answer (p=0.75 vs runner-up=0.15 -> margin = 0.60 > 0)
        top1_correct = opt.compute_reward([0.75, 0.15, 0.10], sampled_index=0, label_index=0)
        self.assertAlmostEqual(top1_correct.correctness, 1.0)
        self.assertAlmostEqual(top1_correct.margin, 0.60, places=5)
        self.assertGreater(top1_correct.margin, 0.0)

        # Narrow top-1 correct answer (p=0.45 vs runner-up=0.40 -> margin = 0.05)
        narrow_correct = opt.compute_reward([0.45, 0.40, 0.15], sampled_index=0, label_index=0)
        self.assertAlmostEqual(narrow_correct.margin, 0.05, places=5)
        self.assertGreater(top1_correct.total, narrow_correct.total)

        # Correct answer that is NOT top-1 (p=0.20 vs runner-up=0.70 -> margin = 0.0)
        suboptimal_correct = opt.compute_reward([0.20, 0.70, 0.10], sampled_index=0, label_index=0)
        self.assertEqual(suboptimal_correct.margin, 0.0)

        # Wrong answer always has 0.0 margin
        wrong = opt.compute_reward([0.10, 0.80, 0.10], sampled_index=1, label_index=0)
        self.assertEqual(wrong.correctness, -1.0)
        self.assertEqual(wrong.margin, 0.0)

    def test_policy_gradient_step_with_old_logits_and_multi_epoch_clipping(self):
        opt = RLCDOptimizer(
            learning_rate=1.5,
            clip_eps=0.1,
            kl_coef=0.05,
            group_size=6,
            seed=7,
        )

        old_logits = torch.tensor([0.2, 1.2, 0.5, -0.3])
        logits = torch.tensor([1.4, 0.4, 0.3, -0.5], requires_grad=True)
        sgd = torch.optim.SGD([logits], lr=opt.learning_rate)

        stats = opt.policy_gradient_step(
            logits,
            label_index=0,
            optimizer=sgd,
            old_logits=old_logits,
            num_epochs=3,
        )

        self.assertEqual(stats["num_epochs"], 3)
        self.assertGreater(stats["kl"], 0.0, "KL divergence against old_logits must be strictly positive")
        self.assertGreater(stats["grad_norm"], 0.0, "Gradients must flow across PPO epochs")
        self.assertTrue(
            any(abs(r - 1.0) > opt.clip_eps for r in stats["ratios"]),
            f"Expected active PPO ratio clipping outside [1-eps, 1+eps], got ratios={stats['ratios']}",
        )
        self.assertGreater(stats["clipped_fraction"], 0.0)

        # Also test multi-epoch without old_logits (reference snapshotted before epoch 1):
        # By epoch 3, logits have moved away from the epoch-0 snapshot, so KL > 0 and ratios != 1.
        logits_multi = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
        sgd_multi = torch.optim.SGD([logits_multi], lr=2.0)
        stats_multi = opt.policy_gradient_step(
            logits_multi,
            label_index=0,
            optimizer=sgd_multi,
            old_logits=None,
            num_epochs=3,
        )
        self.assertGreater(stats_multi["kl"], 0.0)
        self.assertTrue(any(abs(r - 1.0) > 1e-4 for r in stats_multi["ratios"]))


class RangeLossCalibrationTests(unittest.TestCase):
    def _dummy_range_sequence(self) -> PackedRhenSequence:
        meta = PackedBranchMeta(
            question_text="Expected recovery duration",
            options=[],
            branch_token_len=3,
            decide_idx=4,
            option_end_indices=[],
            num_options=0,
            primitive_type="range",
            metadata={"min_val": 0.0, "max_val": 100.0},
        )
        return PackedRhenSequence(
            input_ids=torch.zeros((1, 5), dtype=torch.long),
            attention_mask=torch.zeros((1, 1, 5, 5), dtype=torch.float32),
            state_length=2,
            branch_lengths=[3],
            branches=[meta],
        )

    def test_range_loss_supervises_both_mean_and_concentration(self):
        packed = self._dummy_range_sequence()
        dummy_model = torch.nn.Linear(2, 2)

        # Parameterize alpha = c * u, beta = c * (1 - u) where u is the mean (0.5)
        # and c is the total concentration (alpha + beta).
        # Under pure MSE on mu_norm = alpha / (alpha + beta) = u, d(loss)/dc was identically 0!
        u = torch.tensor(0.5, requires_grad=True)
        concentration = torch.tensor(4.0, requires_grad=True)
        alpha = (concentration * u).reshape(1, 1)
        beta = (concentration * (1.0 - u)).reshape(1, 1)

        precomputed = [
            {
                "branch_index": 0,
                "primitive": "range",
                "min_val": 0.0,
                "max_val": 100.0,
                "expected_value": 50.0,
                "alpha_tensor": alpha,
                "beta_tensor": beta,
                "probabilities": [50.0],
            }
        ]

        # 1. Even when mean matches the target (target=40.0 -> norm_target=0.40, u=0.5),
        # both mean u AND concentration c receive non-zero gradients.
        loss, breakdown = compute_rhen_multi_task_loss(
            model=dummy_model,
            packed_sequence=packed,
            branch_labels=[{"value": 40.0}],
            device=torch.device("cpu"),
            precomputed_results=precomputed,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("b0_range_mse", breakdown)
        self.assertIn("b0_range_var_calib", breakdown)
        self.assertGreater(breakdown["b0_range_var_calib"], 0.0)

        loss.backward()
        self.assertIsNotNone(u.grad)
        self.assertIsNotNone(concentration.grad)
        self.assertGreater(abs(float(u.grad)), 0.0, "Mean must receive gradient")
        self.assertGreater(abs(float(concentration.grad)), 0.0, "Concentration (alpha+beta) must receive gradient")

        # 2. When mean is exact (norm_target=0.5 == u), gradient descent on concentration
        # increases concentration (sharpens the Beta distribution, so d(loss)/d(concentration) < 0).
        u_exact = torch.tensor(0.5, requires_grad=True)
        c_exact = torch.tensor(4.0, requires_grad=True)
        precomputed_exact = [
            {
                **precomputed[0],
                "alpha_tensor": (c_exact * u_exact).reshape(1, 1),
                "beta_tensor": (c_exact * (1.0 - u_exact)).reshape(1, 1),
            }
        ]
        loss_exact, breakdown_exact = compute_rhen_multi_task_loss(
            model=dummy_model,
            packed_sequence=packed,
            branch_labels=[{"value": 50.0}],
            device=torch.device("cpu"),
            precomputed_results=precomputed_exact,
        )
        self.assertAlmostEqual(breakdown_exact["b0_range_mse"], 0.0, places=7)
        self.assertGreater(breakdown_exact["b0_range_var_calib"], 0.0)
        loss_exact.backward()
        self.assertLess(
            float(c_exact.grad),
            0.0,
            "When mean is accurate, negative gradient must push concentration (alpha+beta) higher",
        )


class CollateAndBatchedTrainTests(unittest.TestCase):
    def test_collate_packed_sequences_builds_4d_batch_mask(self):
        tok = get_tokenizer()
        packer = get_rhen_packer(tok)

        seq1 = packer.pack(
            state={"service": "payments", "latency_ms": 120},
            questions_and_options=[("Action?", ["Retry", "Escalate"], "choice", None)],
        )
        seq2 = packer.pack(
            state={"service": "payments-eu-west-long-context", "latency_ms": 4800, "region": "eu-west-1"},
            questions_and_options=[
                ("Action?", ["Retry", "Escalate", "Failover"], "choice", None),
                ("Page on-call?", ["Yes", "No"], "noul", None),
            ],
        )

        len1 = int(seq1.input_ids.shape[1])
        len2 = int(seq2.input_ids.shape[1])
        max_len = max(len1, len2)

        collated = collate_packed_sequences([seq1, seq2], pad_token_id=0)
        self.assertEqual(tuple(collated.input_ids.shape), (2, max_len))
        self.assertEqual(tuple(collated.attention_mask.shape), (2, 1, max_len, max_len))
        self.assertTrue(torch.equal(collated.input_ids[0, :len1], seq1.input_ids[0]))
        self.assertTrue(torch.equal(collated.input_ids[1, :len2], seq2.input_ids[0]))
        if len1 < max_len:
            self.assertTrue(bool((collated.input_ids[0, len1:] == 0).all()))
            self.assertTrue(bool((collated.attention_mask[0, 0, :len1, len1:] == float("-inf")).all()))
        self.assertTrue(torch.equal(collated.attention_mask[0, 0, :len1, :len1], seq1.attention_mask[0, 0]))
        self.assertTrue(torch.equal(collated.attention_mask[1, 0, :len2, :len2], seq2.attention_mask[0, 0]))

    def test_training_module_groups_equal_length_examples_into_batched_forward_rhen(self):
        tok = get_tokenizer()
        torch.manual_seed(0)
        cfg = EOSConfig(**preset_config("tiny", vocab_size=tok.vocab_size))
        model = EOSModel(cfg)
        module = TrainingModule(model, tokenizer=tok, order_consistency_coef=0.1, decision_num_loops=1)

        # Two examples with identical token lengths & branch structure + one with a different length
        ex1 = DecisionExample(
            state={"code": 10},
            questions=[DecisionQuestion("Route", ["A", "B"], "choice", 0)],
        )
        ex2 = DecisionExample(
            state={"code": 20},
            questions=[DecisionQuestion("Route", ["A", "B"], "choice", 1)],
        )
        ex3 = DecisionExample(
            state={"code": 30, "extra_field": "longer_state_context_for_different_length"},
            questions=[DecisionQuestion("Route", ["A", "B", "C"], "choice", 2)],
        )

        batch_sizes_seen = []
        orig_forward_rhen = model.forward_rhen

        def spy_forward_rhen(packed_or_state, *args, **kwargs):
            if hasattr(packed_or_state, "input_ids"):
                batch_sizes_seen.append(int(packed_or_state.input_ids.shape[0]))
            return orig_forward_rhen(packed_or_state, *args, **kwargs)

        model.forward_rhen = spy_forward_rhen
        try:
            loss, stats = module("decision", ([ex1, ex2, ex3], torch.Generator().manual_seed(0)))
        finally:
            model.forward_rhen = orig_forward_rhen

        self.assertIn(2, batch_sizes_seen, f"Expected ex1 and ex2 to be batched together, got {batch_sizes_seen}")
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(stats["decision_loss"], 0.0)
        self.assertGreater(stats["order_consistency"], 0.0)
        loss.backward()
        self.assertIsNotNone(model.embed_tokens.weight.grad)


if __name__ == "__main__":
    unittest.main()
