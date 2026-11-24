"""Tests for Effort Levels ("low" | "medium" | "high") and Intra-Branch Option Isolation.

Covers:
1. EOSConfig.from_effort("low" | "medium" | "high"), EFFORT_LEVELS, overrides, roundtrip.
2. RhenEngine(effort=...) and RhenEngine.from_effort(...), per-call effort / num_loops
   override across all primitives and top-level rhen.decide / rhen.evaluate.
3. CPU latency of 77-option choice() under effort="low" (< 0.5s).
4. RhenPacker option token cache and hoisted opt_end_token_id.
5. Rhen2DBranchMask intra-branch option isolation mask entries (-inf across options,
   0.0 for state/question/self-causal/<|decide|>).
6. RhenPacker position_ids reset per option and end-to-end permutation invariance
   (max probability diff < 1e-5) for both packer-enabled and model-config-enforced
   option isolation, including multi-loop models with SWA + SSM highway + Diff-MLA.
"""

from __future__ import annotations

import random
import time
import unittest

import torch

from rhen.config import EFFORT_LEVELS, EOSConfig, RhenEngineConfig
from rhen.model import EOSModel
from rhen.optim.rhen_mask import Rhen2DBranchMask
from rhen.rhen import (
    Choice,
    Extract,
    MultiChoice,
    Noul,
    Range,
    RhenEngine,
    Score,
    decide,
    evaluate,
)
from rhen.rhen_packer import RhenPacker, extract_branch_option_spans
from rhen.tokenizer import get_tokenizer


class TestEffortLevelsConfig(unittest.TestCase):
    def test_effort_levels_constant_and_defaults(self):
        self.assertEqual(EFFORT_LEVELS, ("low", "medium", "high"))
        default_cfg = EOSConfig()
        self.assertEqual(default_cfg.effort, "high")
        self.assertFalse(default_cfg.isolate_branch_options)
        self.assertFalse(default_cfg.rhen_engine.isolate_branch_options)

    def test_from_effort_low(self):
        cfg = EOSConfig.from_effort("low")
        self.assertEqual(cfg.effort, "low")
        self.assertEqual(cfg.hidden_size, 128)
        self.assertEqual(cfg.intermediate_size, 256)
        self.assertEqual(cfg.num_pre_layers, 1)
        self.assertEqual(cfg.num_recurrent_layers, 1)
        self.assertEqual(cfg.num_post_layers, 1)
        self.assertEqual(cfg.max_recurrent_loops, 1)
        self.assertEqual(cfg.min_recurrent_loops, 1)
        self.assertEqual(cfg.thinking_budget_min_loops, 1)
        self.assertEqual(cfg.thinking_budget_max_loops, 1)
        self.assertEqual(cfg.num_attention_heads, 4)
        self.assertEqual(cfg.num_key_value_heads, 2)
        self.assertEqual(cfg.kv_lora_rank, 32)
        self.assertEqual(cfg.q_lora_rank, 64)
        self.assertEqual(cfg.rhen_hidden_dim, 64)
        self.assertEqual(cfg.rhen_pointer_rank, 32)
        self.assertFalse(cfg.use_moe)
        self.assertFalse(cfg.use_shared_expert_moe)

    def test_from_effort_medium(self):
        cfg = EOSConfig.from_effort("medium")
        self.assertEqual(cfg.effort, "medium")
        self.assertEqual(cfg.hidden_size, 256)
        self.assertEqual(cfg.intermediate_size, 512)
        self.assertEqual(cfg.num_pre_layers, 1)
        self.assertEqual(cfg.num_recurrent_layers, 2)
        self.assertEqual(cfg.num_post_layers, 1)
        self.assertEqual(cfg.max_recurrent_loops, 4)
        self.assertEqual(cfg.min_recurrent_loops, 2)
        self.assertEqual(cfg.act_min_loops, 2)
        self.assertEqual(cfg.thinking_budget_min_loops, 2)
        self.assertEqual(cfg.thinking_budget_max_loops, 4)
        self.assertEqual(cfg.num_attention_heads, 4)
        self.assertEqual(cfg.num_key_value_heads, 2)
        self.assertEqual(cfg.kv_lora_rank, 64)
        self.assertEqual(cfg.q_lora_rank, 128)
        self.assertEqual(cfg.rhen_hidden_dim, 128)
        self.assertEqual(cfg.rhen_pointer_rank, 64)
        self.assertFalse(cfg.use_moe)
        self.assertFalse(cfg.use_shared_expert_moe)

    def test_from_effort_high_matches_deep_default(self):
        cfg_high = EOSConfig.from_effort("high")
        cfg_default = EOSConfig()
        self.assertEqual(cfg_high.effort, "high")
        self.assertEqual(cfg_high.hidden_size, cfg_default.hidden_size)
        self.assertEqual(cfg_high.max_recurrent_loops, 12)
        self.assertEqual(cfg_high.num_recurrent_layers, 4)
        self.assertTrue(cfg_high.use_moe)
        self.assertTrue(cfg_high.use_shared_expert_moe)

    def test_from_effort_overrides_and_invalid(self):
        cfg = EOSConfig.from_effort("low", isolate_branch_options=True, hidden_size=64)
        self.assertTrue(cfg.isolate_branch_options)
        self.assertTrue(cfg.rhen_engine.isolate_branch_options)
        self.assertEqual(cfg.hidden_size, 64)
        self.assertEqual(cfg.effort, "low")

        # Dict roundtrip preserves effort and isolate_branch_options
        rebuilt = EOSConfig.from_dict(cfg.to_dict())
        self.assertEqual(rebuilt.effort, "low")
        self.assertTrue(rebuilt.isolate_branch_options)

        # Nested group construction also works
        cfg_nested = EOSConfig(rhen_engine=RhenEngineConfig(isolate_branch_options=True))
        self.assertTrue(cfg_nested.isolate_branch_options)

        with self.assertRaises(ValueError):
            EOSConfig.from_effort("ultra")


class TestEffortLevelsEngineAndLatency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_tokenizer()
        cls.cpu = torch.device("cpu")

    def test_engine_effort_constructors_and_loop_resolution(self):
        torch.manual_seed(0)
        eng_low = RhenEngine(effort="low", tokenizer=self.tokenizer, device=self.cpu)
        self.assertEqual(eng_low.effort, "low")
        self.assertEqual(eng_low.model.config.hidden_size, 128)
        self.assertEqual(eng_low.model.config.max_recurrent_loops, 1)

        eng_from = RhenEngine.from_effort(
            "medium", tokenizer=self.tokenizer, device=self.cpu, isolate_branch_options=True
        )
        self.assertEqual(eng_from.effort, "medium")
        self.assertEqual(eng_from.model.config.max_recurrent_loops, 4)
        self.assertTrue(eng_from.packer.isolate_branch_options)

        # Loop resolution helper
        self.assertEqual(eng_from._resolve_effort_loops(effort="low"), 1)
        self.assertEqual(eng_from._resolve_effort_loops(effort="medium"), 2)
        self.assertEqual(eng_from._resolve_effort_loops(effort="high"), 4)
        self.assertEqual(eng_from._resolve_effort_loops(effort="low", num_loops=3), 3)
        self.assertIsNone(eng_from._resolve_effort_loops())

        with self.assertRaises(ValueError):
            eng_from._resolve_effort_loops(effort="invalid")

    def test_per_call_effort_controls_recurrent_loop_count(self):
        torch.manual_seed(0)
        cfg = EOSConfig.from_effort("medium", max_recurrent_loops=6, hidden_size=64, intermediate_size=128)
        eng = RhenEngine(config=cfg, tokenizer=self.tokenizer, device=self.cpu)

        call_counts = []
        handle = eng.model.recurrent_core.register_forward_hook(lambda m, i, o: call_counts.append(1))
        try:
            state = {"ticket": "I lost my debit card abroad"}
            opts = ["lost_card", "pin_reset", "transfer_delay"]

            call_counts.clear()
            eng.choice(state, "Intent?", opts, effort="low")
            self.assertEqual(len(call_counts), 1)

            call_counts.clear()
            eng.choice(state, "Intent?", opts, effort="medium")
            self.assertEqual(len(call_counts), 3)  # min(4, 6 // 2) == 3

            call_counts.clear()
            eng.choice(state, "Intent?", opts, effort="high")
            self.assertEqual(len(call_counts), 6)

            call_counts.clear()
            eng.noul(state, "Is this urgent?", effort="low")
            self.assertEqual(len(call_counts), 1)

            call_counts.clear()
            eng.score(state, "Severity?", effort="low")
            self.assertEqual(len(call_counts), 1)

            call_counts.clear()
            eng.multichoice(state, "Tags?", opts, effort="low")
            self.assertEqual(len(call_counts), 1)

            call_counts.clear()
            eng.range(state, "Minutes?", effort="low")
            self.assertEqual(len(call_counts), 1)

            call_counts.clear()
            eng.extract(state, "ticket", effort="low")
            self.assertEqual(len(call_counts), 1)

            call_counts.clear()
            eng.decide_parallel(state, [Choice("Intent?", opts), Noul("Urgent?")], effort="low")
            self.assertEqual(len(call_counts), 1)

            call_counts.clear()
            decide(state, Choice("Intent?", opts), engine=eng, effort="low")
            self.assertEqual(len(call_counts), 1)

            call_counts.clear()
            evaluate(state, Choice("Intent?", opts), engine=eng, num_loops=2)
            self.assertEqual(len(call_counts), 2)
        finally:
            handle.remove()

    def test_mock_model_without_num_loops_param_is_safe(self):
        class _LegacyModel(torch.nn.Module):
            def evaluate_rhen(self, packed_or_state, questions_and_options, tokenizer):
                return [
                    {
                        "chosen_index": 0,
                        "chosen_option": "A",
                        "probabilities": [0.8, 0.2],
                        "confidence": 0.8,
                        "temperature": 1.0,
                    }
                ]

        eng = RhenEngine(model=_LegacyModel(), tokenizer=self.tokenizer, device=self.cpu)
        res = eng.choice("state", "Pick?", ["A", "B"], effort="low")
        self.assertEqual(res.choice_label, "A")

    def test_low_effort_77_options_cpu_latency_under_500ms(self):
        torch.manual_seed(0)
        eng = RhenEngine(effort="low", tokenizer=self.tokenizer, device=self.cpu)
        options_77 = [f"banking_intent_category_{i:02d}" for i in range(77)]
        state = "I tried to withdraw cash from an ATM and my card was declined twice."

        # Warmup (populates _opt_token_cache and PyTorch CPU thread pool)
        res_warm = eng.choice(state, "Classify customer banking intent", options_77)
        self.assertEqual(len(res_warm.raw_probabilities), 77)
        self.assertGreaterEqual(len(eng.packer._opt_token_cache), 77)

        t0 = time.perf_counter()
        res = eng.choice(state, "Classify customer banking intent", options_77)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(res.raw_probabilities), 77)
        self.assertAlmostEqual(sum(res.raw_probabilities), 1.0, places=5)
        self.assertLess(
            elapsed,
            0.5,
            f"77-option low-effort choice() took {elapsed * 1000:.1f}ms (expected < 500ms)",
        )


class TestIntraBranchOptionIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_tokenizer()
        cls.cpu = torch.device("cpu")

    def test_mask_blocks_cross_option_attention_and_keeps_decide_visible(self):
        # State: 0..3 (len 4)
        # Branch 0: 4..14 (len 11): question [4, 5], opt0 [6, 8], opt1 [9, 10], opt2 [11, 13], decide [14]
        spans = [[(6, 8), (9, 10), (11, 13)]]
        mask = Rhen2DBranchMask.create_mask(
            state_length=4,
            branch_lengths=[11],
            branch_option_spans=spans,
            isolate_options=True,
        )[0, 0]

        # Cross-option entries must all be -inf
        for a, (sa, ea) in enumerate(spans[0]):
            for b, (sb, eb) in enumerate(spans[0]):
                if a != b:
                    block = mask[sa : ea + 1, sb : eb + 1]
                    self.assertTrue(
                        bool(torch.isinf(block).all() and (block < 0).all()),
                        f"Expected -inf between opt {a} and opt {b}",
                    )
                else:
                    # Intra-option causal
                    for i in range(sa, ea + 1):
                        for j in range(sa, ea + 1):
                            if j <= i:
                                self.assertEqual(mask[i, j].item(), 0.0)
                            else:
                                self.assertTrue(torch.isinf(mask[i, j]) and mask[i, j] < 0)

            # Each option attends to state [0..3] and question [4..5]
            self.assertTrue(bool((mask[sa : ea + 1, :6] == 0.0).all()))

        # <|decide|> at index 14 attends to state, question, all options, and itself
        self.assertTrue(bool((mask[14, :15] == 0.0).all()))

    def test_packer_position_ids_and_cache(self):
        packer = RhenPacker(self.tokenizer, isolate_branch_options=True)
        options = ["short", "a much longer candidate option label", "mid label"]
        packed = packer.pack(
            state="Customer reports unauthorized charge on credit card.",
            questions_and_options=[("Select category", options)],
        )

        self.assertIsNotNone(packed.position_ids)
        pos = packed.position_ids[0].tolist()
        seq_len = packed.input_ids.shape[1]
        self.assertEqual(len(pos), seq_len)

        # State tokens: 0 .. state_len - 1
        state_len = packed.state_length
        self.assertEqual(pos[:state_len], list(range(state_len)))

        spans = extract_branch_option_spans(packed.branches)[0]
        self.assertEqual(len(spans), 3)
        first_opt_start = spans[0][0]
        q_len = first_opt_start - state_len
        q_end_pos = state_len + q_len - 1

        # Question prefix starts at state_len
        self.assertEqual(pos[state_len:first_opt_start], list(range(state_len, state_len + q_len)))

        # Every option resets its position counter to q_end_pos + 1
        max_opt_len = 0
        for s_k, e_k in spans:
            opt_len = e_k - s_k + 1
            max_opt_len = max(max_opt_len, opt_len)
            self.assertEqual(pos[s_k : e_k + 1], list(range(q_end_pos + 1, q_end_pos + 1 + opt_len)))

        # <|decide|> gets q_end_pos + 1 + max_opt_len
        decide_idx = packed.branches[0].decide_idx
        self.assertEqual(pos[decide_idx], q_end_pos + 1 + max_opt_len)

        # Verify mask inside packed sequence blocks cross-option attention
        m = packed.attention_mask[0, 0]
        for a, (sa, ea) in enumerate(spans):
            for b, (sb, eb) in enumerate(spans):
                if a != b:
                    self.assertTrue(bool(torch.isinf(m[sa : ea + 1, sb : eb + 1]).all()))
        self.assertTrue(bool((m[decide_idx, : decide_idx + 1] == 0.0).all()))

        # Verify _opt_token_cache caches each option string
        self.assertEqual(len(packer._opt_token_cache), 3)

    def test_permutation_invariance_low_effort_and_multiloop_swa_ssm(self):
        options = [
            "card_not_working",
            "lost_or_stolen_card_urgent_replacement",
            "pin",
            "exchange_rate_inquiry",
            "pending_cash_withdrawal",
        ]
        state = {"message": "My card got declined at the grocery store and PIN failed."}
        question = "What is the customer intent?"

        configs = [
            ("low_effort", EOSConfig.from_effort("low", isolate_branch_options=True)),
            (
                "multiloop_swa_ssm_diff",
                EOSConfig.from_effort(
                    "low",
                    isolate_branch_options=True,
                    num_recurrent_layers=2,
                    max_recurrent_loops=3,
                    interleave_swa=True,
                    swa_window_size=64,
                    enable_ssm_highway=True,
                    use_diff_mla=True,
                ),
            ),
        ]

        perms = [
            list(reversed(options)),
            options[2:] + options[:2],
            [options[3], options[0], options[4], options[1], options[2]],
        ]

        for name, cfg in configs:
            with self.subTest(config=name):
                torch.manual_seed(42)
                engine = RhenEngine(config=cfg, tokenizer=self.tokenizer, device=self.cpu)
                base_res = engine.choice(state, question, options)
                for perm in perms:
                    perm_res = engine.choice(state, question, perm)
                    diffs = [
                        abs(base_res.probabilities[o] - perm_res.probabilities[o])
                        for o in options
                    ]
                    max_diff = max(diffs)
                    self.assertLess(
                        max_diff,
                        1e-5,
                        f"[{name}] permutation max_diff={max_diff:.2e} >= 1e-5",
                    )

    def test_forward_rhen_enforces_isolation_even_on_non_isolated_packed_input(self):
        torch.manual_seed(7)
        cfg = EOSConfig.from_effort("low", isolate_branch_options=True)
        model = EOSModel(cfg).eval()
        plain_packer = RhenPacker(self.tokenizer, isolate_branch_options=False)

        options = ["alpha_option", "beta_very_long_option_text", "gamma"]
        perm = ["gamma", "alpha_option", "beta_very_long_option_text"]
        state = "Server CPU spiked above 95% for ten minutes."
        question = "Choose remediation action"

        packed_a = plain_packer.pack(state, [(question, options)])
        packed_b = plain_packer.pack(state, [(question, perm)])
        self.assertIsNone(packed_a.position_ids)
        self.assertIsNone(packed_b.position_ids)

        res_a = model.evaluate_rhen(packed_a)[0]
        res_b = model.evaluate_rhen(packed_b)[0]
        probs_a = dict(zip(options, res_a["probabilities"]))
        probs_b = dict(zip(perm, res_b["probabilities"]))
        max_diff = max(abs(probs_a[o] - probs_b[o]) for o in options)
        self.assertLess(max_diff, 1e-5, f"max_diff={max_diff:.2e} >= 1e-5")

    def test_permutation_invariance_77_options(self):
        torch.manual_seed(123)
        engine = RhenEngine.from_effort(
            "low",
            tokenizer=self.tokenizer,
            device=self.cpu,
            isolate_branch_options=True,
        )
        options = [f"intent_{i:02d}" + ("_extra" if i % 3 == 0 else "") for i in range(77)]
        rng = random.Random(99)
        perm = list(options)
        rng.shuffle(perm)

        state = "Why was I charged a foreign transaction fee on my debit card?"
        res_a = engine.choice(state, "Classify intent", options)
        res_b = engine.choice(state, "Classify intent", perm)

        max_diff = max(abs(res_a.probabilities[o] - res_b.probabilities[o]) for o in options)
        self.assertLess(max_diff, 1e-5, f"77-option permutation max_diff={max_diff:.2e} >= 1e-5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
