"""
Regression tests for the model core in ``model.py``.

Covers the causal LM path (prefix invariance, causal SSM memory, relative RoPE),
the decision path (Rhen only when requested, state-only reset, full loop budget),
loss plumbing (the Score RPS target, chunked CE with z-loss and checkpointing, straight-through aux cap,
torch-seeded loop sampling), the cache and generation contracts, the adaptive loop
gate, memory modules, and the mUP / thinking-budget wiring.

Every model is a tiny CPU configuration; the whole file runs in well under a minute.
"""

import unittest
import warnings

import torch

from config import EOSConfig
from model import (
    AdaptiveLoopRouter,
    DynamicThinkingBudgetController,
    EOSModel,
    ModernHopfieldEpisodicBank,
    MultiLatentAttention,
    RotaryEmbedding,
    StateSpaceMemoryHighway,
    TitansNeuralMemory,
    apply_repetition_penalty,
)
from optim.checkpoint import CheckpointedRecurrentCore
from rhen_loss import _ordinal_cdf_target
from rhen_packer import RhenPacker
from tokenizer import get_tokenizer


def _tiny(**overrides) -> EOSConfig:
    base = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        kv_lora_rank=8,
        q_lora_rank=16,
        num_pre_layers=1,
        num_recurrent_layers=2,
        num_post_layers=1,
        max_recurrent_loops=3,
        max_position_embeddings=64,
        rhen_hidden_dim=16,
        rhen_pointer_rank=8,
        rhen_num_slots=4,
        rhen_num_heads=2,
        num_routed_experts=4,
        num_active_routed=2,
        num_experts=4,
        enable_act=False,
        enable_thinking_budget=False,
        enable_mtp=False,
        min_recurrent_loops=1,
    )
    base.update(overrides)
    return EOSConfig(**base)


def _rope_tables(inv_freq, start, length):
    """cos/sin tables for absolute positions start .. start+length-1."""
    pos = torch.arange(start, start + length, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class CausalLMPathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_lm_logits_prefix_invariance(self):
        model = EOSModel(_tiny()).eval()
        ids = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            full = model(ids, num_loops=3).logits
            prefix = model(ids[:, :5], num_loops=3).logits
        diff = float((full[:, :5] - prefix).abs().max())
        self.assertTrue(torch.allclose(full[:, :5], prefix, atol=1e-5), f"max diff {diff}")

    def test_ssm_highway_memory_is_causal(self):
        ssm = StateSpaceMemoryHighway(hidden_size=16, state_dim=8).eval()
        h = torch.randn(1, 6, 16)
        h_future = h.clone()
        h_future[:, 4:] += 5.0
        with torch.no_grad():
            out_a = ssm(h, step_idx=1)
            out_b = ssm(h_future, step_idx=1)  # step 1 resets the memory
        self.assertTrue(torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-6),
                        "positions before the change must not see future tokens")
        self.assertFalse(torch.allclose(out_a[:, 4:], out_b[:, 4:]))
        self.assertEqual(tuple(ssm.memory_state.shape), (1, 6, 8))
        self.assertFalse(ssm.memory_state.requires_grad, "memory must be stored detached")

    def test_mla_scores_depend_only_on_relative_position(self):
        # The standard (non-differential) path: rotating q and the full k with the same
        # tables makes the output invariant to a common shift of all positions.
        cfg = EOSConfig(hidden_size=32, num_attention_heads=4, head_dim=8, kv_lora_rank=8,
                        q_lora_rank=16, use_diff_mla=False, use_differential_attention=False)
        attn = MultiLatentAttention(cfg, layer_idx=0).eval()
        rope = RotaryEmbedding(dim=8, base=10000.0, yarn_scale_factor=1.0)
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            out0, _ = attn(x, *_rope_tables(rope.inv_freq, 0, 6))
            out7, _ = attn(x, *_rope_tables(rope.inv_freq, 7, 6))
        self.assertTrue(torch.allclose(out0, out7, atol=1e-5),
                        f"max diff {float((out0 - out7).abs().max())}")

    def test_lm_path_never_runs_rhen(self):
        model = EOSModel(_tiny()).train()
        ids = torch.randint(0, 64, (2, 8))
        model(ids, labels=ids, num_loops=2)
        self.assertIsNone(model.recurrent_core._last_rhen_conf)


def _mla_config(**overrides) -> EOSConfig:
    base = dict(hidden_size=32, num_attention_heads=4, head_dim=8, kv_lora_rank=8, q_lora_rank=16)
    base.update(overrides)
    return EOSConfig(**base)


class ConfigFlagTests(unittest.TestCase):
    def test_explicit_differential_false_selects_standard_attention(self):
        torch.manual_seed(0)
        ref = MultiLatentAttention(_mla_config(use_diff_mla=False, use_differential_attention=False), layer_idx=0).eval()
        explicit = MultiLatentAttention(_mla_config(use_diff_mla=True, use_differential_attention=False), layer_idx=0).eval()
        explicit.load_state_dict(ref.state_dict())
        rope = RotaryEmbedding(dim=8, base=10000.0, yarn_scale_factor=1.0)
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            out_ref = ref(x, *_rope_tables(rope.inv_freq, 0, 6))[0]
            out_explicit = explicit(x, *_rope_tables(rope.inv_freq, 0, 6))[0]
        self.assertTrue(torch.allclose(out_ref, out_explicit, atol=1e-6),
                        "an explicit use_differential_attention=False must win over use_diff_mla=True")

    def test_mla_accepts_a_leading_singleton_position_table(self):
        torch.manual_seed(0)
        attn = MultiLatentAttention(_mla_config(use_diff_mla=False, use_differential_attention=False), layer_idx=0).eval()
        rope = RotaryEmbedding(dim=8, base=10000.0, yarn_scale_factor=1.0)
        cos, sin = _rope_tables(rope.inv_freq, 0, 6)
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            flat = attn(x, cos, sin)[0]
            batched = attn(x, cos.unsqueeze(0), sin.unsqueeze(0))[0]
            with self.assertRaises(ValueError):
                attn(x, cos.unsqueeze(0).repeat(2, 1, 1), sin.unsqueeze(0).repeat(2, 1, 1))
        self.assertTrue(torch.allclose(flat, batched, atol=1e-6))

    def test_selective_checkpoint_wraps_only_under_gradient_checkpointing(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            off = EOSModel(_tiny(enable_selective_checkpoint=True, gradient_checkpointing=False))
        self.assertNotIsInstance(off.recurrent_core, CheckpointedRecurrentCore)
        self.assertTrue(any("enable_selective_checkpoint" in str(w.message) for w in caught),
                        "selective checkpointing without gradient checkpointing must warn")

        # The SSM memory and the aux-free congestion bias are per-sequence state that
        # checkpoint recomputation does not replay; they are off so the backward is stable.
        on = EOSModel(_tiny(enable_selective_checkpoint=True, gradient_checkpointing=True,
                            enable_ssm_highway=False, use_aux_free_balancing=False)).train()
        self.assertIsInstance(on.recurrent_core, CheckpointedRecurrentCore)
        self.assertTrue(on.recurrent_core.gradient_checkpointing)
        ids = torch.randint(0, 64, (2, 8))
        on(ids, labels=ids, num_loops=2).loss.backward()
        self.assertIsNotNone(on.embed_tokens.weight.grad)

        plain = EOSModel(_tiny())
        self.assertNotIsInstance(plain.recurrent_core, CheckpointedRecurrentCore)

    def test_gradient_checkpointing_flag_survives_construction(self):
        cfg = _tiny(gradient_checkpointing=True, enable_ssm_highway=False, use_aux_free_balancing=False)
        model = EOSModel(cfg).train()
        self.assertTrue(cfg.gradient_checkpointing, "transformers must not consume the flag")
        ids = torch.randint(0, 64, (2, 8))
        model(ids, labels=ids, num_loops=2).loss.backward()  # previously raised AttributeError
        self.assertTrue(EOSModel(cfg).config.gradient_checkpointing)


class GradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_first_recurrent_layer_router_gets_gradient(self):
        model = EOSModel(_tiny(loop_dropout_prob=0.0)).train()
        ids = torch.randint(0, 64, (2, 8))
        model(ids, labels=ids, num_loops=2).loss.backward()
        grad = model.recurrent_core.layers[0].mlp.router.weight.grad
        self.assertIsNotNone(grad, "first recurrent layer router received no gradient")
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_titans_memory_is_stored_detached_in_train_mode(self):
        mem = TitansNeuralMemory(hidden_size=16, d_mem=4).train()
        h = torch.randn(2, 5, 16, requires_grad=True)
        mem(h)
        self.assertIsNotNone(mem.memory_matrix)
        self.assertFalse(mem.memory_matrix.requires_grad)

    def test_adaptive_gate_full_update_for_active_none_for_frozen(self):
        router = AdaptiveLoopRouter(hidden_size=4, min_loops=0, freeze_threshold=0.5)
        with torch.no_grad():
            router.halt_proj.weight.zero_()
            router.halt_proj.weight[0, 0] = 50.0  # p = sigmoid(50 * x0)
            router.halt_proj.bias.zero_()
        h = torch.randn(1, 4, 4)
        nxt = torch.randn(1, 4, 4)
        nxt[0, :, 0] = torch.tensor([-1.0, -2.0, 1.0, 2.0])  # tokens 0,1 active; 2,3 frozen
        updated, active = router.step(h, nxt, loop_idx=1)
        self.assertEqual(active.tolist(), [[True, True, False, False]])
        self.assertTrue(torch.allclose(updated[0, :2], nxt[0, :2], atol=1e-5),
                        "active tokens must take the full update")
        self.assertTrue(torch.allclose(updated[0, 2:], h[0, 2:], atol=1e-6),
                        "frozen tokens must receive no update")
        updated.sum().backward()
        self.assertGreater(float(router.halt_proj.weight.grad.abs().sum()), 0.0,
                           "the straight-through gate must pass a gradient to the halting head")

    def test_checkpointed_chunked_ce_matches_eager_in_train_mode(self):
        # Off here so the eval and train passes compute the same function: the default
        # ("forward"-mode) aux-free bias moves during the training forward.
        cfg = _tiny(chunked_ce=True, chunked_ce_chunk_size=4, loop_dropout_prob=0.0,
                    use_aux_free_balancing=False, enable_ssm_highway=False)
        model = EOSModel(cfg)
        ids = torch.randint(0, 64, (2, 9))
        model.eval()
        with torch.no_grad():
            eager_ce = model(ids, labels=ids, num_loops=2, output_loop_stats=True)[2]["ce_loss"]
        model.train()
        loss, stats = model(ids, labels=ids, num_loops=2, output_loop_stats=True)[1:]
        loss.backward()
        self.assertAlmostEqual(stats["ce_loss"], eager_ce, places=5)
        self.assertIsNotNone(model.embed_tokens.weight.grad)

    def test_loop_depth_and_dropout_follow_torch_seed(self):
        model = EOSModel(_tiny(use_aux_free_balancing=False)).train()  # loop_dropout_prob stays on
        ids = torch.randint(0, 64, (2, 8))
        torch.manual_seed(11)
        first = model(ids, labels=ids, output_loop_stats=True)[2]
        torch.manual_seed(11)
        second = model(ids, labels=ids, output_loop_stats=True)[2]
        self.assertEqual(first["max_configured_loops"], second["max_configured_loops"])
        self.assertAlmostEqual(first["ce_loss"], second["ce_loss"], places=6)


class ChunkedLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_chunked_ce_gives_positive_z_loss_and_no_full_logits(self):
        model = EOSModel(_tiny(chunked_ce=True, chunked_ce_chunk_size=5)).train()
        ids = torch.randint(0, 64, (2, 9))
        logits, loss, stats = model(ids, labels=ids, num_loops=2, output_loop_stats=True)
        self.assertIsNone(logits, "chunked CE with labels must not materialise full logits")
        self.assertGreater(stats["z_loss_val"], 0.0, "z-loss must be non-zero in chunked mode")
        self.assertTrue(torch.isfinite(loss))

    def test_chunked_ce_matches_full_vocabulary_ce(self):
        model = EOSModel(_tiny(chunked_ce=True, chunked_ce_chunk_size=5)).eval()
        ids = torch.randint(0, 64, (2, 9))
        labels = ids.clone()
        labels[0, 3:6] = -100  # ignored positions inside one chunk
        with torch.no_grad():
            _, _, chunked = model(ids, labels=labels, num_loops=2, output_loop_stats=True)
            model.config.chunked_ce = False
            full_logits, _, full = model(ids, labels=labels, num_loops=2, output_loop_stats=True)
        self.assertIsNotNone(full_logits)
        self.assertAlmostEqual(chunked["ce_loss"], full["ce_loss"], places=5)


class CacheAndGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = EOSModel(_tiny()).eval()
        self.ids = torch.randint(0, 64, (1, 6))

    def _cached_logits(self, model, ids, prefill_len, num_loops):
        out = model(ids[:, :prefill_len], num_loops=num_loops, use_cache=True)
        cache, steps = out.past_key_values, [out.logits]
        for t in range(prefill_len, ids.shape[1]):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            steps.append(out.logits)
        return torch.cat(steps, dim=1), cache

    def test_cached_decoding_matches_the_full_forward(self):
        for overrides in ({}, {"enable_act": True}, {"enable_thinking_budget": True}):
            with self.subTest(**overrides):
                torch.manual_seed(0)
                model = EOSModel(_tiny(**overrides)).eval()
                ids = torch.randint(0, 64, (2, 11))
                with torch.no_grad():
                    full = model(ids, num_loops=3).logits
                    cached, cache = self._cached_logits(model, ids, prefill_len=5, num_loops=3)
                self.assertEqual(cache.seq_len, 11)
                self.assertEqual(cache.num_loops, 3)
                self.assertLess((full - cached).abs().max().item(), 1e-5)

    def test_cache_keeps_the_prefill_loop_count(self):
        with torch.no_grad():
            cache = self.model(self.ids, num_loops=2, use_cache=True).past_key_values
            with self.assertRaises(ValueError):
                self.model(self.ids[:, :1], past_key_values=cache, num_loops=3, use_cache=True)

    def test_cache_refuses_configurations_it_cannot_reproduce(self):
        with self.assertRaises(NotImplementedError):
            self.model(self.ids, use_cache=True, attention_mask=torch.zeros(1, 1, 6, 6))
        titans = EOSModel(_tiny(enable_titans_memory=True)).eval()
        self.assertIn("Titans memory is enabled", titans.kv_cache_incompatibilities())
        with self.assertRaises(NotImplementedError):
            titans(self.ids, use_cache=True)
        self.model.train()
        with self.assertRaises(NotImplementedError):
            self.model(self.ids, use_cache=True)

    def test_generate_adds_requested_tokens_without_cache(self):
        out, stats = self.model.generate(self.ids, max_new_tokens=2, num_loops=2, use_kv_cache=False)
        self.assertEqual(out.shape[1], self.ids.shape[1] + 2)
        self.assertEqual(len(stats), 2)

    def test_cached_and_uncached_greedy_generation_agree(self):
        cached, stats = self.model.generate(self.ids, max_new_tokens=5, temperature=0.0, num_loops=2, use_kv_cache=True)
        uncached, _ = self.model.generate(self.ids, max_new_tokens=5, temperature=0.0, num_loops=2, use_kv_cache=False)
        self.assertTrue(torch.equal(cached, uncached))
        self.assertTrue(stats[0]["prefill"])
        self.assertFalse(stats[1]["prefill"])

    def test_generate_uses_the_cache_by_default_when_supported(self):
        self.assertTrue(self.model.supports_kv_cache())
        _, stats = self.model.generate(self.ids, max_new_tokens=1, num_loops=2)
        self.assertTrue(stats[0].get("kv_cache"))

    def test_forcing_the_cache_on_an_unsupported_model_raises(self):
        titans = EOSModel(_tiny(enable_titans_memory=True)).eval()
        with self.assertRaises(NotImplementedError):
            titans.generate(self.ids, max_new_tokens=1, use_kv_cache=True)
        _, stats = titans.generate(self.ids, max_new_tokens=1, num_loops=2)
        self.assertNotIn("kv_cache", stats[0])

    def test_generate_restores_training_mode(self):
        self.model.train()
        self.model.generate(self.ids, max_new_tokens=1, num_loops=2)
        self.assertTrue(self.model.training)


class RepetitionPenaltyTests(unittest.TestCase):
    def test_penalty_is_sign_aware(self):
        logits = torch.tensor([2.0, -2.0, 1.0, -1.0])
        out = apply_repetition_penalty(logits, torch.tensor([0, 1]), 2.0)
        self.assertTrue(torch.allclose(out, torch.tensor([1.0, -4.0, 1.0, -1.0])))
        self.assertTrue(torch.equal(logits, torch.tensor([2.0, -2.0, 1.0, -1.0])),
                        "the input logits must not be modified in place")

    def test_identity_for_penalty_one_or_empty_history(self):
        logits = torch.tensor([2.0, -2.0, 1.0])
        self.assertTrue(torch.equal(apply_repetition_penalty(logits, [0], 1.0), logits))
        self.assertTrue(torch.equal(apply_repetition_penalty(logits, [], 2.0), logits))

    def test_batched_rows_share_the_seen_ids(self):
        batch = torch.tensor([[2.0, -2.0, 1.0], [4.0, -1.0, 0.5]])
        out = apply_repetition_penalty(batch, [0], 2.0)
        self.assertTrue(torch.allclose(out, torch.tensor([[1.0, -2.0, 1.0], [2.0, -1.0, 0.5]])))


class RhenDecisionPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_tokenizer()
        cls.state = "The system parameter status is nominal and battery voltage is 14.2V."
        cls.questions = [
            ("What is the battery voltage?", ["14.2V", "12.0V", "11.5V"]),
            ("Is system nominal?", ["Yes", "No"]),
        ]

    def _model(self, **overrides) -> EOSModel:
        torch.manual_seed(0)
        kwargs = dict(vocab_size=self.tokenizer.vocab_size)  # packed ids reach ~50k
        kwargs.update(overrides)
        return EOSModel(_tiny(**kwargs)).eval()

    def test_forward_rhen_leaves_every_tensor_unchanged(self):
        model = self._model()
        packed = RhenPacker(self.tokenizer).pack(self.state, self.questions)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            results = model.forward_rhen(packed)
        self.assertEqual(len(results), 2)
        after = model.state_dict()
        self.assertEqual(set(before), set(after))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, after[name]), f"{name} changed during forward_rhen")

    def test_evaluate_rhen_restores_training_mode(self):
        model = self._model().train()
        model.evaluate_rhen(packed_or_state=self.state, questions_and_options=self.questions,
                            tokenizer=self.tokenizer)
        self.assertTrue(model.training)

    def test_selective_checkpoint_wrapper_keeps_the_decision_path(self):
        model = self._model(enable_selective_checkpoint=True, gradient_checkpointing=True)
        self.assertIsInstance(model.recurrent_core, CheckpointedRecurrentCore)
        packed = RhenPacker(self.tokenizer).pack(self.state, self.questions)
        with torch.no_grad():
            results = model.forward_rhen(packed)
        self.assertEqual(len(results), 2)

    def test_range_branch_reports_no_free_energy(self):
        model = self._model()
        results = model.evaluate_rhen(
            packed_or_state=self.state,
            questions_and_options=[("How many minutes to recovery?", ["0", "100"], "range",
                                    {"min_val": 0.0, "max_val": 100.0})],
            tokenizer=self.tokenizer,
        )
        self.assertEqual(results[0]["primitive"], "range")
        self.assertIsNone(results[0]["free_energy"])

    def test_range_and_extract_carry_energy_only_from_logits(self):
        model = self._model()
        results = model.evaluate_rhen(
            packed_or_state=self.state,
            questions_and_options=[
                ("How many minutes to recovery?", ["0", "100"], "range",
                 {"min_val": 0.0, "max_val": 100.0}),
                ("What is the failing host name?", [], "extract", {"slot_name": "target_host"}),
            ],
            tokenizer=self.tokenizer,
        )
        by_primitive = {r["primitive"]: r for r in results}
        self.assertNotIn("scores", by_primitive["range"], "a Beta interval has no logits to score")
        self.assertIsNone(by_primitive["range"]["free_energy"])
        self.assertNotIn("scores", by_primitive["extract"], "span confidences must not be read as logits")
        self.assertIsInstance(by_primitive["extract"]["free_energy"], float,
                              "span energy must come from the start/end logits")

    def test_decide_needs_options_when_rhen_is_disabled(self):
        model = self._model(enable_rhen=False)
        ids = torch.randint(0, 64, (1, 6))
        with self.assertRaises(ValueError):
            model.decide(ids)
        out = model.decide(ids, option_embeddings=torch.randn(1, 3, 32))
        self.assertTrue(torch.allclose(out["probabilities"].sum(-1), torch.ones(1)))

    def test_decide_runs_the_full_loop_budget_and_restores_training_mode(self):
        model = self._model().train()
        calls = []
        handle = model.recurrent_core.register_forward_hook(lambda mod, inp, out: calls.append(1))
        try:
            model.decide(torch.randint(0, 64, (1, 6)))
        finally:
            handle.remove()
        self.assertEqual(len(calls), model.config.max_recurrent_loops)
        self.assertTrue(model.training, "decide() must restore the previous training mode")


class ConfigAndMemoryWiringTests(unittest.TestCase):
    def test_thinking_controller_honours_config_budget(self):
        cfg = _tiny(thinking_budget_max_loops=2)
        self.assertEqual(DynamicThinkingBudgetController(cfg).max_loops, 2)
        self.assertEqual(DynamicThinkingBudgetController(cfg, max_loops=5).max_loops, 5)

    def test_mup_model_constructs_and_runs(self):
        torch.manual_seed(0)
        model = EOSModel(_tiny(mup_enabled=True)).eval()
        ids = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            logits = model(ids, num_loops=2).logits
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_default_state_dict_has_no_titans_or_episodic_keys(self):
        model = EOSModel(_tiny())
        keys = list(model.state_dict().keys())
        self.assertFalse([k for k in keys if "titans" in k or "episodic_memory_bank" in k])
        self.assertIsNone(model.recurrent_core.titans_memory)
        self.assertIsNone(model.episodic_memory_bank)

    def test_titans_and_episodic_bank_are_opt_in(self):
        cfg = _tiny()
        cfg.enable_titans_memory = True
        cfg.enable_episodic_memory = True
        keys = set(EOSModel(cfg).state_dict().keys())
        self.assertTrue(any("titans" in k for k in keys))
        self.assertTrue(any("episodic_memory_bank" in k for k in keys))

    def test_episodic_store_is_a_vectorised_ring_buffer(self):
        bank = ModernHopfieldEpisodicBank(hidden_dim=4, max_memories=4)
        rows = torch.arange(24, dtype=torch.float32).view(6, 4)
        bank.store(rows[:3])
        bank.store(rows[3:6])
        self.assertEqual(int(bank.memory_count), 6)
        # rows 3, 4, 5 land in slots 3, 0, 1; row 2 stays in slot 2
        self.assertTrue(torch.equal(bank.memory_bank[0], rows[4]))
        self.assertTrue(torch.equal(bank.memory_bank[1], rows[5]))
        self.assertTrue(torch.equal(bank.memory_bank[2], rows[2]))
        self.assertTrue(torch.equal(bank.memory_bank[3], rows[3]))

        # A batch larger than the capacity keeps only its newest max_memories rows.
        big = torch.arange(24, 48, dtype=torch.float32).view(6, 4)
        bank.store(big)
        self.assertEqual(int(bank.memory_count), 12)
        for slot, row in enumerate(big[2:]):
            self.assertTrue(torch.equal(bank.memory_bank[(6 + 2 + slot) % 4], row))


class PointerTemperatureTests(unittest.TestCase):
    """The decision temperature must not grow with model width (it used to be ~hidden_size ** 0.5)."""

    def test_initial_temperature_is_near_one_at_every_width(self):
        for hidden in (32, 64, 256):
            with self.subTest(hidden=hidden):
                torch.manual_seed(0)
                model = EOSModel(_tiny(hidden_size=hidden, intermediate_size=2 * hidden))
                head = model._core_block.rhen.pointer_head
                details = head(torch.randn(4, 1, hidden), torch.randn(4, 3, hidden), return_details=True)
                tau = details["temperature"]
                self.assertTrue(bool(((tau >= 1.0) & (tau < 1.6)).all()), tau.flatten().tolist())

    def test_a_handful_of_decisions_can_be_fit(self):
        from rhen_loss import compute_rhen_multi_task_loss
        from rhen_packer import get_rhen_packer

        tok = get_tokenizer()
        packer = get_rhen_packer(tok)
        torch.manual_seed(0)
        model = EOSModel(_tiny(vocab_size=tok.vocab_size)).train()
        # Grounding decisions: the answer is the option whose text appears in the state.
        examples = [
            ({"city": "Perth"}, ("City on file", ["Quito", "Perth", "Lisbon"], "choice", None), 1),
            ({"city": "Oslo"}, ("City on file", ["Oslo", "Lima", "Doha"], "choice", None), 0),
            ({"city": "Riga"}, ("City on file", ["Kyoto", "Hanoi", "Riga"], "choice", None), 2),
            ({"city": "Lagos"}, ("City on file", ["Lagos", "Perth", "Quito"], "choice", None), 0),
        ]
        packs = [packer.pack(state=s, questions_and_options=[q]) for s, q, _ in examples]
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        history = []
        for _ in range(25):
            total = 0.0
            for packed, (_, _, label) in zip(packs, examples):
                loss, _ = compute_rhen_multi_task_loss(model, packed, [label], torch.device("cpu"))
                total = total + loss / len(examples)
            history.append(total.item())
            opt.zero_grad()
            total.backward()
            opt.step()
        # With the width-scaled temperature, or the <|/opt|>-only readout, the loss barely moved.
        self.assertLess(history[-1], 0.25 * history[0], history)


class OptionSpanReadoutTests(unittest.TestCase):
    """Options are read from their whole span, not from the shared <|/opt|> token alone."""

    def setUp(self):
        from rhen_packer import get_rhen_packer

        self.tok = get_tokenizer()
        self.packed = get_rhen_packer(self.tok).pack(
            state={"city": "Perth"}, questions_and_options=[("City on file", ["Quito", "Perth", "Lisbon"], "choice", None)]
        )

    def test_packer_records_where_each_option_starts(self):
        branch = self.packed.branches[0]
        ids = self.packed.input_ids[0].tolist()
        self.assertEqual(len(branch.option_start_indices), 3)
        for start, end in zip(branch.option_start_indices, branch.option_end_indices):
            self.assertEqual(ids[start], self.tok.opt_token_id)
            self.assertEqual(ids[end], self.tok.opt_end_token_id)
            self.assertLess(start, end)

    def test_option_representations_are_distinguishable(self):
        import torch.nn.functional as F

        torch.manual_seed(0)
        model = EOSModel(_tiny(vocab_size=self.tok.vocab_size)).eval()
        shallow = model.evaluate_rhen(self.packed)[0]["opt_layers"][0][0]
        sims = F.normalize(shallow.float(), dim=-1) @ F.normalize(shallow.float(), dim=-1).T
        off_diag = sims[~torch.eye(3, dtype=torch.bool)]
        # The <|/opt|> token alone gave cosine similarities of about 0.9999 here.
        self.assertLess(float(off_diag.max()), 0.98)

    def test_hand_built_branches_fall_back_to_the_end_token(self):
        from model import _option_span_reprs
        from rhen_packer import PackedBranchMeta

        hidden = torch.randn(1, 10, 4)
        meta = PackedBranchMeta("q", ["a", "b"], 6, 9, [4, 7], 2)
        self.assertTrue(torch.equal(_option_span_reprs(hidden, meta), hidden[:, [4, 7], :]))
        meta.option_start_indices = [2, 5]
        expected = torch.stack([hidden[:, 3:5].mean(1), hidden[:, 6:8].mean(1)], dim=1)
        self.assertTrue(torch.allclose(_option_span_reprs(hidden, meta), expected))


class ThinkingBudgetLimitTests(unittest.TestCase):
    def test_budget_never_exceeds_the_configured_loop_count(self):
        # The controller's own floor (4) is above max_recurrent_loops (2); the estimate must
        # still stay within the per-step embeddings, with or without ACT early exit.
        torch.manual_seed(0)
        model = EOSModel(_tiny(max_recurrent_loops=2, enable_thinking_budget=True, enable_act=False)).eval()
        hidden = torch.randn(1, 5, 32)
        self.assertLessEqual(model.thinking_controller.estimate_budget(hidden, base_max_loops=2), 2)
        with torch.no_grad():
            _, _, stats = model(torch.randint(0, 64, (1, 5)), output_loop_stats=True)
        self.assertLessEqual(stats["max_configured_loops"], 2)


class MoeBalanceAndCheckpointTests(unittest.TestCase):
    """Aux-free balancing modes, checkpoint recomputation and the state-dict layout."""

    CFG = dict(loop_sampling="fixed", loop_dropout_prob=0.0)

    def _grads(self, state, checkpointing):
        model = EOSModel(_tiny(gradient_checkpointing=checkpointing, **self.CFG)).train()
        model.load_state_dict(state)
        model.set_moe_balance_update_mode("step")
        ids = torch.randint(0, 64, (2, 10), generator=torch.Generator().manual_seed(1))
        out = model(ids, labels=ids, num_loops=3)
        out.loss.backward()
        grads = {
            name.replace("recurrent_core.core.", "recurrent_core."): p.grad.clone()
            for name, p in model.named_parameters() if p.grad is not None
        }
        return out.loss.item(), grads, [m.pending_calls for m in model.moe_balance_modules()]

    def test_default_mode_is_the_historical_per_forward_update(self):
        torch.manual_seed(0)
        model = EOSModel(_tiny(**self.CFG)).train()
        self.assertEqual({m.balance_update_mode for m in model.moe_balance_modules()}, {"forward"})
        before = [m.congestion_bias.clone() for m in model.moe_balance_modules()]
        model(torch.randint(0, 64, (1, 8)), num_loops=2)
        after = [m.congestion_bias for m in model.moe_balance_modules()]
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))

    def test_gradient_checkpointing_selects_step_mode(self):
        model = EOSModel(_tiny(gradient_checkpointing=True))
        self.assertEqual({m.balance_update_mode for m in model.moe_balance_modules()}, {"step"})

    def test_checkpointed_gradients_equal_plain_gradients(self):
        torch.manual_seed(0)
        state = EOSModel(_tiny(**self.CFG)).state_dict()
        loss_plain, grads_plain, calls_plain = self._grads(state, checkpointing=False)
        loss_ckpt, grads_ckpt, calls_ckpt = self._grads(state, checkpointing=True)
        self.assertEqual(loss_plain, loss_ckpt)
        self.assertEqual(set(grads_plain), set(grads_ckpt))
        for name in grads_plain:
            self.assertTrue(torch.allclose(grads_plain[name], grads_ckpt[name], atol=1e-6), name)
        # Recomputation in backward does not record the routed load a second time.
        self.assertEqual(calls_plain, calls_ckpt)

    def test_step_mode_moves_the_bias_once_from_the_mean_load(self):
        torch.manual_seed(0)
        model = EOSModel(_tiny(**self.CFG)).train()
        model.set_moe_balance_update_mode("step")
        modules = model.moe_balance_modules()
        before = [m.congestion_bias.clone() for m in modules]
        with torch.no_grad():
            model(torch.randint(0, 64, (1, 8)), num_loops=2)
        self.assertTrue(all(torch.equal(m.congestion_bias, b) for m, b in zip(modules, before)))
        self.assertTrue(all(m.pending_calls > 0 for m in modules))
        self.assertEqual(model.apply_moe_balance_updates(), len(modules))
        self.assertTrue(all(m.pending_calls == 0 for m in modules))
        self.assertTrue(any(not torch.equal(m.congestion_bias, b) for m, b in zip(modules, before)))
        self.assertEqual(model.apply_moe_balance_updates(), 0)

    def test_reduce_fn_sets_the_update_on_every_rank(self):
        torch.manual_seed(0)
        model = EOSModel(_tiny(**self.CFG)).train()
        model.set_moe_balance_update_mode("step")
        uniform = lambda load: torch.full_like(load, 1.0 / load.numel())
        before = [m.congestion_bias.clone() for m in model.moe_balance_modules()]
        # A rank that saw no tokens still joins the reduction; a uniform mean load is no update.
        self.assertEqual(model.apply_moe_balance_updates(reduce_fn=uniform), len(before))
        for module, bias in zip(model.moe_balance_modules(), before):
            self.assertTrue(torch.allclose(module.congestion_bias, bias))

    def test_state_dict_layout_is_canonical_and_portable(self):
        torch.manual_seed(0)
        ckpt_model = EOSModel(_tiny(gradient_checkpointing=True))
        state = ckpt_model.state_dict()
        self.assertFalse(any(k.startswith("recurrent_core.core.") for k in state))
        self.assertTrue(any(k.startswith("recurrent_core.layers.") for k in state))
        plain = EOSModel(_tiny())
        plain.load_state_dict(state, strict=True)
        legacy = {
            ("recurrent_core.core." + k[len("recurrent_core."):] if k.startswith("recurrent_core.") else k): v
            for k, v in state.items()
        }
        plain.load_state_dict(dict(legacy), strict=True)
        ckpt_model.load_state_dict(dict(legacy), strict=True)


class TestScoreRpsTargetIsOrdinalCdf(unittest.TestCase):
    """The Score RPS target must be a cumulative distribution: 0 below the label, 1 from it upwards."""

    CPU = torch.device("cpu")

    def test_integer_label_is_a_point_mass_cdf(self):
        cdf = _ordinal_cdf_target(5, 2.0, self.CPU)
        self.assertEqual(cdf.tolist(), [0.0, 0.0, 1.0, 1.0, 1.0])

    def test_fractional_label_splits_mass_between_neighbours(self):
        cdf = _ordinal_cdf_target(5, 2.5, self.CPU)
        self.assertEqual(cdf.tolist(), [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_perfect_prediction_has_zero_rps_at_every_level(self):
        for level in range(5):
            probs = torch.zeros(5)
            probs[level] = 1.0
            pred_cdf = torch.cumsum(probs, dim=-1)
            true_cdf = _ordinal_cdf_target(5, float(level), self.CPU)
            rps = torch.mean((pred_cdf[:-1] - true_cdf[:-1]) ** 2).item()
            self.assertEqual(rps, 0.0, f"level {level}")

    def test_out_of_range_labels_are_clamped_to_the_scale(self):
        self.assertEqual(_ordinal_cdf_target(5, -3.0, self.CPU).tolist(), [1.0] * 5)
        self.assertEqual(_ordinal_cdf_target(5, 9.0, self.CPU).tolist(), [0.0, 0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
