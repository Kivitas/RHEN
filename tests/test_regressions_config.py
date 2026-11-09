"""Regression tests for the configuration, optimizer and checkpoint helper fixes.

Covers:
  * EOSConfig equality and hashing (the decorator-generated __eq__ made all
    configs compare equal).
  * use_differential_attention: explicit values win; None follows use_diff_mla;
    no sticky state across re-initialisation.
  * diff_mla_lambda_init is the only DIFF lambda; diff_attn_lambda_init is gone.
  * Titans / episodic memory flags and titans_d_mem defaults.
  * Changed defaults (attn_logit_softcapping 0.0).
  * Removed no-op keys are no longer config fields (see test_config_legacy_keys
    for the accept-and-warn behaviour).
  * muon_update_scale and the Muon step that uses it.
  * CheckpointedRecurrentCore (config precedence, training-only recompute,
    gradient equivalence, RNG-preserving recompute, delegation).
  * AdvancedOptimizations init path, the optim package import without
    optim.kernels, and removal of the selective-checkpoint helpers.
"""

import math
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

import optim
import optim.checkpoint as optim_checkpoint
from config import (
    ActBudgetConfig,
    AttentionMLAConfig,
    EOSConfig,
    LoopControlConfig,
    OrchestratorConfig,
    RecurrentLoopConfig,
    RhenEngineConfig,
    YaRNConfig,
)
from optim import (
    AdvancedOptimizations,
    apply_advanced_optimizations,
    create_mup_parameter_groups,
)
from optim.checkpoint import CheckpointedRecurrentCore
from optimizers import Muon, muon_update_scale, zeropower_via_newtonschulz5
from test_config_roundtrip import _historical_flat_defaults

ROOT = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "tests"
    else os.path.dirname(os.path.abspath(__file__))
)

#: Removed no-op keys as (group class that used to own them, key). All 21 were
#: removed in the release after 1.0.0; none may reappear as a field of its group.
REMOVED_KEYS = [
    (RecurrentLoopConfig, "enable_symplectic_loop"),
    (AttentionMLAConfig, "use_flash_attn"),
    (AttentionMLAConfig, "diff_attn_lambda_init"),
    (AttentionMLAConfig, "enable_prefix_lm_mask"),
    (YaRNConfig, "yarn_scale_factor"),
    (RhenEngineConfig, "rhen_calib_loss_coef"),
    (RhenEngineConfig, "rhen_enable_deep_reasoning"),
    (RhenEngineConfig, "rhen_confidence_threshold"),
    (RhenEngineConfig, "rhen_enable_qwen_reasoning"),
    (RhenEngineConfig, "rhen_enable_deepseek_reasoning"),
    (RhenEngineConfig, "rhen_meta_reasoning_cache_size"),
    (RhenEngineConfig, "rhen_enable_three_layer_pipeline"),
    (RhenEngineConfig, "rhen_semantic_layer_weight"),
    (RhenEngineConfig, "rhen_reasoning_layer_weight"),
    (RhenEngineConfig, "rhen_generation_layer_weight"),
    (RhenEngineConfig, "rhen_enable_confidence_routing"),
    (RhenEngineConfig, "rhen_min_semantic_confidence"),
    (RhenEngineConfig, "rhen_max_reasoning_depth"),
    (OrchestratorConfig, "use_fused_recurrent_kernels"),
    (OrchestratorConfig, "checkpoint_policy"),
    (OrchestratorConfig, "use_triton"),
]


def _tiny_config(**overrides) -> EOSConfig:
    kwargs = dict(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        enable_grouped_gemm=False,
    )
    kwargs.update(overrides)
    return EOSConfig(**kwargs)


class TestEOSConfigEqualityAndHash(unittest.TestCase):
    def test_distinct_configs_are_unequal(self):
        self.assertTrue(EOSConfig(hidden_size=256) != EOSConfig(hidden_size=512))

    def test_equal_configs_are_equal(self):
        self.assertEqual(EOSConfig(hidden_size=256), EOSConfig(hidden_size=256))

    def test_mutation_breaks_equality(self):
        cfg = EOSConfig()
        other = EOSConfig()
        other.hidden_size = 999
        self.assertNotEqual(cfg, other)

    def test_hash_works(self):
        self.assertIsInstance(hash(EOSConfig()), int)
        self.assertEqual(hash(EOSConfig(hidden_size=256)), hash(EOSConfig(hidden_size=256)))

    def test_set_semantics_follow_equality(self):
        self.assertEqual(len({EOSConfig(hidden_size=256), EOSConfig(hidden_size=256)}), 1)
        self.assertEqual(len({EOSConfig(hidden_size=256), EOSConfig(hidden_size=512)}), 2)

    def test_hash_is_stable_under_mutation(self):
        cfg = EOSConfig()
        before = hash(cfg)
        cfg.hidden_size = 999
        self.assertEqual(hash(cfg), before)


class TestDifferentialAttentionFlag(unittest.TestCase):
    def test_default_follows_use_diff_mla(self):
        self.assertIsNone(AttentionMLAConfig().use_differential_attention)
        self.assertIs(EOSConfig().use_differential_attention, True)
        self.assertIs(EOSConfig(use_diff_mla=False).use_differential_attention, False)

    def test_explicit_false_wins_over_diff_mla(self):
        cfg = EOSConfig(use_diff_mla=True, use_differential_attention=False)
        self.assertIs(cfg.use_differential_attention, False)

    def test_explicit_false_wins_through_group_object(self):
        cfg = EOSConfig(attention=AttentionMLAConfig(use_diff_mla=True, use_differential_attention=False))
        self.assertIs(cfg.use_differential_attention, False)

    def test_explicit_true_wins_when_diff_mla_false(self):
        cfg = EOSConfig(use_diff_mla=False, use_differential_attention=True)
        self.assertIs(cfg.use_differential_attention, True)

    def test_no_sticky_value_on_reinitialisation(self):
        cfg = EOSConfig(use_differential_attention=True, use_diff_mla=False)
        cfg.__init__(use_diff_mla=False)
        self.assertIs(cfg.use_differential_attention, False)


class TestDiffLambdaInit(unittest.TestCase):
    def test_diff_mla_lambda_init_is_the_diff_lambda(self):
        cfg = EOSConfig(diff_mla_lambda_init=0.3)
        self.assertEqual(cfg.diff_mla_lambda_init, 0.3)
        self.assertEqual(cfg.attention.diff_mla_lambda_init, 0.3)
        self.assertEqual(EOSConfig().diff_mla_lambda_init, 0.8)

    def test_removed_diff_attn_lambda_init_is_not_an_attribute(self):
        with self.assertWarns(FutureWarning):
            cfg = EOSConfig(diff_attn_lambda_init=0.5)
        self.assertFalse(hasattr(cfg, "diff_attn_lambda_init"))
        self.assertFalse(hasattr(cfg.attention, "diff_attn_lambda_init"))
        self.assertEqual(cfg.diff_mla_lambda_init, 0.8)


class TestTitansAndEpisodicMemoryFlags(unittest.TestCase):
    def test_flat_defaults(self):
        cfg = EOSConfig()
        self.assertIs(cfg.enable_titans_memory, False)
        self.assertIs(cfg.enable_episodic_memory, False)
        self.assertEqual(cfg.titans_d_mem, 64)

    def test_group_defaults(self):
        group = RecurrentLoopConfig()
        self.assertIs(group.enable_titans_memory, False)
        self.assertIs(group.enable_episodic_memory, False)
        self.assertEqual(group.titans_d_mem, 64)

    def test_flat_kwargs_reach_the_recurrent_group(self):
        cfg = EOSConfig(enable_titans_memory=True, enable_episodic_memory=True, titans_d_mem=32)
        self.assertIs(cfg.recurrent.enable_titans_memory, True)
        self.assertIs(cfg.recurrent.enable_episodic_memory, True)
        self.assertEqual(cfg.recurrent.titans_d_mem, 32)

    def test_nested_group_values_reach_flat_attributes(self):
        cfg = EOSConfig(recurrent=RecurrentLoopConfig(enable_episodic_memory=True, titans_d_mem=16))
        self.assertIs(cfg.enable_episodic_memory, True)
        self.assertEqual(cfg.titans_d_mem, 16)

    def test_recurrent_snapshot_equals_group_defaults(self):
        # Every RecurrentLoopConfig field must be flattened, including the new ones.
        self.assertEqual(EOSConfig().recurrent, RecurrentLoopConfig())


class TestChangedDefaults(unittest.TestCase):
    def test_fused_recurrent_kernel_flag_is_removed(self):
        self.assertNotIn("use_fused_recurrent_kernels", OrchestratorConfig.__dataclass_fields__)
        self.assertFalse(hasattr(EOSConfig(), "use_fused_recurrent_kernels"))

    def test_attn_logit_softcapping_default_zero(self):
        self.assertEqual(EOSConfig().attn_logit_softcapping, 0.0)
        self.assertEqual(AttentionMLAConfig().attn_logit_softcapping, 0.0)

    def test_group_snapshots_equal_group_defaults(self):
        cfg = EOSConfig()
        self.assertEqual(cfg.yarn, YaRNConfig())
        self.assertEqual(cfg.act_budget, ActBudgetConfig())
        self.assertEqual(cfg.rhen_engine, RhenEngineConfig())
        self.assertEqual(cfg.loop_control, LoopControlConfig())


class TestRemovedNoOpKeys(unittest.TestCase):
    def test_removed_keys_are_not_group_fields(self):
        for group_cls, key in REMOVED_KEYS:
            with self.subTest(key=key):
                self.assertNotIn(key, group_cls.__dataclass_fields__)

    def test_removed_keys_are_not_flat_attributes(self):
        cfg = EOSConfig()
        for _group_cls, key in REMOVED_KEYS:
            with self.subTest(key=key):
                self.assertFalse(hasattr(cfg, key))

    def test_removed_keys_are_not_in_the_flat_signature(self):
        import inspect

        params = inspect.signature(EOSConfig.__init__).parameters
        for _group_cls, key in REMOVED_KEYS:
            with self.subTest(key=key):
                self.assertNotIn(key, params)


class TestMuonUpdateScale(unittest.TestCase):
    def test_wide_matrix_scale_is_one(self):
        self.assertEqual(muon_update_scale(4, 16), 1.0)
        self.assertEqual(muon_update_scale(8, 8), 1.0)

    def test_tall_matrix_scale_is_sqrt_aspect_ratio(self):
        self.assertAlmostEqual(muon_update_scale(16, 4), math.sqrt(4.0))
        self.assertAlmostEqual(muon_update_scale(10, 4), math.sqrt(2.5))

    def test_step_applies_the_scale_for_tall_and_wide_matrices(self):
        lr = 0.1
        for rows, cols in [(4, 16), (16, 4)]:
            with self.subTest(shape=(rows, cols)):
                torch.manual_seed(0)
                g = torch.randn(rows, cols)
                p = nn.Parameter(torch.zeros(rows, cols))
                p.grad = g.clone()
                Muon([p], lr=lr, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0).step()

                # First step: momentum buffer = g, Nesterov update = g + 0.95 * g.
                u = zeropower_via_newtonschulz5(g.add(g, alpha=0.95), steps=5)
                expected = -lr * muon_update_scale(rows, cols) * u
                torch.testing.assert_close(p.detach(), expected, rtol=1e-5, atol=1e-6)


class TestCheckpointedRecurrentCore(unittest.TestCase):
    def test_constructs_from_config_and_forwards(self):
        core = nn.Linear(4, 4)
        wrapper = CheckpointedRecurrentCore(core, EOSConfig(gradient_checkpointing=True))
        self.assertIs(wrapper.core, core)
        self.assertTrue(wrapper.gradient_checkpointing)
        wrapper.train()
        x = torch.randn(3, 4, requires_grad=True)
        out = wrapper(x)
        torch.testing.assert_close(out, core(x))
        out.sum().backward()
        self.assertIsNotNone(core.weight.grad)

    def test_config_takes_precedence_over_explicit_flag(self):
        core = nn.Linear(2, 2)
        cfg_off = SimpleNamespace(gradient_checkpointing=False)
        self.assertFalse(CheckpointedRecurrentCore(core, cfg_off, gradient_checkpointing=True).gradient_checkpointing)

    def test_explicit_flag_used_without_config_attribute(self):
        core = nn.Linear(2, 2)
        self.assertFalse(CheckpointedRecurrentCore(core, config=object(), gradient_checkpointing=False).gradient_checkpointing)
        self.assertTrue(CheckpointedRecurrentCore(core).gradient_checkpointing)

    def test_checkpoint_used_only_in_training_mode(self):
        real_checkpoint = torch.utils.checkpoint.checkpoint
        wrapper = CheckpointedRecurrentCore(nn.Linear(4, 4), EOSConfig(gradient_checkpointing=True))
        x = torch.randn(2, 4, requires_grad=True)

        with mock.patch("torch.utils.checkpoint.checkpoint", wraps=real_checkpoint) as spy:
            wrapper.train()
            wrapper(x)
            self.assertEqual(spy.call_count, 1)
            self.assertIs(spy.call_args.kwargs.get("use_reentrant"), False)

            wrapper.eval()
            wrapper(x)
            self.assertEqual(spy.call_count, 1)

    def test_checkpoint_skipped_when_disabled(self):
        real_checkpoint = torch.utils.checkpoint.checkpoint
        wrapper = CheckpointedRecurrentCore(nn.Linear(4, 4), EOSConfig(gradient_checkpointing=False))
        wrapper.train()
        with mock.patch("torch.utils.checkpoint.checkpoint", wraps=real_checkpoint) as spy:
            wrapper(torch.randn(2, 4, requires_grad=True))
        self.assertEqual(spy.call_count, 0)

    def test_gradients_match_the_direct_core(self):
        core = nn.Linear(4, 4)
        wrapper = CheckpointedRecurrentCore(core, EOSConfig(gradient_checkpointing=True))
        wrapper.train()
        x = torch.randn(3, 4)

        core.zero_grad()
        core(x).sum().backward()
        grad_direct = core.weight.grad.clone()

        core.zero_grad()
        wrapper(x).sum().backward()
        torch.testing.assert_close(core.weight.grad, grad_direct)

    def test_recompute_reproduces_dropout_masks(self):
        # Recomputation during backward must redraw the same dropout mask as the
        # forward pass, so gradients match the un-checkpointed core exactly.
        core = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5))
        wrapper = CheckpointedRecurrentCore(core, EOSConfig(gradient_checkpointing=True))
        wrapper.train()
        x = torch.randn(16, 8)

        torch.manual_seed(7)
        direct_out = core(x)
        direct_out.sum().backward()
        grad_direct = core[0].weight.grad.clone()
        core.zero_grad()

        torch.manual_seed(7)
        wrapped_out = wrapper(x)
        wrapped_out.sum().backward()
        grad_wrapped = core[0].weight.grad.clone()

        torch.testing.assert_close(wrapped_out, direct_out)
        torch.testing.assert_close(grad_wrapped, grad_direct)

    def test_reset_special_parameters_delegates(self):
        class _Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)
                self.resets = 0

            def reset_special_parameters(self):
                self.resets += 1

        core = _Core()
        CheckpointedRecurrentCore(core).reset_special_parameters()
        self.assertEqual(core.resets, 1)
        # A core without the hook is a no-op, not an error.
        CheckpointedRecurrentCore(nn.Linear(2, 2)).reset_special_parameters()


class TestAdvancedOptimizationsInit(unittest.TestCase):
    def test_fused_kernels_stay_none_and_swiglu_is_built(self):
        opt = AdvancedOptimizations(_tiny_config())
        self.assertIsNone(opt.fused_kernels)
        self.assertIsNone(opt.grouped_gemm_moe)
        self.assertIsNotNone(opt.fused_swiglu)
        self.assertEqual(opt.fused_swiglu.hidden_size, 32)
        self.assertEqual(opt.fused_swiglu.intermediate_size, 64)

    def test_apply_wraps_recurrent_core_once(self):
        cfg = _tiny_config(enable_selective_checkpoint=True, gradient_checkpointing=True)
        model = nn.Module()
        model.recurrent_core = nn.Linear(32, 32)
        opt = AdvancedOptimizations(cfg)

        opt.apply_to_model(model)
        self.assertIsInstance(model.recurrent_core, CheckpointedRecurrentCore)
        self.assertTrue(opt.checkpointed_core)

        opt.apply_to_model(model)  # a second pass must not nest the wrapper
        self.assertIsInstance(model.recurrent_core.core, nn.Linear)

    def test_no_wrap_when_selective_checkpointing_disabled(self):
        cfg = _tiny_config(enable_selective_checkpoint=False)
        model = nn.Module()
        model.recurrent_core = nn.Linear(32, 32)
        opt = AdvancedOptimizations(cfg)
        opt.apply_to_model(model)
        self.assertIsInstance(model.recurrent_core, nn.Linear)
        self.assertFalse(opt.checkpointed_core)

    def test_apply_advanced_optimizations_returns_the_model(self):
        cfg = _tiny_config()
        model = nn.Sequential(nn.Linear(32, 32))
        self.assertIs(apply_advanced_optimizations(model, config=cfg), model)

    def test_mup_parameter_groups_return_muon_and_adamw(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        for mup in (False, True):
            with self.subTest(mup_enabled=mup):
                groups = create_mup_parameter_groups(model, EOSConfig(mup_enabled=mup), base_lr=1e-3)
                self.assertIsInstance(groups[0], Muon)
                self.assertIsInstance(groups[1], torch.optim.AdamW)


class TestOptimPackageImports(unittest.TestCase):
    def test_package_imports_without_optim_kernels(self):
        # optim/__init__ must not depend on optim.kernels, which is being removed.
        code = (
            "import sys; sys.modules['optim.kernels'] = None; "
            "import optim, optim.checkpoint; print('IMPORT_OK')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT, capture_output=True, text=True, timeout=240,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("IMPORT_OK", proc.stdout)

    def test_selective_checkpoint_helpers_are_removed(self):
        for name in ("SelectiveCheckpoint", "selective_checkpointing",
                     "apply_selective_checkpointing", "checkpointed_forward"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(optim_checkpoint, name))
                self.assertFalse(hasattr(optim, name))

    def test_package_all_resolves(self):
        for name in optim.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(optim, name))


class TestConfigFlatSurfaceCoverage(unittest.TestCase):
    def test_round_trip_expectations_cover_every_flat_kwarg(self):
        """The round-trip expectations must list every flat signature kwarg."""
        import inspect

        signature = inspect.signature(EOSConfig.__init__)
        group_kwargs = {
            "recurrent", "moe", "attention", "yarn", "act_budget",
            "rhen_engine", "loop_control", "orchestrator",
        }
        flat = {n for n in signature.parameters if n not in {"self", "kwargs"} | group_kwargs}
        covered = set(_historical_flat_defaults()) | {"vocab_size", "hidden_size", "intermediate_size"}
        self.assertEqual(flat, covered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
