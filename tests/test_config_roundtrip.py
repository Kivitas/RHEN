"""Phase B1 unit tests: nested config groups and the flat-signature contract.

Locks the invariants the decomposition depends on:

1. Flat defaults match the pre-B1 EOSConfig defaults, with the deliberate B1
   changes applied (see ``_historical_flat_defaults``).
2. Flat kwargs still override (and win over) nested group values.
3. Nested construction works and untouched groups keep their defaults.
4. Group snapshots are live views derived from flat state, never instance
   state, so they cannot leak into ``to_dict()`` or checkpoint serialization.
5. The full ``EOSConfig(**flat_kwargs).to_dict()`` round-trip is lossless
   across the entire 141-parameter flat surface.
6. The legacy ``LoopedLLMConfig`` alias is still the same class.

Run directly (script mode, battery style) or via pytest/unittest.
"""

from __future__ import annotations

import unittest

from config import (
    ActBudgetConfig,
    AttentionMLAConfig,
    EOSConfig,
    LoopedLLMConfig,
    LoopControlConfig,
    MoEConfig,
    OrchestratorConfig,
    RecurrentLoopConfig,
    RhenEngineConfig,
    YaRNConfig,
)


def _historical_flat_defaults() -> dict:
    """Expected flat defaults: the pre-B1 EOSConfig values with the deliberate B1
    changes applied (fused kernels off, softcapping 0.0, Titans/episodic keys).
    Keys that entered the flat surface with the nested-groups refactor or later
    use their committed group defaults.
    """
    return {
        "num_pre_layers": 1,
        "num_recurrent_layers": 4,
        "num_post_layers": 1,
        "max_recurrent_loops": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 70,
        "kv_lora_rank": 140,
        "q_lora_rank": 280,
        "use_moe": True,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_aux_loss_coef": 0.01,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "use_yarn": True,
        "yarn_scale": 4.0,
        "yarn_original_max_position_embeddings": 2048,
        "yarn_beta_fast": 32.0,
        "yarn_beta_slow": 1.0,
        "yarn_attn_factor": 0.1,
        "tie_word_embeddings": True,
        "enable_act": True,
        "act_threshold": 0.90,
        "act_ponder_penalty": 0.0001,
        "use_step_modulation": True,
        "use_sdpa": True,
        "use_flash_attn": True,
        "use_qk_norm": False,
        "z_loss_coef": 1e-4,
        "initializer_range": 0.02,
        "min_recurrent_loops": 4,
        "loop_sampling": "uniform",
        "act_prior_lambda": 0.02,
        "act_early_stop": True,
        "act_min_loops": 5,
        "gradient_checkpointing": False,
        "chunked_ce": True,
        "chunked_ce_chunk_size": 512,
        "use_fused_recurrent_kernels": False,  # B1: deprecated no-op, default False
        "enable_grouped_gemm": True,
        "quantized_kv_cache": False,
        "kv_quant_dtype": "fp8",
        "enable_selective_checkpoint": False,
        "checkpoint_policy": "mla,moera",
        "mup_enabled": False,
        "mup_base_hidden": 1024,
        "mup_base_loops": 16,
        "enable_cross_loop_attention": False,
        "cross_loop_memory_dim": 32,
        "enable_speculative_decoding": False,
        "draft_loops": 2,
        "enable_loop_bypass": False,
        "enable_titans_memory": False,
        "enable_episodic_memory": False,
        "titans_d_mem": 64,
        "fuse_multiloop_attn": False,
        "ring_size": 2,
        "max_draft_tokens": 4,
        "router_type": "sigmoid",
        "use_triton": True,
        "lr_scale_factor": 1.0,
        "attn_logit_softcapping": 0.0,  # B1: 0.0 keeps the fast SDPA path
        "final_logit_softcapping": 30.0,
        "yarn_scale_factor": 1.0,
        "contraction_loss_coef": 0.01,
        "loop_dropout_prob": 0.10,
        "router_z_loss_coef": 0.001,
        "enable_ssm_highway": True,
        "ssm_state_dim": 64,
        "enable_symplectic_loop": True,
        "use_shared_expert_moe": True,
        "num_shared_experts": 1,
        "num_routed_experts": 8,
        "num_active_routed": 2,
        "use_aux_free_balancing": True,
        "aux_free_balance_rate": 0.001,
        # moe_intermediate_size falls back to intermediate_size (1680)
        "moe_intermediate_size": 1680,
        "enable_mtp": True,
        "mtp_loss_coef": 0.3,
        "mtp_num_layers": 1,
        "use_diff_mla": True,
        "diff_mla_lambda_init": 0.8,
        "use_differential_attention": True,  # derived: bool(use_diff_mla) while the group value is None
        "diff_attn_lambda_init": 0.8,
        "diff_attn_learnable_lambda": True,
        "diff_v2_head_lambda": True,
        "enable_prefix_lm_mask": True,
        "interleave_swa": True,
        "swa_window_size": 512,
        "use_qkv_bias": True,
        "enable_thinking_budget": True,
        "thinking_budget_min_loops": 4,
        "thinking_budget_max_loops": 12,
        "enable_step_size_control": False,
        "loop_step_alpha": 0.25,
        "loop_step_min_alpha": 0.05,
        "loop_step_anneal": True,
        "loop_step_overshoot_rejection": True,
        "loop_full_update": True,
        "enable_flash_loop_sparsity": False,
        "flash_loop_threshold": 1e-3,
        "flash_loop_warmup_loops": 2,
        "enable_loop_curriculum": False,
        "loop_curriculum_warmup_steps": 0,
        "enable_adaptive_loop_router": False,
        "adaptive_loop_min_loops": 2,
        "adaptive_loop_freeze_threshold": 0.9,
        "adaptive_loop_savings_target": 0.35,
        "enable_clam_bridge": False,
        "clam_memory_dim": 64,
        "clam_init_gate": 0.0,
        "enable_dynamic_ntk_yarn": False,
        "dynamic_ntk_yarn_max_context": 16384,
        "dynamic_ntk_yarn_snap_factor": 0.25,
        "enable_rhen": True,
        "rhen_num_slots": 8,
        "rhen_num_heads": 4,
        "rhen_hidden_dim": 256,
        "rhen_pointer_rank": 128,
        "rhen_ortho_loss_coef": 0.01,
        "rhen_calib_loss_coef": 0.05,
        "rhen_halt_confidence_threshold": 0.85,
        "rhen_enable_kv_modulation": True,
        "rhen_enable_deep_reasoning": True,
        "rhen_confidence_threshold": 0.75,
        "rhen_enable_qwen_reasoning": True,
        "rhen_enable_deepseek_reasoning": True,
        "rhen_meta_reasoning_cache_size": 1000,
        "rhen_enable_meta_cognitive": True,
        "rhen_enable_three_layer_pipeline": True,
        "rhen_semantic_layer_weight": 0.4,
        "rhen_reasoning_layer_weight": 0.35,
        "rhen_generation_layer_weight": 0.25,
        "rhen_enable_confidence_routing": True,
        "rhen_min_semantic_confidence": 0.85,
        "rhen_max_reasoning_depth": 3,
    }


#: The eight group properties and their backing classes.
GROUPS = {
    "recurrent": RecurrentLoopConfig,
    "moe": MoEConfig,
    "attention": AttentionMLAConfig,
    "yarn": YaRNConfig,
    "act_budget": ActBudgetConfig,
    "rhen_engine": RhenEngineConfig,
    "loop_control": LoopControlConfig,
    "orchestrator": OrchestratorConfig,
}


class TestEOSConfigRoundTrip(unittest.TestCase):
    """Flat-signature and serialization contract for the nested EOSConfig."""

    def test_flat_defaults_match_history(self):
        cfg = EOSConfig()
        expected = _historical_flat_defaults()
        bad = {k: getattr(cfg, k) for k, v in expected.items() if getattr(cfg, k) != v}
        self.assertEqual(bad, {}, "flat defaults drifted from historical values")

    def test_explicit_flat_kwargs_override(self):
        cfg = EOSConfig(
            hidden_size=256,
            rhen_num_slots=4,
            enable_clam_bridge=True,
            use_moe=False,
        )
        self.assertEqual(cfg.hidden_size, 256)
        self.assertEqual(cfg.rhen_num_slots, 4)
        self.assertTrue(cfg.enable_clam_bridge)
        self.assertFalse(cfg.use_moe)
        # Derivations preserved
        self.assertEqual(cfg.head_dim, 32)  # 256 // 8
        self.assertEqual(cfg.moe_intermediate_size, 1680)  # fallback to intermediate_size

    def test_nested_construction_and_flat_precedence(self):
        cfg = EOSConfig(
            moe=MoEConfig(num_routed_experts=16, num_active_routed=4),
            rhen_engine=RhenEngineConfig(rhen_num_slots=2),
            loop_control=LoopControlConfig(enable_clam_bridge=True, clam_memory_dim=128),
        )
        self.assertEqual(cfg.num_routed_experts, 16)
        self.assertEqual(cfg.num_active_routed, 4)
        self.assertEqual(cfg.rhen_num_slots, 2)
        self.assertTrue(cfg.enable_clam_bridge)
        self.assertEqual(cfg.clam_memory_dim, 128)
        self.assertTrue(cfg.use_moe)  # untouched group keeps its default

        # A flat kwarg beats a nested group value.
        cfg2 = EOSConfig(moe=MoEConfig(num_routed_experts=16), num_routed_experts=8)
        self.assertEqual(cfg2.num_routed_experts, 8)

    def test_group_snapshots_are_live_views_not_state(self):
        cfg = EOSConfig(enable_clam_bridge=True, clam_memory_dim=99)
        snap = cfg.loop_control
        self.assertTrue(snap.enable_clam_bridge)
        self.assertEqual(snap.clam_memory_dim, 99)
        # Snapshots are derived properties: never instance state, never equal
        # objects, so they cannot leak into to_dict() or checkpoints.
        self.assertNotIn("loop_control", vars(cfg))
        self.assertIsNot(cfg.loop_control, cfg.loop_control)

    def test_to_dict_roundtrip_full_flat_surface(self):
        """EOSConfig(**all 141 flat kwargs).to_dict() -> EOSConfig(**flat) is lossless."""
        all_kwargs = _historical_flat_defaults()
        all_kwargs.update({
            "vocab_size": 50048,
            "hidden_size": 560,
            "intermediate_size": 1680,
            # Values that exercise the derivations and non-default branches:
            "head_dim": 70,
            "moe_intermediate_size": 512,   # explicit (must not be clobbered)
            "enable_selective_checkpoint": True,
            "min_recurrent_loops": 3,
            "act_min_loops": 3,
            "num_experts_per_tok": 1,
            "use_differential_attention": False,  # explicit False wins over use_diff_mla=True
            "enable_titans_memory": True,
            "enable_episodic_memory": True,
            "titans_d_mem": 32,
        })
        cfg = EOSConfig(**all_kwargs)

        d = cfg.to_dict()
        # Group snapshots must never leak into serialization.
        for group_name in GROUPS:
            self.assertNotIn(group_name, d)

        flat_keys = set(all_kwargs) | {"vocab_size", "hidden_size", "intermediate_size"}
        rebuilt = EOSConfig(**{k: v for k, v in d.items() if k in flat_keys})
        for key in sorted(flat_keys):
            self.assertEqual(
                getattr(rebuilt, key), getattr(cfg, key),
                f"round-trip drift on {key!r}",
            )

    def test_legacy_alias_unchanged(self):
        self.assertIs(LoopedLLMConfig, EOSConfig)
        self.assertEqual(EOSConfig.model_type, "eos")

    def test_group_snapshot_covers_every_field(self):
        """Every group field must exist as a flat attribute (flattening is total)."""
        cfg = EOSConfig()
        for group_name, group_cls in GROUPS.items():
            snapshot = getattr(cfg, group_name)
            self.assertIsInstance(snapshot, group_cls)
            for field in group_cls.__dataclass_fields__:
                self.assertTrue(
                    hasattr(cfg, field),
                    f"{group_name}.{field} missing from flattened EOSConfig",
                )


if __name__ == "__main__":
    print("=" * 70)
    print("  PHASE B1: EOS CONFIG NESTED-GROUP CONTRACT VERIFICATION")
    print("=" * 70)
    unittest.main(verbosity=2)
