"""Configuration for EOS & Rhen.

Structure (Phase B1)
--------------------
Every functional area of the architecture owns one dataclass group whose
defaults are defined exactly once:

    RecurrentLoopConfig   recurrent core shape & per-loop traversal
    MoEConfig             sparse / shared-expert MoE routing and balancing
    AttentionMLAConfig    (Diff-)MLA attention, kernels, KV quantization
    YaRNConfig            RoPE / YaRN long-context interpolation
    ActBudgetConfig       ACT halting, loop sampling, thinking budget,
                          token-level adaptive exit
    RhenEngineConfig      Rhen latent decision engine & three-layer routing
    LoopControlConfig     Rhen-1.0 recurrent-loop controls (default-off)
    OrchestratorConfig    training / runtime / serialization plumbing

``EOSConfig`` keeps its flat keyword signature (one kwarg per parameter, plus
the eight group keywords). Groups are *constructed and flattened* inside
``EOSConfig.__init__``: the single source of truth for flat attribute values
remains the ``EOSConfig`` instance itself, and each group is exposed as a
**read-only snapshot property** (``config.recurrent``, ``config.moe``, ...)
assembled live from the flat attributes. Snapshots are deliberately not stored
on the instance, so ``to_dict()`` / ``save_pretrained`` serialization and
checkpoint state dicts are unchanged.

Precedence when both are given: an explicit flat kwarg wins over the value
inside a nested group object; a group's own defaults fill everything the
caller left unset. Derived values (e.g. ``head_dim`` fallback, loop clamps,
``moe_intermediate_size`` fallback) are applied during flattening exactly as
before, so the flattened view of ``EOSConfig()`` is identical to the
historical defaults.
"""

import dataclasses
import functools
import warnings
from dataclasses import dataclass
from typing import Optional

try:
    from transformers import PretrainedConfig
except ImportError:
    PretrainedConfig = object


# ---------------------------------------------------------------------------
# Removed no-op keys. RHEN 1.0.0 deleted these knobs because no code read them.
# Configs and saved checkpoints written earlier may still carry them, so every
# constructor below accepts them, drops them, and emits one DeprecationWarning
# per call that names the keys. Nothing else about the config changes.
# ---------------------------------------------------------------------------
_REMOVED_LEGACY_KEYS = (
    # RecurrentLoopConfig
    "enable_symplectic_loop",
    # AttentionMLAConfig
    "use_flash_attn",
    "diff_attn_lambda_init",
    "enable_prefix_lm_mask",
    # YaRNConfig
    "yarn_scale_factor",
    # RhenEngineConfig
    "rhen_calib_loss_coef",
    "rhen_enable_deep_reasoning",
    "rhen_confidence_threshold",
    "rhen_enable_qwen_reasoning",
    "rhen_enable_deepseek_reasoning",
    "rhen_meta_reasoning_cache_size",
    "rhen_enable_three_layer_pipeline",
    "rhen_semantic_layer_weight",
    "rhen_reasoning_layer_weight",
    "rhen_generation_layer_weight",
    "rhen_enable_confidence_routing",
    "rhen_min_semantic_confidence",
    "rhen_max_reasoning_depth",
    # OrchestratorConfig
    "use_fused_recurrent_kernels",
    "checkpoint_policy",
    "use_triton",
)


def _pop_removed_legacy_keys(owner: str, *mappings: dict) -> None:
    """Remove ``_REMOVED_LEGACY_KEYS`` from each mapping in place, then warn once.

    ``owner`` names the constructor in the message. The warning lists every
    removed key found across all mappings, in tuple order, without repeats.
    """
    found = []
    for mapping in mappings:
        for key in _REMOVED_LEGACY_KEYS:
            if key in mapping:
                del mapping[key]
                found.append(key)
    if found:
        warnings.warn(
            f"{owner}: ignoring removed no-op config key(s) "
            f"{', '.join(dict.fromkeys(found))}; they had no effect and can be deleted.",
            DeprecationWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Nested config groups. Field defaults here are THE defaults; EOSConfig's flat
# signature forwards to them, so each default exists in exactly one place.
# ---------------------------------------------------------------------------


@dataclass
class RecurrentLoopConfig:
    """Recurrent core shape and per-loop traversal."""

    num_pre_layers: int = 1
    num_recurrent_layers: int = 4
    num_post_layers: int = 1
    max_recurrent_loops: int = 12     # Effective depth: 1 + (4 * 12) + 1 = 50
    loop_dropout_prob: float = 0.10
    enable_ssm_highway: bool = True   # 1D state-space memory channel across loops
    ssm_state_dim: int = 64
    use_step_modulation: bool = True  # FiLM step modulation per loop iteration
    fuse_multiloop_attn: bool = False
    enable_cross_loop_attention: bool = False
    cross_loop_memory_dim: int = 32
    enable_loop_bypass: bool = False
    enable_titans_memory: bool = False    # Titans MAG neural long-term memory
    enable_episodic_memory: bool = False  # Modern-Hopfield episodic memory bank
    titans_d_mem: int = 64                # Titans memory slot dimension


@dataclass
class MoEConfig:
    """Sparse / shared-expert Mixture-of-Experts routing and balancing."""

    use_moe: bool = True
    num_experts: int = 4              # 4 SwiGLU FFN experts (legacy mode)
    num_experts_per_tok: int = 2      # Top-2 routing per token
    moe_aux_loss_coef: float = 0.01
    router_type: str = "sigmoid"
    router_z_loss_coef: float = 0.001
    use_shared_expert_moe: bool = True   # invariant shared expert + fine-grained routed
    num_shared_experts: int = 1
    num_routed_experts: int = 8
    num_active_routed: int = 2        # active routed experts per token (Top-2 of 8)
    use_aux_free_balancing: bool = True  # aux-loss-free congestion-bias balancing
    aux_free_balance_rate: float = 0.001
    moe_intermediate_size: Optional[int] = None  # falls back to intermediate_size
    enable_grouped_gemm: bool = True


@dataclass
class AttentionMLAConfig:
    """Multi-Latent / differential attention and attention kernels."""

    num_attention_heads: int = 8
    num_key_value_heads: int = 2      # GQA/MLA head ratio
    head_dim: Optional[int] = None    # falls back to hidden_size // num_attention_heads
    kv_lora_rank: int = 140           # MLA latent KV compression rank
    q_lora_rank: int = 280            # MLA latent query compression rank
    use_sdpa: bool = True             # fused scaled_dot_product_attention
    use_qk_norm: bool = False
    # 0.0 keeps the fast SDPA attention path; >0 applies tanh softcapping on the manual path.
    attn_logit_softcapping: float = 0.0
    final_logit_softcapping: float = 30.0
    use_diff_mla: bool = True
    diff_mla_lambda_init: float = 0.8
    # Dual subspace noise-canceling attention. None follows use_diff_mla; an explicit bool wins.
    use_differential_attention: Optional[bool] = None
    diff_attn_learnable_lambda: bool = True
    quantized_kv_cache: bool = False
    kv_quant_dtype: str = "fp8"
    ring_size: int = 2
    use_qkv_bias: bool = True
    interleave_swa: bool = True
    swa_window_size: int = 512
    diff_v2_head_lambda: bool = True


@dataclass
class YaRNConfig:
    """RoPE base and YaRN ultra-long-context interpolation."""

    max_position_embeddings: int = 8192   # 8k optimized context window
    rope_theta: float = 500000.0
    use_yarn: bool = True
    yarn_scale: float = 4.0               # 8192 / 2048
    yarn_original_max_position_embeddings: int = 2048
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_attn_factor: float = 0.1


@dataclass
class ActBudgetConfig:
    """Adaptive computation time, loop sampling and dynamic loop budgets."""

    enable_act: bool = True
    act_threshold: float = 0.90
    act_ponder_penalty: float = 0.0001
    act_prior_lambda: float = 0.02    # PonderNet geometric prior target
    act_early_stop: bool = True
    act_min_loops: int = 5
    min_recurrent_loops: int = 4      # lower bound of stochastic loop sampling
    loop_sampling: str = "uniform"    # "fixed" | "uniform" - random depth per batch
    enable_thinking_budget: bool = True
    thinking_budget_min_loops: int = 4
    thinking_budget_max_loops: int = 12
    enable_adaptive_loop_router: bool = False  # per-token halting, frozen on trivia
    adaptive_loop_min_loops: int = 2
    adaptive_loop_freeze_threshold: float = 0.9
    adaptive_loop_savings_target: float = 0.35


@dataclass
class RhenEngineConfig:
    """Rhen latent decision engine (System One) and three-layer routing."""

    enable_rhen: bool = True
    rhen_num_slots: int = 8           # K orthogonal latent decision slots
    rhen_num_heads: int = 4           # H multi-head specialized bilinear pointer heads
    rhen_hidden_dim: int = 256
    rhen_pointer_rank: int = 128
    rhen_ortho_loss_coef: float = 0.01
    rhen_halt_confidence_threshold: float = 0.85
    rhen_enable_kv_modulation: bool = True
    rhen_enable_meta_cognitive: bool = True          # live: gates MetaCognitiveMonitor self-correction flags


@dataclass
class LoopControlConfig:
    """Rhen-1.0 recurrent-loop controls (see eos_loop.py). All default-off."""

    enable_step_size_control: bool = False  # arXiv:2609.16665 trust-region step
    loop_step_alpha: float = 0.25           # paper's effective quarter step
    loop_step_min_alpha: float = 0.05
    loop_step_anneal: bool = True
    loop_step_overshoot_rejection: bool = True
    loop_full_update: bool = True           # True = legacy full-step behaviour
    enable_flash_loop_sparsity: bool = False  # arXiv:2609.29812 token-sparse loops
    flash_loop_threshold: float = 1e-3
    flash_loop_warmup_loops: int = 2
    enable_loop_curriculum: bool = False    # arXiv:2603.02615 depth schedule
    loop_curriculum_warmup_steps: int = 0
    enable_clam_bridge: bool = False        # cross-loop compressed KV memory
    clam_memory_dim: int = 64
    clam_init_gate: float = 0.0             # initial lambda_clam (starts closed)
    enable_dynamic_ntk_yarn: bool = False   # rescale RoPE from actual runtime length
    dynamic_ntk_yarn_max_context: int = 16384
    dynamic_ntk_yarn_snap_factor: float = 0.25


@dataclass
class OrchestratorConfig:
    """Training / runtime / serialization plumbing (not architecture shape)."""

    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    z_loss_coef: float = 1e-4
    contraction_loss_coef: float = 0.01
    enable_mtp: bool = True             # multi-token prediction head
    mtp_loss_coef: float = 0.3
    mtp_num_layers: int = 1
    gradient_checkpointing: bool = False
    chunked_ce: bool = True
    chunked_ce_chunk_size: int = 512
    enable_selective_checkpoint: Optional[bool] = None  # falls back to gradient_checkpointing
    mup_enabled: bool = False
    mup_base_hidden: int = 1024
    mup_base_loops: int = 16
    enable_speculative_decoding: bool = False
    draft_loops: int = 2
    max_draft_tokens: int = 4
    lr_scale_factor: float = 1.0


# Group dataclasses accept the removed keys as well, so direct construction such as
# RhenEngineConfig(rhen_confidence_threshold=...) still works. Each generated __init__
# is wrapped once here, and the warning names the group class.
for _group_cls in (
    RecurrentLoopConfig, MoEConfig, AttentionMLAConfig, YaRNConfig,
    ActBudgetConfig, RhenEngineConfig, LoopControlConfig, OrchestratorConfig,
):
    def _accept_removed_keys(self, *args, _generated_init=_group_cls.__init__, **kwargs):
        _pop_removed_legacy_keys(type(self).__name__, kwargs)
        _generated_init(self, *args, **kwargs)

    _group_cls.__init__ = functools.wraps(_group_cls.__init__)(_accept_removed_keys)
del _group_cls


#: The eight group keywords accepted alongside the flat kwargs.
CONFIG_GROUP_NAMES = (
    "recurrent",
    "moe",
    "attention",
    "yarn",
    "act_budget",
    "rhen_engine",
    "loop_control",
    "orchestrator",
)


def _merge_group(group, flat: dict):
    """Effective values for one group: explicit flat kwargs win, then the
    group object's values (defaults if no group object was passed).

    ``flat`` maps flat parameter names to the caller-supplied value or None
    (None = not explicitly passed, because group-backed flat parameters use a
    None sentinel default).
    """
    merged = {f.name: getattr(group, f.name) for f in dataclasses.fields(group)}
    for name, value in flat.items():
        if value is not None:
            merged[name] = value
    return merged


class EOSConfig(PretrainedConfig):
    """
    Configuration parameters for EOS.
    Configured for 50,000 Vocab Size, 50-Layer Depth, 8k Context Window (16k Extrapolatable via YaRN),
    and Rhen Non-Autoregressive System One Decision Engine.

    Construction is backward compatible: every historical flat kwarg works
    unchanged. Additionally, whole groups may be passed as nested objects
    (``EOSConfig(recurrent=RecurrentLoopConfig(...), moe=MoEConfig(...))``);
    see the module docstring for the precedence rules.
    """
    model_type = "eos"

    # transformers applies @dataclass(kw_only=True) to every PreTrainedConfig
    # subclass (PreTrainedConfig.__init_subclass__). The generated __eq__ compares
    # only the base fields, so distinct EOSConfigs compared equal. Defining
    # __eq__ in this body keeps the generated one out; equality stays
    # PreTrainedConfig's __dict__ comparison. PreTrainedConfig also sets
    # __hash__ = None. The hash below depends only on the class, so it never
    # goes stale when a config is mutated and equal configs always hash equal.
    __eq__ = PretrainedConfig.__eq__

    def __hash__(self):
        return hash(self.model_type)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Build from a dict. Removed no-op keys are dropped with one warning; the caller's dict is not mutated."""
        config_dict = dict(config_dict)
        _pop_removed_legacy_keys(cls.__name__, config_dict, kwargs)
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        vocab_size: int = 50048,           # 50,048 Vocab Size (accommodates base + special tokens)
        hidden_size: int = 560,            # Hidden Dimension d_model (560 * 50k = 28.0M Embed)
        intermediate_size: int = 1680,     # SwiGLU FFN hidden dimension per expert
        num_pre_layers: Optional[int] = None,
        num_recurrent_layers: Optional[int] = None,
        num_post_layers: Optional[int] = None,
        max_recurrent_loops: Optional[int] = None,  # Effective Depth: 1 + (4 * 12) + 1 = 50 layers!
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,   # GQA/MLA head ratio
        head_dim: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,          # MLA Latent KV compression rank
        q_lora_rank: Optional[int] = None,           # MLA Latent Query compression rank

        # Sparse Mixture of Experts (MoE)
        use_moe: Optional[bool] = None,
        num_experts: Optional[int] = None,           # 4 SwiGLU FFN Experts (legacy mode)
        num_experts_per_tok: Optional[int] = None,   # Top-2 routing per token
        moe_aux_loss_coef: Optional[float] = None,   # Load balancing auxiliary loss coefficient

        rms_norm_eps: Optional[float] = None,
        max_position_embeddings: Optional[int] = None,  # 8k Optimized Context Window!
        rope_theta: Optional[float] = None,             # Extended RoPE Base Theta for 8k/16k stability

        # YaRN & Ultra-Long Context Hardening
        use_yarn: Optional[bool] = None,             # Full YaRN NTK-by-parts frequency interpolation
        yarn_scale: Optional[float] = None,          # Scale factor (8192 / 2048 = 4.0)
        yarn_original_max_position_embeddings: Optional[int] = None,  # Base training length
        yarn_beta_fast: Optional[float] = None,      # High frequency threshold
        yarn_beta_slow: Optional[float] = None,      # Low frequency threshold
        yarn_attn_factor: Optional[float] = None,    # Attention entropy scaling factor

        tie_word_embeddings: Optional[bool] = None,  # Weight-tied LM head
        enable_act: Optional[bool] = None,           # Adaptive Computation Time dynamic halting
        act_threshold: Optional[float] = None,       # Halting threshold
        act_ponder_penalty: Optional[float] = None,  # Ponder penalty
        use_step_modulation: Optional[bool] = None,  # FiLM step modulation per loop iteration

        # Attention Acceleration & Kernels
        use_sdpa: Optional[bool] = None,             # Fused scaled_dot_product_attention
        use_qk_norm: Optional[bool] = None,          # RMSNorm on Q/K before dot-product
        z_loss_coef: Optional[float] = None,         # logit z-loss
        initializer_range: Optional[float] = None,   # Base init std

        # Looping control
        min_recurrent_loops: Optional[int] = None,   # Lower bound of stochastic loop sampling
        loop_sampling: Optional[str] = None,         # "fixed" | "uniform" - random depth per batch
        act_prior_lambda: Optional[float] = None,    # PonderNet geometric prior target
        act_early_stop: Optional[bool] = None,       # Halt loop early once every token has exited
        act_min_loops: Optional[int] = None,         # Enforce at least this many core applications

        # Memory / runtime switches
        gradient_checkpointing: Optional[bool] = None,
        chunked_ce: Optional[bool] = None,
        chunked_ce_chunk_size: Optional[int] = None,
        enable_grouped_gemm: Optional[bool] = None,
        quantized_kv_cache: Optional[bool] = None,
        kv_quant_dtype: Optional[str] = None,
        enable_selective_checkpoint: Optional[bool] = None,
        mup_enabled: Optional[bool] = None,
        mup_base_hidden: Optional[int] = None,
        mup_base_loops: Optional[int] = None,
        enable_cross_loop_attention: Optional[bool] = None,
        cross_loop_memory_dim: Optional[int] = None,
        enable_speculative_decoding: Optional[bool] = None,
        draft_loops: Optional[int] = None,
        enable_loop_bypass: Optional[bool] = None,
        fuse_multiloop_attn: Optional[bool] = None,

        # Sequence parallelism & execution
        ring_size: Optional[int] = None,
        max_draft_tokens: Optional[int] = None,
        router_type: Optional[str] = None,
        lr_scale_factor: Optional[float] = None,

        # Architectural enhancements
        attn_logit_softcapping: Optional[float] = None,
        final_logit_softcapping: Optional[float] = None,
        contraction_loss_coef: Optional[float] = None,
        loop_dropout_prob: Optional[float] = None,
        router_z_loss_coef: Optional[float] = None,

        # Experimental Looping Upgrades
        enable_ssm_highway: Optional[bool] = None,   # State-Space Memory channel across loops
        ssm_state_dim: Optional[int] = None,
        enable_titans_memory: Optional[bool] = None,    # Titans MAG neural long-term memory
        enable_episodic_memory: Optional[bool] = None,  # Modern-Hopfield episodic memory bank
        titans_d_mem: Optional[int] = None,             # Titans memory slot dimension

        # 1. Isolated Shared-Expert Fine-Grained MoE (SE-MoE)
        use_shared_expert_moe: Optional[bool] = None,
        num_shared_experts: Optional[int] = None,
        num_routed_experts: Optional[int] = None,
        num_active_routed: Optional[int] = None,
        use_aux_free_balancing: Optional[bool] = None,
        aux_free_balance_rate: Optional[float] = None,
        moe_intermediate_size: Optional[int] = None,  # Fine-grained expert intermediate size

        # 2. Multi-Token Autoregressive Depth Prediction (MTP)
        enable_mtp: Optional[bool] = None,
        mtp_loss_coef: Optional[float] = None,
        mtp_num_layers: Optional[int] = None,

        # 3. Differential Multi-Latent Attention (Diff-MLA)
        use_diff_mla: Optional[bool] = None,
        diff_mla_lambda_init: Optional[float] = None,
        use_differential_attention: Optional[bool] = None,
        diff_attn_learnable_lambda: Optional[bool] = None,
        use_qkv_bias: Optional[bool] = None,
        interleave_swa: Optional[bool] = None,
        swa_window_size: Optional[int] = None,
        diff_v2_head_lambda: Optional[bool] = None,

        # 4. Dynamic Test-Time Thinking Budget Controller
        enable_thinking_budget: Optional[bool] = None,
        thinking_budget_min_loops: Optional[int] = None,
        thinking_budget_max_loops: Optional[int] = None,

        # 5/6/7/8. Rhen-1.0 loop controls, adaptive exit, CLAM, dynamic NTK
        enable_step_size_control: Optional[bool] = None,
        loop_step_alpha: Optional[float] = None,
        loop_step_min_alpha: Optional[float] = None,
        loop_step_anneal: Optional[bool] = None,
        loop_step_overshoot_rejection: Optional[bool] = None,
        loop_full_update: Optional[bool] = None,
        enable_flash_loop_sparsity: Optional[bool] = None,
        flash_loop_threshold: Optional[float] = None,
        flash_loop_warmup_loops: Optional[int] = None,
        enable_loop_curriculum: Optional[bool] = None,
        loop_curriculum_warmup_steps: Optional[int] = None,
        enable_adaptive_loop_router: Optional[bool] = None,
        adaptive_loop_min_loops: Optional[int] = None,
        adaptive_loop_freeze_threshold: Optional[float] = None,
        adaptive_loop_savings_target: Optional[float] = None,
        enable_clam_bridge: Optional[bool] = None,
        clam_memory_dim: Optional[int] = None,
        clam_init_gate: Optional[float] = None,
        enable_dynamic_ntk_yarn: Optional[bool] = None,
        dynamic_ntk_yarn_max_context: Optional[int] = None,
        dynamic_ntk_yarn_snap_factor: Optional[float] = None,

        # Rhen Latent Decision Engine (System One Parallel Reasoner)
        enable_rhen: Optional[bool] = None,
        rhen_num_slots: Optional[int] = None,
        rhen_num_heads: Optional[int] = None,
        rhen_hidden_dim: Optional[int] = None,
        rhen_pointer_rank: Optional[int] = None,
        rhen_ortho_loss_coef: Optional[float] = None,
        rhen_halt_confidence_threshold: Optional[float] = None,
        rhen_enable_kv_modulation: Optional[bool] = None,
        rhen_enable_meta_cognitive: Optional[bool] = None,

        # Nested group objects (Phase B1). See module docstring.
        recurrent: Optional[RecurrentLoopConfig] = None,
        moe: Optional[MoEConfig] = None,
        attention: Optional[AttentionMLAConfig] = None,
        yarn: Optional[YaRNConfig] = None,
        act_budget: Optional[ActBudgetConfig] = None,
        rhen_engine: Optional[RhenEngineConfig] = None,
        loop_control: Optional[LoopControlConfig] = None,
        orchestrator: Optional[OrchestratorConfig] = None,
        **kwargs
    ):
        # ---- Removed no-op keys: accept, drop, warn (see _REMOVED_LEGACY_KEYS) ----
        _pop_removed_legacy_keys(type(self).__name__, kwargs)

        # ---- Resolve the eight groups -------------------------------------
        # Defaults live in the group dataclasses; explicit flat kwargs (the
        # non-None sentinel values below) override them; a passed group object
        # supplies values for everything the caller left unset.
        recurrent_g = RecurrentLoopConfig() if recurrent is None else recurrent
        moe_g = MoEConfig() if moe is None else moe
        attention_g = AttentionMLAConfig() if attention is None else attention
        yarn_g = YaRNConfig() if yarn is None else yarn
        act_budget_g = ActBudgetConfig() if act_budget is None else act_budget
        rhen_engine_g = RhenEngineConfig() if rhen_engine is None else rhen_engine
        loop_control_g = LoopControlConfig() if loop_control is None else loop_control
        orchestrator_g = OrchestratorConfig() if orchestrator is None else orchestrator

        recurrent_g = RecurrentLoopConfig(**_merge_group(recurrent_g, dict(
            num_pre_layers=num_pre_layers,
            num_recurrent_layers=num_recurrent_layers,
            num_post_layers=num_post_layers,
            max_recurrent_loops=max_recurrent_loops,
            loop_dropout_prob=loop_dropout_prob,
            enable_ssm_highway=enable_ssm_highway,
            ssm_state_dim=ssm_state_dim,
            use_step_modulation=use_step_modulation,
            fuse_multiloop_attn=fuse_multiloop_attn,
            enable_cross_loop_attention=enable_cross_loop_attention,
            cross_loop_memory_dim=cross_loop_memory_dim,
            enable_loop_bypass=enable_loop_bypass,
            enable_titans_memory=enable_titans_memory,
            enable_episodic_memory=enable_episodic_memory,
            titans_d_mem=titans_d_mem,
        )))
        moe_g = MoEConfig(**_merge_group(moe_g, dict(
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_aux_loss_coef=moe_aux_loss_coef,
            router_type=router_type,
            router_z_loss_coef=router_z_loss_coef,
            use_shared_expert_moe=use_shared_expert_moe,
            num_shared_experts=num_shared_experts,
            num_routed_experts=num_routed_experts,
            num_active_routed=num_active_routed,
            use_aux_free_balancing=use_aux_free_balancing,
            aux_free_balance_rate=aux_free_balance_rate,
            moe_intermediate_size=moe_intermediate_size,
            enable_grouped_gemm=enable_grouped_gemm,
        )))
        attention_g = AttentionMLAConfig(**_merge_group(attention_g, dict(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            use_sdpa=use_sdpa,
            use_qk_norm=use_qk_norm,
            attn_logit_softcapping=attn_logit_softcapping,
            final_logit_softcapping=final_logit_softcapping,
            use_diff_mla=use_diff_mla,
            diff_mla_lambda_init=diff_mla_lambda_init,
            use_differential_attention=use_differential_attention,
            diff_attn_learnable_lambda=diff_attn_learnable_lambda,
            quantized_kv_cache=quantized_kv_cache,
            kv_quant_dtype=kv_quant_dtype,
            ring_size=ring_size,
            use_qkv_bias=use_qkv_bias,
            interleave_swa=interleave_swa,
            swa_window_size=swa_window_size,
            diff_v2_head_lambda=diff_v2_head_lambda,
        )))
        yarn_g = YaRNConfig(**_merge_group(yarn_g, dict(
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            use_yarn=use_yarn,
            yarn_scale=yarn_scale,
            yarn_original_max_position_embeddings=yarn_original_max_position_embeddings,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            yarn_attn_factor=yarn_attn_factor,
        )))
        act_budget_g = ActBudgetConfig(**_merge_group(act_budget_g, dict(
            enable_act=enable_act,
            act_threshold=act_threshold,
            act_ponder_penalty=act_ponder_penalty,
            act_prior_lambda=act_prior_lambda,
            act_early_stop=act_early_stop,
            act_min_loops=act_min_loops,
            min_recurrent_loops=min_recurrent_loops,
            loop_sampling=loop_sampling,
            enable_thinking_budget=enable_thinking_budget,
            thinking_budget_min_loops=thinking_budget_min_loops,
            thinking_budget_max_loops=thinking_budget_max_loops,
            enable_adaptive_loop_router=enable_adaptive_loop_router,
            adaptive_loop_min_loops=adaptive_loop_min_loops,
            adaptive_loop_freeze_threshold=adaptive_loop_freeze_threshold,
            adaptive_loop_savings_target=adaptive_loop_savings_target,
        )))
        rhen_engine_g = RhenEngineConfig(**_merge_group(rhen_engine_g, dict(
            enable_rhen=enable_rhen,
            rhen_num_slots=rhen_num_slots,
            rhen_num_heads=rhen_num_heads,
            rhen_hidden_dim=rhen_hidden_dim,
            rhen_pointer_rank=rhen_pointer_rank,
            rhen_ortho_loss_coef=rhen_ortho_loss_coef,
            rhen_halt_confidence_threshold=rhen_halt_confidence_threshold,
            rhen_enable_kv_modulation=rhen_enable_kv_modulation,
            rhen_enable_meta_cognitive=rhen_enable_meta_cognitive,
        )))
        loop_control_g = LoopControlConfig(**_merge_group(loop_control_g, dict(
            enable_step_size_control=enable_step_size_control,
            loop_step_alpha=loop_step_alpha,
            loop_step_min_alpha=loop_step_min_alpha,
            loop_step_anneal=loop_step_anneal,
            loop_step_overshoot_rejection=loop_step_overshoot_rejection,
            loop_full_update=loop_full_update,
            enable_flash_loop_sparsity=enable_flash_loop_sparsity,
            flash_loop_threshold=flash_loop_threshold,
            flash_loop_warmup_loops=flash_loop_warmup_loops,
            enable_loop_curriculum=enable_loop_curriculum,
            loop_curriculum_warmup_steps=loop_curriculum_warmup_steps,
            enable_clam_bridge=enable_clam_bridge,
            clam_memory_dim=clam_memory_dim,
            clam_init_gate=clam_init_gate,
            enable_dynamic_ntk_yarn=enable_dynamic_ntk_yarn,
            dynamic_ntk_yarn_max_context=dynamic_ntk_yarn_max_context,
            dynamic_ntk_yarn_snap_factor=dynamic_ntk_yarn_snap_factor,
        )))
        orchestrator_g = OrchestratorConfig(**_merge_group(orchestrator_g, dict(
            tie_word_embeddings=tie_word_embeddings,
            rms_norm_eps=rms_norm_eps,
            initializer_range=initializer_range,
            z_loss_coef=z_loss_coef,
            contraction_loss_coef=contraction_loss_coef,
            enable_mtp=enable_mtp,
            mtp_loss_coef=mtp_loss_coef,
            mtp_num_layers=mtp_num_layers,
            gradient_checkpointing=gradient_checkpointing,
            chunked_ce=chunked_ce,
            chunked_ce_chunk_size=chunked_ce_chunk_size,
            enable_selective_checkpoint=enable_selective_checkpoint,
            mup_enabled=mup_enabled,
            mup_base_hidden=mup_base_hidden,
            mup_base_loops=mup_base_loops,
            enable_speculative_decoding=enable_speculative_decoding,
            draft_loops=draft_loops,
            max_draft_tokens=max_draft_tokens,
            lr_scale_factor=lr_scale_factor,
        )))

        # ---- Core dimensions ----------------------------------------------
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # ---- Flatten: RecurrentLoopConfig ----------------------------------
        self.num_pre_layers = recurrent_g.num_pre_layers
        self.num_recurrent_layers = recurrent_g.num_recurrent_layers
        self.num_post_layers = recurrent_g.num_post_layers
        self.max_recurrent_loops = recurrent_g.max_recurrent_loops
        self.loop_dropout_prob = recurrent_g.loop_dropout_prob
        self.enable_ssm_highway = recurrent_g.enable_ssm_highway
        self.ssm_state_dim = recurrent_g.ssm_state_dim
        self.use_step_modulation = recurrent_g.use_step_modulation
        self.fuse_multiloop_attn = recurrent_g.fuse_multiloop_attn
        self.enable_cross_loop_attention = recurrent_g.enable_cross_loop_attention
        self.cross_loop_memory_dim = recurrent_g.cross_loop_memory_dim
        self.enable_loop_bypass = recurrent_g.enable_loop_bypass
        self.enable_titans_memory = recurrent_g.enable_titans_memory
        self.enable_episodic_memory = recurrent_g.enable_episodic_memory
        self.titans_d_mem = recurrent_g.titans_d_mem

        # ---- Flatten: MoEConfig ---------------------------------------------
        self.use_moe = moe_g.use_moe
        self.num_experts = moe_g.num_experts
        self.num_experts_per_tok = min(moe_g.num_experts_per_tok, moe_g.num_experts)
        self.moe_aux_loss_coef = moe_g.moe_aux_loss_coef
        self.router_type = moe_g.router_type
        self.router_z_loss_coef = moe_g.router_z_loss_coef
        self.use_shared_expert_moe = moe_g.use_shared_expert_moe
        self.num_shared_experts = moe_g.num_shared_experts
        self.num_routed_experts = moe_g.num_routed_experts
        self.num_active_routed = moe_g.num_active_routed
        self.use_aux_free_balancing = moe_g.use_aux_free_balancing
        self.aux_free_balance_rate = moe_g.aux_free_balance_rate
        self.moe_intermediate_size = moe_g.moe_intermediate_size or intermediate_size
        self.enable_grouped_gemm = moe_g.enable_grouped_gemm

        # ---- Flatten: AttentionMLAConfig ------------------------------------
        self.num_attention_heads = attention_g.num_attention_heads
        self.num_key_value_heads = attention_g.num_key_value_heads
        self.head_dim = attention_g.head_dim or (hidden_size // attention_g.num_attention_heads)
        self.kv_lora_rank = attention_g.kv_lora_rank
        self.q_lora_rank = attention_g.q_lora_rank
        self.use_sdpa = attention_g.use_sdpa
        self.use_qk_norm = attention_g.use_qk_norm
        self.attn_logit_softcapping = attention_g.attn_logit_softcapping
        self.final_logit_softcapping = attention_g.final_logit_softcapping
        self.use_diff_mla = attention_g.use_diff_mla
        self.diff_mla_lambda_init = attention_g.diff_mla_lambda_init
        # An explicit value wins (False included, even with use_diff_mla=True);
        # when unset, differential attention follows use_diff_mla.
        self.use_differential_attention = (
            attention_g.use_differential_attention
            if attention_g.use_differential_attention is not None
            else bool(attention_g.use_diff_mla)
        )
        self.diff_attn_learnable_lambda = attention_g.diff_attn_learnable_lambda
        self.quantized_kv_cache = attention_g.quantized_kv_cache
        self.kv_quant_dtype = attention_g.kv_quant_dtype
        self.ring_size = attention_g.ring_size
        self.use_qkv_bias = attention_g.use_qkv_bias
        self.interleave_swa = attention_g.interleave_swa
        self.swa_window_size = attention_g.swa_window_size
        self.diff_v2_head_lambda = attention_g.diff_v2_head_lambda

        # ---- Flatten: YaRNConfig ---------------------------------------------
        self.max_position_embeddings = yarn_g.max_position_embeddings
        self.rope_theta = yarn_g.rope_theta
        self.use_yarn = yarn_g.use_yarn
        self.yarn_scale = yarn_g.yarn_scale
        self.yarn_original_max_position_embeddings = yarn_g.yarn_original_max_position_embeddings
        self.yarn_beta_fast = yarn_g.yarn_beta_fast
        self.yarn_beta_slow = yarn_g.yarn_beta_slow
        self.yarn_attn_factor = yarn_g.yarn_attn_factor

        # ---- Flatten: ActBudgetConfig (with the historical clamps) -----------
        self.enable_act = act_budget_g.enable_act
        self.act_threshold = act_budget_g.act_threshold
        self.act_ponder_penalty = act_budget_g.act_ponder_penalty
        self.act_prior_lambda = act_budget_g.act_prior_lambda
        self.act_early_stop = act_budget_g.act_early_stop
        self.act_min_loops = max(0, min(act_budget_g.act_min_loops, self.max_recurrent_loops - 1))
        self.min_recurrent_loops = max(1, min(act_budget_g.min_recurrent_loops, self.max_recurrent_loops))
        self.loop_sampling = act_budget_g.loop_sampling
        self.enable_thinking_budget = act_budget_g.enable_thinking_budget
        self.thinking_budget_min_loops = act_budget_g.thinking_budget_min_loops
        self.thinking_budget_max_loops = act_budget_g.thinking_budget_max_loops
        self.enable_adaptive_loop_router = act_budget_g.enable_adaptive_loop_router
        self.adaptive_loop_min_loops = max(0, int(act_budget_g.adaptive_loop_min_loops))
        self.adaptive_loop_freeze_threshold = float(act_budget_g.adaptive_loop_freeze_threshold)
        self.adaptive_loop_savings_target = float(act_budget_g.adaptive_loop_savings_target)

        # ---- Flatten: RhenEngineConfig ----------------------------------------
        for _f in dataclasses.fields(RhenEngineConfig):
            setattr(self, _f.name, getattr(rhen_engine_g, _f.name))

        # ---- Flatten: LoopControlConfig ----------------------------------------
        self.enable_step_size_control = loop_control_g.enable_step_size_control
        self.loop_step_alpha = loop_control_g.loop_step_alpha
        self.loop_step_min_alpha = loop_control_g.loop_step_min_alpha
        self.loop_step_anneal = loop_control_g.loop_step_anneal
        self.loop_step_overshoot_rejection = loop_control_g.loop_step_overshoot_rejection
        self.loop_full_update = loop_control_g.loop_full_update
        self.enable_flash_loop_sparsity = loop_control_g.enable_flash_loop_sparsity
        self.flash_loop_threshold = loop_control_g.flash_loop_threshold
        self.flash_loop_warmup_loops = loop_control_g.flash_loop_warmup_loops
        self.enable_loop_curriculum = loop_control_g.enable_loop_curriculum
        self.loop_curriculum_warmup_steps = loop_control_g.loop_curriculum_warmup_steps
        self.enable_clam_bridge = loop_control_g.enable_clam_bridge
        self.clam_memory_dim = int(loop_control_g.clam_memory_dim)
        self.clam_init_gate = float(loop_control_g.clam_init_gate)
        self.enable_dynamic_ntk_yarn = loop_control_g.enable_dynamic_ntk_yarn
        self.dynamic_ntk_yarn_max_context = int(loop_control_g.dynamic_ntk_yarn_max_context)
        self.dynamic_ntk_yarn_snap_factor = float(loop_control_g.dynamic_ntk_yarn_snap_factor)

        # ---- Flatten: OrchestratorConfig ---------------------------------------
        self.tie_word_embeddings = orchestrator_g.tie_word_embeddings
        self.rms_norm_eps = orchestrator_g.rms_norm_eps
        self.initializer_range = orchestrator_g.initializer_range
        self.z_loss_coef = orchestrator_g.z_loss_coef
        self.contraction_loss_coef = orchestrator_g.contraction_loss_coef
        self.enable_mtp = orchestrator_g.enable_mtp
        self.mtp_loss_coef = orchestrator_g.mtp_loss_coef
        self.mtp_num_layers = orchestrator_g.mtp_num_layers
        self.gradient_checkpointing = orchestrator_g.gradient_checkpointing
        self.chunked_ce = orchestrator_g.chunked_ce
        self.chunked_ce_chunk_size = orchestrator_g.chunked_ce_chunk_size
        self.enable_selective_checkpoint = (
            orchestrator_g.enable_selective_checkpoint
            if orchestrator_g.enable_selective_checkpoint is not None
            else orchestrator_g.gradient_checkpointing
        )
        self.mup_enabled = orchestrator_g.mup_enabled
        self.mup_base_hidden = orchestrator_g.mup_base_hidden
        self.mup_base_loops = orchestrator_g.mup_base_loops
        self.enable_speculative_decoding = orchestrator_g.enable_speculative_decoding
        self.draft_loops = orchestrator_g.draft_loops
        self.max_draft_tokens = orchestrator_g.max_draft_tokens
        self.lr_scale_factor = orchestrator_g.lr_scale_factor

        if hasattr(super(), "__init__"):
            super().__init__(**kwargs)

    # -------------------------------------------------------------------------
    # Read-only group snapshots. These are properties (never instance state)
    # so they never leak into __dict__, to_dict() or saved checkpoints.
    # -------------------------------------------------------------------------

    def _group_snapshot(self, group_cls):
        names = (f.name for f in dataclasses.fields(group_cls))
        return group_cls(**{name: getattr(self, name) for name in names})

    @property
    def recurrent(self) -> RecurrentLoopConfig:
        """Live snapshot of the recurrent-loop group (read-only view)."""
        return self._group_snapshot(RecurrentLoopConfig)

    @property
    def moe(self) -> MoEConfig:
        """Live snapshot of the MoE group (read-only view)."""
        return self._group_snapshot(MoEConfig)

    @property
    def attention(self) -> AttentionMLAConfig:
        """Live snapshot of the attention group (read-only view)."""
        return self._group_snapshot(AttentionMLAConfig)

    @property
    def yarn(self) -> YaRNConfig:
        """Live snapshot of the YaRN group (read-only view)."""
        return self._group_snapshot(YaRNConfig)

    @property
    def act_budget(self) -> ActBudgetConfig:
        """Live snapshot of the ACT/budget group (read-only view)."""
        return self._group_snapshot(ActBudgetConfig)

    @property
    def rhen_engine(self) -> RhenEngineConfig:
        """Live snapshot of the Rhen engine group (read-only view)."""
        return self._group_snapshot(RhenEngineConfig)

    @property
    def loop_control(self) -> LoopControlConfig:
        """Live snapshot of the loop-control group (read-only view)."""
        return self._group_snapshot(LoopControlConfig)

    @property
    def orchestrator(self) -> OrchestratorConfig:
        """Live snapshot of the orchestrator group (read-only view)."""
        return self._group_snapshot(OrchestratorConfig)

    # -------------------------------------------------------------------------
    # Backward-compatible dimension aliases.
    # -------------------------------------------------------------------------

    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def d_ff(self) -> int:
        return self.intermediate_size

    @property
    def n_heads(self) -> int:
        return self.num_attention_heads

    @property
    def n_layers(self) -> int:
        return self.num_recurrent_layers

    @property
    def max_loops(self) -> int:
        return self.max_recurrent_loops


# ---------------------------------------------------------------------------
# Backward-compatibility aliases.
# EOS was previously published as "EOS Looped LLM". The canonical public name
# is now `EOSConfig`; the legacy name is retained so existing checkpoints and
# downstream imports keep working.
# ---------------------------------------------------------------------------
LoopedLLMConfig = EOSConfig
RhenConfig = EOSConfig

__all__ = [
    "RhenConfig",
    "EOSConfig",
    "LoopedLLMConfig",
    "RecurrentLoopConfig",
    "MoEConfig",
    "AttentionMLAConfig",
    "YaRNConfig",
    "ActBudgetConfig",
    "RhenEngineConfig",
    "LoopControlConfig",
    "OrchestratorConfig",
]
