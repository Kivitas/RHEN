"""Named model sizes for experiments.

``tiny`` runs in seconds on a CPU (tests and smoke runs), ``small`` fits a single consumer
accelerator for ablations, and ``base`` is the default ``EOSConfig`` (about 246M parameters).
Every preset keeps the full architecture (MLA, shared-expert MoE, recurrent core, SSM highway,
Rhen) so results at small scale exercise the same code as the full model. Values are
``EOSConfig`` keyword arguments; a run's ``model_overrides`` are applied on top.
"""

from __future__ import annotations

from typing import Any, Dict

PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        kv_lora_rank=16,
        q_lora_rank=32,
        num_pre_layers=1,
        num_recurrent_layers=2,
        num_post_layers=1,
        max_recurrent_loops=3,
        min_recurrent_loops=1,
        max_position_embeddings=512,
        rhen_hidden_dim=32,
        rhen_pointer_rank=16,
        rhen_num_slots=4,
        rhen_num_heads=2,
        num_experts=4,
        num_routed_experts=4,
        num_active_routed=2,
        enable_mtp=False,
    ),
    "small": dict(
        hidden_size=256,
        intermediate_size=704,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        kv_lora_rank=64,
        q_lora_rank=128,
        num_pre_layers=1,
        num_recurrent_layers=2,
        num_post_layers=1,
        max_recurrent_loops=6,
        min_recurrent_loops=2,
        max_position_embeddings=2048,
        rhen_hidden_dim=128,
        rhen_pointer_rank=64,
        rhen_num_slots=8,
        rhen_num_heads=4,
        num_experts=8,
        num_routed_experts=8,
        num_active_routed=2,
    ),
    "base": {},
}


def preset_config(name: str, **overrides: Any) -> Dict[str, Any]:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    merged = dict(PRESETS[name])
    merged.update(overrides)
    return merged
