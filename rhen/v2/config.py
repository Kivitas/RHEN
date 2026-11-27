"""Configuration, presets and parameter accounting for RHEN-2 (Cached-State Deliberation Lattice).

One flat dataclass (under 40 fields) describes a model. The module code is identical at every
scale; only these values change. ``count_parameters`` computes the parameter count from the
same shapes the modules use, so a test can compare it with the instantiated model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

NUM_QUESTION_TYPES = 6
QUESTION_TYPES = ("choice", "noul", "score", "multi", "range", "extract")
TYPE_INDEX = {name: i for i, name in enumerate(QUESTION_TYPES)}
NUM_QUANTILES = 9
QUANTILE_LEVELS = tuple(round(0.1 * (i + 1), 1) for i in range(NUM_QUANTILES))


@dataclass
class Rhen2Config:
    # shape
    vocab_size: int = 50048
    d_model: int = 512
    d_embed: int = 192
    n_heads: int = 8
    n_state_layers: int = 8
    n_core_layers: int = 3
    n_set_layers: int = 1
    ffn_mult: float = 8.0 / 3.0
    d_readout: int = 256
    n_scratch: int = 8
    # loops
    r_min: int = 2
    r_mean: int = 4
    r_max: int = 8
    k_bptt: int = 2
    # limits
    max_state_tokens: int = 2048
    max_q_tokens: int = 128
    max_opt_tokens: int = 48
    max_options: int = 255
    max_levels: int = 64
    max_span: int = 64
    # attention layout
    global_every: int = 0          # 0: every state layer is global; n: every n-th is global, rest local
    local_window: int = 128        # symmetric window (tokens on each side is window // 2)
    evidence_n: int = 0            # 0: off; otherwise options read this many state positions
    evidence_budget: int = 4_000_000
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    null_bias_init: float = 2.0
    # ablation switches (each one is a single flag, see plan section 10.4)
    use_set_layer: bool = True
    use_null: bool = True
    use_numeric: bool = True
    use_reinjection: bool = True
    use_damping: bool = True
    use_sandwich_norm: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head dimension must be even for rotary embeddings")
        if not (1 <= self.r_min <= self.r_mean <= self.r_max):
            raise ValueError("need 1 <= r_min <= r_mean <= r_max")
        if not (1 <= self.k_bptt):
            raise ValueError("k_bptt must be >= 1")
        if self.max_options > 255:
            raise ValueError("max_options is capped at 255")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        return int(self.ffn_mult * self.d_model)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rhen2Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown Rhen2Config keys: {sorted(unknown)}")
        return cls(**data)


PRESETS: Dict[str, Dict[str, Any]] = {
    "r2-tiny": dict(
        d_model=64, d_embed=32, n_heads=4, n_state_layers=2, n_core_layers=1, n_set_layers=1,
        d_readout=32, n_scratch=2, r_min=1, r_mean=2, r_max=3, k_bptt=2,
        max_state_tokens=256, max_q_tokens=48, max_opt_tokens=24,
    ),
    "r2-50m": dict(
        d_model=512, d_embed=192, n_heads=8, n_state_layers=8, n_core_layers=3, n_set_layers=1,
        n_scratch=8, r_min=2, r_mean=4, r_max=8, max_state_tokens=2048, global_every=0, evidence_n=0,
    ),
    "r2-300m": dict(
        d_model=1024, d_embed=384, n_heads=16, n_state_layers=16, n_core_layers=4, n_set_layers=2,
        n_scratch=8, r_min=2, r_mean=4, r_max=8, max_state_tokens=8192, global_every=3, evidence_n=256,
    ),
    "r2-1b": dict(
        d_model=1536, d_embed=768, n_heads=24, n_state_layers=24, n_core_layers=6, n_set_layers=2,
        n_scratch=16, r_min=2, r_mean=4, r_max=12, max_state_tokens=16384, global_every=3, evidence_n=384,
    ),
    "r2-8b": dict(
        d_model=3072, d_embed=1536, n_heads=48, n_state_layers=54, n_core_layers=10, n_set_layers=2,
        n_scratch=32, r_min=2, r_mean=4, r_max=16, max_state_tokens=32768, global_every=3, evidence_n=512,
    ),
}


def preset(name: str, **overrides: Any) -> Rhen2Config:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    values = dict(PRESETS[name])
    values.update(overrides)
    return Rhen2Config(**values)


def count_parameters(cfg: Rhen2Config) -> Dict[str, int]:
    """Parameter count by component, from the shapes the modules allocate (exact)."""
    d, h, dr, hd = cfg.d_model, cfg.ffn_hidden, cfg.d_readout, cfg.head_dim
    attn = 4 * d * d + 2 * hd                     # q,k,v,o + per-head q/k RMSNorm weights
    ffn = 3 * d * h
    block = attn + ffn + 2 * d                    # encoder / set block with two norms
    core_layer = 2 * attn + ffn + 3 * d + (d if cfg.use_sandwich_norm else 0)
    readout = (
        3 * d * dr + 3 * dr + 2                   # wq, wo, wn, their norms, log_scale, null_bias
        + d + 1                                   # score thresholds
        + d * (d // 2) + d // 2 + d // 2 + 1      # score location MLP
        + d * dr + dr + dr * (NUM_QUANTILES + 1) + NUM_QUANTILES + 1   # range MLP
        + d * (d // 4) + d // 4 + d // 4 + 1      # halting head
        + 3 * d * dr + dr                         # span projections and norm
    )
    parts = {
        "embedding": cfg.vocab_size * cfg.d_embed + cfg.d_embed * d + 3 * d + len(TYPE_INDEX) * d,
        "numeric": (4 * (d // 4) + (d // 4) * d) if cfg.use_numeric else 0,
        "state_encoder": cfg.n_state_layers * block + d,
        "core": cfg.n_core_layers * core_layer + cfg.n_scratch * d + 1,
        "pools": 2 * (d + 1),
        "rank_emb": cfg.max_levels * d,
        "set_layer": d + d + (cfg.n_set_layers * block if cfg.use_set_layer else 0),
        "readout": readout,
        "mask_memory": d,
    }
    parts["total"] = sum(parts.values())
    return parts
