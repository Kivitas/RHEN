import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from rhen.config import EOSConfig
    from rhen.eos_loop import FlashLoopUpdater, LoopCurriculum, StepSizeController
except ImportError:
    from config import EOSConfig
    from eos_loop import FlashLoopUpdater, LoopCurriculum, StepSizeController
from optim.grouped_gemm import GroupedGEMMWrapper
from optim.quantized_kv import QuantizedMultiLatentAttention
from optim.checkpoint import CheckpointedRecurrentCore
from optim.mup import apply_mup_init, create_mup_config
from optim.advanced_loops import (
    create_loop_bypass_router, create_cross_loop_attention, create_self_speculative_decoder
)
from optim.ring_mla import RingMLAAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) with YaRN NTK-by-parts context scaling and cached tables.
    Supports 8k target window and smooth extrapolation to 16k+ context without quality degradation.
    """
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 8192, 
        base: float = 500000.0,
        yarn_scale_factor: float = 4.0,
        yarn_original_max_position: int = 2048,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_attn_factor: float = 0.1,
        enable_dynamic_ntk: bool = False,
        dynamic_max_context: int = 16384,
        dynamic_snap_factor: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.yarn_scale_factor = yarn_scale_factor
        self.yarn_original_max_position = yarn_original_max_position
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_attn_factor = yarn_attn_factor

        # Dynamic Runtime NTK YaRN. Off by default, so the legacy fixed-scale
        # path stays bit-identical unless explicitly enabled.
        self.enable_dynamic_ntk = bool(enable_dynamic_ntk)
        self.dynamic_max_context = int(dynamic_max_context)
        self.dynamic_snap_factor = float(dynamic_snap_factor)
        self._cache_scale = 1.0

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        # Unscaled frequencies, retained so dynamic NTK can interpolate from the
        # pristine table rather than from the already-YaRN-warped one.
        self.base_inv_freq = inv_freq.clone()
        
        if self.yarn_scale_factor > 1.0:
            # YaRN NTK-by-parts interpolation: split frequencies into 3 bands
            # 1. High-freq (short wavelength): no interpolation (preserves crisp local syntax)
            # 2. Low-freq (long wavelength): full scale interpolation
            # 3. Mid-freq: smooth ramp transition
            low_freq_wavelen = self.yarn_original_max_position / self.yarn_beta_slow
            high_freq_wavelen = self.yarn_original_max_position / self.yarn_beta_fast
            wavelen = 2.0 * math.pi / inv_freq
            
            smooth = (wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen + 1e-6)
            smooth = torch.clamp(smooth, 0.0, 1.0)
            
            inv_freq_interp = inv_freq / self.yarn_scale_factor
            inv_freq = (1.0 - smooth) * inv_freq + smooth * inv_freq_interp

            # Attention temperature scaling factor
            self.mscale = float(self.yarn_attn_factor * math.log(self.yarn_scale_factor) + 1.0)
        else:
            self.mscale = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos() * self.mscale
        self.sin_cached = emb.sin() * self.mscale

    # -- Dynamic Runtime NTK YaRN ------------------------------------------------

    def dynamic_ntk_scale(self, runtime_len: int) -> float:
        """RoPE scale factor implied by the *actual* runtime sequence length.

        YaRN's fixed ``yarn_scale_factor`` is tuned for one nominal window. When
        the runtime length is much shorter, that fixed scale over-stretches the
        high-frequency band and needlessly blurs local syntax; when it is longer,
        the model has no headroom left. Dynamic NTK scaling instead derives the
        scale from the observed length:

            s(runtime) = max(1, runtime / original_max_position)

        The result is **quantised** to ``snap_factor`` steps. Without snapping, a
        runtime length that jitters by one token (batching, streaming) would
        rebuild the whole frequency table and perturb the rotary phase slightly
        each time, so cached positions would not be comparable across calls.
        Snapping makes the scale piecewise-constant, which keeps the cache valid.

        Args:
            runtime_len: Observed sequence length for this forward pass.

        Returns:
            The scale to apply, ``>= 1.0``.
        """
        if runtime_len <= 0:
            return 1.0
        raw = max(1.0, float(runtime_len) / float(max(self.yarn_original_max_position, 1)))
        snap = float(self.dynamic_snap_factor)
        if snap > 0.0:
            raw = math.ceil(raw / snap) * snap
        return max(1.0, raw)

    def _ntk_interpolated_inv_freq(self, scale: float) -> torch.Tensor:
        """NTK-aware inverse frequencies for a runtime scale ``scale``.

        NTK-by-parts base scaling adjusts ``theta`` itself rather than dividing
        every frequency uniformly:

            theta' = theta * scale^(dim / (dim - 2))
            inv_freq' = 1 / (theta'^(2i / dim))

        Because low frequencies are scaled more aggressively than high ones, the
        long-range bands gain the resolution they need while local syntax keeps
        its crispness.
        """
        if scale <= 1.0 + 1e-9:
            return self.base_inv_freq
        dim = self.dim
        theta = self.base * (scale ** (dim / max(dim - 2, 1)))
        return 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    def _build_dynamic_cache(
        self,
        seq_len: int,
        device: torch.device,
        scale: float,
    ) -> None:
        inv_freq = self._ntk_interpolated_inv_freq(scale)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        mscale = float(self.yarn_attn_factor * math.log(max(scale, 1.0)) + 1.0)
        self.cos_cached = emb.cos() * mscale
        self.sin_cached = emb.sin() * mscale
        self._cache_scale = scale

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_dynamic_ntk:
            scale = self.dynamic_ntk_scale(seq_len)
            needs_rebuild = (
                self.cos_cached.numel() == 0
                or self.cos_cached.shape[0] < seq_len
                or self.cos_cached.device != x.device
                or abs(scale - self._cache_scale) > 1e-6
            )
            if needs_rebuild:
                self._build_dynamic_cache(
                    max(seq_len, min(self.max_position_embeddings, self.dynamic_max_context)),
                    x.device,
                    scale,
                )
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

        if (
            self.cos_cached.numel() == 0
            or self.cos_cached.shape[0] < seq_len
            or self.cos_cached.device != x.device
        ):
            self._build_cache(max(seq_len, self.max_position_embeddings), x.device)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape [B, H, S, D] by position tables ``cos``/``sin`` of shape [S, D]."""
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class DynamicStepAdapter(nn.Module):
    """
    Dynamic Low-Rank Step Adapter with Learnable Residual Gating.
    Morphs weights dynamically for each loop step k: Delta_W_k = tanh(gate_k) * (A_k * B_k)
    Allows shared layers to specialize per loop step without residual scale drift.
    """
    def __init__(self, hidden_size: int, max_loops: int, r: int = 64):
        super().__init__()
        self.max_loops = max_loops
        self.r = r
        self.lora_A = nn.Embedding(max_loops + 1, hidden_size * r)
        self.lora_B = nn.Embedding(max_loops + 1, r * hidden_size)
        self.gate = nn.Embedding(max_loops + 1, 1)

    def reset_parameters(self):
        nn.init.zeros_(self.lora_A.weight)
        nn.init.normal_(self.lora_B.weight, std=0.01)
        nn.init.zeros_(self.gate.weight)

    def forward(self, hidden_states: torch.Tensor, step_idx) -> torch.Tensor:
        if isinstance(step_idx, int):
            step_tensor = torch.tensor([step_idx], device=hidden_states.device, dtype=torch.long)
        else:
            step_tensor = step_idx

        A = self.lora_A(step_tensor).view(-1, hidden_states.shape[-1], self.r)
        B = self.lora_B(step_tensor).view(-1, self.r, hidden_states.shape[-1])
        gamma = torch.tanh(self.gate(step_tensor)).view(-1, 1, 1)

        delta = torch.matmul(torch.matmul(hidden_states, A), B)
        return hidden_states + gamma * delta


class StepModulation(nn.Module):
    """FiLM Step Modulation combined with Dynamic Step Adapter."""
    def __init__(self, hidden_size: int, max_loops: int):
        super().__init__()
        self.step_embedding = nn.Embedding(max_loops + 1, hidden_size * 2)
        self.adapter = DynamicStepAdapter(hidden_size, max_loops)

    def reset_parameters(self):
        nn.init.zeros_(self.step_embedding.weight)
        self.adapter.reset_parameters()

    def forward(self, hidden_states: torch.Tensor, step_idx) -> torch.Tensor:
        if isinstance(step_idx, int):
            step_tensor = torch.tensor([step_idx], device=hidden_states.device, dtype=torch.long)
        else:
            step_tensor = step_idx

        emb = self.step_embedding(step_tensor)
        scale, shift = emb.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        hidden_states = hidden_states * (1.0 + scale) + shift
        return self.adapter(hidden_states, step_idx)


class StateSpaceMemoryHighway(nn.Module):
    """
    1D Continuous State-Space Memory channel bridging recurrent loop iterations.
    Maintains a persistent memory state m_l across loop iterations l=1..L,
    allowing deep reasoning loops to recall earlier loop representations without gradient vanishing.
    """
    def __init__(self, hidden_size: int, state_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        
        self.in_proj = nn.Linear(hidden_size, state_dim, bias=False)
        self.a_gate = nn.Linear(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size)
        
        self.register_buffer("memory_state", None, persistent=False)

    def reset_state(self):
        self.memory_state = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_idx: int,
        prefix_state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Inject the loop memory into ``hidden_states``.

        ``prefix_state`` is used by cached decoding: it holds the running float32 sum and
        count of every earlier position seen at this loop step, so the causal mean of a new
        token equals the one the full-sequence forward would compute. It is updated in place.
        """
        _, seq_len, _ = hidden_states.shape
        # Causal pooling: position t summarises tokens 0..t only (running mean), so
        # the memory never reads future positions.
        prev_count = 0
        prev_sum = None
        if prefix_state is not None:
            prev_count = int(prefix_state.get("count", 0))
            prev_sum = prefix_state.get("sum")
        counts = torch.arange(
            prev_count + 1, prev_count + seq_len + 1, device=hidden_states.device, dtype=torch.float32
        ).view(1, seq_len, 1)
        h_sum = hidden_states.float().cumsum(dim=1)
        if prev_sum is not None:
            h_sum = h_sum + prev_sum
        if prefix_state is not None:
            prefix_state["sum"] = h_sum[:, -1:, :].detach()
            prefix_state["count"] = prev_count + seq_len
        h_cum = (h_sum / counts).to(hidden_states.dtype)  # [B, S, H]
        x_m = self.in_proj(h_cum)                                                        # [B, S, state_dim]

        # One memory vector per position, carried across loop iterations.
        prev = self.memory_state
        if prev is None or prev.shape != x_m.shape or step_idx <= 1:
            mem = torch.tanh(x_m)
        else:
            alpha = torch.sigmoid(self.a_gate(prev))
            mem = alpha * prev + (1.0 - alpha) * torch.tanh(x_m)
        # Stored detached so the memory does not extend the autograd graph across loops.
        self.memory_state = mem.detach()

        m_inject = self.out_proj(mem)  # [B, S, H]
        return hidden_states + 0.1 * self.norm(m_inject)


class TitansNeuralMemory(nn.Module):
    """
    Google Titans Neural Long-Term Memory (MAG: Memory As Gating).
    Maintains associative memory matrix M in R^(d_mem x d_mem) updated via surprise/momentum Hebbian updates:
        S_t = k_t v_t^T - M_{t-1} k_t k_t^T
        M_t = (1 - alpha_t) M_{t-1} + eta_t S_t
    where alpha_t = sigmoid(W_alpha h_t) and eta_t = sigmoid(W_eta h_t).
    Outputs y_t = M_t q_t and returns memory-gated state:
        g_t = sigmoid(W_g h_t)
        h_out = g_t * h_t + (1 - g_t) * RMSNorm(W_out y_t)
    """
    def __init__(self, hidden_size: int, d_mem: int = 64, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_mem = d_mem
        self.k_proj = nn.Linear(hidden_size, d_mem, bias=False)
        self.v_proj = nn.Linear(hidden_size, d_mem, bias=False)
        self.q_proj = nn.Linear(hidden_size, d_mem, bias=False)
        self.w_alpha = nn.Linear(hidden_size, d_mem, bias=True)
        self.w_eta = nn.Linear(hidden_size, d_mem, bias=True)
        self.w_g = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(d_mem, hidden_size, bias=False)
        self.out_norm = RMSNorm(hidden_size, eps=eps)
        
        self.register_buffer("memory_matrix", None, persistent=False)

    def reset_state(self):
        self.memory_matrix = None

    def forward(self, hidden_states: torch.Tensor, step_idx: Optional[int] = None) -> torch.Tensor:
        B, S, H = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        if self.memory_matrix is None or self.memory_matrix.shape[0] != B:
            M = torch.zeros(B, self.d_mem, self.d_mem, device=device, dtype=dtype)
        else:
            M = self.memory_matrix.to(device=device, dtype=dtype)

        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = self.q_proj(hidden_states)
        alpha = torch.sigmoid(self.w_alpha(hidden_states))
        eta = torch.sigmoid(self.w_eta(hidden_states))
        g = torch.sigmoid(self.w_g(hidden_states))

        y_list = []
        for t in range(S):
            kt = k[:, t, :].unsqueeze(-1)       # [B, d_mem, 1]
            vt = v[:, t, :].unsqueeze(1)        # [B, 1, d_mem]
            qt = q[:, t, :].unsqueeze(-1)       # [B, d_mem, 1]
            at = alpha[:, t, :].unsqueeze(-1)   # [B, d_mem, 1]
            et = eta[:, t, :].unsqueeze(-1)      # [B, d_mem, 1]

            Mk = torch.matmul(M, kt)             # [B, d_mem, 1]
            St = torch.matmul(kt, vt) - torch.matmul(Mk, kt.transpose(-1, -2))  # [B, d_mem, d_mem]

            M = (1.0 - at) * M + et * St
            yt = torch.matmul(M, qt).squeeze(-1) # [B, d_mem]
            y_list.append(yt)

        self.memory_matrix = M.detach()
        y = torch.stack(y_list, dim=1)           # [B, S, d_mem]
        y_proj = self.out_norm(self.out_proj(y)) # [B, S, H]
        h_out = g * hidden_states + (1.0 - g) * y_proj
        return h_out


class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention (MLA) with fused SDPA Flash kernels, learnable Differential Attention,
    DeepSeek-V3 Decoupled MLA & Matrix Absorption Cache, Qwen 2.5 Dual QK-Norm & QKV Bias,
    and Gemma 2 Interleaved SWA & Soft-Capping.
    """
    def __init__(self, config: EOSConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank

        use_qkv_bias = getattr(config, "use_qkv_bias", True)
        use_qk_norm = getattr(config, "use_qk_norm", True)

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=use_qkv_bias)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=use_qkv_bias)

        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.head_dim, bias=use_qkv_bias)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.head_dim + self.head_dim), bias=use_qkv_bias)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.o_proj._residual_out = True

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if use_qk_norm else None

        # DIFF V2 per-token per-head dynamic lambda projection
        self.lambda_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # Learnable per-head Differential Attention lambda fallback
        if getattr(config, "diff_attn_learnable_lambda", True):
            init_lambda = float(getattr(config, "diff_mla_lambda_init", 0.8))
            self.lambda_param = nn.Parameter(torch.full((1, self.num_heads, 1, 1), init_lambda))
        else:
            self.lambda_param = None

        # DIFF sub-maps are built from rotation-closed sets of RoPE pairs (index i together with
        # i + head_dim/2). Each sub-map then stays a function of relative position, which a
        # contiguous half split would break by cutting pairs across the two halves.
        half = self.head_dim // 2
        pairs1 = half // 2
        if self.config.use_differential_attention:
            if pairs1 == 0 or half - pairs1 == 0:
                raise ValueError("DIFF attention needs head_dim >= 4 to form two RoPE-closed sub-maps")
            idx1 = list(range(pairs1)) + list(range(half, half + pairs1))
            idx2 = list(range(pairs1, half)) + list(range(half + pairs1, 2 * half))
            self.register_buffer("_diff_idx1", torch.tensor(idx1, dtype=torch.long), persistent=False)
            self.register_buffer("_diff_idx2", torch.tensor(idx2, dtype=torch.long), persistent=False)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        lambda_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_dtype = q.dtype
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)

        softcap = float(self.config.attn_logit_softcapping)
        # The config resolves use_differential_attention (an explicit False wins over use_diff_mla).
        use_diff = bool(self.config.use_differential_attention)

        if use_diff:
            # Differential Latent Attention (DIFF V2)
            if self._diff_idx1 is not None:
                q1, q2 = q.index_select(-1, self._diff_idx1), q.index_select(-1, self._diff_idx2)
                k1, k2 = k.index_select(-1, self._diff_idx1), k.index_select(-1, self._diff_idx2)

                if lambda_t is None:
                    if self.lambda_param is not None:
                        lambda_t = 2.0 * torch.sigmoid(self.lambda_param).to(q.dtype)
                    else:
                        lambda_t = float(getattr(self.config, "diff_mla_lambda_init", 0.8))
                elif self.lambda_param is not None:
                    lambda_t = lambda_t + 0.0 * self.lambda_param.to(lambda_t.dtype)

                # DIFF V2 FlashAttention / SDPA Kernel Path. is_causal is only correct
                # for square, unmasked attention; cached or custom-masked calls take the
                # manual path below, which applies the offset causal mask.
                can_sdpa = (
                    self.config.use_sdpa
                    and softcap <= 0.0
                    and attention_mask is None
                    and q.shape[2] == k.shape[2]
                )
                if can_sdpa:
                    out1 = F.scaled_dot_product_attention(
                        q1, k1, v, attn_mask=None, dropout_p=0.0, is_causal=True
                    )
                    out2 = F.scaled_dot_product_attention(
                        q2, k2, v, attn_mask=None, dropout_p=0.0, is_causal=True
                    )
                    return out1 - lambda_t * out2

                # Manual matmul fallback path (when softcapping > 0 or custom masks used)
                scores1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(q1.shape[-1])
                scores2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(q2.shape[-1])
                if softcap > 0.0:
                    scores1 = softcap * torch.tanh(scores1 / softcap)
                    scores2 = softcap * torch.tanh(scores2 / softcap)
                seq_len = q.shape[2]
                kv_len = k.shape[2]
                if attention_mask is None:
                    causal = torch.full((seq_len, kv_len), float("-inf"), device=q.device)
                    causal = torch.triu(causal, diagonal=kv_len - seq_len + 1)
                    mask = causal.unsqueeze(0).unsqueeze(0)
                else:
                    mask = attention_mask
                scores1 = scores1 + mask
                scores2 = scores2 + mask
                weights1 = F.softmax(scores1, dim=-1, dtype=torch.float32)
                weights2 = F.softmax(scores2, dim=-1, dtype=torch.float32)

                diff_weights = (weights1 - lambda_t * weights2).to(q.dtype)
                return torch.matmul(diff_weights, v)

        # Fast FlashAttention / SDPA path
        can_sdpa = (
            self.config.use_sdpa
            and attention_mask is None
            and q.shape[2] == k.shape[2]
            and softcap <= 0.0
        )
        if can_sdpa:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if softcap > 0.0:
            scores = softcap * torch.tanh(scores / softcap)

        seq_len = q.shape[2]
        kv_len = k.shape[2]
        if attention_mask is None:
            causal = torch.full((seq_len, kv_len), float("-inf"), device=q.device)
            causal = torch.triu(causal, diagonal=kv_len - seq_len + 1)
            attention_mask = causal.unsqueeze(0).unsqueeze(0)
        scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_modulation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # DIFF V2 per-token per-head dynamic lambda
        if getattr(self.config, "diff_v2_head_lambda", True):
            sig = torch.sigmoid(self.lambda_proj(hidden_states))  # [B, S, num_heads]
            if sig.dim() == 3:
                lambda_t = sig.transpose(1, 2).unsqueeze(-1)  # [B, num_heads, S, 1]
            else:
                lambda_t = sig.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, num_heads, 1, 1]
        else:
            lambda_t = None

        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_latent).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rope = torch.split(kv_compressed, [self.kv_lora_rank, self.head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)

        # DeepSeek-V3 Decoupled MLA & Matrix Absorption Cache:
        # During generation, store ONLY (kv_latent, k_rope) of shape [B, S_seq, kv_lora_rank] and [B, S_seq, head_dim]
        if kv_cache is not None:
            prev_1, prev_2 = kv_cache
            if prev_1 is not None and prev_1.shape[-1] == self.kv_lora_rank:
                cached_kv_latent = torch.cat([prev_1, kv_latent], dim=1)
                cached_k_rope = torch.cat([prev_2, k_rope], dim=1)
            else:
                cached_kv_latent = kv_latent
                cached_k_rope = k_rope
        else:
            cached_kv_latent = kv_latent
            cached_k_rope = k_rope

        new_kv_cache = (cached_kv_latent, cached_k_rope) if kv_cache is not None else None

        kv_expanded = self.kv_b_proj(cached_kv_latent).view(
            batch_size, -1, self.num_heads, self.head_dim * 2
        )
        k_unrope, v = torch.chunk(kv_expanded, 2, dim=-1)

        k_unrope = k_unrope.transpose(1, 2)
        v = v.transpose(1, 2)
        k_rope_exp = cached_k_rope.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k_unrope = self.k_norm(k_unrope)

        # RoPE is applied to the full key (content + rope parts) and to the query with
        # the same cos/sin, so q.k depends only on relative position. cos/sin are [S, D]
        # tables (a leading singleton batch dim is accepted) and must cover every key
        # position; the queries are the trailing seq_len of them.
        if cos.dim() == 3 and cos.shape[0] == 1:
            cos, sin = cos[0], sin[0]
        if cos.dim() != 2 or sin.dim() != 2:
            raise ValueError(f"cos/sin must be [S, D] or [1, S, D], got {tuple(cos.shape)}")
        k = k_unrope + k_rope_exp
        k_len = k.shape[2]
        if cos.shape[0] < k_len:
            raise ValueError(f"cos/sin must cover all {k_len} key positions, got {cos.shape[0]}")
        cos_k, sin_k = cos[:k_len], sin[:k_len]
        q = _rope_rotate(q, cos_k[k_len - seq_len:], sin_k[k_len - seq_len:])
        k = _rope_rotate(k, cos_k, sin_k)

        # Apply bidirectional Rhen KV modulation if active
        if kv_modulation is not None:
            if kv_modulation.shape[-1] == self.num_heads * self.head_dim:
                kv_mod = kv_modulation.view(kv_modulation.size(0), self.num_heads, 1, self.head_dim)
            else:
                kv_mod = kv_modulation.unsqueeze(1)
            k = k * kv_mod
            v = v * kv_mod

        # Gemma 2 Interleaved SWA & Soft-Capping
        if getattr(self.config, "interleave_swa", True) and (self.layer_idx % 2 == 1):
            window_size = getattr(self.config, "swa_window_size", 512)
            device = hidden_states.device
            r_idx = torch.arange(seq_len, device=device).unsqueeze(1)
            c_idx = torch.arange(k.shape[2], device=device).unsqueeze(0)
            offset = k.shape[2] - seq_len
            causal_b = (c_idx <= r_idx + offset)
            swa_b = causal_b & (c_idx > r_idx + offset - window_size)

            swa_mask = torch.full((seq_len, k.shape[2]), float("-inf"), device=device, dtype=q.dtype)
            swa_mask = swa_mask.masked_fill(swa_b, 0.0).unsqueeze(0).unsqueeze(0)

            if attention_mask is None:
                attention_mask = swa_mask
            else:
                attention_mask = attention_mask + swa_mask

        output = self._attention(q, k, v, attention_mask, lambda_t=lambda_t)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(output), new_kv_cache


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, config: EOSConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.down_proj._residual_out = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoESwiGLUFFN(nn.Module):
    """Sparse Mixture of Experts (MoE) FFN with Top-k Routing & vectorized Load Balancing Loss."""
    def __init__(self, config: EOSConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = min(config.num_experts_per_tok, config.num_experts)

        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFFN(config) for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        router_logits = self.router(x_flat)
        rz_coef = float(self.config.router_z_loss_coef)
        router_z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean() * rz_coef

        router_scores = torch.sigmoid(router_logits)
        weights, selected_experts = torch.topk(router_scores, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        assign_counts = torch.bincount(
            selected_experts.reshape(-1), minlength=self.num_experts
        ).to(router_scores.dtype)
        expert_freq = assign_counts / selected_experts.numel()
        mean_prob = router_scores.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(expert_freq * mean_prob) + router_z_loss

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_idx, top_k_idx = torch.where(selected_experts == i)
            if token_idx.numel() > 0:
                expert_input = x_flat[token_idx]
                expert_out = expert(expert_input)
                expert_weights = weights[token_idx, top_k_idx].unsqueeze(-1)
                output.index_add_(0, token_idx, expert_out * expert_weights)

        return output.view(batch_size, seq_len, hidden_size), aux_loss


def _in_autograd_backward() -> bool:
    """True while autograd runs a backward pass (e.g. a checkpoint recomputation)."""
    graph_task_id = getattr(torch._C, "_current_graph_task_id", None)
    if graph_task_id is None:
        return False
    try:
        return graph_task_id() != -1
    except RuntimeError:
        return False


MOE_BALANCE_MODES = ("forward", "step")


class SharedExpertFineGrainedMoE(nn.Module):
    """
    Isolated Shared-Expert Fine-Grained Mixture of Experts (SE-MoE).

    Aux-loss-free balancing has two update modes (``balance_update_mode``):

    - ``"forward"`` (default, the historical behaviour): the congestion bias moves inside
      every training forward call. This is not safe under gradient checkpointing, because the
      recomputed forward sees a bias that has already moved and may route differently.
    - ``"step"``: forward calls only record the routed load (skipped while autograd recomputes
      a checkpointed forward), and ``apply_pending_balance_update`` moves the bias once, from
      the mean load, after the optimizer step. The bias is then constant between a forward and
      its recomputation, and data-parallel ranks can average their loads first.
    """
    def __init__(self, config: EOSConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.top_k = min(config.num_active_routed, self.num_routed_experts)
        self.use_aux_free = config.use_aux_free_balancing
        self.balance_rate = float(config.aux_free_balance_rate)

        self.shared_experts = nn.ModuleList([
            SwiGLUFFN(config) for _ in range(self.num_shared_experts)
        ])

        self.routed_experts = nn.ModuleList([
            SwiGLUFFN(config) for _ in range(self.num_routed_experts)
        ])

        self.router = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)

        self.register_buffer(
            "congestion_bias",
            torch.zeros(self.num_routed_experts, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "load_distribution",
            torch.full((self.num_routed_experts,), 1.0 / self.num_routed_experts, dtype=torch.float32),
            persistent=False,
        )
        self.balance_update_mode = "forward"
        self.register_buffer(
            "pending_load_sum",
            torch.zeros(self.num_routed_experts, dtype=torch.float32),
            persistent=False,
        )
        self._pending_calls = 0

    def set_balance_update_mode(self, mode: str) -> None:
        if mode not in MOE_BALANCE_MODES:
            raise ValueError(f"balance_update_mode must be one of {MOE_BALANCE_MODES}, got {mode!r}")
        self.balance_update_mode = mode
        self.pending_load_sum.zero_()
        self._pending_calls = 0

    @property
    def pending_calls(self) -> int:
        return self._pending_calls

    @torch.no_grad()
    def apply_pending_balance_update(self, mean_load: Optional[torch.Tensor] = None) -> bool:
        """Move the congestion bias once from the mean routed load since the last update.

        ``mean_load`` overrides the locally accumulated mean (a data-parallel trainer passes
        the all-reduced mean so every rank applies the same update). Returns whether an update
        was applied.
        """
        if not self.use_aux_free:
            self.pending_load_sum.zero_()
            self._pending_calls = 0
            return False
        if mean_load is None:
            if self._pending_calls == 0:
                return False
            mean_load = self.pending_load_sum / self._pending_calls
        target_load = 1.0 / self.num_routed_experts
        delta = (target_load - mean_load.to(self.congestion_bias.device, torch.float32))
        self.congestion_bias.add_(self.balance_rate * delta)
        self.pending_load_sum.zero_()
        self._pending_calls = 0
        return True

    def get_load_distribution(self) -> torch.Tensor:
        """Returns normalized load across routed experts."""
        return self.load_distribution.clone()

    def reset_congestion_bias(self):
        """Zeroes out self.congestion_bias."""
        self.congestion_bias.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        num_tokens = x_flat.shape[0]

        # Shared Expert computation
        shared_output = torch.zeros_like(x_flat)
        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(x_flat)

        # Router scores
        router_logits = self.router(x_flat)
        rz_coef = float(self.config.router_z_loss_coef)
        router_z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean() * rz_coef

        raw_scores = torch.sigmoid(router_logits)

        if self.use_aux_free:
            biased_scores = raw_scores + self.congestion_bias.to(raw_scores.device)
            _, selected_experts = torch.topk(biased_scores, self.top_k, dim=-1)
        else:
            _, selected_experts = torch.topk(raw_scores, self.top_k, dim=-1)

        unbiased_weights = torch.gather(raw_scores, dim=-1, index=selected_experts)
        unbiased_weights = unbiased_weights / (unbiased_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if num_tokens > 0:
            with torch.no_grad():
                token_counts = torch.bincount(
                    selected_experts.reshape(-1), minlength=self.num_routed_experts
                ).float()
                actual_load = token_counts / max(1, token_counts.sum().item())
                self.load_distribution.copy_(actual_load.to(self.load_distribution.device))
                if self.training and self.use_aux_free:
                    if self.balance_update_mode == "step":
                        if not _in_autograd_backward():
                            self.pending_load_sum.add_(actual_load.to(self.pending_load_sum.device))
                            self._pending_calls += 1
                    else:
                        target_load = 1.0 / self.num_routed_experts
                        congestion_delta = (target_load - actual_load).to(self.congestion_bias.device)
                        self.congestion_bias.add_(self.balance_rate * congestion_delta)

        routed_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            token_idx, top_k_idx = torch.where(selected_experts == i)
            if token_idx.numel() > 0:
                expert_input = x_flat[token_idx]
                expert_out = expert(expert_input)
                expert_weights = unbiased_weights[token_idx, top_k_idx].unsqueeze(-1)
                routed_output.index_add_(0, token_idx, expert_out * expert_weights)

        total_output = shared_output + routed_output
        return total_output.view(batch_size, seq_len, hidden_size), router_z_loss


class EntropyGatedACTHalting(nn.Module):
    """Entropy-Gated Adaptive Computation Time (ACT) Halting Unit."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def reset_parameters(self):
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_prob = torch.sigmoid(self.gate(hidden_states)).squeeze(-1)

        if velocity is not None:
            norm_v = velocity / math.sqrt(hidden_states.shape[-1])
            decay_boost = torch.exp(-2.0 * norm_v)
            base_prob = torch.clamp(base_prob + 0.25 * decay_boost, 0.0, 1.0)

        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            confidence_boost = torch.exp(-0.5 * entropy)
            base_prob = torch.clamp(base_prob + 0.2 * confidence_boost, 0.0, 1.0)

        return base_prob


class AdaptiveLoopRouter(nn.Module):
    """Token-Level Adaptive Recurrent Exit with DeepSeek-R1 Entropy-Guided Loop Pondering.

    Monitors online token predictive entropy H_l = -sum p_l log p_l and Helmholtz free energy
    F_l = -tau log sum e^(s_i / tau) per loop.
    If relative entropy reduction Delta H_l = H_{l-1} - H_l < eps and confidence p_max > theta,
    token/decision exits early; otherwise allocate up to L_max = 16 loops for hard tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        min_loops: int = 2,
        freeze_threshold: float = 0.9,
        eps: float = 0.01,
        theta: float = 0.8,
        tau: float = 1.0,
        max_loops: int = 16,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.min_loops = max(0, int(min_loops))
        self.freeze_threshold = float(freeze_threshold)
        self.eps = float(eps)
        self.theta = float(theta)
        self.tau = float(tau)
        self.max_loops = int(max_loops)

        self.halt_proj = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.halt_proj.bias, -1.0)

        # Steepness of the straight-through softening
        self._soft_scale = 10.0

        # Per-forward-pass state.
        self.frozen: Optional[torch.Tensor] = None
        self.active_loops = 0
        self.total_loops = 0
        self.prev_entropy: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear per-sequence freezing state. Call at the start of a forward pass."""
        self.frozen = None
        self.active_loops = 0
        self.total_loops = 0
        self.prev_entropy = None

    def halting_probs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``h_halt,l(t) = sigma(W_halt h_l(t) + b)``, shape ``[B, S]``."""
        return torch.sigmoid(self.halt_proj(hidden_states)).squeeze(-1)

    def compute_entropy_and_free_energy(
        self,
        logits_or_probs: torch.Tensor,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monitors online token predictive entropy H_l = -sum p_l log p_l
        and Helmholtz free energy F_l = -tau log sum e^(s_i / tau).
        """
        tau_val = tau if tau is not None else self.tau
        if logits_or_probs.ndim == 3:
            scaled = logits_or_probs / max(tau_val, 1e-6)
            probs = F.softmax(scaled, dim=-1)
            free_energy = -tau_val * torch.logsumexp(scaled, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            p_max, _ = torch.max(probs, dim=-1)
        else:
            p = logits_or_probs.clamp(1e-6, 1.0 - 1e-6)
            probs = torch.stack([p, 1.0 - p], dim=-1)
            free_energy = -tau_val * torch.log(torch.exp(p / tau_val) + torch.exp((1.0 - p) / tau_val) + 1e-9)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            p_max, _ = torch.max(probs, dim=-1)

        return entropy, free_energy, p_max

    def eval_ponder_exit(
        self,
        hidden_states_or_logits: torch.Tensor,
        loop_idx: int,
    ) -> torch.Tensor:
        """
        DeepSeek-R1 Entropy-Guided Loop Pondering early exit evaluation.
        If Delta H_l = H_{l-1} - H_l < eps and confidence p_max > theta, token/decision exits early.
        """
        if hidden_states_or_logits.shape[-1] == self.hidden_size:
            p = self.halting_probs(hidden_states_or_logits)
            entropy, free_energy, p_max = self.compute_entropy_and_free_energy(p)
        else:
            entropy, free_energy, p_max = self.compute_entropy_and_free_energy(hidden_states_or_logits)

        if self.prev_entropy is None or loop_idx <= self.min_loops:
            self.prev_entropy = entropy
            return torch.zeros_like(p_max, dtype=torch.bool)

        delta_h = self.prev_entropy - entropy
        self.prev_entropy = entropy
        exit_mask = (delta_h < self.eps) & (p_max > self.theta)
        return exit_mask

    def step(
        self,
        hidden_states: torch.Tensor,
        next_states: torch.Tensor,
        loop_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.halting_probs(next_states)
        ponder_exit = self.eval_ponder_exit(next_states, loop_idx)

        if loop_idx <= self.min_loops:
            # Warm-up: nothing may freeze before min_loops.
            active = torch.ones_like(p, dtype=torch.bool)
        else:
            active = (p < self.freeze_threshold) & (~ponder_exit)

        # Tokens frozen in an earlier loop stay frozen.
        if self.frozen is not None:
            active = active & (~self.frozen)

        # Straight-through gate. The forward value is the exact 0/1 active mask: active
        # tokens take the full update, frozen tokens keep their state. The backward pass
        # flows through a soft sigmoid that is large for tokens below the threshold.
        soft_mask = torch.sigmoid((self.freeze_threshold - p) * self._soft_scale)
        update = next_states - hidden_states
        hard_mask = active.to(next_states.dtype)
        gate = hard_mask + soft_mask - soft_mask.detach()
        updated = hidden_states + gate.unsqueeze(-1) * update

        self.frozen = active.logical_not() if self.frozen is None else (self.frozen | ~active)
        self.active_loops += int(active.sum().item())
        self.total_loops += active.numel()
        return updated, active

    def savings(self) -> Dict[str, Any]:
        """Halting-mask statistics for the last forward pass.

        Returns the fraction of (token, loop) update slots that the mask excluded
        ("saved_fraction") plus the absolute counts. The update is applied densely
        with a mask, so these are mask statistics, not measured compute savings.
        """
        if self.total_loops == 0:
            return {
                "computed_fraction": 1.0,
                "saved_fraction": 0.0,
                "update_slots": 0,
                "skipped_slots": 0,
            }
        saved = self.total_loops - self.active_loops
        return {
            "computed_fraction": self.active_loops / self.total_loops,
            "saved_fraction": saved / self.total_loops,
            "update_slots": self.total_loops,
            "skipped_slots": saved,
        }


class CLAMBridge(nn.Module):
    """Cross-Loop Attentive Memory Highway (CLAM Bridge).

    In a parameter-shared recurrent trunk, loop ``l`` re-derives its
    representation from scratch. After ~50 effective layer applications this
    drives **representation collapse**: the residual stream converges and
    additional depth stops adding information.

    The bridge lets loop ``l`` attend to *compressed* keys and values captured at
    loop ``l-1``:

        k_{l-1}, v_{l-1} = compress_k(h_{l-1}), compress_v(h_{l-1})
        a_l              = softmax( q_l k_{l-1}^T / sqrt(d) ) v_{l-1}
        h_l             <- h_l + lambda_clam * a_l

    ``lambda_clam`` is a learnable scalar gate squashed through a sigmoid and
    initialised closed (``clam_init_gate``, default 0), so a freshly-initialised
    model reproduces the legacy update exactly and the bridge only contributes
    once trained. Keeping the gate in ``[0, 1]`` also means the memory can never
    dominate the current computation.

    Compression (projection down to ``memory_dim``) keeps the stored memory low-rank
    rather than a full KV cache. The attention itself scores an ``S x S_prev`` matrix
    per head, so the cost is **quadratic in sequence length** per loop.

    Args:
        hidden_size: Model hidden width.
        memory_dim: Width of the compressed key/value summary.
        num_heads: Attention heads used for the cross-loop lookup.
        init_gate: Pre-sigmoid initial value of ``lambda_clam``.
    """

    #: Pre-sigmoid bias yielding an effectively closed gate. ``sigmoid(0) = 0.5``,
    #: so a literal 0.0 would leave the bridge half-open on a fresh model; this
    #: maps to ~4.5e-5, i.e. numerically closed while staying finite and
    #: learnable (an exact 0 would be a permanently dead gradient).
    _CLOSED_LOGIT: float = -10.0

    def __init__(
        self,
        hidden_size: int,
        memory_dim: int = 64,
        num_heads: int = 4,
        init_gate: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.memory_dim = max(1, int(memory_dim))
        self.num_heads = max(1, int(num_heads))
        self.head_dim = max(1, self.memory_dim // self.num_heads)

        self.query = nn.Linear(hidden_size, self.memory_dim, bias=False)
        self.compress_k = nn.Linear(hidden_size, self.memory_dim, bias=False)
        self.compress_v = nn.Linear(hidden_size, self.memory_dim, bias=False)
        self.out_proj = nn.Linear(self.memory_dim, hidden_size, bias=False)

        # lambda_clam, squashed. Initialised closed so enabling the bridge is a
        # strict no-op until the model learns to rely on it.
        # ``sigmoid(0) = 0.5``, so a literal ``init_gate=0.0`` would leave the
        # bridge half-open on a fresh model. This maps to ~4.5e-5 -- numerically
        # closed, while staying finite so the gate keeps a live gradient (an
        # exact 0.0 would be a permanently dead parameter).
        closed_logit = math.log(float(init_gate) / (1.0 - float(init_gate))) if 0.0 < float(init_gate) < 1.0 else self._CLOSED_LOGIT
        self.gate_logit = nn.Parameter(torch.tensor(float(closed_logit)))
        self._memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.last_gate: float = 0.0

    @property
    def gate(self) -> torch.Tensor:
        """The squashed learnable gate ``lambda_clam`` in ``[0, 1]``."""
        return torch.sigmoid(self.gate_logit)

    def reset(self) -> None:
        """Drop the stored previous-loop memory."""
        self._memory = None

    def store(self, hidden_states: torch.Tensor) -> None:
        """Compress and store ``h_l`` so loop ``l+1`` can attend to it."""
        self._memory = (
            self.compress_k(hidden_states),
            self.compress_v(hidden_states),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Attend to the stored previous-loop memory and add the gated result.

        Returns ``(hidden_states, stats)``. When no memory is stored the input is
        returned unchanged with a zero gate, so callers need no special-casing.
        """
        lam = self.gate
        self.last_gate = float(lam.detach().item())

        if self._memory is None:
            return hidden_states, {"gate": self.last_gate, "applied": 0.0}

        k_prev, v_prev = self._memory
        q = self.query(hidden_states)                      # [B, S, d_m]

        b, s, _ = q.shape
        h = self.num_heads
        hd = self.head_dim
        q_h = q.view(b, s, h, hd).transpose(1, 2)          # [B, H, S, hd]
        k_h = k_prev.view(b, -1, h, hd).transpose(1, 2)    # [B, H, S_prev, hd]
        v_h = v_prev.view(b, -1, h, hd).transpose(1, 2)

        attn = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(hd)
        attn = F.softmax(attn, dim=-1)
        attended = torch.matmul(attn, v_h)                 # [B, H, S, hd]
        attended = attended.transpose(1, 2).reshape(b, s, self.memory_dim)

        delta = self.out_proj(attended)
        updated = hidden_states + lam * delta
        return updated, {"gate": self.last_gate, "applied": float(lam.detach().item())}


class TransformerLayer(nn.Module):
    """Standard Transformer Layer with MLA & MoE."""
    def __init__(self, config: EOSConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if config.quantized_kv_cache:
            self.self_attn = QuantizedMultiLatentAttention(config)
        elif config.fuse_multiloop_attn:
            self.self_attn = RingMLAAttention(config)
        else:
            self.self_attn = MultiLatentAttention(config, layer_idx=layer_idx)
        
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.use_moe:
            if config.use_shared_expert_moe:
                self.mlp = SharedExpertFineGrainedMoE(config)
            elif config.enable_grouped_gemm:
                self.mlp = GroupedGEMMWrapper(
                    num_experts=config.num_experts,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    use_fused_swiglu=True,
                )
            else:
                self.mlp = SparseMoESwiGLUFFN(config)
        else:
            self.mlp = SwiGLUFFN(config)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_modulation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        normed_input = self.input_layernorm(hidden_states)
        attn_out, new_kv_cache = self.self_attn(
            normed_input, cos, sin, attention_mask=attention_mask, kv_cache=kv_cache, kv_modulation=kv_modulation
        )
        hidden_states = hidden_states + attn_out * residual_scale

        normed_attn = self.post_attention_layernorm(hidden_states)

        if self.config.use_moe:
            mlp_out, aux_loss = self.mlp(normed_attn)
        else:
            mlp_out = self.mlp(normed_attn)
            aux_loss = hidden_states.new_zeros(())

        hidden_states = hidden_states + mlp_out * residual_scale

        return hidden_states, new_kv_cache, aux_loss

    def forward(self, *args, **kwargs):
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            hidden_states = args[0]
            cos = kwargs.get("cos", args[1] if len(args) > 1 else None)
            sin = kwargs.get("sin", args[2] if len(args) > 2 else None)
            attention_mask = kwargs.get("attention_mask", args[3] if len(args) > 3 else None)
            residual_scale = kwargs.get("residual_scale", args[4] if len(args) > 4 else 1.0)
            kv_modulation = kwargs.get("kv_modulation", None)

            def ckpt_fn(hs):
                out, _, aux = self._forward(hs, cos, sin, attention_mask, residual_scale, kv_modulation=kv_modulation)
                return out, aux

            out, aux = torch.utils.checkpoint.checkpoint(ckpt_fn, hidden_states, use_reentrant=False)
            return out, None, aux

        return self._forward(*args, **kwargs)


class ContinuousLatentAdapter(nn.Module):
    """
    Pillar 1: Continuous Latent Grounding Adapter.
    Projects raw continuous modality embeddings (vector DB embeddings, image/sensor features)
    directly into the model's hidden dimension without text serialization.
    """
    def __init__(self, in_features: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_size)
        self.norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_latents, in_features] or [B, in_features]
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        return self.norm(self.proj(x))


class LatentTransitionOperator(nn.Module):
    """
    Pillar 2: Latent World-Model Transition Operator (T_theta).
    Simulates prospective future states in latent space conditioned on current state and candidate option:
    z_hat^(t+1) = z_S^(t) + sigma(W_gate [z_S, z_o]) * tanh(W_trans [z_S, z_o])
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.trans = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim, eps=eps)

    def forward(self, z_s: torch.Tensor, z_o: torch.Tensor) -> torch.Tensor:
        # z_s: [B, D] or [B, 1, D], z_o: [B, D] or [B, 1, D]
        if z_s.ndim == 1:
            z_s = z_s.unsqueeze(0)
        elif z_s.ndim == 3:
            z_s = z_s.squeeze(1)
        if z_o.ndim == 1:
            z_o = z_o.unsqueeze(0)
        elif z_o.ndim == 3:
            z_o = z_o.squeeze(1)
        cat = torch.cat([z_s, z_o], dim=-1)
        delta = torch.sigmoid(self.gate(cat)) * torch.tanh(self.trans(cat))
        return self.norm(z_s + delta)


class ProspectiveValueHead(nn.Module):
    """
    Pillar 2: Prospective Value Head V(z_hat).
    Scores the expected utility / safety of an anticipated latent state:
    V(z_hat) = W_v GELU(W_vh z_hat)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            z = z.squeeze(1)
        elif z.ndim == 1:
            z = z.unsqueeze(0)
        return self.net(z).squeeze(-1)  # [B]


class ModernHopfieldEpisodicBank(nn.Module):
    """
    Pillar 7: Modern Hopfield Episodic Memory Bank.
    Associative memory matrix M_episodic with energy-based content retrieval:
    E(z) = -beta^-1 log sum_i exp(beta z^T M_i) + 0.5 ||z||^2
    """
    def __init__(self, hidden_dim: int, max_memories: int = 256, beta: Optional[float] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_memories = max_memories
        self.beta = beta if beta is not None else (1.0 / math.sqrt(hidden_dim))
        self.register_buffer("memory_bank", torch.zeros(max_memories, hidden_dim))
        self.register_buffer("memory_count", torch.tensor(0, dtype=torch.long))

    def store(self, key_repr: torch.Tensor) -> None:
        # key_repr: [B, D] or [D]. Vectorised ring-buffer write: only the newest
        # max_memories rows can survive, and the counter advances by every row.
        flat = key_repr.detach().reshape(-1, self.hidden_dim).to(device=self.memory_bank.device, dtype=self.memory_bank.dtype)
        n = flat.shape[0]
        if n == 0:
            return
        keep = min(n, self.max_memories)
        offset = n - keep
        slots = torch.arange(keep, device=flat.device)
        idx = (self.memory_count + offset + slots) % self.max_memories
        self.memory_bank.index_copy_(0, idx, flat[offset:])
        self.memory_count.add_(n)

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query: [B, D] or [B, 1, D]
        q = query.squeeze(1) if query.ndim == 3 else query
        N = min(int(self.memory_count.item()), self.max_memories)
        if N == 0:
            return q, torch.zeros(q.shape[0], 1, device=q.device, dtype=q.dtype)
        memories = self.memory_bank[:N].to(device=q.device, dtype=q.dtype)  # [N, D]
        # Attention scores: beta * (q @ memories.T)
        scores = self.beta * torch.matmul(q, memories.transpose(0, 1))  # [B, N]
        attn = F.softmax(scores, dim=-1)  # [B, N]
        retrieved = torch.matmul(attn, memories)  # [B, D]
        # Energy
        energy = -(1.0 / self.beta) * torch.logsumexp(scores, dim=-1, keepdim=True) + 0.5 * torch.sum(q**2, dim=-1, keepdim=True)
        return retrieved, energy


class SchemaMutationHead(nn.Module):
    """
    Pillar 5: Dynamic DAG Schema Mutation Head.
    Predicts schema transitions (e.g., spawn sub-questions, switch primitive, branch activation):
    e_schema = W_schema z_decision
    """
    def __init__(self, hidden_dim: int, num_schema_types: int = 8):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_schema_types)

    def forward(self, z_decision: torch.Tensor) -> torch.Tensor:
        # z_decision: [B, D] or [B, 1, D]
        z = z_decision.squeeze(1) if z_decision.ndim == 3 else z_decision
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return self.linear(z)


class MultiHeadRhenPointerHead(nn.Module):
    """
    Multi-Head Bilinear Factorized Pointer Head for Rhen System One decisions.
    
    Contains H=4 specialized query-option alignment heads:
    1. Head 0: Semantic Conceptual Match
    2. Head 1: Policy Constraint Verification
    3. Head 2: Magnitude & Severity Ordering
    4. Head 3: Entity & Role Classification
    
    Each head computes:
        s_h(q, o_i) = (RMSNorm(h_q W_{Q,h})) · (RMSNorm(h_o W_{O,h}))^T / sqrt(rank)
    
    Aggregated alignment:
        s(q, o_i) = sum_{h=1}^H w_h * s_h(q, o_i) where w = Softmax(head_logits)
    
    Question-adaptive temperature calibration:
        tau(q_m) = tau_0 * exp(beta * Entropy(h_q_m))
        P(o_i | q, S) = exp(s(q_m, o_{m,k}) / tau(q_m)) / sum_j exp(s(q_m, o_{m,j}) / tau(q_m))
    """
    def __init__(self, hidden_size: int, num_heads: int = 4, rank: int = 128, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.rank = rank
        self.hidden_size = hidden_size
        self.tau_0 = 1.0
        self.beta = 0.5
        
        # Specialized bilinear projection heads
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_size, rank, bias=False) for _ in range(num_heads)
        ])
        self.o_projs = nn.ModuleList([
            nn.Linear(hidden_size, rank, bias=False) for _ in range(num_heads)
        ])
        self.norm_q = nn.ModuleList([RMSNorm(rank, eps=eps) for _ in range(num_heads)])
        self.norm_o = nn.ModuleList([RMSNorm(rank, eps=eps) for _ in range(num_heads)])
        
        # Learnable head aggregation weights
        self.head_logits = nn.Parameter(torch.zeros(num_heads))
        
        # Learnable layer fusion weights for Hierarchical Multi-Layer Latent Fusion (HMLF)
        self.layer_fusion = nn.Linear(hidden_size, 1, bias=False)

        # Dynamic question-adaptive temperature: tau = tau_0 * exp(beta * entropy)
        self.entropy_proj = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Dynamic Latent Head Probing (DLHP) projections
        self.probe_q = nn.Linear(hidden_size, rank, bias=False)
        self.probe_k = nn.Linear(hidden_size, rank, bias=False)
        self.probe_v = nn.Linear(hidden_size, rank, bias=False)
        self.probe_o = nn.Linear(hidden_size, rank, bias=False)
        self.norm_probe_q = RMSNorm(rank, eps=eps)
        self.norm_probe_k = RMSNorm(rank, eps=eps)
        self.norm_probe_o = RMSNorm(rank, eps=eps)
        self.alpha_probe = nn.Parameter(torch.tensor(0.1))

        # MultiChoice dynamic threshold predictor
        self.multichoice_thresh_proj = nn.Linear(hidden_size, 1)

        # Bounded continuous interval regression head (Beta distribution parameters alpha, beta)
        self.range_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )

        # Extractive span pointer projections
        self.span_start_proj = nn.Linear(hidden_size, rank, bias=False)
        self.span_end_proj = nn.Linear(hidden_size, rank, bias=False)
        self.span_state_proj = nn.Linear(hidden_size, rank, bias=False)
        self.norm_span_q = RMSNorm(rank, eps=eps)
        self.norm_span_state = RMSNorm(rank, eps=eps)

        # Pillar 5: Dynamic DAG Schema Mutation Head
        self.schema_mutation_head = SchemaMutationHead(hidden_size, num_schema_types=8)

    def fuse_layers(self, multi_layer_reprs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Fuses multi-layer representations (shallow, middle, deep) via learned attention weights.
        """
        if isinstance(multi_layer_reprs, torch.Tensor):
            return multi_layer_reprs
        if not multi_layer_reprs:
            raise ValueError("multi_layer_reprs cannot be empty")
        if len(multi_layer_reprs) == 1:
            return multi_layer_reprs[0]
        # Stack along a new layer dimension: [B, N, L, H]
        stack = torch.stack(multi_layer_reprs, dim=2)
        logits = self.layer_fusion(stack).squeeze(-1)  # [B, N, L]
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)  # [B, N, L, 1]
        return (stack * weights).sum(dim=2)  # [B, N, H]

    def forward(
        self,
        q_repr: Union[torch.Tensor, List[torch.Tensor]],
        option_reprs: Union[torch.Tensor, List[torch.Tensor]],
        kv_latent: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            q_repr: [B, 1, H] question query representation at <|decide|> or list of multi-layer tensors.
            option_reprs: [B, NumOptions, H] candidate option representations at <|/opt|> or list of multi-layer tensors.
            kv_latent: Optional [B, S_kv, H] Key/Value latent buffer or state representation for DLHP.
            return_details: If True, returns full dict with probabilities, temperature, and head scores.
        """
        if isinstance(q_repr, list):
            q_repr = self.fuse_layers([q.unsqueeze(1) if q.ndim == 2 else q for q in q_repr])
        elif q_repr.ndim == 2:
            q_repr = q_repr.unsqueeze(1)

        if isinstance(option_reprs, list):
            option_reprs = self.fuse_layers([o.unsqueeze(0) if o.ndim == 2 else o for o in option_reprs])
        elif option_reprs.ndim == 2:
            option_reprs = option_reprs.unsqueeze(0)

        # Compute per-head bilinear pointer scores
        head_scores_list = []
        for h in range(self.num_heads):
            proj_q = self.norm_q[h](self.q_projs[h](q_repr))          # [B, 1, rank]
            proj_o = self.norm_o[h](self.o_projs[h](option_reprs))    # [B, NumOptions, rank]
            s_h = torch.matmul(proj_q, proj_o.transpose(-1, -2)).squeeze(1) / math.sqrt(self.rank)  # [B, NumOptions]
            head_scores_list.append(s_h)

        head_scores = torch.stack(head_scores_list, dim=-1)  # [B, NumOptions, num_heads]
        head_weights = F.softmax(self.head_logits, dim=0)    # [num_heads]
        
        # Weighted aggregate score
        raw_scores = torch.sum(head_scores * head_weights.view(1, 1, -1), dim=-1)  # [B, NumOptions]

        # Dynamic Latent Head Probing (DLHP)
        if kv_latent is not None and option_reprs.shape[1] > 0 and kv_latent.shape[1] > 0:
            pq = self.norm_probe_q(self.probe_q(q_repr))
            pk = self.norm_probe_k(self.probe_k(kv_latent))
            pv = self.probe_v(kv_latent)
            probe_attn = F.softmax(torch.matmul(pq, pk.transpose(-1, -2)) / math.sqrt(self.rank), dim=-1)
            h_probe = torch.matmul(probe_attn, pv)
            po = self.norm_probe_o(self.probe_o(option_reprs))
            s_probe = torch.matmul(h_probe, po.transpose(-1, -2)).squeeze(1) / math.sqrt(self.rank)
            raw_scores = raw_scores + self.alpha_probe * s_probe

        # Question-adaptive temperature: tau(q) = tau_0 * (1 + 0.2 * softplus(proj(q))), about 1.14 at init.
        #
        # An earlier version also multiplied by exp(beta * Entropy(softmax(q / ||q||))). A softmax
        # over a unit-normalised hidden vector is uniform to within ~1e-3, so that "entropy" is the
        # constant ln(hidden_size) for every input: it scaled tau to hidden_size ** beta (about 9 at
        # width 64 and 27 at the default 560), flattened every decision towards uniform and made the
        # head's initial sharpness depend on model width. It is still reported as ``entropy`` (for
        # diagnostics) but no longer scales the temperature.
        q_flat = q_repr.squeeze(1)  # [B, H]
        entropy_boost = F.softplus(self.entropy_proj(q_flat))  # [B, 1]
        q_norm = q_flat / (q_flat.norm(dim=-1, keepdim=True) + 1e-8)
        q_probs = F.softmax(q_norm, dim=-1)  # [B, H]
        q_entropy = -torch.sum(q_probs * torch.log(q_probs + 1e-8), dim=-1, keepdim=True)  # [B, 1]
        tau = self.tau_0 * (1.0 + 0.2 * entropy_boost)  # [B, 1]
        
        calibrated_scores = raw_scores / tau
        probabilities = F.softmax(calibrated_scores, dim=-1)     # [B, NumOptions]
        free_energy = -tau * torch.logsumexp(raw_scores / tau, dim=-1, keepdim=True)  # [B, 1]
        schema_logits = self.schema_mutation_head(q_repr)

        if return_details:
            return {
                "scores": raw_scores,
                "calibrated_scores": calibrated_scores,
                "probabilities": probabilities,
                "temperature": tau,
                "head_scores": head_scores,
                "head_weights": head_weights,
                "entropy": q_entropy,
                "free_energy": free_energy,
                "schema_logits": schema_logits,
            }
        return calibrated_scores

    def compute_multichoice(
        self,
        raw_scores: torch.Tensor,
        tau: torch.Tensor,
        q_repr: torch.Tensor,
        threshold_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes independent calibrated sigmoid probabilities per option for MultiChoice / Subset Selection.
        """
        q_flat = q_repr.squeeze(1) if q_repr.ndim == 3 else q_repr
        theta_learned = torch.sigmoid(self.multichoice_thresh_proj(q_flat))  # [B, 1]
        if threshold_override is not None:
            theta = 0.8 * torch.tensor(threshold_override, device=raw_scores.device, dtype=raw_scores.dtype) + 0.2 * theta_learned
        else:
            theta = theta_learned
        probs = torch.sigmoid((raw_scores - theta) / tau)
        return probs, theta

    def compute_range(
        self,
        q_repr: torch.Tensor,
        min_val: float = 0.0,
        max_val: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Bounded continuous interval regression: predicts Beta(alpha, beta) parameters
        over normalized [0, 1] interval and scales to [min_val, max_val].
        """
        q_flat = q_repr.squeeze(1) if q_repr.ndim == 3 else q_repr
        out = self.range_head(q_flat)  # [B, 2]
        alpha = F.softplus(out[..., 0:1]) + 1.0  # [B, 1] >= 1.0
        beta = F.softplus(out[..., 1:2]) + 1.0   # [B, 1] >= 1.0
        mu_norm = alpha / (alpha + beta)
        var_norm = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1.0))
        sigma_norm = torch.sqrt(var_norm)

        scale = max_val - min_val
        expected_val = min_val + mu_norm * scale
        interval_radius = 1.96 * sigma_norm * scale
        interval_low = torch.clamp(expected_val - interval_radius, min=min_val, max=max_val)
        interval_high = torch.clamp(expected_val + interval_radius, min=min_val, max=max_val)
        confidence = torch.exp(-2.0 * sigma_norm).clamp(0.0, 1.0)

        return {
            "expected_value": expected_val,
            "interval_low": interval_low,
            "interval_high": interval_high,
            "alpha": alpha,
            "beta": beta,
            "confidence": confidence,
            "sigma_norm": sigma_norm,
        }

    def compute_extract_spans(
        self,
        q_repr: torch.Tensor,
        state_reprs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extractive slot-filling over state token representations.
        Computes non-autoregressive start and end token position logits.
        """
        if q_repr.ndim == 2:
            q_repr = q_repr.unsqueeze(1)
        if state_reprs.ndim == 2:
            state_reprs = state_reprs.unsqueeze(0)

        q_start = self.norm_span_q(self.span_start_proj(q_repr))  # [B, 1, rank]
        q_end = self.norm_span_q(self.span_end_proj(q_repr))      # [B, 1, rank]
        s_state = self.norm_span_state(self.span_state_proj(state_reprs))  # [B, S_state, rank]

        start_logits = torch.matmul(q_start, s_state.transpose(-1, -2)).squeeze(1) / math.sqrt(self.rank)  # [B, S_state]
        end_logits = torch.matmul(q_end, s_state.transpose(-1, -2)).squeeze(1) / math.sqrt(self.rank)      # [B, S_state]
        return start_logits, end_logits


# Backward compatibility alias
RhenPointerHead = MultiHeadRhenPointerHead


def _option_span_reprs(hidden: torch.Tensor, branch: Any) -> torch.Tensor:
    """Option representations ``[B, num_options, H]`` for one packed decision branch.

    Each option is the mean hidden state over its span: the tokens after ``<|opt|>`` up to and
    including ``<|/opt|>``. Reading the ``<|/opt|>`` token alone (the earlier readout) gives
    nearly identical vectors for every option, because that token and the space before it are
    the same in every option; the per-option gradients then cancel and decision training
    collapses to a uniform guess. Branches built without start indices fall back to the
    ``<|/opt|>`` token.
    """
    ends = list(branch.option_end_indices)
    starts = getattr(branch, "option_start_indices", None)
    if not starts or len(starts) != len(ends):
        return torch.stack([hidden[:, e, :] for e in ends], dim=1)
    reprs = []
    for s, e in zip(starts, ends):
        lo = min(s + 1, e)
        reprs.append(hidden[:, lo:e + 1, :].mean(dim=1))
    return torch.stack(reprs, dim=1)


def create_rhen_branch_mask(
    state_len: int,
    branch_lens: List[int],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Constructs a 2D Block-Causal Branch Mask for parallel multi-question evaluation over shared state.
    """
    from optim.rhen_mask import Rhen2DBranchMask
    return Rhen2DBranchMask.create_mask(state_len, branch_lens, device=device, dtype=dtype)


class RhenLatentDecisionEngine(nn.Module):
    """
    Rhen Latent Decision Engine (System One Parallel Reasoner).
    Evaluates continuous hypotheses in parallel, provides multi-head factorized pointer scoring,
    and supports Noul, Choice, and Score primitives with zero token generation overhead.
    
    Implements 5 bidirectional communication channels between Recurrent Core and Rhen:
    Channel A: Forward State Extraction (Core → Rhen) - Cross-attention from slots to hidden states
    Channel B: Backward Steering (Rhen → Core Residual Stream) - Content-adaptive dynamic steering
    Channel C: Bidirectional Attention KV Modulation (Rhen → MLA Attention Stream) - Multiplicative gating
    Channel D: Gated Slot Trajectory Memory (Rhen Internal Loop Memory) - Discrete gated recurrent update
    Channel E: Instant Confidence Feedback to ACT Halting (Rhen → ACT) - Certainty metric
    """
    def __init__(self, config: EOSConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.rhen_hidden_dim = config.rhen_hidden_dim
        self.pointer_rank = config.rhen_pointer_rank
        self.num_heads = config.rhen_num_heads
        self.num_slots = config.rhen_num_slots
        self.ortho_loss_coef = float(config.rhen_ortho_loss_coef)
        self.halt_threshold = float(config.rhen_halt_confidence_threshold)

        # K learnable base latent decision slots (hypotheses bank)
        # D_base ∈ R^{1×K×d_rhen} with Frobenius orthogonality regularization
        self.d_slots_base = nn.Parameter(torch.randn(1, self.num_slots, self.rhen_hidden_dim) * 0.02)
        # Learnable slot role embeddings to specialize slot functionality
        self.slot_role_embeds = nn.Parameter(torch.randn(1, self.num_slots, self.rhen_hidden_dim) * 0.05)
        self.d_slots = self.d_slots_base  # Default active slots
        
        # Store active slots for recurrent trajectory memory
        self._active_slots: Optional[torch.Tensor] = None
        self.trajectory_buffer: List[torch.Tensor] = []
        self.hopfield_beta = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.rhen_hidden_dim)))
        self.hopfield_gate = nn.Parameter(torch.tensor(0.15))

        # ===== Bridge 1: Input Layer → Rhen (Context State Priming) =====
        # s_0 = Pool(H_0[1:|S|]) ∈ R^{B×1×d}
        # D_slots^(0) = D_base + RMSNorm(s_0 W_prime), W_prime ∈ R^{d×d_rhen}
        self.prime_proj = nn.Linear(self.hidden_dim, self.rhen_hidden_dim, bias=False)
        self.prime_norm = RMSNorm(self.rhen_hidden_dim, eps=config.rms_norm_eps)

        # ===== Bridge 2: Channel A - Forward State Extraction (Core → Rhen) =====
        # Cross-attention projections: Q_l = D_slots^(l) W_Q^rhen, K_l = H_l W_K^rhen, V_l = H_l W_V^rhen
        self.q_proj = nn.Linear(self.rhen_hidden_dim, self.rhen_hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.rhen_hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.rhen_hidden_dim, bias=False)

        # ===== Bridge 2: Channel D - Gated Slot Trajectory Memory =====
        # α_l = Sigmoid(W_α · Z_l) ∈ R^{B×K×d_rhen}
        # D_slots^(l+1) = α_l ⊙ D_slots^(l) + (1 - α_l) ⊙ Z_l
        self.slot_gate = nn.Sequential(
            nn.Linear(self.rhen_hidden_dim * 2, self.rhen_hidden_dim),
            nn.Sigmoid()
        )

        # ===== Bridge 2: Channel B - Backward Steering (Rhen → Core Residual Stream) =====
        # γ(H_l) = Sigmoid(W_γ,2 · SiLU(W_γ,1 · H_l)) × 0.1 ∈ R^{B×S×1}
        # ΔH_l = RMSNorm(W_steer · Z̄_l), where Z̄_l = mean(Z_l, dim=1)
        # H_l ← H_l + γ(H_l) ⊙ ΔH_l
        self.gate_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.rhen_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.rhen_hidden_dim, 1),
            nn.Sigmoid()
        )
        self.steer_proj = nn.Linear(self.rhen_hidden_dim, self.hidden_dim, bias=False)
        self.slot_steer_proj = nn.Linear(self.rhen_hidden_dim, self.hidden_dim, bias=False)
        self.steer_norm = RMSNorm(self.hidden_dim, eps=config.rms_norm_eps)

        # ===== Bridge 2: Channel C - Bidirectional KV Modulation =====
        # g_KV^(l) = Sigmoid(W_kv · Z̄_l) ∈ R^{B×1×d}
        # K_MLA^(l+1) ← K_MLA^(l+1) ⊙ g_KV^(l), V_MLA^(l+1) ← V_MLA^(l+1) ⊙ g_KV^(l)
        if config.rhen_enable_kv_modulation:
            self.kv_gate_proj = nn.Linear(self.rhen_hidden_dim, self.hidden_dim, bias=False)
        else:
            self.kv_gate_proj = None

        # ===== Bridge 2: Channel E - Instant Confidence Feedback to ACT =====
        # c_l = Sigmoid(W_c · Z̄_l) ∈ [0, 1]
        # Exposed as a diagnostic only; EOSModel.forward no longer boosts the ACT gate with it.
        self.conf_head = nn.Sequential(
            RMSNorm(self.rhen_hidden_dim, eps=config.rms_norm_eps),
            nn.Linear(self.rhen_hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # ===== Bridge 3: Rhen → Output Layer (Dual-Path Output) =====
        # Path 2: Standalone Rhen System One Decision API
        # Output projection back to model hidden dimension
        self.out_proj = nn.Linear(self.rhen_hidden_dim, self.hidden_dim, bias=False)
        self.out_norm = RMSNorm(self.hidden_dim, eps=config.rms_norm_eps)

        # Multi-head factorized pointer head (H=4 specialized heads)
        self.pointer_head = MultiHeadRhenPointerHead(
            self.hidden_dim,
            num_heads=self.num_heads,
            rank=self.pointer_rank,
            eps=config.rms_norm_eps
        )

    def prime_slots(self, initial_state_emb: Optional[torch.Tensor] = None, batch_size: int = 1):
        """
        Primes the hypothesis slots D_slots^(0) at the input layer before recurrent looping.
        Incorporates learned slot role embeddings to specialize slot functionality.
        """
        slots = (self.d_slots_base + self.slot_role_embeds).expand(batch_size, -1, -1)
        if initial_state_emb is not None:
            if initial_state_emb.ndim == 3:
                s0 = initial_state_emb.mean(dim=1, keepdim=True)
            else:
                s0 = initial_state_emb.unsqueeze(1)
            prime_delta = self.prime_norm(self.prime_proj(s0))
            slots = slots + prime_delta
        self._active_slots = slots
        self.trajectory_buffer = []
        return slots

    def reset_state(self):
        self._active_slots = None
        self.trajectory_buffer = []

    def compute_ortho_loss(self) -> torch.Tensor:
        """Frobenius norm slot orthogonality penalty to prevent slot collapse."""
        slots = self.d_slots_base[0]  # [K, d_rhen]
        slots_norm = slots / (slots.norm(dim=-1, keepdim=True) + 1e-6)
        gram = torch.matmul(slots_norm, slots_norm.transpose(-1, -2))  # [K, K]
        eye = torch.eye(self.num_slots, device=slots.device, dtype=slots.dtype)
        return torch.norm(gram - eye, p="fro").pow(2) / max(self.num_slots, 1)

    def forward(
        self,
        h_loop: torch.Tensor,
        step_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Bidirectional communication between Recurrent Core and Rhen across 5 channels.
        
        Returns:
            h_steered: Core residual stream after Rhen steering (Channel B)
            confidence: Instant confidence feedback to ACT halting (Channel E)
            ortho_loss: Slot orthogonality penalty (Channel D)
            kv_modulation: Multiplicative KV gate for MLA attention (Channel C)
            steer_vector: Content-adaptive steering vector for diagnostic access
        """
        batch_size, seq_len, _ = h_loop.shape
        
        # Channel D: Gated Slot Trajectory Memory (internal recurrent update)
        # Slots persist across loops; first call initializes from base bank
        if self._active_slots is None or self._active_slots.shape[0] != batch_size:
            slots = self.d_slots_base.expand(batch_size, -1, -1)
            self.trajectory_buffer = []
        else:
            slots = self._active_slots

        if step_idx == 1:
            self.trajectory_buffer = []

        # Channel A: Forward State Extraction (Core → Rhen)
        # Q_l = D_slots^(l) W_Q^rhen, K_l = H_l W_K^rhen, V_l = H_l W_V^rhen
        q = self.q_proj(slots)                           # [B, K, d_rhen]
        k = self.k_proj(h_loop)                          # [B, S, d_rhen]
        v = self.v_proj(h_loop)                          # [B, S, d_rhen]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rhen_hidden_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)    # [B, K, S]
        z_decision = torch.matmul(attn_weights, v)       # [B, K, d_rhen]

        # Channel D: Gated Slot Trajectory Memory (internal recurrent update)
        # α_l = Sigmoid(W_α · Z_l) ∈ R^{B×K×d_rhen}
        # D_slots^(l+1) = α_l ⊙ D_slots^(l) + (1 - α_l) ⊙ Z_l
        alpha = self.slot_gate(torch.cat([slots, z_decision], dim=-1))
        self._active_slots = alpha * slots + (1.0 - alpha) * z_decision

        # Channel D+ (Modern Hopfield Associative Trajectory Memory)
        self.trajectory_buffer.append(self._active_slots)
        if len(self.trajectory_buffer) > 1:
            M = torch.cat(self.trajectory_buffer, dim=1)  # [B, total_slots, d_rhen]
            hop_scores = self.hopfield_beta * torch.matmul(self._active_slots, M.transpose(-1, -2))
            hop_attn = F.softmax(hop_scores, dim=-1)
            z_hop = torch.matmul(hop_attn, M)
            lambda_hop = torch.sigmoid(self.hopfield_gate)
            self._active_slots = (1.0 - lambda_hop) * self._active_slots + lambda_hop * z_hop

        # Channel B+ (Multi-Slot Layer Steering)
        z_bar = z_decision.mean(dim=1, keepdim=True)     # [B, 1, d_rhen]
        slot_idx = ((step_idx - 1) % self.num_slots) if step_idx is not None else 0
        slot_k = z_decision[:, slot_idx:slot_idx+1, :]
        z_steer = 0.5 * z_bar + 0.5 * slot_k
        gamma = self.gate_net(h_loop) * 0.1              # [B, S, 1]
        steer_vector = self.steer_norm(self.steer_proj(z_steer))
        h_steered = h_loop + gamma * steer_vector

        # Channel C: Bidirectional KV Modulation (Rhen → MLA Attention Stream)
        # g_KV^(l) = Sigmoid(W_kv · Z̄_l) ∈ R^{B×1×d}
        kv_modulation = None
        if self.kv_gate_proj is not None:
            kv_modulation = torch.sigmoid(self.kv_gate_proj(z_bar))  # [B, 1, H]

        # Channel E: Instant Confidence Feedback to ACT Halting
        # c_l = Sigmoid(W_c · Z̄_l) ∈ [0, 1]
        confidence = self.conf_head(z_bar).mean(dim=1)   # [B, 1]

        ortho_loss = self.compute_ortho_loss() * self.ortho_loss_coef

        return h_steered, confidence, ortho_loss, kv_modulation, steer_vector

    def score_options(
        self,
        context_vector: torch.Tensor,
        option_embeddings: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Evaluates query representation against option embeddings using multi-head pointer head.
        """
        details = self.pointer_head(context_vector, option_embeddings, return_details=True)

        slots = self.d_slots_base.expand(context_vector.shape[0], -1, -1)
        q = self.q_proj(slots)
        k = self.k_proj(context_vector)
        v = self.v_proj(context_vector)
        attn_weights = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rhen_hidden_dim), dim=-1)
        z_decision = torch.matmul(attn_weights, v)
        confidence = self.conf_head(z_decision).mean(dim=1)

        details["confidence"] = confidence
        return details


class RecurrentCoreBlock(nn.Module):
    """
    Parameter-Shared Recurrent Core Block with State-Space Memory Highway,
    Symplectic Leapfrog Dynamics, FiLM Step Modulation, and Rhen Latent Decision Engine.
    """
    def __init__(self, config: EOSConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerLayer(config, layer_idx=i) for i in range(config.num_recurrent_layers)
        ])
        if config.use_step_modulation:
            self.step_mod = StepModulation(config.hidden_size, config.max_recurrent_loops)
        else:
            self.step_mod = None

        self.damped_beta = nn.Parameter(torch.zeros(config.hidden_size))
        
        # State-Space Memory Highway across loop iterations
        if config.enable_ssm_highway:
            self.ssm_highway = StateSpaceMemoryHighway(config.hidden_size, config.ssm_state_dim)
        else:
            self.ssm_highway = None

        # Cross-Loop Attention (Memory Grid)
        if config.enable_cross_loop_attention:
            self.cross_loop_attn = create_cross_loop_attention(config)
        else:
            self.cross_loop_attn = None
        
        # Loop-Bypass Router
        if config.enable_loop_bypass:
            self.loop_bypass_router = create_loop_bypass_router(config)
        else:
            self.loop_bypass_router = None
        
        # Google Titans Neural Long-Term Memory (MAG: Memory As Gating). Default-off.
        if getattr(config, "enable_titans_memory", False):
            d_mem = int(getattr(config, "titans_d_mem", 64))
            self.titans_memory = TitansNeuralMemory(config.hidden_size, d_mem=d_mem, eps=config.rms_norm_eps)
        else:
            self.titans_memory = None

        # Rhen Latent Decision Engine
        if config.enable_rhen:
            self.rhen = RhenLatentDecisionEngine(config)
        else:
            self.rhen = None
        self._last_rhen_conf = None

    def reset_special_parameters(self):
        """Reset per-sequence state only (SSM memory, Titans memory, Rhen slots and trajectory).

        No parameter is modified here.
        """
        if self.ssm_highway is not None:
            self.ssm_highway.reset_state()
        if self.titans_memory is not None:
            self.titans_memory.reset_state()
        if self.rhen is not None:
            self.rhen.reset_state()

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
        use_rhen: bool = False,
        layer_caches: Optional[List[Any]] = None,
        ssm_prefix: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step.

        ``layer_caches`` (cached decoding only) is a list with one ``(kv_latent, k_rope)``
        entry per core layer for this loop step; it is updated in place. ``ssm_prefix`` is the
        SSM highway's running prefix for this loop step (see ``StateSpaceMemoryHighway``).
        """
        if layer_caches is not None and len(layer_caches) != len(self.layers):
            raise ValueError(f"layer_caches needs {len(self.layers)} entries, got {len(layer_caches)}")

        input_states = hidden_states
        # Rhen slots are not reset here: forward_rhen resets them once per decision and then
        # primes them from the state, and resetting here would discard that priming.
        if step_idx <= 1:
            if self.titans_memory is not None:
                self.titans_memory.reset_state()
            if self.ssm_highway is not None:
                self.ssm_highway.reset_state()

        if self.step_mod is not None:
            hidden_states = self.step_mod(hidden_states, step_idx)

        total_aux_loss = hidden_states.new_zeros(())

        # Google Titans Neural Long-Term Memory (MAG)
        if self.titans_memory is not None:
            hidden_states = self.titans_memory(hidden_states, step_idx=step_idx)

        # State-Space Memory Highway
        if self.ssm_highway is not None:
            hidden_states = self.ssm_highway(hidden_states, step_idx=step_idx, prefix_state=ssm_prefix)

        # Cross-Loop Attention
        if self.cross_loop_attn is not None and step_idx > 1:
            hidden_states = self.cross_loop_attn(
                hidden_states, step_idx, attention_mask=attention_mask
            )
        
        # Rhen Latent Decision Engine bidirectional communication. It runs only on the
        # decision path (use_rhen=True); the LM path never mixes positions through Rhen.
        kv_modulation = None
        self._last_rhen_conf = None
        if use_rhen and self.rhen is not None:
            hidden_states, rhen_conf, rhen_ortho, kv_modulation, _ = self.rhen(hidden_states, step_idx=step_idx)
            total_aux_loss = total_aux_loss + rhen_ortho
            self._last_rhen_conf = rhen_conf

        # Core layers evaluation
        loop_drop = float(self.config.loop_dropout_prob)
        for layer_idx, layer in enumerate(self.layers):
            if self.training and loop_drop > 0.0 and float(torch.rand(())) < loop_drop:
                continue
            if layer_caches is not None:
                hidden_states, new_cache, aux_loss = layer(
                    hidden_states, cos, sin, attention_mask=attention_mask, residual_scale=residual_scale,
                    kv_cache=layer_caches[layer_idx], kv_modulation=kv_modulation,
                )
                layer_caches[layer_idx] = new_cache
            else:
                hidden_states, _, aux_loss = layer(
                    hidden_states, cos, sin, attention_mask=attention_mask, residual_scale=residual_scale, kv_modulation=kv_modulation
                )
            total_aux_loss = total_aux_loss + aux_loss

        if self.cross_loop_attn is not None:
            self.cross_loop_attn.store_memory(hidden_states, step_idx)

        # Damped Anderson Fixed-Point combination
        beta = (0.05 + 0.90 * torch.sigmoid(self.damped_beta)).view(1, 1, -1)
        output_states = (1.0 - beta) * input_states + beta * hidden_states

        return output_states, total_aux_loss


class MTPStage(nn.Module):
    """Single stage in DeepSeek-V3 Chained MTP with depth conditioning."""
    def __init__(self, config: EOSConfig, stage_depth: int):
        super().__init__()
        self.stage_depth = stage_depth
        self.norm_h = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_emb = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fusion_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.depth_embed = nn.Embedding(16, config.hidden_size)
        self.predictor_layers = nn.ModuleList([
            TransformerLayer(config, layer_idx=0) for _ in range(config.mtp_num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        token_emb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        depth_idx = torch.tensor([self.stage_depth], device=hidden.device, dtype=torch.long)
        d_emb = self.depth_embed(depth_idx).unsqueeze(0)
        fused = self.fusion_proj(torch.cat([
            self.norm_h(hidden),
            self.norm_emb(token_emb)
        ], dim=-1)) + d_emb
        
        h = fused
        for layer in self.predictor_layers:
            h, _, _ = layer(h, cos, sin, attention_mask=attention_mask)
        return self.final_norm(h)


class MultiTokenPredictionHead(nn.Module):
    """
    DeepSeek-V3 Chained Multi-Token Prediction (MTP) Head.
    Chains sequential MTP prediction heads (MTP1 -> MTP2) with depth conditioning.
    """
    def __init__(self, config: EOSConfig, num_stages: int = 2):
        super().__init__()
        self.config = config
        self.num_stages = num_stages
        self.stages = nn.ModuleList([
            MTPStage(config, stage_depth=i + 1) for i in range(num_stages)
        ])

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        next_token_emb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        next_token_emb_2: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        h_mtp1 = self.stages[0](trunk_hidden, next_token_emb, cos, sin, attention_mask=attention_mask)
        if next_token_emb_2 is None or len(self.stages) < 2:
            return h_mtp1

        min_len = min(h_mtp1.shape[1], next_token_emb_2.shape[1])
        h_mtp2 = self.stages[1](
            h_mtp1[:, :min_len, :],
            next_token_emb_2[:, :min_len, :],
            cos[:min_len],
            sin[:min_len],
            attention_mask=attention_mask[:, :, :min_len, :min_len] if attention_mask is not None else None
        )
        return h_mtp1, h_mtp2


class DynamicThinkingBudgetController(nn.Module):
    """DeepSeek-R1 Entropy-Guided Test-Time Thinking Budget Controller."""
    def __init__(
        self,
        config: EOSConfig,
        eps: float = 0.01,
        theta: float = 0.8,
        tau: float = 1.0,
        max_loops: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.min_loops = config.thinking_budget_min_loops
        self.max_loops = max_loops if max_loops is not None else config.thinking_budget_max_loops
        self.eps = float(eps)
        self.theta = float(theta)
        self.tau = float(tau)
        self.complexity_gate = nn.Linear(config.hidden_size, 1)
        self.prev_entropy: Optional[torch.Tensor] = None

    def reset(self):
        self.prev_entropy = None

    def compute_entropy_and_free_energy(
        self,
        logits_or_probs: torch.Tensor,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau_val = tau if tau is not None else self.tau
        if logits_or_probs.ndim == 3:
            scaled = logits_or_probs / max(tau_val, 1e-6)
            probs = F.softmax(scaled, dim=-1)
            free_energy = -tau_val * torch.logsumexp(scaled, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            p_max, _ = torch.max(probs, dim=-1)
        else:
            p = logits_or_probs.clamp(1e-6, 1.0 - 1e-6)
            probs = torch.stack([p, 1.0 - p], dim=-1)
            free_energy = -tau_val * torch.log(torch.exp(p / tau_val) + torch.exp((1.0 - p) / tau_val) + 1e-9)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            p_max, _ = torch.max(probs, dim=-1)

        return entropy, free_energy, p_max

    def check_ponder_exit(
        self,
        logits_or_hidden: torch.Tensor,
        loop_idx: int,
    ) -> bool:
        """
        DeepSeek-R1 Entropy-Guided Loop Pondering check:
        If Delta H_l = H_{l-1} - H_l < eps and confidence p_max > theta, token/decision exits early;
        otherwise allocate up to L_max = 16 loops for hard tokens.
        """
        entropy, free_energy, p_max = self.compute_entropy_and_free_energy(logits_or_hidden)
        if self.prev_entropy is None or loop_idx <= self.min_loops:
            self.prev_entropy = entropy.detach()
            return False

        delta_h = (self.prev_entropy - entropy).mean().item()
        self.prev_entropy = entropy.detach()
        mean_pmax = p_max.mean().item()

        if delta_h < self.eps and mean_pmax > self.theta:
            return True
        return False

    def estimate_budget(
        self,
        hidden_states: torch.Tensor,
        early_velocity: Optional[torch.Tensor] = None,
        base_max_loops: Optional[int] = None,
    ) -> int:
        max_limit = base_max_loops or self.max_loops
        with torch.no_grad():
            gate_score = torch.sigmoid(self.complexity_gate(hidden_states.mean(dim=(0, 1)))).item()
            if early_velocity is not None:
                vel_norm = early_velocity.mean().item() / math.sqrt(hidden_states.shape[-1])
                complexity = 0.5 * gate_score + 0.5 * min(1.0, vel_norm * 2.0)
            else:
                complexity = gate_score
            
            # The floor never exceeds the limit: a budget above max_recurrent_loops would index
            # past the per-step embeddings, which only ACT early exit used to hide.
            min_limit = min(self.min_loops, max_limit)
            allocated = int(min_limit + complexity * (max_limit - min_limit))
            return max(min_limit, min(max_limit, allocated))


def apply_repetition_penalty(
    logits: torch.Tensor,
    seen_token_ids: Any,
    penalty: float,
) -> torch.Tensor:
    """Sign-aware repetition penalty for already-generated tokens.

    For every seen token id, a positive logit is divided by ``penalty`` and a negative
    logit is multiplied by it, so the token becomes less likely in both cases. A penalty
    of 1.0 returns the logits unchanged. ``logits`` has the vocabulary on its last
    dimension; ``seen_token_ids`` is applied to every leading row. Returns a new tensor.
    """
    if penalty == 1.0:
        return logits
    seen = torch.as_tensor(seen_token_ids, dtype=torch.long, device=logits.device).reshape(-1)
    if seen.numel() == 0:
        return logits
    picked = logits.index_select(-1, seen)
    adjusted = torch.where(picked > 0, picked / penalty, picked * penalty)
    return logits.index_copy(-1, seen, adjusted)


_WRAPPED_CORE = "recurrent_core.core."
_PLAIN_CORE = "recurrent_core."


def _canonical_core_state_dict_hook(module, state_dict, prefix, local_metadata):
    """Save core keys as ``recurrent_core.X`` whether or not the checkpoint wrapper is present."""
    wrapped = prefix + _WRAPPED_CORE
    for key in [k for k in state_dict if k.startswith(wrapped)]:
        state_dict[prefix + _PLAIN_CORE + key[len(wrapped):]] = state_dict.pop(key)
    return state_dict


class EOSKVCache:
    """Decoding cache for the causal LM path.

    The recurrent core reuses one block of layers for every loop step, but each step sees
    different hidden states, so each (loop step, core layer) pair keeps its own MLA cache:
    the compressed ``kv_latent`` and ``k_rope`` of every earlier position. The SSM highway's
    causal mean keeps a running sum per loop step. The loop count is fixed when the cache is
    filled (prefill) and reused for every later token, and adaptive early exit is not taken
    while caching, so cached decoding matches a full forward run with the same ``num_loops``.
    """

    def __init__(self, num_pre_layers: int, num_post_layers: int, num_core_layers: int):
        self.pre: List[Any] = [(None, None)] * num_pre_layers
        self.post: List[Any] = [(None, None)] * num_post_layers
        self.num_core_layers = num_core_layers
        self.core: List[List[Any]] = []
        self.ssm: List[Dict[str, Any]] = []
        self.num_loops: Optional[int] = None
        self.seq_len = 0

    def allocate_loops(self, num_loops: int) -> None:
        self.num_loops = int(num_loops)
        self.core = [[(None, None)] * self.num_core_layers for _ in range(self.num_loops)]
        self.ssm = [{} for _ in range(self.num_loops)]

    def get_seq_length(self) -> int:
        return self.seq_len

    def __len__(self) -> int:
        return self.seq_len


class EOSModel(PreTrainedModel):
    """
    EOS with Rhen System One Decision Engine.
    HuggingFace PreTrainedModel compatible.
    """
    config_class = EOSConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: EOSConfig):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.depth_embed = nn.Embedding(16, config.hidden_size)
        self.decision_cross_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.decision_gate_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.rope_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            yarn_scale_factor=config.yarn_scale if config.use_yarn else 1.0,
            yarn_original_max_position=config.yarn_original_max_position_embeddings,
            yarn_beta_fast=config.yarn_beta_fast,
            yarn_beta_slow=config.yarn_beta_slow,
            yarn_attn_factor=config.yarn_attn_factor,
            enable_dynamic_ntk=config.enable_dynamic_ntk_yarn,
            dynamic_max_context=config.dynamic_ntk_yarn_max_context,
            dynamic_snap_factor=config.dynamic_ntk_yarn_snap_factor,
        )

        self.pre_layers = nn.ModuleList([
            TransformerLayer(config, layer_idx=0)
            for _ in range(config.num_pre_layers)
        ])

        base_core = RecurrentCoreBlock(config)
        # Core-level (selective) checkpointing only exists under gradient checkpointing:
        # the wrapper reads config.gradient_checkpointing, so it is active iff it is built.
        if config.enable_selective_checkpoint and config.gradient_checkpointing:
            self.recurrent_core = CheckpointedRecurrentCore(base_core, config)
        else:
            self.recurrent_core = base_core
            if config.enable_selective_checkpoint:
                warnings.warn(
                    "enable_selective_checkpoint has no effect without gradient_checkpointing",
                    stacklevel=2,
                )
        # The checkpoint wrapper adds a "core." level to the core's keys. Checkpoints always
        # use the unwrapped layout, and loading accepts either layout, so a model trained with
        # checkpointing loads into one without it and vice versa.
        self._register_state_dict_hook(_canonical_core_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._core_layout_load_pre_hook)
        # The per-call ("forward") congestion-bias update moves the bias between a forward and
        # its checkpoint recomputation, which then routes differently and fails. Checkpointed
        # models therefore record loads and update once per step (applied at the next training
        # forward unless a trainer applies it after the optimizer step).
        self._checkpointing_at_init = bool(config.gradient_checkpointing)

        self.post_layers = nn.ModuleList([
            TransformerLayer(config, layer_idx=config.num_pre_layers + config.num_recurrent_layers)
            for _ in range(config.num_post_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        if config.enable_act:
            self.act_halting = EntropyGatedACTHalting(config.hidden_size)
        else:
            self.act_halting = None

        # Token-Level Adaptive Recurrent Exit. Default-off so the legacy
        # `h_loop = h_next` update stays bit-identical until enabled.
        self.adaptive_loop_router: Optional[AdaptiveLoopRouter] = None
        if config.enable_adaptive_loop_router:
            self.adaptive_loop_router = AdaptiveLoopRouter(
                hidden_size=config.hidden_size,
                min_loops=config.adaptive_loop_min_loops,
                freeze_threshold=config.adaptive_loop_freeze_threshold,
            )

        # Cross-Loop Attentive Memory Highway. Default-off, and the gate starts
        # closed even when enabled, so it is a no-op until trained.
        self.clam_bridge: Optional[CLAMBridge] = None
        if config.enable_clam_bridge:
            self.clam_bridge = CLAMBridge(
                hidden_size=config.hidden_size,
                memory_dim=config.clam_memory_dim,
                num_heads=config.num_attention_heads,
                init_gate=config.clam_init_gate,
            )

        # Multi-Token Prediction Head
        if config.enable_mtp:
            self.mtp_head = MultiTokenPredictionHead(config)
        else:
            self.mtp_head = None

        # Dynamic Thinking Budget Controller
        if config.enable_thinking_budget:
            self.thinking_controller = DynamicThinkingBudgetController(config)
        else:
            self.thinking_controller = None

        if config.enable_speculative_decoding:
            self.speculative_decoder = create_self_speculative_decoder(self, config)
        else:
            self.speculative_decoder = None

        self.continuous_latent_adapter = ContinuousLatentAdapter(
            # The Rhen latent input width; the historical fallback default,
            # previously shadowed behind a getattr ghost key.
            in_features=768,
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.latent_transition_operator = LatentTransitionOperator(config.hidden_size, eps=config.rms_norm_eps)
        self.prospective_value_head = ProspectiveValueHead(config.hidden_size)
        # Modern Hopfield episodic bank. Default-off, so a stock model has no bank buffers.
        if getattr(config, "enable_episodic_memory", False):
            self.episodic_memory_bank = ModernHopfieldEpisodicBank(config.hidden_size)
        else:
            self.episodic_memory_bank = None

        self.post_init()
        if config.mup_enabled:
            apply_mup_init(self, mup_config=create_mup_config(config))
        self._last_kv_cache = None
        self.auto_apply_moe_balance = True
        if self._checkpointing_at_init:
            self.set_moe_balance_update_mode("step")

        # Rhen-1.0 recurrent-loop control. Both are None unless explicitly
        # enabled, so a default EOSModel keeps the exact legacy update
        # `h_loop = h_next` and produces bit-identical results to before.
        self.loop_step_controller: Optional[StepSizeController] = None
        if config.enable_step_size_control:
            self.loop_step_controller = StepSizeController(
                hidden_size=config.hidden_size,
                base_alpha=config.loop_step_alpha,
                min_alpha=config.loop_step_min_alpha,
                anneal=config.loop_step_anneal,
                overshoot_rejection=config.loop_step_overshoot_rejection,
                enabled=True,
            )
            # A per-parameter module would otherwise be registered as a child
            # of EOSModel and change the checkpoint's state_dict.
            object.__setattr__(self.loop_step_controller, "_modules", {})

        self.loop_flash_updater: Optional[FlashLoopUpdater] = None
        if config.enable_flash_loop_sparsity:
            self.loop_flash_updater = FlashLoopUpdater(
                threshold=config.flash_loop_threshold,
                warmup_loops=config.flash_loop_warmup_loops,
                enabled=True,
            )
            object.__setattr__(self.loop_flash_updater, "_modules", {})

        self.loop_curriculum: Optional[LoopCurriculum] = None
        if config.enable_loop_curriculum:
            self.loop_curriculum = LoopCurriculum(
                min_loops=config.min_recurrent_loops,
                max_loops=config.max_recurrent_loops,
                warmup_steps=config.loop_curriculum_warmup_steps,
            )

    @property
    def _core_block(self) -> "RecurrentCoreBlock":
        """The RecurrentCoreBlock itself, looking through the optional checkpoint wrapper."""
        core = self.recurrent_core
        return core.core if isinstance(core, CheckpointedRecurrentCore) else core

    def loop_diagnostics(self) -> Dict[str, Any]:
        """Report the Rhen-1.0 loop controls for the most recent forward pass."""
        out: Dict[str, Any] = {
            "step_size_control": bool(self.loop_step_controller is not None),
            "flash_loop_sparsity": bool(self.loop_flash_updater is not None),
            "adaptive_loop_router": bool(self.adaptive_loop_router is not None),
            "clam_bridge": bool(self.clam_bridge is not None),
            "dynamic_ntk_yarn": bool(getattr(self.rope_emb, "enable_dynamic_ntk", False)),
        }
        if self.loop_step_controller is not None:
            out["step"] = self.loop_step_controller.summary()
        if self.loop_flash_updater is not None:
            out["flash"] = self.loop_flash_updater.savings()
        if self.adaptive_loop_router is not None:
            # Halting-mask statistics of the token-level exit (not compute saved).
            out["adaptive_loop"] = self.adaptive_loop_router.savings()
        if self.clam_bridge is not None:
            out["clam"] = {"gate": self.clam_bridge.last_gate}
        return out

    def _backward_compatibility_gradient_checkpointing(self):
        # transformers would call gradient_checkpointing_enable() here and delete
        # config.gradient_checkpointing. EOSModel reads that flag itself (TransformerLayer
        # and the selective core wrapper), so it must stay on the config; deleting it would
        # break the next forward pass and any later construction from the same config.
        return None

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "_residual_out") and module._residual_out:
                module.weight.data.normal_(mean=0.0, std=std / math.sqrt(2.0 * self.config.num_recurrent_layers))
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head
        return self.embed_tokens

    def count_parameters(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed_params = sum(p.numel() for p in self.embed_tokens.parameters() if p.requires_grad)
        layer_core = total_params - embed_params
        effective_depth = self.config.num_pre_layers + (self.config.num_recurrent_layers * self.config.max_recurrent_loops) + self.config.num_post_layers
        return {
            "total_parameters": total_params,
            "layer_core_parameters": layer_core,
            "total_effective_depth": effective_depth,
        }

    def _core_layout_load_pre_hook(self, state_dict, prefix, *args, **kwargs):
        """Map core keys to this model's layout (wrapped or not) before loading."""
        wrapped_here = isinstance(self.recurrent_core, CheckpointedRecurrentCore)
        wrapped, plain = prefix + _WRAPPED_CORE, prefix + _PLAIN_CORE
        for key in list(state_dict.keys()):
            if wrapped_here and key.startswith(plain) and not key.startswith(wrapped):
                state_dict[wrapped + key[len(plain):]] = state_dict.pop(key)
            elif not wrapped_here and key.startswith(wrapped):
                state_dict[plain + key[len(wrapped):]] = state_dict.pop(key)

    # ------------------------------------------------------------------
    # KV cache support
    # ------------------------------------------------------------------
    def kv_cache_incompatibilities(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        depth_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """Reasons why cached decoding would not match a full forward pass (empty if it would).

        Cached decoding supports the default architecture. Components that mix positions in
        ways the cache does not track are listed here and make ``use_cache`` raise.
        """
        reasons: List[str] = []
        if self.training:
            reasons.append("the model is in training mode")
        if attention_mask is not None:
            reasons.append("a custom attention_mask is given")
        if depth_ids is not None:
            reasons.append("depth_ids are given")
        if labels is not None:
            reasons.append("labels are given")
        cfg = self.config
        if cfg.quantized_kv_cache:
            reasons.append("quantized_kv_cache uses a different attention module")
        if cfg.fuse_multiloop_attn:
            reasons.append("fuse_multiloop_attn uses a different attention module")
        core = self._core_block
        if core.titans_memory is not None:
            reasons.append("Titans memory is enabled")
        if core.cross_loop_attn is not None:
            reasons.append("cross-loop attention is enabled")
        if core.loop_bypass_router is not None:
            reasons.append("the loop bypass router is enabled")
        if getattr(self, "clam_bridge", None) is not None:
            reasons.append("the CLAM bridge is enabled")
        if getattr(self, "adaptive_loop_router", None) is not None:
            reasons.append("the adaptive loop router is enabled")
        if getattr(self, "loop_flash_updater", None) is not None:
            reasons.append("the flash loop updater is enabled")
        if getattr(self, "loop_step_controller", None) is not None:
            reasons.append("the loop step controller is enabled")
        if getattr(self.rope_emb, "enable_dynamic_ntk", False):
            reasons.append("dynamic NTK RoPE rescales earlier positions as the sequence grows")
        return reasons

    def supports_kv_cache(self) -> bool:
        return not self.kv_cache_incompatibilities()

    # ------------------------------------------------------------------
    # MoE balance updates (see SharedExpertFineGrainedMoE)
    # ------------------------------------------------------------------
    def moe_balance_modules(self) -> List["SharedExpertFineGrainedMoE"]:
        return [m for m in self.modules() if isinstance(m, SharedExpertFineGrainedMoE)]

    def set_moe_balance_update_mode(self, mode: str) -> None:
        """``"forward"`` (default) or ``"step"``; ``"step"`` is required with gradient checkpointing."""
        for module in self.moe_balance_modules():
            module.set_balance_update_mode(mode)

    @torch.no_grad()
    def apply_moe_balance_updates(self, reduce_fn=None) -> int:
        """Apply pending ``"step"``-mode congestion-bias updates. Call after the optimizer step.

        ``reduce_fn`` maps a local mean-load tensor to the mean across data-parallel ranks, so
        every rank applies the same update. Returns the number of modules updated.
        """
        updated = 0
        for module in self.moe_balance_modules():
            if module.balance_update_mode != "step" or not module.use_aux_free:
                continue
            calls = module.pending_calls
            local_mean = (
                module.pending_load_sum / calls
                if calls > 0
                else torch.full_like(module.pending_load_sum, 1.0 / module.num_routed_experts)
            )
            if reduce_fn is not None:
                # Every rank must join the collective, even one that saw no tokens.
                mean_load = reduce_fn(local_mean)
            elif calls > 0:
                mean_load = local_mean
            else:
                continue
            if module.apply_pending_balance_update(mean_load=mean_load):
                updated += 1
        return updated

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_loops: Optional[int] = None,
        output_loop_stats: bool = False,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        depth_ids: Optional[torch.Tensor] = None,
        decision_latent: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.training and getattr(self, "auto_apply_moe_balance", True) and not _in_autograd_backward():
            # Step-mode MoE balancing: apply the update recorded by the previous training
            # forward if no trainer applied it explicitly. A no-op in the default "forward" mode.
            # Trainers that apply updates once per optimizer step (eos_research.train) turn this off.
            self.apply_moe_balance_updates()

        cache: Optional[EOSKVCache] = None
        if use_cache or past_key_values is not None:
            problems = self.kv_cache_incompatibilities(
                attention_mask=attention_mask, depth_ids=depth_ids, labels=labels
            )
            if problems:
                raise NotImplementedError("KV cache is not supported here: " + "; ".join(problems))
            if past_key_values is None:
                cache = EOSKVCache(len(self.pre_layers), len(self.post_layers), len(self._core_block.layers))
            elif isinstance(past_key_values, EOSKVCache):
                cache = past_key_values
            else:
                raise TypeError(
                    "past_key_values must be the EOSKVCache returned by an earlier forward call, "
                    f"got {type(past_key_values).__name__}"
                )

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        if depth_ids is not None:
            d_ids = depth_ids if isinstance(depth_ids, torch.Tensor) else torch.tensor(depth_ids, device=hidden_states.device)
            if d_ids.ndim == 1:
                d_ids = d_ids.unsqueeze(0)
            hidden_states = hidden_states + self.depth_embed(d_ids[:, :hidden_states.shape[1]].clamp(0, 15))

        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # With a cache, the RoPE tables cover every cached position plus the new ones; the
        # attention layers rotate the new queries by the trailing positions.
        total_len = seq_len + (cache.seq_len if cache is not None else 0)
        cos_full, sin_full = self.rope_emb(hidden_states, total_len)
        cos = cos_full[:total_len]
        sin = sin_full[:total_len]

        # Pre-Loop Layers
        for layer_idx, layer in enumerate(self.pre_layers):
            hidden_states, layer_kv, _ = layer(
                hidden_states, cos, sin, attention_mask=attention_mask,
                kv_cache=cache.pre[layer_idx] if cache is not None else None,
            )
            if cache is not None:
                cache.pre[layer_idx] = layer_kv

        # Dynamic Recurrent Loop Allocation
        if cache is not None and cache.num_loops is not None:
            # Decoding reuses the loop count fixed at prefill: every cached position has one
            # KV entry per loop step, so a different count would not line up with them.
            if num_loops is not None and int(num_loops) != cache.num_loops:
                raise ValueError(
                    f"num_loops={num_loops} differs from the {cache.num_loops} loops this cache was filled with"
                )
            max_loops = cache.num_loops
        elif num_loops is not None:
            max_loops = num_loops
        elif self.training and self.config.loop_sampling == "uniform":
            min_l = self.config.min_recurrent_loops
            max_l = self.config.max_recurrent_loops
            max_loops = int(torch.randint(min_l, max_l + 1, (1,)).item())
        elif self.thinking_controller is not None and not self.training:
            max_loops = self.thinking_controller.estimate_budget(hidden_states, base_max_loops=self.config.max_recurrent_loops)
        else:
            max_loops = self.config.max_recurrent_loops

        if cache is not None and cache.num_loops is None:
            cache.allocate_loops(max_loops)

        h_loop = hidden_states
        h_prev = None
        contract_loss = hidden_states.new_zeros(())
        num_contract_steps = 0

        # Rhen-1.0 loop control. `step_controller` implements the trust-region
        # step from arXiv:2609.16665 ("Right Direction, Wrong Step"), and
        # `flash_updater` the token-sparse cross-loop update from arXiv:2609.29812.
        # Both default to disabled so legacy behaviour is bit-identical until a
        # config flag turns them on.
        step_controller = getattr(self, "loop_step_controller", None)
        flash_updater = getattr(self, "loop_flash_updater", None)
        if step_controller is not None:
            step_controller.set_budget(max_loops)
            step_controller.reset()
        if flash_updater is not None:
            flash_updater.reset()

        # Token-level adaptive exit and the cross-loop memory highway. Both are
        # None unless explicitly enabled, so the default model keeps the exact
        # legacy `h_loop = h_next` recurrent update.
        loop_router = getattr(self, "adaptive_loop_router", None)
        clam = getattr(self, "clam_bridge", None)
        if loop_router is not None:
            loop_router.reset()
        if clam is not None:
            clam.reset()

        ponder_cost = hidden_states.new_zeros(batch_size, seq_len)
        accum_prob = hidden_states.new_zeros(batch_size, seq_len)
        step_counts = hidden_states.new_zeros(batch_size, seq_len)
        has_halted = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        h_halted = torch.zeros_like(hidden_states)
        exit_step_probs = []

        total_moe_aux_loss = hidden_states.new_zeros(())

        loop_bypass_router = getattr(self._core_block, "loop_bypass_router", None)

        for step in range(1, max_loops + 1):
            if loop_bypass_router is not None:
                bypass_mask, router_loss = loop_bypass_router(h_loop, step)
                total_moe_aux_loss = total_moe_aux_loss + router_loss
                h_loop_active = h_loop
            else:
                bypass_mask = None
                h_loop_active = h_loop

            cache_kwargs = {}
            if cache is not None:
                cache_kwargs = {"layer_caches": cache.core[step - 1], "ssm_prefix": cache.ssm[step - 1]}
            h_next, aux_loss = self.recurrent_core(
                h_loop_active,
                step_idx=step,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                residual_scale=1.0 / math.sqrt(max_loops),
                use_rhen=False,
                **cache_kwargs,
            )
            total_moe_aux_loss = total_moe_aux_loss + aux_loss

            if bypass_mask is not None:
                h_next = torch.where(bypass_mask, h_loop, h_next)

            # Rhen-1.0: hold frozen tokens and take a trust-region step.
            if flash_updater is not None:
                flash_updater.observe(h_loop)
                h_next = flash_updater.apply(h_loop, h_next)
            if step_controller is not None:
                h_next, _step_stats = step_controller.apply(h_loop, h_next, step)
            elif not self.config.loop_full_update:
                # Opt-in safety: never take an unmitigated full step.
                h_next = h_loop + 0.25 * (h_next - h_loop)

            if h_prev is not None:
                step_contract = F.mse_loss(h_next, h_prev)
                contract_loss = contract_loss + step_contract
                num_contract_steps += 1
            h_prev = h_next.detach()

            # CLAM Bridge: let this loop attend to the previous loop's compressed
            # keys/values. The gate is initialised closed, so on a freshly
            # initialised model this is exactly a no-op.
            if clam is not None:
                h_next, _clam_stats = clam(h_next)

            # Token-level adaptive exit: freeze trivial tokens so the remaining
            # loops spend their compute on the reasoning tokens instead.
            if loop_router is not None:
                h_next, _active_mask = loop_router.step(h_loop, h_next, step)

            velocity = torch.norm(h_next - h_loop, dim=-1) if self.config.enable_act else None

            if self.config.enable_act and self.act_halting is not None:
                p = self.act_halting(h_next, velocity=velocity)
                exit_step_probs.append(p)

                if step < max_loops:
                    new_accum = accum_prob + p
                    is_last_step = (new_accum >= self.config.act_threshold) & (~has_halted)
                    
                    if step < self.config.act_min_loops:
                        is_last_step = torch.zeros_like(is_last_step)

                    prob_to_add = torch.where(is_last_step, 1.0 - accum_prob, p)
                    prob_to_add = torch.where(has_halted, torch.zeros_like(p), prob_to_add)

                    h_halted = h_halted + prob_to_add.unsqueeze(-1) * h_next
                    accum_prob = accum_prob + prob_to_add
                    ponder_cost = ponder_cost + (~has_halted).float()
                    step_counts = torch.where(~has_halted, step_counts + 1, step_counts)

                    has_halted = has_halted | is_last_step
                else:
                    remainder = 1.0 - accum_prob
                    remainder = torch.where(has_halted, torch.zeros_like(remainder), remainder)
                    h_halted = h_halted + remainder.unsqueeze(-1) * h_next
                    ponder_cost = ponder_cost + (~has_halted).float()
                    step_counts = torch.where(~has_halted, step_counts + 1, step_counts)

                # While caching, every loop step runs: later tokens attend to this step's keys
                # for the current positions, which a full forward would also compute.
                if self.config.act_early_stop and not self.training and cache is None and bool(has_halted.all()):
                    break

            h_loop = h_next

            # Compress and store this loop's state for the next one to attend to.
            if clam is not None:
                clam.store(h_loop)

        if self.config.enable_act and self.act_halting is not None:
            hidden_states = h_halted
        else:
            hidden_states = h_loop
            step_counts.fill_(max_loops)

        # Cross-layer decision conditioning (Rhen -> Generation bridge)
        if decision_latent is not None:
            dec_emb = self.decision_cross_proj(decision_latent)  # [B, 1, H]
            if dec_emb.ndim == 2:
                dec_emb = dec_emb.unsqueeze(1)
            gate = torch.sigmoid(self.decision_gate_proj(hidden_states))  # [B, S, H]
            hidden_states = hidden_states + gate * dec_emb

        # Post-Loop Layers
        for layer_idx, layer in enumerate(self.post_layers):
            hidden_states, layer_kv, _ = layer(
                hidden_states, cos, sin, attention_mask=attention_mask,
                kv_cache=cache.post[layer_idx] if cache is not None else None,
            )
            if cache is not None:
                cache.post[layer_idx] = layer_kv

        if cache is not None:
            cache.seq_len += seq_len

        hidden_states = self.norm(hidden_states)

        # Logit Projection & Loss Calculation
        head_weight = self.get_output_embeddings().weight
        softcap = float(self.config.final_logit_softcapping)

        chunk_z_loss: Optional[torch.Tensor] = None
        if labels is not None and self.config.chunked_ce:
            chunk_size = self.config.chunked_ce_chunk_size
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            flat_hidden = shift_hidden.view(-1, self.config.hidden_size)
            flat_labels = shift_labels.view(-1)
            use_ckpt = self.training and torch.is_grad_enabled()

            def _chunk_stats(chunk_h: torch.Tensor, chunk_l: torch.Tensor):
                # CE sum and z-loss sum (logsumexp^2 over valid tokens) for one chunk.
                # Checkpointed when training, so the chunk's logits are recomputed in backward.
                chunk_logits = F.linear(chunk_h, head_weight)
                if softcap > 0.0:
                    chunk_logits = softcap * torch.tanh(chunk_logits / softcap)
                chunk_logits = chunk_logits.float()
                valid = (chunk_l != -100).to(chunk_logits.dtype)
                ce_sum = F.cross_entropy(chunk_logits, chunk_l, ignore_index=-100, reduction="sum")
                z_sum = (torch.logsumexp(chunk_logits, dim=-1).pow(2) * valid).sum()
                return ce_sum, z_sum

            total_ce = hidden_states.new_zeros((), dtype=torch.float32)
            total_z = hidden_states.new_zeros((), dtype=torch.float32)
            num_valid = 0
            for i in range(0, flat_hidden.shape[0], chunk_size):
                chunk_h = flat_hidden[i:i + chunk_size]
                chunk_l = flat_labels[i:i + chunk_size]
                n_valid = int((chunk_l != -100).sum().item())
                if n_valid == 0:
                    continue
                if use_ckpt:
                    ce_sum, z_sum = torch.utils.checkpoint.checkpoint(
                        _chunk_stats, chunk_h, chunk_l, use_reentrant=False
                    )
                else:
                    ce_sum, z_sum = _chunk_stats(chunk_h, chunk_l)
                total_ce = total_ce + ce_sum
                total_z = total_z + z_sum
                num_valid += n_valid

            ce_loss = total_ce / max(num_valid, 1)
            chunk_z_loss = total_z / max(num_valid, 1)
            logits = None  # full-vocabulary logits are never materialised on this path
        else:
            logits = F.linear(hidden_states, head_weight)
            if softcap > 0.0:
                logits = softcap * torch.tanh(logits / softcap)
            ce_loss = None

        loss = None
        z_loss = hidden_states.new_zeros(())
        mtp_loss = hidden_states.new_zeros(())
        act_prior = hidden_states.new_zeros(())
        ponder_loss = hidden_states.new_zeros(())
        moe_loss = hidden_states.new_zeros(())
        prior_loss = hidden_states.new_zeros(())
        contract_coef = float(self.config.contraction_loss_coef)
        z_coef = float(self.config.z_loss_coef)

        if labels is not None:
            if ce_loss is None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

            # Multi-Token Prediction Loss (t+2 prediction)
            if self.mtp_head is not None and hidden_states.shape[1] > 2:
                trunk_slice = hidden_states[..., :-2, :].contiguous()
                next_tokens = labels[..., 1:-1].clamp(min=0)
                next_emb = self.embed_tokens(next_tokens)
                mtp_hidden = self.mtp_head(trunk_slice, next_emb, cos[:trunk_slice.shape[1]], sin[:trunk_slice.shape[1]])
                mtp_logits = F.linear(mtp_hidden, head_weight)
                if softcap > 0.0:
                    mtp_logits = softcap * torch.tanh(mtp_logits / softcap)
                target_tokens = labels[..., 2:].contiguous()
                mtp_loss = F.cross_entropy(
                    mtp_logits.view(-1, self.config.vocab_size),
                    target_tokens.view(-1),
                    ignore_index=-100
                )

            ponder_loss = self.config.act_ponder_penalty * ponder_cost.mean()
            moe_loss = self.config.moe_aux_loss_coef * total_moe_aux_loss

            if self.config.enable_act and exit_step_probs:
                stacked_probs = torch.stack(exit_step_probs, dim=-1)
                num_steps = stacked_probs.shape[-1]
                steps = torch.arange(1, num_steps + 1, device=device).float()
                geom_prior = (1.0 - self.config.act_prior_lambda) ** (steps - 1) * self.config.act_prior_lambda
                geom_prior = geom_prior / geom_prior.sum()
                act_prior = F.kl_div(
                    torch.log(stacked_probs.mean(dim=(0, 1)) + 1e-9),
                    geom_prior,
                    reduction="batchmean"
                )
                prior_loss = 0.01 * act_prior
            else:
                act_prior = hidden_states.new_zeros(())
                prior_loss = hidden_states.new_zeros(())

            contract_coef = float(self.config.contraction_loss_coef)
            if num_contract_steps > 0:
                contract_loss = contract_loss / num_contract_steps

            z_coef = float(self.config.z_loss_coef)
            if chunk_z_loss is not None:
                z_loss = chunk_z_loss
            elif logits is not None and z_coef > 0.0:
                z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

            mtp_coef = float(self.config.mtp_loss_coef)

            # Straight-through cap: the value is min(raw_aux, aux_cap), but the gradient
            # of raw_aux is kept (torch.min would zero it whenever the cap binds).
            aux_cap = 0.15 * ce_loss.detach()
            raw_aux = ponder_loss + moe_loss + prior_loss + (contract_coef * contract_loss)
            clamped_aux = raw_aux - (raw_aux - aux_cap).clamp(min=0.0).detach()

            loss = ce_loss + clamped_aux + (z_coef * z_loss) + (mtp_coef * mtp_loss)

        # Logits are already None on the chunked-CE path (labels given); no full-vocabulary
        # projection is re-run here.
        loop_stats = None
        if output_loop_stats:
            # Diagnostics (and their .item() syncs) are only built when requested.
            loop_stats = {
                "avg_loops_per_token": step_counts.mean().item(),
                "min_loops": step_counts.min().item(),
                "max_loops": step_counts.max().item(),
                "ponder_cost": ponder_cost.mean().item(),
                "moe_aux_loss": total_moe_aux_loss.item(),
                "act_prior_loss": act_prior.item(),
                "max_configured_loops": max_loops,
                "ce_loss": ce_loss.item() if labels is not None else 0.0,
                "ponder_loss": ponder_loss.item() if labels is not None else 0.0,
                "moe_loss": moe_loss.item() if labels is not None else 0.0,
                "prior_loss": prior_loss.item() if labels is not None else 0.0,
                "contract_loss": (contract_coef * contract_loss).item() if labels is not None else 0.0,
                "z_loss_val": (z_coef * z_loss).item() if labels is not None else 0.0,
                "mtp_loss": mtp_loss.item() if labels is not None else 0.0,
                "rhen_conf": getattr(self._core_block, "_last_rhen_conf", None).mean().item() if getattr(self._core_block, "_last_rhen_conf", None) is not None else 0.0,
                "rhen_calib_loss": 0.0,  # Rhen calibration no longer enters the LM objective
            }

        # With output_loop_stats the return value is a tuple; the cache is then read from here.
        self._last_kv_cache = cache

        if output_loop_stats:
            return logits, loss, loop_stats

        if return_dict or return_dict is None:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=cache,
                hidden_states=None,
                attentions=None,
            )
        return (loss, logits) if loss is not None else logits

    def forward_rhen(
        self,
        packed_or_state: Any,
        questions_and_options: Optional[List[Tuple[str, List[str]]]] = None,
        tokenizer = None,
        num_loops: Optional[int] = None,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Differentiable forward pass for Rhen System One decisions.
        Extracts multi-layer representations (shallow, middle, deep) and passes them to
        the MultiHeadRhenPointerHead to return differentiable probability and logit tensors.
        """
        try:
            from rhen.rhen_packer import PackedRhenSequence, get_rhen_packer
        except ImportError:
            from rhen_packer import PackedRhenSequence, get_rhen_packer

        if isinstance(packed_or_state, PackedRhenSequence):
            packed = packed_or_state
        else:
            packer = get_rhen_packer(tokenizer)
            packed = packer.pack(state=packed_or_state, questions_and_options=questions_and_options or [])

        input_ids = packed.input_ids.to(self.embed_tokens.weight.device)
        attention_mask = packed.attention_mask.to(self.embed_tokens.weight.device)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Reset recurrent special states
        self.recurrent_core.reset_special_parameters()

        # Input Embedding & RoPE
        hidden_states = self.embed_tokens(input_ids)
        if getattr(packed, "depth_ids", None) is not None and len(packed.depth_ids) > 0:
            d_ids = packed.depth_ids.to(device) if isinstance(packed.depth_ids, torch.Tensor) else torch.tensor(packed.depth_ids, device=device)
            if d_ids.ndim == 1:
                d_ids = d_ids.unsqueeze(0)
            hidden_states = hidden_states + self.depth_embed(d_ids[:, :seq_len].clamp(0, 15))

        # Continuous Latent Injection (Pillar 1)
        if getattr(packed, "latent_tensors", None) is not None and getattr(packed, "latent_slot_indices", None) is not None:
            latent_tensors = packed.latent_tensors
            slot_indices = packed.latent_slot_indices
            if len(slot_indices) > 0:
                raw_tensors = []
                if isinstance(latent_tensors, dict):
                    raw_tensors = list(latent_tensors.values())
                elif isinstance(latent_tensors, (list, tuple)):
                    raw_tensors = list(latent_tensors)
                elif isinstance(latent_tensors, torch.Tensor):
                    raw_tensors = [latent_tensors]

                slot_tensors = []
                for t in raw_tensors:
                    if not isinstance(t, torch.Tensor):
                        continue
                    t = t.to(device=device, dtype=hidden_states.dtype)
                    if t.ndim == 3:
                        for s in range(t.shape[1]):
                            slot_tensors.append(t[:, s, :])
                    elif t.ndim == 2:
                        if t.shape[0] > 1 and len(slot_indices) > 1 and batch_size == 1:
                            for s in range(t.shape[0]):
                                slot_tensors.append(t[s:s+1, :])
                        else:
                            slot_tensors.append(t)
                    elif t.ndim == 1:
                        slot_tensors.append(t.unsqueeze(0))

                hidden_states = hidden_states.clone()
                for i, slot_idx in enumerate(slot_indices):
                    if slot_idx < seq_len and i < len(slot_tensors):
                        raw_slot = slot_tensors[i]
                        if raw_slot.shape[0] == 1 and batch_size > 1:
                            raw_slot = raw_slot.expand(batch_size, -1)
                        if raw_slot.shape[-1] == self.config.hidden_size:
                            proj_slot = self.continuous_latent_adapter.norm(raw_slot)
                        else:
                            proj_slot = self.continuous_latent_adapter(raw_slot)
                        if proj_slot.ndim == 3:
                            proj_slot = proj_slot.squeeze(1)
                        hidden_states[:, slot_idx, :] = proj_slot

        cos_full, sin_full = self.rope_emb(hidden_states, seq_len)
        cos = cos_full[:seq_len]
        sin = sin_full[:seq_len]

        # State descriptor priming for hypothesis slots
        if packed.state_length > 0 and self._core_block.rhen is not None:
            state_slice = hidden_states[:, :packed.state_length, :]
            self._core_block.rhen.prime_slots(state_slice, batch_size=batch_size)

        # Pre-Layers -> Shallow representation H_shallow
        for layer in self.pre_layers:
            hidden_states, _, _ = layer(hidden_states, cos, sin, attention_mask=attention_mask)
        h_shallow = hidden_states

        # Recurrent Core with 2D Block-Causal Masking -> Middle representation H_middle
        max_loops = num_loops or self.config.max_recurrent_loops
        residual_scale = 1.0 / math.sqrt(max_loops)
        for loop_idx in range(1, max_loops + 1):
            hidden_states, _ = self.recurrent_core(
                hidden_states,
                step_idx=loop_idx,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                residual_scale=residual_scale,
                use_rhen=True,
            )
        h_middle = hidden_states

        # Post-Layers & Norm -> Deep representation H_deep
        for layer in self.post_layers:
            hidden_states, _, _ = layer(hidden_states, cos, sin, attention_mask=attention_mask)
        h_deep = self.norm(hidden_states)  # [1, seq_len, hidden_dim]

        # Non-autoregressive decision extraction per branch using Hierarchical Multi-Layer Latent Fusion
        rhen_engine = self._core_block.rhen
        pointer_head = rhen_engine.pointer_head if rhen_engine is not None else MultiHeadRhenPointerHead(self.config.hidden_size).to(device)

        results = []
        state_len = packed.state_length
        kv_state_latent = h_middle[:, :state_len, :] if state_len > 0 else None

        for b_idx, branch in enumerate(packed.branches):
            decide_pos = branch.decide_idx
            opt_end_positions = branch.option_end_indices
            num_opts = len(opt_end_positions)
            prim_type = getattr(branch, "primitive_type", "choice")
            meta = getattr(branch, "metadata", None) or {}

            # Hierarchical multi-layer query representation [shallow, middle, deep]
            q_layers = [
                h_shallow[:, decide_pos:decide_pos+1, :],
                h_middle[:, decide_pos:decide_pos+1, :],
                h_deep[:, decide_pos:decide_pos+1, :],
            ]
            q_fused = pointer_head.fuse_layers(q_layers)

            if prim_type == "range":
                min_v = float(meta.get("min_val", 0.0))
                max_v = float(meta.get("max_val", 100.0))
                range_info = pointer_head.compute_range(q_fused, min_val=min_v, max_val=max_v)
                expected_val = float(range_info["expected_value"][0].item())
                int_low = float(range_info["interval_low"][0].item())
                int_high = float(range_info["interval_high"][0].item())
                conf = float(range_info["confidence"][0].item())
                schema_logits = pointer_head.schema_mutation_head(q_fused)

                branch_res = {
                    "branch_index": b_idx,
                    "question": branch.question_text,
                    "primitive": "range",
                    "expected_value": expected_val,
                    "interval_low": int_low,
                    "interval_high": int_high,
                    "min_val": min_v,
                    "max_val": max_v,
                    "confidence": conf,
                    "temperature": 1.0,
                    "probabilities": [expected_val],
                    # A Beta interval has no logits, so it carries no scores and no energy.
                    "alpha_tensor": range_info["alpha"],
                    "beta_tensor": range_info["beta"],
                    "free_energy": None,
                    "schema_logits": schema_logits,
                }
                results.append(branch_res)

            elif prim_type == "extract":
                slot_name = meta.get("slot_name", branch.question_text)
                schema_logits = pointer_head.schema_mutation_head(q_fused)
                if state_len > 0:
                    state_fused = pointer_head.fuse_layers([
                        h_shallow[:, :state_len, :],
                        h_middle[:, :state_len, :],
                        h_deep[:, :state_len, :],
                    ])
                    start_logits, end_logits = pointer_head.compute_extract_spans(q_fused, state_fused)
                    p_start = F.softmax(start_logits, dim=-1)
                    p_end = F.softmax(end_logits, dim=-1)

                    i_best = int(torch.argmax(p_start[0]).item())
                    j_candidates = p_end[0, i_best:]
                    j_best = i_best + int(torch.argmax(j_candidates).item())
                    span_conf = float((p_start[0, i_best] * p_end[0, j_best]).item())

                    extracted_ids = input_ids[0, i_best:j_best + 1].tolist()
                    if tokenizer is not None and hasattr(tokenizer, "decode"):
                        extracted_str = tokenizer.decode(extracted_ids).strip()
                    else:
                        extracted_str = f"span_{i_best}_{j_best}"
                    extract_fe = float((-0.5 * (torch.logsumexp(start_logits, dim=-1) + torch.logsumexp(end_logits, dim=-1)))[0].item())
                else:
                    i_best, j_best, span_conf = 0, 0, 0.5
                    extracted_str = ""
                    start_logits = torch.zeros(1, 1, device=device)
                    end_logits = torch.zeros(1, 1, device=device)
                    extract_fe = None  # no start/end logits in this branch, so no energy

                branch_res = {
                    "branch_index": b_idx,
                    "question": branch.question_text,
                    "primitive": "extract",
                    "slot_name": slot_name,
                    "extracted_value": extracted_str,
                    "start_idx": i_best,
                    "end_idx": j_best,
                    "confidence": span_conf,
                    "temperature": 1.0,
                    "probabilities": [span_conf],
                    # The span's energy comes from start/end logits only; no scores key here.
                    "start_logits": start_logits,
                    "end_logits": end_logits,
                    "free_energy": extract_fe,
                    "schema_logits": schema_logits,
                }
                results.append(branch_res)

            elif prim_type == "multichoice":
                opt_layers = [
                    _option_span_reprs(h_shallow, branch),
                    _option_span_reprs(h_middle, branch),
                    _option_span_reprs(h_deep, branch),
                ]
                details = pointer_head(q_layers, opt_layers, kv_latent=kv_state_latent, return_details=True)
                raw_scores = details["scores"]
                tau = details["temperature"]
                thresh_override = meta.get("threshold", 0.5)

                probs, theta = pointer_head.compute_multichoice(
                    raw_scores, tau, q_fused, threshold_override=thresh_override
                )
                prob_list = probs[0].detach().cpu().tolist()
                thresh_val = float(theta.item()) if theta.numel() == 1 else thresh_override

                selected_indices = [k for k, p in enumerate(prob_list) if p >= thresh_val]
                selected_options = [branch.options[k] for k in selected_indices if k < len(branch.options)]
                margin_conf = float(torch.mean(torch.abs(probs[0] - 0.5) * 2.0).item())

                branch_res = {
                    "branch_index": b_idx,
                    "question": branch.question_text,
                    "options": branch.options,
                    "primitive": "multichoice",
                    "selected_indices": selected_indices,
                    "selected_options": selected_options,
                    "threshold": thresh_val,
                    "probs_tensor": probs,
                    "logits_tensor": raw_scores,
                    "scores_tensor": raw_scores,
                    "temperature_tensor": tau,
                    "probabilities": prob_list,
                    "scores": raw_scores[0].detach().cpu().tolist(),
                    "temperature": float(tau[0].item()),
                    "confidence": margin_conf,
                    "chosen_index": selected_indices[0] if selected_indices else 0,
                    "chosen_option": selected_options[0] if selected_options else (branch.options[0] if branch.options else None),
                    "free_energy": float(details["free_energy"][0].item()),
                    "schema_logits": details["schema_logits"],
                    # Live multi-layer tensors for DDPO / permutation-invariance losses
                    "q_layers": q_layers,
                    "opt_layers": opt_layers,
                    "kv_state_latent": kv_state_latent,
                }
                results.append(branch_res)

            else:
                # Hierarchical multi-layer option representations [shallow, middle, deep]
                opt_layers = [
                    _option_span_reprs(h_shallow, branch),
                    _option_span_reprs(h_middle, branch),
                    _option_span_reprs(h_deep, branch),
                ]

                details = pointer_head(q_layers, opt_layers, kv_latent=kv_state_latent, return_details=True)
                probs = details["probabilities"]  # [B, NumOptions]
                logits = details["calibrated_scores"]  # [B, NumOptions]
                raw_scores = details["scores"]  # [B, NumOptions]
                tau = details["temperature"]  # [B, 1]

                chosen_idx = int(torch.argmax(probs[0]).item())

                # Official TypeSafe confidence calculation
                p_max = float(torch.max(probs[0]).item())
                K = max(num_opts, 2)
                c_typesafe = max(0.0, min(1.0, (p_max - 1.0 / K) / (1.0 - 1.0 / K)))

                branch_res = {
                    "branch_index": b_idx,
                    "question": branch.question_text,
                    "primitive": prim_type,
                    "options": branch.options,
                    "chosen_index": chosen_idx,
                    "chosen_option": branch.options[chosen_idx] if chosen_idx < len(branch.options) else None,
                    "probs_tensor": probs,
                    "logits_tensor": logits,
                    "scores_tensor": raw_scores,
                    "temperature_tensor": tau,
                    "probabilities": probs[0].detach().cpu().tolist(),
                    "scores": raw_scores[0].detach().cpu().tolist(),
                    "temperature": float(tau[0].item()),
                    "confidence": p_max,
                    "typesafe_confidence": c_typesafe,
                    "free_energy": float(details["free_energy"][0].item()),
                    "schema_logits": details["schema_logits"],
                    # Live multi-layer tensors for DDPO / permutation-invariance losses
                    "q_layers": q_layers,
                    "opt_layers": opt_layers,
                    "kv_state_latent": kv_state_latent,
                }
                results.append(branch_res)

        if getattr(packed, "is_speculative_tree", False):
            root_result = results[0] if len(results) > 0 else {}
            speculative_results: Dict[Any, List[Dict[str, Any]]] = {}
            for b_idx in range(1, len(results)):
                branch_meta = packed.branches[b_idx]
                cond_idx = branch_meta.condition_option_idx
                if cond_idx is None:
                    cond_idx = 0
                if cond_idx not in speculative_results:
                    speculative_results[cond_idx] = []
                speculative_results[cond_idx].append(results[b_idx])
            return {
                "root_result": root_result,
                "speculative_results": speculative_results,
                "all_results": results,
            }

        return results

    @torch.no_grad()
    def evaluate_rhen(
        self,
        packed_or_state: Any,
        questions_and_options: Optional[List[Tuple[str, List[str]]]] = None,
        tokenizer = None,
        num_loops: Optional[int] = None,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Evaluates multiple typed questions (Noul, Choice, Score) in a single forward pass
        over shared state using 2D block-causal branch masking and multi-head pointer heads.

        Args:
            packed_or_state: Either a pre-constructed `PackedRhenSequence` or a raw state (str, dict, list).
            questions_and_options: Optional list of tuples `(question_text, [option_candidates])` if passing raw state.
            tokenizer: Optional custom tokenizer instance.
            num_loops: Optional recurrent loop override (defaults to config.max_recurrent_loops).

        Returns:
            List of decision results per branch with probabilities, chosen option, confidence, and calibrated scores.
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward_rhen(
                packed_or_state=packed_or_state,
                questions_and_options=questions_and_options,
                tokenizer=tokenizer,
                num_loops=num_loops,
            )
        finally:
            self.train(was_training)

    def simulate_latent_rollout(
        self,
        state_repr: torch.Tensor,
        option_reprs: torch.Tensor,
        horizon: int = 2,
    ) -> Dict[str, Any]:
        """
        Pillar 2: Latent World-Model Simulation (Simulate Latent Rollout).
        For each candidate option in option_reprs, rolls forward through
        self.latent_transition_operator for horizon steps, evaluating prospective
        values at each step with self.prospective_value_head.

        Args:
            state_repr: [hidden_size], [1, hidden_size], [B, hidden_size], or [B, 1, hidden_size]
            option_reprs: [NumOptions, hidden_size], [1, NumOptions, hidden_size], or [B, NumOptions, hidden_size]
            horizon: Number of forward simulation steps (default 2).

        Returns:
            Dict containing:
                - "trajectories": List of predicted latent state trajectories per option [B, horizon + 1, hidden_size]
                - "predicted_trajectories": List of predicted latent state trajectories per option
                - "prospective_values": Prospective value scores per option [B, NumOptions]
                - "step_values": Value progression per step per option
                - "recommended_option": Index of the highest prospective value option
        """
        device = next(self.parameters()).device

        # Align state_repr: ensure shape [B, hidden_size]
        if state_repr.ndim == 1:
            z_init = state_repr.unsqueeze(0).to(device)
        elif state_repr.ndim == 3:
            z_init = state_repr.squeeze(1).to(device)
        else:
            z_init = state_repr.to(device)

        batch_size = z_init.shape[0]

        # Align option_reprs: ensure shape [B, NumOptions, hidden_size]
        opts = option_reprs.to(device)
        if opts.ndim == 2:
            opts = opts.unsqueeze(0)  # [1, NumOptions, hidden_size]
        if opts.shape[0] == 1 and batch_size > 1:
            opts = opts.expand(batch_size, -1, -1)

        num_options = opts.shape[1]

        all_trajectories: List[torch.Tensor] = []
        all_step_values: List[torch.Tensor] = []
        final_values: List[torch.Tensor] = []

        for i in range(num_options):
            opt_repr = opts[:, i, :]  # [B, hidden_size]
            z_curr = z_init
            traj = [z_curr]
            step_vals = []

            for _ in range(horizon):
                z_curr = self.latent_transition_operator(z_curr, opt_repr)  # [B, hidden_size]
                val = self.prospective_value_head(z_curr)  # [B]
                traj.append(z_curr)
                step_vals.append(val)

            traj_tensor = torch.stack(traj, dim=1)  # [B, horizon + 1, hidden_size]
            all_trajectories.append(traj_tensor)

            if step_vals:
                vals_tensor = torch.stack(step_vals, dim=-1)  # [B, horizon]
                all_step_values.append(vals_tensor)
                final_values.append(vals_tensor[:, -1])  # [B]
            else:
                v0 = self.prospective_value_head(z_curr)
                final_values.append(v0)

        prospective_values = torch.stack(final_values, dim=-1)  # [B, NumOptions]
        recommended_idx = int(torch.argmax(prospective_values[0]).item())

        return {
            "trajectories": all_trajectories,
            "predicted_trajectories": all_trajectories,
            "prospective_values": prospective_values,
            "step_values": all_step_values,
            "recommended_option": recommended_idx,
        }

    @torch.no_grad()
    def decide(
        self,
        input_ids: torch.Tensor,
        option_embeddings: Optional[torch.Tensor] = None,
        num_choices: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes a pure non-autoregressive System One parallel decision pass (Rhen architecture).
        Returns normalized probabilities summing to 1.0 and calibrated confidence without generating tokens.

        The recurrent core runs the full loop budget on the Rhen decision path. Without
        the Rhen engine, ``option_embeddings`` must be supplied. ``num_choices`` is kept
        for backward compatibility and has no effect.
        """
        rhen_engine = getattr(self._core_block, "rhen", None)
        if option_embeddings is None and rhen_engine is None:
            raise ValueError("decide() needs option_embeddings when the Rhen engine is disabled")

        was_training = self.training
        self.eval()
        try:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

            hidden_states = self.embed_tokens(input_ids)
            cos_full, sin_full = self.rope_emb(hidden_states, seq_len)
            cos = cos_full[:seq_len]
            sin = sin_full[:seq_len]

            for layer in self.pre_layers:
                hidden_states, _, _ = layer(hidden_states, cos, sin)

            max_loops = self.config.max_recurrent_loops
            for loop_idx in range(1, max_loops + 1):
                hidden_states, _ = self.recurrent_core(
                    hidden_states,
                    step_idx=loop_idx,
                    cos=cos,
                    sin=sin,
                    residual_scale=1.0 / math.sqrt(max_loops),
                    use_rhen=True,
                )
            context_vector = self.norm(hidden_states).mean(dim=1, keepdim=True)  # [B, 1, HiddenDim]

            if option_embeddings is None:
                option_embeddings = rhen_engine.out_norm(rhen_engine.out_proj(rhen_engine.d_slots_base)).expand(batch_size, -1, -1)

            if rhen_engine is not None:
                out = rhen_engine.score_options(context_vector, option_embeddings)
            else:
                scores = torch.matmul(option_embeddings, context_vector.transpose(-1, -2)).squeeze(-1)
                probs = F.softmax(scores / math.sqrt(self.config.hidden_size), dim=-1)
                out = {"probabilities": probs, "confidence": torch.ones(batch_size, 1, device=device) * 0.9}

            out["decision_index"] = torch.argmax(out["probabilities"], dim=-1)
            return out
        finally:
            self.train(was_training)

    @torch.no_grad()
    def decide_noul(self, state: Any, proposition: str, tokenizer = None) -> Dict[str, Any]:
        """
        Noul Primitive: Evaluates a binary semantic truth proposition (Yes/No).
        Returns probability of truth P in [0, 1].
        """
        results = self.evaluate_rhen(
            packed_or_state=state,
            questions_and_options=[(proposition, ["Yes", "No"])],
            tokenizer=tokenizer,
        )
        res = results[0]
        p_true = res["probabilities"][0]
        return {
            "primitive": "noul",
            "proposition": proposition,
            "p_true": p_true,
            "decision": bool(p_true >= 0.5),
            "confidence": abs(p_true - 0.5) * 2.0,
            "temperature": res["temperature"],
        }

    @torch.no_grad()
    def decide_choice(self, state: Any, question: str, options: List[str], tokenizer = None) -> Dict[str, Any]:
        """
        Choice Primitive: Evaluates categorical option selection (1 of K).
        Returns winning option, probability distribution summing to 1.0, and confidence.
        """
        results = self.evaluate_rhen(
            packed_or_state=state,
            questions_and_options=[(question, options)],
            tokenizer=tokenizer,
        )
        res = results[0]
        K = max(len(options), 2)
        p_max = float(res["confidence"])
        typesafe_conf = max(0.0, min(1.0, (p_max - 1.0 / K) / (1.0 - 1.0 / K)))
        return {
            "primitive": "choice",
            "question": question,
            "choice": res["chosen_index"],
            "choice_label": res["chosen_option"],
            "probabilities": res["probabilities"],
            "confidence": typesafe_conf,
            "temperature": res["temperature"],
        }

    @torch.no_grad()
    def decide_score(
        self,
        state: Any,
        question: str,
        levels: Optional[List[str]] = None,
        min_score: float = 0.0,
        max_score: float = 10.0,
        tokenizer = None,
    ) -> Dict[str, Any]:
        """
        Score Primitive: Evaluates an ordered discrete spectrum of levels.
        Returns expected continuous score, level probability distribution, and confidence.
        """
        if levels is None:
            levels = [f"Level {i}" for i in range(5)]
        results = self.evaluate_rhen(
            packed_or_state=state,
            questions_and_options=[(question, levels)],
            tokenizer=tokenizer,
        )
        res = results[0]
        probs = res["probabilities"]
        num_levels = len(probs)
        # Expected continuous score mapping to [min_score, max_score]
        norm_weights = [i / max(num_levels - 1, 1) for i in range(num_levels)]
        expected_norm = sum(p * w for p, w in zip(probs, norm_weights))
        expected_score = min_score + expected_norm * (max_score - min_score)

        # Modal ordinal dispersion confidence
        k_mode = int(max(range(num_levels), key=lambda i: probs[i])) if probs else 0
        disp_mode = sum(p * abs(k - k_mode) for k, p in enumerate(probs))
        disp_uniform = sum(abs(k - k_mode) for k in range(num_levels)) / max(num_levels, 1)
        score_conf = max(0.0, min(1.0, 1.0 - (disp_mode / max(disp_uniform, 1e-6))))

        return {
            "primitive": "score",
            "question": question,
            "score": expected_score,
            "min_score": min_score,
            "max_score": max_score,
            "level_probabilities": probs,
            "confidence": score_conf,
            "temperature": res["temperature"],
        }

    @torch.no_grad()
    def decide_multichoice(
        self,
        state: Any,
        question: str,
        options: List[str],
        threshold: float = 0.5,
        tokenizer = None,
    ) -> Dict[str, Any]:
        """
        MultiChoice Primitive: Evaluates independent multi-label selection above calibrated threshold.
        """
        results = self.evaluate_rhen(
            packed_or_state=state,
            questions_and_options=[(question, options, "multichoice", {"threshold": threshold})],
            tokenizer=tokenizer,
        )
        res = results[0]
        prob_dict = {opt: p for opt, p in zip(options, res["probabilities"])}
        return {
            "primitive": "multichoice",
            "question": question,
            "selected_indices": res.get("selected_indices", []),
            "selected_options": res.get("selected_options", []),
            "probabilities": prob_dict,
            "raw_probabilities": res["probabilities"],
            "confidence": res["confidence"],
            "threshold": res.get("threshold", threshold),
            "temperature": res["temperature"],
        }

    @torch.no_grad()
    def decide_range(
        self,
        state: Any,
        question: str,
        min_val: float = 0.0,
        max_val: float = 100.0,
        tokenizer = None,
    ) -> Dict[str, Any]:
        """
        Range Primitive: Evaluates bounded continuous interval regression [min_val, max_val].
        """
        results = self.evaluate_rhen(
            packed_or_state=state,
            questions_and_options=[(question, [str(min_val), str(max_val)], "range", {"min_val": min_val, "max_val": max_val})],
            tokenizer=tokenizer,
        )
        res = results[0]
        return {
            "primitive": "range",
            "question": question,
            "expected_value": res.get("expected_value", (min_val + max_val) / 2.0),
            "interval_low": res.get("interval_low", min_val),
            "interval_high": res.get("interval_high", max_val),
            "min_val": min_val,
            "max_val": max_val,
            "confidence": res["confidence"],
            "temperature": res["temperature"],
        }

    @torch.no_grad()
    def decide_extract(
        self,
        state: Any,
        slot_name: str,
        query: str = "",
        tokenizer = None,
    ) -> Dict[str, Any]:
        """
        Extract Primitive: Non-autoregressively extracts structured token span for slot_name from state.
        """
        q_text = query or slot_name
        results = self.evaluate_rhen(
            packed_or_state=state,
            questions_and_options=[(q_text, [], "extract", {"slot_name": slot_name})],
            tokenizer=tokenizer,
        )
        res = results[0]
        return {
            "primitive": "extract",
            "slot_name": slot_name,
            "extracted_value": res.get("extracted_value", ""),
            "start_idx": res.get("start_idx", 0),
            "end_idx": res.get("end_idx", 0),
            "confidence": res["confidence"],
            "temperature": res["temperature"],
        }

    @torch.no_grad()
    def decide_parallel(
        self,
        state: Any,
        questions_and_options: List[Tuple[str, List[str]]],
        tokenizer = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates multiple typed questions concurrently over a single state
        in one forward pass using 2D block-causal branch masking.
        """
        return self.evaluate_rhen(
            packed_or_state=state,
            questions_and_options=questions_and_options,
            tokenizer=tokenizer,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        num_loops: Optional[int] = None,
        use_kv_cache: Optional[bool] = None,
        decision_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Autoregressive sampling.

        ``use_kv_cache``: ``None`` (default) uses the cache whenever the configuration supports
        it (``supports_kv_cache``), ``True`` requires it and raises with the reasons when it is
        not supported, and ``False`` recomputes the full prefix for every token. With the cache,
        the loop count is fixed at prefill (pass ``num_loops`` to choose it), so the output
        equals uncached decoding with the same ``num_loops``. The model runs in eval mode for
        the call and its previous training mode is restored afterwards.
        """
        was_training = self.training
        self.eval()
        try:
            problems = self.kv_cache_incompatibilities()
            if use_kv_cache and problems:
                raise NotImplementedError("KV cache is not supported here: " + "; ".join(problems))
            cached = bool(use_kv_cache) if use_kv_cache is not None else not problems
            if self.speculative_decoder is not None and self.config.enable_speculative_decoding:
                cached = False  # the self-speculative decoder runs its own verification passes
            generate_fn = self._generate_tokens_cached if cached else self._generate_tokens
            return generate_fn(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
                num_loops=num_loops,
                decision_latent=decision_latent,
            )
        finally:
            self.train(was_training)

    def _sample_next_token(
        self,
        next_token_logits: torch.Tensor,
        generated: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Pick the next token from last-position logits ([B, V]); shared by both decoders."""
        next_token_logits = next_token_logits / max(temperature, 1e-5)

        if repetition_penalty != 1.0:
            for b in range(generated.shape[0]):
                seen = torch.unique(generated[b])
                next_token_logits[b] = apply_repetition_penalty(next_token_logits[b], seen, repetition_penalty)

        if top_k is not None and 0 < top_k < next_token_logits.shape[-1]:
            kth_val = torch.topk(next_token_logits, top_k, dim=-1).values[:, -1:]
            next_token_logits[next_token_logits < kth_val] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        if temperature < 1e-4:
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)
        probs = F.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def _generate_tokens_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        num_loops: Optional[int] = None,
        decision_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """KV-cached decoding: one prefill pass, then one position per step."""
        generated = input_ids.clone()
        gen_stats: List[Dict] = []
        max_pos = int(self.config.max_position_embeddings)
        cache: Optional[EOSKVCache] = None
        next_input = generated[:, -max_pos:]

        for _ in range(max_new_tokens):
            if cache is not None and cache.seq_len + next_input.shape[1] > max_pos:
                # The context window is full: refill the cache from the most recent window,
                # which is what the uncached decoder sees at this point.
                cache = None
                next_input = generated[:, -max_pos:]
            prefill = cache is None
            out = self.forward(
                next_input,
                num_loops=num_loops if prefill else None,
                past_key_values=cache,
                use_cache=True,
                decision_latent=decision_latent,
                return_dict=True,
            )
            cache = out.past_key_values
            next_token = self._sample_next_token(
                out.logits[:, -1, :], generated, temperature, top_p, top_k, repetition_penalty
            )
            generated = torch.cat([generated, next_token], dim=-1)
            gen_stats.append({
                "kv_cache": True,
                "prefill": prefill,
                "max_configured_loops": cache.num_loops,
                "cached_tokens": cache.seq_len,
            })
            next_input = next_token

            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        return generated, gen_stats

    @torch.no_grad()
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        num_loops: Optional[int] = None,
        decision_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        if self.speculative_decoder is not None and self.config.enable_speculative_decoding:
            generated = self.speculative_decoder(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            return generated, [{"speculative_decoding": True}]

        generated = input_ids.clone()
        gen_stats: List[Dict] = []

        for _ in range(max_new_tokens):
            curr_input = generated[:, -self.config.max_position_embeddings:]

            logits, _, stats = self.forward(
                curr_input,
                num_loops=num_loops,
                output_loop_stats=True,
                decision_latent=decision_latent,
            )

            next_token = self._sample_next_token(
                logits[:, -1, :], generated, temperature, top_p, top_k, repetition_penalty
            )

            generated = torch.cat([generated, next_token], dim=-1)
            gen_stats.append(stats)

            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        return generated, gen_stats


# ---------------------------------------------------------------------------
# Backward-compatibility alias.
# EOS was previously published as "EOS Looped LLM"; the canonical public class
# name is now `EOSModel`. The legacy name is retained so existing imports and
# saved checkpoints continue to resolve.
# ---------------------------------------------------------------------------
EOSLoopedLLM = EOSModel
RhenModel = EOSModel

__all__ = [
    "RhenModel",
    "EOSModel",
    "EOSLoopedLLM",
    "MultiHeadRhenPointerHead",
    "RhenLatentDecisionEngine",
    "ContinuousLatentAdapter",
    "LatentTransitionOperator",
    "ProspectiveValueHead",
    "ModernHopfieldEpisodicBank",
    "SchemaMutationHead",
]
