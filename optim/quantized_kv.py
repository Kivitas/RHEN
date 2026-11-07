import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import warnings

class QuantizedKVCache(nn.Module):
    """
    FP8/INT8 Quantized KV Cache for Multi-Latent Attention (MLA).
    
    Implements efficient quantization/dequantization of key-value caches
    for inference with reduced memory footprint. Supports FP8 (e4m3fn) 
    and INT8 quantization with per-tensor scaling.
    
    Features:
    - Per-tensor scale factors for better precision
    - On-the-fly dequantization during attention computation
    - Compatible with MLA latent KV format
    - Automatic fallback for CPU environments
    """
    
    def __init__(
        self,
        kv_lora_rank: int,
        num_heads: int,
        head_dim: int,
        quant_dtype: str = "fp8",  # "fp8" or "int8"
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.quant_dtype = quant_dtype
        self.device = device if device is not None else torch.device("cpu")
        
        # Scale factors for quantization (one per tensor)
        self.register_buffer("k_scale", torch.tensor(1.0, device=self.device))
        self.register_buffer("v_scale", torch.tensor(1.0, device=self.device))
        
        # Check quantization support
        self.fp8_supported = hasattr(torch, 'float8_e4m3fn') and torch.cuda.is_available()
        self.int8_supported = torch.cuda.is_available()  # INT8 is widely supported
        
        if quant_dtype == "fp8" and not self.fp8_supported:
            warnings.warn(
                "FP8 quantization not supported on this device. "
                "Falling back to INT8 or FP16."
            )
            self.quant_dtype = "int8" if self.int8_supported else "fp16"
        elif quant_dtype == "int8" and not self.int8_supported:
            warnings.warn(
                "INT8 quantization not supported on this device. "
                "Falling back to FP16."
            )
            self.quant_dtype = "fp16"
    
    def quantize(self, x: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to the target dtype with per-tensor scaling.
        
        Args:
            x: Input tensor to quantize
            scale: Scale factor for quantization
            
        Returns:
            quantized: Quantized tensor
            scale: Updated scale factor
        """
        if self.quant_dtype == "fp8" and self.fp8_supported:
            # FP8 E4M3 quantization
            max_val = x.abs().max()
            scale = max_val / 448.0  # FP8 E4M3 max representable value ~448
            scale = scale.clamp(min=1e-8)
            
            quantized = (x / scale).to(torch.float8_e4m3fn)
            return quantized, scale
            
        elif self.quant_dtype == "int8" and self.int8_supported:
            # INT8 dynamic quantization
            max_val = x.abs().max()
            scale = max_val / 127.0
            scale = scale.clamp(min=1e-8)
            
            quantized = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
            return quantized, scale
            
        else:
            # No quantization, return original
            return x, torch.tensor(1.0, device=x.device, dtype=x.dtype)
    
    def dequantize(
        self, 
        quantized: torch.Tensor, 
        scale: torch.Tensor, 
        target_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """
        Dequantize a tensor back to the target dtype.
        
        Args:
            quantized: Quantized tensor
            scale: Scale factor used during quantization
            target_dtype: Target dtype for dequantization
            
        Returns:
            Dequantized tensor in target_dtype
        """
        if self.quant_dtype == "fp8" and self.fp8_supported:
            return (quantized.to(target_dtype) * scale).to(target_dtype)
        elif self.quant_dtype == "int8" and self.int8_supported:
            return (quantized.to(target_dtype) * scale).to(target_dtype)
        else:
            return quantized.to(target_dtype)
    
    def quantize_kv_cache(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Quantize key and value cache tensors.
        
        Args:
            k: Key tensor [B, H, S, D]
            v: Value tensor [B, H, S, D]
            
        Returns:
            quantized_cache: Tuple of (quantized_k, quantized_v)
            scales: Tuple of (k_scale, v_scale)
        """
        # Per-head channel-wise scaling along trailing dimensions
        max_val = 448.0 if self.quant_dtype == "fp8" else 127.0
        if k.dim() >= 3:
            k_scale = k.abs().amax(dim=(-2, -1), keepdim=True) / max_val
            v_scale = v.abs().amax(dim=(-2, -1), keepdim=True) / max_val
        else:
            k_scale = k.abs().max() / max_val
            v_scale = v.abs().max() / max_val

        k_scale = k_scale.clamp(min=1e-8)
        v_scale = v_scale.clamp(min=1e-8)
        
        if self.quant_dtype == "fp8" and self.fp8_supported:
            quantized_k = (k / k_scale).to(torch.float8_e4m3fn)
            quantized_v = (v / v_scale).to(torch.float8_e4m3fn)
        elif self.quant_dtype == "int8" and self.int8_supported:
            quantized_k = torch.round(k / k_scale).clamp(-128, 127).to(torch.int8)
            quantized_v = torch.round(v / v_scale).clamp(-128, 127).to(torch.int8)
        else:
            return (k, v), (torch.tensor(1.0, device=k.device, dtype=k.dtype), 
                         torch.tensor(1.0, device=v.device, dtype=v.dtype))
        
        return (quantized_k, quantized_v), (k_scale, v_scale)
    
    def dequantize_kv_cache(
        self,
        quantized_cache: Tuple[torch.Tensor, torch.Tensor],
        scales: Tuple[torch.Tensor, torch.Tensor],
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize key and value cache tensors.
        
        Args:
            quantized_cache: Tuple of (quantized_k, quantized_v)
            scales: Tuple of (k_scale, v_scale)
            target_dtype: Target dtype for dequantization
            
        Returns:
            Dequantized (k, v) tensors
        """
        quantized_k, quantized_v = quantized_cache
        k_scale, v_scale = scales
        
        if self.quant_dtype == "fp8" and self.fp8_supported:
            k = (quantized_k.to(target_dtype) * k_scale).to(target_dtype)
            v = (quantized_v.to(target_dtype) * v_scale).to(target_dtype)
        elif self.quant_dtype == "int8" and self.int8_supported:
            k = (quantized_k.to(target_dtype) * k_scale).to(target_dtype)
            v = (quantized_v.to(target_dtype) * v_scale).to(target_dtype)
        else:
            return quantized_k.to(target_dtype), quantized_v.to(target_dtype)
        
        return k, v


class QuantizedMultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention with quantized KV cache support.
    
    Extends the standard MLA with optional FP8/INT8 KV cache quantization
    for inference memory savings.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        
        # Projections
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, 
            self.kv_lora_rank + self.head_dim, 
            bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, 
            self.num_heads * (self.head_dim + self.head_dim), 
            bias=False
        )
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.o_proj._residual_out = True
        
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        
        # Quantized KV cache
        self.use_quantized_kv = config.quantized_kv_cache
        if self.use_quantized_kv:
            self.kv_quantizer = QuantizedKVCache(
                kv_lora_rank=config.kv_lora_rank,
                num_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                quant_dtype=config.kv_quant_dtype,
            )
        else:
            self.kv_quantizer = None
    
    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard attention computation with optional SDPA."""
        target_dtype = q.dtype
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)
        
        can_sdpa = (
            self.config.use_sdpa
            and attention_mask is None
            and q.shape[2] == k.shape[2]
        )
        
        if can_sdpa:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
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
        kv_cache: Optional[Union[
            Tuple[torch.Tensor, torch.Tensor],  # Standard cache
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]  # Quantized cache
        ]] = None,
        is_inference: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query computation
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_latent).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Key-Value computation (compressed)
        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rope = torch.split(
            kv_compressed, 
            [self.kv_lora_rank, self.head_dim], 
            dim=-1
        )
        kv_latent = self.kv_a_layernorm(kv_latent)
        
        kv_expanded = self.kv_b_proj(kv_latent).view(
            batch_size, seq_len, self.num_heads, self.head_dim * 2
        )
        k_unrope, v = torch.chunk(kv_expanded, 2, dim=-1)
        
        k_unrope = k_unrope.transpose(1, 2)
        v = v.transpose(1, 2)
        k_rope = k_rope.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        if self.q_norm is not None:
            q = self.q_norm(q)
            k_unrope = self.k_norm(k_unrope)
        
        # Apply rotary embeddings
        q, k_rope = apply_rotary_pos_emb(q, k_rope, cos, sin)
        k = k_unrope + k_rope
        
        # Handle KV cache (quantized or standard)
        new_kv_cache = None
        if kv_cache is not None:
            if self.use_quantized_kv and is_inference and isinstance(kv_cache[0], tuple):
                # Dequantize cached KV
                (prev_k, prev_v), (k_scale, v_scale) = kv_cache
                prev_k, prev_v = self.kv_quantizer.dequantize_kv_cache(
                    (prev_k, prev_v), (k_scale, v_scale), target_dtype=q.dtype
                )
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
                
                # Re-quantize for next step if inference
                if is_inference:
                    quantized_cache, scales = self.kv_quantizer.quantize_kv_cache(k, v)
                    new_kv_cache = (quantized_cache, scales)
            else:
                # Standard cache
                prev_k, prev_v = kv_cache
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
                
                if is_inference and self.use_quantized_kv:
                    quantized_cache, scales = self.kv_quantizer.quantize_kv_cache(k, v)
                    new_kv_cache = (quantized_cache, scales)
        else:
            if is_inference and self.use_quantized_kv:
                quantized_cache, scales = self.kv_quantizer.quantize_kv_cache(k, v)
                new_kv_cache = (quantized_cache, scales)
        
        # Standard attention
        output = self._attention(q, k, v, attention_mask)
        
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        return self.o_proj(output), new_kv_cache


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings."""
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
    if sin.ndim == 2:
        sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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


def create_quantized_mla(
    config,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> QuantizedMultiLatentAttention:
    """
    Factory function to create quantized MLA module.
    
    Args:
        config: LoopedLLMConfig containing model configuration
        dtype: Data type for computations
        device: Device to place on
        
    Returns:
        QuantizedMultiLatentAttention instance
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return QuantizedMultiLatentAttention(config).to(device=device)