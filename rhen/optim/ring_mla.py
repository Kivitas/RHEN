import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

class RingMLA(nn.Module):
    """
    Vectorized Ring-MLA (Multi-Latent Attention) for distributed context.
    
    Implements distributed KV cache sharing across multiple GPUs in a ring topology,
    enabling effective context lengths beyond single-GPU memory limits.
    
    Key features:
    - Ring-based all-gather communication for KV latents
    - Compressed latents reduce communication bandwidth
    - Compatible with standard MLA architecture
    - Single-GPU fallback for environments without distributed setup
    - NCCL backend with Gloo fallback for non-GPU environments
    """
    
    def __init__(
        self,
        kv_lora_rank: int,
        num_heads: int,
        head_dim: int,
        ring_size: int = 2,
        compression_dim: int = 32,
        backend: str = "nccl",  # "nccl", "gloo"
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ring_size = ring_size
        self.compression_dim = compression_dim
        self.backend = backend
        self.device = device if device is not None else torch.device("cpu")
        
        # Check distributed availability
        self.dist_available = False
        if torch.distributed.is_available():
            self.dist_available = True
        
        # Get process info
        self.rank = 0
        self.world_size = 1
        if self.dist_available and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        
        # Compression for ring communication
        self.compress_proj = nn.Linear(kv_lora_rank, compression_dim, bias=False)
        self.decompress_proj = nn.Linear(compression_dim, kv_lora_rank, bias=False)
        
        # Ring cache: stores compressed latents from all ranks
        self.ring_cache = None
        self.cache_size = 0
    
    def init_ring_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ):
        """Initialize ring cache for distributed KV storage."""
        if self.world_size == 1:
            # Single GPU - no ring needed
            self.ring_cache = None
            return
        
        # Allocate ring cache: [world_size, B, max_seq_len, compression_dim]
        self.ring_cache = torch.zeros(
            self.world_size, batch_size, max_seq_len, self.compression_dim,
            device=device, dtype=torch.float32,
        )
        self.cache_size = 0
    
    def gather_kv_latents(
        self,
        kv_latents: torch.Tensor,  # [B, S, kv_lora_rank]
    ) -> torch.Tensor:
        """
        Gather compressed KV latents from all ranks in the ring.
        
        Uses a ring-based all-gather pattern where each rank:
        1. Compresses its local KV latents
        2. Sends to next rank
        3. Receives from previous rank
        4. After N steps, all ranks have all latents
        
        Args:
            kv_latents: Local KV latents [B, S, kv_lora_rank]
            
        Returns:
            All KV latents from all ranks [world_size, B, S, kv_lora_rank]
        """
        if self.world_size == 1:
            # Single GPU - return local latents wrapped
            return kv_latents.unsqueeze(0)  # [1, B, S, kv_lora_rank]
        
        if not self.dist_available:
            warnings.warn(
                "Distributed not available, falling back to single-GPU mode"
            )
            return kv_latents.unsqueeze(0)
        
        # Compress local latents for efficient communication
        compressed = self.compress_proj(kv_latents.to(self.device))  # [B, S, compression_dim]
        
        # Allocate ring buffer
        B, S, _ = compressed.shape
        ring_buffer = torch.zeros(
            self.world_size, B, S, self.compression_dim,
            device=self.device, dtype=compressed.dtype,
        )
        
        # Ring all-gather
        send_tensor = compressed.contiguous()
        
        for step in range(self.world_size - 1):
            # Send to next rank, receive from previous
            next_rank = (self.rank + 1) % self.world_size
            prev_rank = (self.rank - 1 + self.world_size) % self.world_size
            
            recv_tensor = torch.empty_like(send_tensor)
            
            # Use batch_isend_irecv for async communication
            if self.backend == "nccl" and torch.cuda.is_available():
                ops = [
                    torch.distributed.P2POp(
                        torch.distributed.isend, send_tensor, next_rank
                    ),
                    torch.distributed.P2POp(
                        torch.distributed.irecv, recv_tensor, prev_rank
                    ),
                ]
                for req in torch.distributed.batch_isend_irecv(ops):
                    req.wait()
            else:
                # Fallback to synchronous ops
                torch.distributed.send(send_tensor, next_rank)
                torch.distributed.recv(recv_tensor, prev_rank)
            
            # Store received data
            origin_rank = (self.rank - step - 1 + self.world_size) % self.world_size
            ring_buffer[origin_rank] = recv_tensor
            send_tensor = recv_tensor
        
        # Include our own data
        ring_buffer[self.rank] = compressed
        
        # Decompress all latents
        # [world_size, B, S, compression_dim] -> [world_size, B, S, kv_lora_rank]
        decompressed = self.decompress_proj(ring_buffer.view(-1, self.compression_dim))
        decompressed = decompressed.view(
            self.world_size, B, S, self.kv_lora_rank
        )
        
        return decompressed
    
    def scatter_kv_latents(
        self,
        all_latents: torch.Tensor,  # [world_size, B, S, kv_lora_rank]
    ) -> torch.Tensor:
        """
        Distribute the gathered latents back to local attention computation.
        
        For Ring-MLA, the local rank typically uses its own latents for attention,
        but can optionally use the full context for cross-rank attention.
        
        Args:
            all_latents: All latents from all ranks
            
        Returns:
            Latents for local attention [B, S, kv_lora_rank]
        """
        if self.world_size == 1:
            return all_latents.squeeze(0)
        
        # Default: use local rank's latents
        local_latents = all_latents[self.rank]
        return local_latents
    
    def compute_ring_attention(
        self,
        q: torch.Tensor,  # [B, H, S, D]
        kv_latents: torch.Tensor,  # [B, S, kv_lora_rank]
        kv_b_proj: nn.Linear,
        k_rope: torch.Tensor,  # [B, 1, S, D]
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention with ring-distributed KV latents.
        
        Implements the full Ring-MLA pattern:
        1. Gather compressed latents from all ranks
        2. Decompress locally
        3. Apply KV_b projection
        4. Compute attention with all gathered context
        
        Args:
            q: Query tensor [B, H, S, D]
            kv_latents: Compressed KV latents [B, S, kv_lora_rank]
            kv_b_proj: KV expansion projection
            k_rope: K rope component [B, 1, S, D]
            cos: Cosine of RoPE
            sin: Sine of RoPE
            attention_mask: Optional attention mask
            
        Returns:
            Attention output [B, S, H*D]
        """
        B, H, S, D = q.shape
        
        if self.world_size == 1:
            # Single GPU - use standard attention
            kv_expanded = kv_b_proj(kv_latents).view(
                B, S, self.num_heads, self.head_dim * 2
            )
            k_unrope, v = torch.chunk(kv_expanded, 2, dim=-1)
            k_unrope = k_unrope.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Apply RoPE
            from rhen.optim.quantized_kv import apply_rotary_pos_emb
            q, k_rope = apply_rotary_pos_emb(q, k_rope, cos, sin)
            k = k_unrope + k_rope
            
            # Standard SDPA
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=True
            )
            return attn_out.transpose(1, 2).contiguous().view(B, S, H * D)
        
        # Multi-GPU: gather latents from all ranks
        all_latents = self.gather_kv_latents(kv_latents)
        
        # Concatenate all latents along sequence dimension
        # [world_size, B, S, kv_lora_rank] -> [B, world_size * S, kv_lora_rank]
        all_latents = all_latents.permute(1, 0, 2, 3).reshape(B, -1, self.kv_lora_rank)
        
        # Apply KV_b projection on all gathered latents
        kv_expanded = kv_b_proj(all_latents).view(
            B, -1, self.num_heads, self.head_dim * 2
        )
        k_unrope, v = torch.chunk(kv_expanded, 2, dim=-1)
        k_unrope = k_unrope.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Expand k_rope to match new sequence length
        k_rope_expanded = k_rope.repeat(1, 1, self.world_size, 1)
        
        # Apply RoPE
        from rhen.optim.quantized_kv import apply_rotary_pos_emb
        cos_expanded = cos.repeat(1, 1, self.world_size, 1) if cos.ndim == 4 else cos.repeat(self.world_size, 1)
        sin_expanded = sin.repeat(1, 1, self.world_size, 1) if sin.ndim == 4 else sin.repeat(self.world_size, 1)
        q, k_rope = apply_rotary_pos_emb(q, k_rope_expanded, cos_expanded, sin_expanded)
        k = k_unrope + k_rope
        
        # Causal attention over the full gathered sequence
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        # Take only the local portion of output
        attn_out = attn_out[:, :, :S, :]
        return attn_out.transpose(1, 2).contiguous().view(B, S, H * D)


class RingMLAAttention(nn.Module):
    """
    Ring-MLA attention module.
    
    Wraps standard MLA with ring-based distributed KV cache sharing.
    Falls back to standard attention in single-GPU environments.
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
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.head_dim + self.head_dim),
            bias=False,
        )
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.o_proj._residual_out = True
        
        # Ring-MLA controller
        self.ring_mla = RingMLA(
            kv_lora_rank=config.kv_lora_rank,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            ring_size=config.ring_size,
            compression_dim=32,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ring-MLA attention."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query computation
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_latent).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # KV computation
        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rope = torch.split(
            kv_compressed,
            [self.kv_lora_rank, self.head_dim],
            dim=-1,
        )
        kv_latent = self.kv_a_layernorm(kv_latent)
        
        # Ring-based attention with distributed latents
        attn_output = self.ring_mla.compute_ring_attention(
            q=q,
            kv_latents=kv_latent,
            kv_b_proj=self.kv_b_proj,
            k_rope=k_rope.unsqueeze(1),
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )
        
        return self.o_proj(attn_output), None


def create_ring_mla(config, device: Optional[torch.device] = None) -> RingMLAAttention:
    """
    Factory function to create Ring-MLA attention module.
    
    Args:
        config: LoopedLLMConfig containing model configuration
        device: Device to place module on
        
    Returns:
        RingMLAAttention instance
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return RingMLAAttention(config).to(device=device)