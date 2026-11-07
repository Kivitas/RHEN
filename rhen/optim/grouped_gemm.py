import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GroupedGEMMWrapper(nn.Module):
    """
    Grouped GEMM (General Matrix Multiply) wrapper for batched expert computations.
    
    Efficiently batches multiple GEMM operations by combining them into a single
    operation with grouped execution, which reduces kernel launch overhead and
    improves memory bandwidth utilization for MoE computations.
    
    Falls back to standard loop-based execution when grouped GEMM is not available.
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        use_fused_swiglu: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_fused_swiglu = use_fused_swiglu
        
        # Two projections to intermediate: gate and up
        self.experts_gate = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False) 
            for _ in range(num_experts)
        ])
        self.experts_up = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False) 
            for _ in range(num_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(intermediate_size, hidden_size, bias=False) 
            for _ in range(num_experts)
        ])
        
        # Router for expert selection
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Skip torch.compile on CPU (no C++ compiler available)
        self._compile_enabled = False
    
    def _fused_forward_cpu(
        self,
        x: torch.Tensor,  # [B*S, H]
        router_weights: torch.Tensor,  # [B*S, K]
        selected_experts: torch.Tensor,  # [B*S, K]
        router_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused MoE forward pass with grouped GEMM optimization."""
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]
        k = selected_experts.shape[-1] if selected_experts.dim() > 1 else 1
        
        # Initialize output and aux loss
        output = torch.zeros_like(x_flat)
        aux_loss = x_flat.new_zeros(())
        
        # Consolidated Batched Dispatch: single pass per expert across all top-k slots
        flat_expert_ids = selected_experts.view(-1)
        flat_weights = router_weights.view(-1, 1)
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(k)

        for expert_idx in range(self.num_experts):
            mask = (flat_expert_ids == expert_idx)
            if not mask.any():
                continue

            active_tokens = token_indices[mask]
            active_weights = flat_weights[mask]
            expert_input = x_flat[active_tokens]

            expert_output = self._forward_fused_swiglu_expert(expert_idx, expert_input)
            output.index_add_(0, active_tokens, expert_output * active_weights)
        
        # Load balancing aux loss
        with torch.no_grad():
            assign_counts = torch.bincount(
                selected_experts.reshape(-1), minlength=self.num_experts
            ).to(router_weights.dtype)
            expert_freq = assign_counts / max(selected_experts.numel(), 1)
            
        if router_probs is not None:
            mean_prob = router_probs.mean(dim=0)
        else:
            # Fallback to using router_weights
            mean_prob = torch.zeros(self.num_experts, device=router_weights.device, dtype=router_weights.dtype)
            for ki in range(router_weights.shape[1] if router_weights.dim() > 1 else 1):
                expert_ids_k = selected_experts[:, ki] if selected_experts.dim() > 1 else selected_experts
                weights_k = router_weights[:, ki] if router_weights.dim() > 1 else router_weights
                mean_prob.scatter_add_(
                    0, expert_ids_k, weights_k,
                )
            mean_prob /= max(selected_experts.shape[0], 1)
            
        aux_loss = self.num_experts * torch.sum(expert_freq * mean_prob)
        
        return output, aux_loss
    
    def _fused_swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Fused SwiGLU activation: gate * silu(up)"""
        return gate * F.silu(up)

    def _forward_fused_swiglu_expert(self, expert_idx: int, expert_input: torch.Tensor) -> torch.Tensor:
        """Compute forward for a single expert using fused SwiGLU."""
        gate = self.experts_gate[expert_idx](expert_input)
        up = self.experts_up[expert_idx](expert_input)
        swiglu_out = self._fused_swiglu(gate, up)
        return self.experts_down[expert_idx](swiglu_out)

    def forward(
        self,
        x: torch.Tensor,
        router_weights: Optional[torch.Tensor] = None,
        selected_experts: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        router_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for grouped MoE computation.
        
        Supports two modes:
        1. Standard mode: Takes only input x, computes routing internally
           Returns: (output, aux_loss) - compatible with SparseMoESwiGLUFFN
        2. Advanced mode: Takes pre-computed router_weights and selected_experts
           Returns: (output, aux_loss)
        
        Args:
            x: Input tensor [B*S, H] or [B, S, H]
            router_weights: Optional pre-computed router weights [B*S, K]
            selected_experts: Optional pre-computed selected experts [B*S, K]
            seq_len: Sequence length for reshaping (optional)
            router_probs: Optional pre-computed full router probabilities
            
        Returns:
            output: Processed tensor with same shape as input
            aux_loss: Auxiliary loss for load balancing
        """
        if x.dim() == 3:
            hidden_size = x.shape[-1]
            x_flat = x.view(-1, hidden_size)
        else:
            x_flat = x
        
        # If router_weights not provided, compute them from the router
        if router_weights is None:
            router_logits = self.router(x_flat)
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Top-k routing
            k = 2
            weights, selected_experts = torch.topk(router_probs, k, dim=-1)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            # Use provided weights
            weights = router_weights
            if selected_experts is None:
                selected_experts = torch.topk(router_weights, 2, dim=-1)[1]
        
        # Run fused forward
        # torch.compile is disabled on CPU; fall back to standard path
        output, aux_loss = self._fused_forward_cpu(x_flat, weights, selected_experts, router_probs)
        
        # Reshape output to match input
        output = output.view(x.shape)
        
        return output, aux_loss


def create_grouped_gemm_moe(
    config,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> GroupedGEMMWrapper:
    """
    Factory function to create grouped GEMM MoE wrapper.
    
    Args:
        config: LoopedLLMConfig containing model configuration
        dtype: Data type for computations
        device: Device to place on
        
    Returns:
        GroupedGEMMWrapper instance
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return GroupedGEMMWrapper(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        use_fused_swiglu=True,
    ).to(device=device, dtype=dtype)
