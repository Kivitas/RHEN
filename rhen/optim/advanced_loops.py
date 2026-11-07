import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

class LoopBypassRouter(nn.Module):
    """
    Dynamic Token-Routing Layer Skipping (Loop-Bypass).
    
    Implements per-token exit loop prediction, allowing tokens to exit
    the recurrent core at different depths based on learned halting signals.
    This replaces the simple binary ACT halting with a predicted target
    loop count per token.
    
    Key features:
    - Predicts target exit loop per token (not just binary halt)
    - Differentiable loop assignment with straight-through estimator
    - PonderNet-style prior on exit distribution
    - Compatible with existing ACT head for backward compatibility
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_loops: int,
        min_loops: int = 1,
        router_type: str = "sigmoid",  # "sigmoid", "softmax", "integer"
        ponder_lambda: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_loops = max_loops
        self.min_loops = min_loops
        self.router_type = router_type
        self.ponder_lambda = ponder_lambda
        
        # Loop prediction head: predicts continuous or discrete loop count
        if router_type == "sigmoid":
            # Predict exit probability at each loop
            self.loop_head = nn.Linear(hidden_size, max_loops)
        elif router_type == "softmax":
            # Predict distribution over loop counts
            self.loop_head = nn.Linear(hidden_size, max_loops)
        elif router_type == "integer":
            # Direct integer prediction (regression)
            self.loop_head = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"Unknown router_type: {router_type}")
        
        # Initialize with bias toward minimum loops
        nn.init.constant_(self.loop_head.bias, 0.0)
        if router_type == "sigmoid":
            # Bias early loops to be more likely
            for i in range(max_loops):
                self.loop_head.bias.data[i] = -0.5 * (i / max(max_loops - 1, 1))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        step_idx: int,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict loop routing for current step.
        
        Args:
            hidden_states: Current hidden states [B, S, H]
            step_idx: Current recurrent step (1-indexed)
            is_training: Whether in training mode
            
        Returns:
            loop_weights: Weight for each token to continue/stop [B, S]
            router_loss: Regularization loss for load balancing
        """
        batch_size, seq_len, _ = hidden_states.shape
        _ = (batch_size, seq_len)
        
        if self.router_type == "sigmoid":
            # Predict exit probability at each loop
            logits = self.loop_head(hidden_states)  # [B, S, max_loops]
            probs = torch.sigmoid(logits)
            
            # Weight for current step
            step_weight = probs[..., step_idx - 1]  # [B, S]
            
        elif self.router_type == "softmax":
            # Predict distribution over loop counts
            logits = self.loop_head(hidden_states)  # [B, S, max_loops]
            probs = F.softmax(logits, dim=-1)  # [B, S, max_loops]
            
            # Weight is probability of exiting at current step
            step_weight = probs[..., step_idx - 1]  # [B, S]
            
        elif self.router_type == "integer":
            # Predict continuous loop count
            raw_loop = self.loop_head(hidden_states).squeeze(-1)  # [B, S]
            # Convert to exit probability at current step
            predicted_loop = torch.clamp(raw_loop, self.min_loops, self.max_loops)
            step_weight = (predicted_loop <= step_idx).float()
        
        # Compute PonderNet-style prior loss for regularization
        router_loss = self._compute_prior_loss(step_weight, step_idx)
        
        return step_weight, router_loss
    
    def _compute_prior_loss(
        self,
        step_weight: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """
        Compute PonderNet geometric prior loss on exit distribution.
        
        Encourages exit distribution to match geometric(ponder_lambda).
        """
        lam = self.ponder_lambda
        
        # Geometric prior: P(exit at k) = lam * (1-lam)^(k-1)
        ks = torch.arange(1, self.max_loops + 1, device=step_weight.device, dtype=step_weight.dtype)
        prior = lam * (1 - lam) ** (ks - 1)
        prior[-1] = (1 - lam) ** max(self.max_loops - 1, 0)  # Remainder
        
        # For current step, compute KL divergence
        if self.router_type in ["sigmoid", "softmax"]:
            # We only have weight for current step, not full distribution
            # Use a simplified prior loss
            prior_step = prior[step_idx - 1]
            # Encourage step_weight to match prior_step
            kl = step_weight * (torch.log(step_weight.clamp(min=1e-9)) - math.log(prior_step))
            return kl.mean()
        
        return step_weight.new_zeros(())


class CrossLoopAttention(nn.Module):
    """
    Cross-Loop Attention (Memory Grid).
    
    Implements attention across recurrent loop iterations, allowing the model
    to attend to hidden states from previous loops. This creates a "memory grid"
    where each loop can access representations from earlier loops.
    
    Key features:
    - Memory-efficient compressed storage of past loop states
    - Cross-attention from current loop to past loop memories
    - Configurable memory dimension for compression
    - Optional positional encoding across loops
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        max_loops: int,
        memory_dim: int = 32,
        use_compression: bool = True,
        use_loop_pos: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_loops = max_loops
        self.memory_dim = memory_dim
        self.use_compression = use_compression
        self.use_loop_pos = use_loop_pos
        
        # Compression projections for memory storage
        if use_compression:
            self.memory_compress = nn.Linear(hidden_size, memory_dim, bias=False)
            self.memory_decompress = nn.Linear(memory_dim, hidden_size, bias=False)
        else:
            self.memory_compress = None
            self.memory_decompress = None
            memory_dim = hidden_size
        
        # Cross-attention projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.o_proj._residual_out = True
        
        # Loop positional embeddings
        if use_loop_pos:
            self.loop_pos_emb = nn.Embedding(max_loops + 1, memory_dim)
            nn.init.normal_(self.loop_pos_emb.weight, std=0.02)
        
        # Memory buffer for storing compressed states
        self.register_buffer("memory_buffer", torch.zeros(0))
        self.register_buffer("memory_valid", torch.zeros(0, dtype=torch.bool))
    
    def init_memory(self, batch_size: int, seq_len: int, device: torch.device):
        """Initialize memory buffer for new sequence."""
        self.memory_buffer = torch.zeros(
            batch_size, self.max_loops, seq_len, self.memory_dim,
            device=device, dtype=torch.float32
        )
        self.memory_valid = torch.zeros(
            batch_size, self.max_loops, dtype=torch.bool, device=device
        )
    
    def store_memory(
        self,
        hidden_states: torch.Tensor,
        loop_idx: int,
    ):
        """
        Store compressed hidden states from a loop iteration.
        
        Args:
            hidden_states: Hidden states [B, S, H]
            loop_idx: Loop index (1-indexed)
        """
        if self.memory_buffer.numel() == 0:
            self.init_memory(hidden_states.shape[0], hidden_states.shape[1], hidden_states.device)
            
        if self.use_compression:
            compressed = self.memory_compress(hidden_states.to(torch.float32))
        else:
            compressed = hidden_states.to(torch.float32)
        
        # Add loop positional encoding
        if self.use_loop_pos:
            pos_emb = self.loop_pos_emb(
                torch.tensor([loop_idx - 1], device=compressed.device)
            )
            compressed = compressed + pos_emb
        
        self.memory_buffer[:, loop_idx - 1] = compressed
        self.memory_valid[:, loop_idx - 1] = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        current_loop: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-loop attention: attend to past loop memories.
        
        Args:
            hidden_states: Current loop hidden states [B, S, H]
            current_loop: Current loop index (1-indexed)
            attention_mask: Optional mask for memory attention
            
        Returns:
            Enhanced hidden states with cross-loop context
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize memory if not done
        if self.memory_buffer.numel() == 0:
            self.init_memory(batch_size, seq_len, device)
        
        # Build memory keys/values from valid past loops
        valid_loops = self.memory_valid[:, :current_loop - 1].any()
        if not valid_loops:
            return hidden_states  # No past memory yet
        
        # Get valid memory states
        memory_states = self.memory_buffer[:, :current_loop - 1]  # [B, L-1, S, M]
        valid_mask = self.memory_valid[:, :current_loop - 1]  # [B, L-1]
        
        # Flatten memory for attention
        B, L, S, M = memory_states.shape
        memory_flat = memory_states.view(B, L * S, M)
        
        # Build valid mask for attention
        memory_mask = valid_mask.unsqueeze(-1).expand(-1, -1, S).reshape(B, L * S)
        
        # Project queries from current hidden states
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project keys/values from memory
        k = self.k_proj(memory_flat).view(B, L * S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_flat).view(B, L * S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if memory_mask is not None:
            memory_mask_exp = memory_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L*S]
            scores = scores.masked_fill(~memory_mask_exp, float("-inf"))
            
        if attention_mask is not None:
            # Combine with sequence attention mask
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v)
        
        # Project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        cross_output = self.o_proj(attn_out)
        
        # Residual connection
        return hidden_states + cross_output


class SelfSpeculativeDecoder(nn.Module):
    """
    Self-Speculative Decoding via Loop-Truncation.
    
    Implements a draft-verify mechanism where:
    - Draft phase: Run with fewer loops (e.g., 2 loops) for fast generation
    - Verify phase: Run with full loops (e.g., 12 loops) and accept/reject
    
    This provides faster inference with maintained quality by accepting
    the draft prefix that matches the full verification.
    
    Key features:
    - Greedy top-k draft (no tree search for simplicity)
    - Parallel verification of multiple tokens
    - Acceptance based on token match
    - Configurable draft vs verify loop counts
    """
    
    def __init__(
        self,
        model: nn.Module,
        draft_loops: int = 2,
        verify_loops: int = 12,
        max_draft_tokens: int = 4,
    ):
        super().__init__()
        self.model = model
        self.draft_loops = draft_loops
        self.verify_loops = verify_loops
        self.max_draft_tokens = max_draft_tokens
    
    def generate_draft(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        **generation_kwargs,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate draft tokens using reduced loop count.
        
        Args:
            input_ids: Input sequence
            num_tokens: Number of tokens to draft
            
        Returns:
            Draft tokens and loop stats
        """
        # Temporarily reduce loop count for draft
        original_loops = self.model.config.max_recurrent_loops
        self.model.config.max_recurrent_loops = self.draft_loops
        
        try:
            draft_tokens, draft_stats = self.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                num_loops=self.draft_loops,
                **generation_kwargs,
            )
        finally:
            # Restore original loop count
            self.model.config.max_recurrent_loops = original_loops
        
        return draft_tokens, draft_stats
    
    def verify_draft(
        self,
        draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        **generation_kwargs,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Verify draft tokens using full loop count.
        
        Args:
            draft_tokens: Draft generated tokens
            input_ids: Original input
            
        Returns:
            Verified tokens and loop stats
        """
        # Run full verification
        verified_tokens, verify_stats = self.model.generate(
            input_ids,
            max_new_tokens=draft_tokens.shape[1] - input_ids.shape[1],
            num_loops=self.verify_loops,
            **generation_kwargs,
        )
        
        return verified_tokens, verify_stats
    
    def accept_prefix(
        self,
        draft: torch.Tensor,
        verify: torch.Tensor,
    ) -> torch.Tensor:
        """
        Accept the matching prefix between draft and verify.
        
        Args:
            draft: Draft tokens [B, S]
            verify: Verified tokens [B, S]
            
        Returns:
            Accepted tokens (matching prefix)
        """
        # Compare only up to min length
        min_len = min(draft.shape[1], verify.shape[1])
        matches = (draft[:, :min_len] == verify[:, :min_len])
        
        # Find minimum match length across batches
        min_match_len = min_len
        for b in range(draft.shape[0]):
            mismatch_pos = torch.where(~matches[b])[0]
            if len(mismatch_pos) > 0:
                min_match_len = min(min_match_len, mismatch_pos[0].item())
        
        return verify[:, :min_match_len]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        **generation_kwargs,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Full speculative decoding forward pass.
        
        Args:
            input_ids: Input sequence
            
        Returns:
            Final tokens and combined stats
        """
        # Generate draft
        draft_tokens, draft_stats = self.generate_draft(
            input_ids, self.max_draft_tokens, **generation_kwargs
        )
        
        # Verify
        verify_tokens, verify_stats = self.verify_draft(
            draft_tokens, input_ids, **generation_kwargs
        )
        
        # Accept matching prefix
        accepted = self.accept_prefix(draft_tokens, verify_tokens)
        
        combined_stats = draft_stats + verify_stats
        return accepted, combined_stats


def create_loop_bypass_router(
    config,
    device: Optional[torch.device] = None,
) -> LoopBypassRouter:
    """Factory for LoopBypassRouter."""
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return LoopBypassRouter(
        hidden_size=config.hidden_size,
        max_loops=config.max_recurrent_loops,
        min_loops=config.min_recurrent_loops,
        router_type="sigmoid",
        ponder_lambda=config.act_prior_lambda,
    ).to(device)


def create_cross_loop_attention(
    config,
    device: Optional[torch.device] = None,
) -> CrossLoopAttention:
    """Factory for CrossLoopAttention."""
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CrossLoopAttention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        max_loops=config.max_recurrent_loops,
        memory_dim=config.cross_loop_memory_dim,
        use_compression=True,
        use_loop_pos=True,
    ).to(device)


def create_self_speculative_decoder(
    model,
    config,
    device: Optional[torch.device] = None,
) -> SelfSpeculativeDecoder:
    """Factory for SelfSpeculativeDecoder."""
    return SelfSpeculativeDecoder(
        model=model,
        draft_loops=config.draft_loops,
        verify_loops=config.max_recurrent_loops,
        max_draft_tokens=4,
    )