"""
Rhen 2D Block-Causal Branch Mask Engine.

Provides exact hybrid attention masking for parallel latent decision evaluation:
1. State Prefix (0 .. |S|-1): Full bidirectional attention across all state tokens.
2. Branch Regions: Intra-branch causal attention + full visibility to State Prefix.
3. Cross-Branch Isolation: Strict -inf masking between distinct question branches,
   achieving O(1) state computation across M concurrent decision questions.
"""

from typing import List, Optional, Tuple
import torch


class Rhen2DBranchMask:
    """
    Constructs 2D block-causal attention masks for Rhen multi-question decision branches.
    
    Structure of attention matrix A [seq_len, seq_len]:
    - S_i -> S_j : 0.0 (Bidirectional state self-attention)
    - Q_{m,i} -> S_j : 0.0 (All branches attend to state prefix)
    - Q_{m,i} -> Q_{m,j} (j <= i) : 0.0 (Causal attention within same branch)
    - Q_{m,i} -> Q_{m,j} (j > i) : -inf (No lookahead within branch)
    - Q_{m,i} -> Q_{k,j} (k != m) : -inf (Strict isolation between branches)
    - Optional intra-branch option isolation: O_{m,a,i} -> O_{m,b,j} (a != b) : -inf,
      while <|decide|> still attends to state, question prefix, and all options.
    """

    @staticmethod
    def apply_option_isolation(
        mask_2d: torch.Tensor,
        branch_offsets: List[Tuple[int, int]],
        branch_option_spans: List[List[Tuple[int, int]]],
    ) -> torch.Tensor:
        """Mask out cross-option attention within each branch in-place on a 2D [L, L] mask."""
        device = mask_2d.device
        for b_idx, (b_start, b_end) in enumerate(branch_offsets):
            if b_idx >= len(branch_option_spans):
                break
            opt_spans = branch_option_spans[b_idx]
            if not opt_spans or len(opt_spans) <= 1:
                continue
            b_len = b_end - b_start
            if b_len <= 0:
                continue
            opt_ids = torch.full((b_len,), -1, dtype=torch.long, device=device)
            for opt_idx, (s_a, e_a) in enumerate(opt_spans):
                if s_a > e_a:
                    continue
                lo = max(0, s_a - b_start)
                hi = min(b_len, e_a - b_start + 1)
                if lo < hi:
                    opt_ids[lo:hi] = opt_idx
            in_opt = opt_ids >= 0
            cross_opt = (
                in_opt.unsqueeze(1)
                & in_opt.unsqueeze(0)
                & (opt_ids.unsqueeze(1) != opt_ids.unsqueeze(0))
            )
            mask_2d[b_start:b_end, b_start:b_end].masked_fill_(cross_opt, float("-inf"))
        return mask_2d

    @staticmethod
    def create_mask(
        state_length: int,
        branch_lengths: List[int],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        branch_option_spans: Optional[List[List[Tuple[int, int]]]] = None,
        isolate_options: bool = False,
    ) -> torch.Tensor:
        """
        Creates a single 2D attention mask for a single sequence with state + multiple branches.

        Args:
            state_length: Length of shared state prefix |S|.
            branch_lengths: List of token lengths for each question branch [L_1, L_2, ..., L_M].
            device: Target torch device.
            dtype: Floating point dtype (default torch.float32).
            branch_option_spans: Optional per-branch list of inclusive (opt_start_abs, opt_end_abs)
                token index pairs in the full sequence.
            isolate_options: If True and ``branch_option_spans`` is provided, masks out all
                cross-option attention within each branch while keeping state, question, and
                ``<|decide|>`` visibility intact.

        Returns:
            torch.Tensor of shape (1, 1, total_len, total_len) where masked entries are -inf
            and allowed entries are 0.0.
        """
        total_len = state_length + sum(branch_lengths)
        mask = torch.full((total_len, total_len), float("-inf"), device=device, dtype=dtype)

        # 1. State Prefix bidirectional attention: S_i <-> S_j
        if state_length > 0:
            mask[:state_length, :state_length] = 0.0

        # 2. Branch regions
        curr_offset = state_length
        branch_offsets: List[Tuple[int, int]] = []
        for b_len in branch_lengths:
            if b_len <= 0:
                branch_offsets.append((curr_offset, curr_offset))
                continue
            b_end = curr_offset + b_len
            branch_offsets.append((curr_offset, b_end))
            
            # Each branch token can attend to the full state prefix
            if state_length > 0:
                mask[curr_offset:b_end, :state_length] = 0.0

            # Each branch token can attend causally to previous/current tokens in its own branch
            branch_causal = torch.triu(torch.full((b_len, b_len), float("-inf"), device=device, dtype=dtype), diagonal=1)
            # allowed positions are 0.0
            mask[curr_offset:b_end, curr_offset:b_end] = branch_causal

            curr_offset = b_end

        # 3. Optional intra-branch option isolation
        if isolate_options and branch_option_spans is not None:
            Rhen2DBranchMask.apply_option_isolation(mask, branch_offsets, branch_option_spans)

        return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, total_len, total_len)

    @staticmethod
    def create_batch_mask(
        batch_state_lengths: List[int],
        batch_branch_lengths: List[List[int]],
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        batch_branch_option_spans: Optional[List[Optional[List[List[Tuple[int, int]]]]]] = None,
        isolate_options: bool = False,
    ) -> torch.Tensor:
        """
        Creates a batched 4D attention mask for a batch of sequences with variable branch layouts.

        Args:
            batch_state_lengths: List of state lengths for each item in batch.
            batch_branch_lengths: List of lists of branch lengths for each item in batch.
            max_seq_len: Optional maximum sequence length to pad all masks to.
            device: Target torch device.
            dtype: Floating point dtype.
            batch_branch_option_spans: Optional per-batch-item branch_option_spans.
            isolate_options: Whether to isolate intra-branch option spans.

        Returns:
            torch.Tensor of shape (batch_size, 1, max_len, max_len).
        """
        batch_size = len(batch_state_lengths)
        total_lengths = [
            s_len + sum(b_lens)
            for s_len, b_lens in zip(batch_state_lengths, batch_branch_lengths)
        ]
        target_len = max_seq_len if max_seq_len is not None else max(total_lengths)

        batch_mask = torch.full(
            (batch_size, 1, target_len, target_len),
            float("-inf"),
            device=device,
            dtype=dtype,
        )

        for b_idx in range(batch_size):
            s_len = batch_state_lengths[b_idx]
            b_lens = batch_branch_lengths[b_idx]
            item_spans = (
                batch_branch_option_spans[b_idx]
                if batch_branch_option_spans is not None and b_idx < len(batch_branch_option_spans)
                else None
            )
            single_mask = Rhen2DBranchMask.create_mask(
                state_length=s_len,
                branch_lengths=b_lens,
                device=device,
                dtype=dtype,
                branch_option_spans=item_spans,
                isolate_options=isolate_options,
            ).squeeze(0).squeeze(0)  # Shape: (L_b, L_b)

            curr_len = single_mask.shape[0]
            batch_mask[b_idx, 0, :curr_len, :curr_len] = single_mask

        return batch_mask


def build_rhen_attention_mask(
    state_len: int,
    branch_lens: List[int],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    branch_option_spans: Optional[List[List[Tuple[int, int]]]] = None,
    isolate_options: bool = False,
) -> torch.Tensor:
    """Convenience functional wrapper for single-sequence Rhen 2D mask."""
    return Rhen2DBranchMask.create_mask(
        state_len,
        branch_lens,
        device=device,
        dtype=dtype,
        branch_option_spans=branch_option_spans,
        isolate_options=isolate_options,
    )
