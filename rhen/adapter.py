"""
Rhen HuggingFace and Custom Backbone Encoder Adapter.

Allows developers to use Rhen's JEV-type non-autoregressive decision layer
(MultiHeadRhenPointerHead + Primitives) with ANY existing HuggingFace transformer
(BERT, RoBERTa, DeBERTa, LLaMA, Mistral, Qwen, Gemma, T5 encoder) or custom PyTorch encoder.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

try:
    from rhen.model import MultiHeadRhenPointerHead
    from rhen.rhen_packer import PackedRhenSequence, get_rhen_packer
except ImportError:
    from model import MultiHeadRhenPointerHead
    from rhen_packer import PackedRhenSequence, get_rhen_packer


class MockEncoderOutput:
    """Mock container mimicking HuggingFace BaseModelOutput."""
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx == 0:
            return self.last_hidden_state
        raise IndexError(idx)


class MockEncoder(nn.Module):
    """
    Lightweight mock encoder with 0 or minimal weights for instant local unit tests
    without downloading HuggingFace models.
    """
    def __init__(self, hidden_size: int = 768, vocab_size: int = 32000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = type("MockConfig", (), {"hidden_size": hidden_size, "d_model": hidden_size})()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MockEncoderOutput:
        safe_ids = input_ids.clamp(0, self.vocab_size - 1)
        h = self.embedding(safe_ids)
        return MockEncoderOutput(h)


class HuggingFaceEncoderAdapter(nn.Module):
    """
    Adapts any HuggingFace PreTrainedModel or custom PyTorch encoder to serve
    as the backbone for Rhen's System One decision layer.

    Args:
        encoder: HuggingFace model or custom nn.Module producing hidden states.
        tokenizer: Associated tokenizer.
        hidden_size: Hidden dimension of encoder (auto-inferred from encoder.config if available).
        num_heads: Number of bilinear pointer heads in Rhen readout (default 4).
        rank: Low-rank bilinear projection dimension (default 128).
        pooling_mode: Span pooling mode for option tokens ('mean', 'last', 'cls'). Default 'mean'.
    """
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: int = 4,
        rank: int = 128,
        pooling_mode: str = "mean",
    ):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer

        if pooling_mode not in ("mean", "last", "cls"):
            raise ValueError(f"Invalid pooling_mode: {pooling_mode!r}. Must be 'mean', 'last', or 'cls'")
        self.pooling_mode = pooling_mode

        if hidden_size is None:
            if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
                hidden_size = encoder.config.hidden_size
            elif hasattr(encoder, "config") and hasattr(encoder.config, "d_model"):
                hidden_size = encoder.config.d_model
            else:
                hidden_size = 768  # Standard fallback

        self.hidden_size = hidden_size
        self.pointer_head = MultiHeadRhenPointerHead(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rank=rank,
        )

    def forward_rhen(
        self,
        packed_or_state: Any,
        questions_and_options: Optional[List[Tuple[str, List[str]]]] = None,
        tokenizer: Optional[Any] = None,
        num_loops: Optional[int] = None,
        pooling_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Differentiable forward pass for Rhen decisions using the external encoder.
        """
        pool = (pooling_mode or self.pooling_mode).lower()
        if pool not in ("mean", "last", "cls"):
            raise ValueError(f"Invalid pooling_mode: {pool!r}. Must be 'mean', 'last', or 'cls'")

        tok = tokenizer or self.tokenizer
        if isinstance(packed_or_state, PackedRhenSequence):
            packed = packed_or_state
        else:
            packer = get_rhen_packer(tok)
            packed = packer.pack(state=packed_or_state, questions_and_options=questions_and_options or [])

        device = next(self.parameters()).device
        input_ids = packed.input_ids.to(device)
        attention_mask = packed.attention_mask.to(device)

        # Extract 2D padding mask properly from attention_mask and tokenizer pad_token_id
        if attention_mask.dim() == 2:
            mask_2d = attention_mask
        elif attention_mask.dim() == 3:
            diag = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
            mask_2d = (diag != 0).long()
            if (mask_2d == 0).all():
                mask_2d = (attention_mask.sum(dim=-1) > 0).long()
        elif attention_mask.dim() == 4:
            diag = torch.diagonal(attention_mask.squeeze(1), dim1=-2, dim2=-1)
            mask_2d = (diag != 0).long()
            if (mask_2d == 0).all():
                mask_2d = (attention_mask.squeeze(1).sum(dim=-1) > 0).long()
        else:
            mask_2d = torch.ones_like(input_ids)

        if tok is not None and getattr(tok, "pad_token_id", None) is not None:
            mask_2d = mask_2d * (input_ids != tok.pad_token_id).long()

        if (mask_2d == 0).all():
            mask_2d = torch.ones_like(input_ids)

        # Run through the external encoder
        if hasattr(self.encoder, "forward"):
            try:
                enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                try:
                    # Many standard HuggingFace transformers expect a 2D attention mask
                    enc_out = self.encoder(input_ids=input_ids, attention_mask=mask_2d)
                except Exception:
                    enc_out = self.encoder(input_ids=input_ids)

            if hasattr(enc_out, "last_hidden_state"):
                h = enc_out.last_hidden_state
            elif isinstance(enc_out, tuple):
                h = enc_out[0]
            else:
                h = enc_out
        else:
            h = self.encoder(input_ids)

        branch_results = []
        branches = getattr(packed, "branches", getattr(packed, "branch_metadata", []))
        for b_idx, branch in enumerate(branches):
            if isinstance(branch, dict):
                opt_starts = branch.get("option_start_indices", None)
                opt_ends = branch.get("option_end_indices", [])
                decide_idx = branch.get("decide_idx", branch.get("question_end_idx", 0))
                q_text = branch.get("question_text", "")
                options = branch.get("options", [])
            else:
                opt_starts = getattr(branch, "option_start_indices", None)
                opt_ends = getattr(branch, "option_end_indices", [])
                decide_idx = getattr(branch, "decide_idx", 0)
                q_text = getattr(branch, "question_text", "")
                options = getattr(branch, "options", [])

            # Option embeddings (span pooling or boundary tokens)
            opt_reps = []
            if opt_starts and len(opt_starts) == len(opt_ends):
                for s, e in zip(opt_starts, opt_ends):
                    if e > s:
                        if pool == "mean":
                            opt_rep = h[:, s:e].mean(dim=1)
                        elif pool == "last":
                            opt_rep = h[:, e - 1]
                        elif pool == "cls":
                            opt_rep = h[:, s]
                    else:
                        opt_rep = h[:, s]
                    opt_reps.append(opt_rep)
            else:
                for e in opt_ends:
                    opt_reps.append(h[:, e])

            if opt_reps:
                h_options = torch.stack(opt_reps, dim=1)  # [B, K, D]
            else:
                h_options = torch.zeros(input_ids.shape[0], 0, self.hidden_size, device=device)

            # Question representation at <|decide|>
            h_q = h[:, decide_idx:decide_idx + 1, :]

            # Pointer forward via MultiHeadRhenPointerHead
            state_len = getattr(packed, "state_length", getattr(packed, "state_token_count", 0))
            kv_latent = h[:, :state_len, :] if state_len > 0 else None

            out = self.pointer_head(
                q_repr=h_q,
                option_reprs=h_options,
                kv_latent=kv_latent,
                return_details=True,
            )

            if isinstance(out, dict):
                out["branch_index"] = b_idx
                out["question"] = q_text
                out["options"] = options
                if "probabilities" in out and isinstance(out["probabilities"], torch.Tensor):
                    probs_list = out["probabilities"].squeeze(0).tolist()
                    out["probabilities_dict"] = {
                        options[i]: probs_list[i] for i in range(min(len(options), len(probs_list)))
                    }
                    if len(probs_list) > 0:
                        best_idx = int(torch.argmax(out["probabilities"].squeeze(0)).item())
                        out["chosen_index"] = best_idx
                        out["chosen_option"] = options[best_idx] if best_idx < len(options) else None
            branch_results.append(out)

        return branch_results

    def evaluate_rhen(
        self,
        packed_or_state: Any,
        questions_and_options: Optional[List[Tuple[str, List[str]]]] = None,
        tokenizer: Optional[Any] = None,
        num_loops: Optional[int] = None,
        pooling_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluation pass for Rhen decisions.
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward_rhen(
                packed_or_state=packed_or_state,
                questions_and_options=questions_and_options,
                tokenizer=tokenizer,
                num_loops=num_loops,
                pooling_mode=pooling_mode,
            )
        finally:
            if was_training:
                self.train()


RhenEncoderAdapter = HuggingFaceEncoderAdapter
