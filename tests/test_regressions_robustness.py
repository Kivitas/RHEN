"""
Regression tests for error-handling hardening.

1. adapter: the external encoder gets one deterministic call with the 2D padding mask. A
   RuntimeError from the encoder propagates instead of being replaced by a later signature
   error. An encoder without an attention_mask keyword falls back to a call without it, and
   the output matches the output of an encoder that accepts the mask.
2. rhen_packer: a failing state-depth function or latent-slot lookup logs one WARNING per
   site and the packed sequence is still produced.
3. rhen_chunker: a tokenizer whose encode raises RuntimeError makes count_tokens return the
   character estimate and log one WARNING.
4. fused_swiglu: a failing torch.compile logs a WARNING and writes nothing to stdout; a
   failing compiled forward logs at DEBUG and falls back to the standard implementation.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rhen.adapter import HuggingFaceEncoderAdapter, MockEncoderOutput
from rhen.rhen_packer import PackedBranchMeta, PackedRhenSequence
import rhen.rhen_chunker as chunker_mod
import rhen.rhen_packer as packer_mod
from rhen.optim.fused_swiglu import FusedSwiGLU
from rhen.research import checkpointing as ckpt
from rhen.research.distributed import DistributedContext

HIDDEN = 16
VOCAB = 50


# --------------------------------------------------------------------------------------
# 1. Adapter
# --------------------------------------------------------------------------------------


class _EmbeddingEncoder(nn.Module):
    """Accepts attention_mask and returns embeddings of input_ids."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.emb = emb

    def forward(self, input_ids, attention_mask=None):
        return MockEncoderOutput(self.emb(input_ids))


class _NoMaskEncoder(nn.Module):
    """Has no attention_mask keyword, so Python raises TypeError when it is passed."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.emb = emb

    def forward(self, input_ids):
        return MockEncoderOutput(self.emb(input_ids))


class _DeviceMismatchEncoder(nn.Module):
    """Raises a RuntimeError on the first call; later calls would succeed."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.emb = emb
        self.calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Expected all tensors to be on the same device, but found cpu and cuda:0")
        return MockEncoderOutput(self.emb(input_ids))


class _BrokenMaskEncoder(nn.Module):
    """Raises a TypeError that is about the mask's content, not a missing keyword."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.calls += 1
        raise TypeError("attention_mask must be a float tensor")


def _packed_sequence(length: int = 8) -> PackedRhenSequence:
    """One state prefix of 2 tokens and one branch with two options and a <|decide|> token."""
    input_ids = (torch.arange(length) % VOCAB).unsqueeze(0)
    attention_mask = torch.zeros(1, 1, length, length)  # 4D, as the packer produces
    branch = PackedBranchMeta(
        question_text="which?",
        options=["a", "b"],
        branch_token_len=length - 2,
        decide_idx=length - 1,
        option_end_indices=[4, 6],
        num_options=2,
        option_start_indices=[2, 4],
    )
    return PackedRhenSequence(
        input_ids=input_ids,
        attention_mask=attention_mask,
        state_length=2,
        branch_lengths=[length - 2],
        branches=[branch],
    )


def _adapter(encoder: nn.Module) -> HuggingFaceEncoderAdapter:
    torch.manual_seed(0)
    return HuggingFaceEncoderAdapter(
        encoder=encoder, hidden_size=HIDDEN, num_heads=2, rank=8,
    )


def test_adapter_runtime_error_propagates_on_first_call():
    emb = nn.Embedding(VOCAB, HIDDEN)
    encoder = _DeviceMismatchEncoder(emb)
    adapter = _adapter(encoder)

    with pytest.raises(RuntimeError, match="same device"):
        adapter.forward_rhen(_packed_sequence())
    # Exactly one attempt: no retry that could hide the real error behind a later failure.
    assert encoder.calls == 1


def test_adapter_non_keyword_type_error_retries_with_2d_mask_then_propagates_chained():
    encoder = _BrokenMaskEncoder()
    adapter = _adapter(encoder)

    with pytest.raises(TypeError, match="float tensor") as info:
        adapter.forward_rhen(_packed_sequence())
    # One attempt with the 4D block mask, one with the 2D padding mask, then the error surfaces.
    assert encoder.calls == 2
    assert info.value.__cause__ is not None


class _MaskRecordingEncoder(nn.Module):
    """Records the rank of every attention_mask it receives."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.emb = emb
        self.mask_ranks: list[int] = []

    def forward(self, input_ids, attention_mask=None):
        self.mask_ranks.append(attention_mask.dim())
        return MockEncoderOutput(self.emb(input_ids))


def test_adapter_sends_packed_4d_block_mask_first():
    """The branch-isolation 4D mask must reach the encoder before any 2D fallback."""
    emb = nn.Embedding(VOCAB, HIDDEN)
    encoder = _MaskRecordingEncoder(emb)
    adapter = _adapter(encoder)

    adapter.forward_rhen(_packed_sequence())
    assert encoder.mask_ranks[0] == 4
    assert len(encoder.mask_ranks) == 1  # succeeds on the first call; no needless retry


def test_adapter_falls_back_when_encoder_has_no_attention_mask_keyword(caplog):
    emb = nn.Embedding(VOCAB, HIDDEN)
    adapter = _adapter(_NoMaskEncoder(emb))
    packed = _packed_sequence()

    with caplog.at_level(logging.DEBUG, logger="rhen.adapter"):
        fallback = adapter.forward_rhen(packed)

    assert any("does not accept attention_mask" in r.getMessage() for r in caplog.records)

    # The same weights behind an encoder that accepts the mask must give the same output.
    adapter.encoder = _EmbeddingEncoder(emb)
    direct = adapter.forward_rhen(packed)

    assert len(fallback) == len(direct) == 1
    tensor_keys = [k for k, v in direct[0].items() if isinstance(v, torch.Tensor)]
    assert tensor_keys, "pointer head returned no tensors"
    for key in tensor_keys:
        assert torch.allclose(fallback[0][key], direct[0][key]), key


# --------------------------------------------------------------------------------------
# 2. Packer fallbacks
# --------------------------------------------------------------------------------------


class _CharTokenizer:
    """Encodes each character as an id. Has no token_to_id, so the latent-slot lookup falls back."""

    def encode(self, text, add_special_tokens=True):
        return [ord(c) % VOCAB for c in text]


def test_packer_depth_failure_logs_once_and_still_packs(monkeypatch, caplog):
    monkeypatch.setattr(packer_mod, "_FALLBACKS_LOGGED", set())

    def failing_depths(text):
        raise ValueError("depth boom")

    monkeypatch.setattr(packer_mod, "compute_structural_depth_ids", failing_depths)
    packer = packer_mod.RhenPacker(tokenizer=_CharTokenizer())
    qs = [("Pick one?", ["x", "y"])]

    with caplog.at_level(logging.WARNING, logger="rhen.rhen_packer"):
        first = packer.pack(state={"a": 1}, questions_and_options=qs)
        second = packer.pack(state={"a": 2}, questions_and_options=qs)

    depth_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "state depths" in r.getMessage()
    ]
    assert len(depth_warnings) == 1
    assert "_pack_sequence" in depth_warnings[0].getMessage()
    assert "ValueError" in depth_warnings[0].getMessage()

    for packed in (first, second):
        assert packed.input_ids.shape[1] == len(packed.depth_ids)
        assert all(d == 0 for d in packed.depth_ids[: packed.state_length])
        assert len(packed.branches) == 1


def test_packer_latent_slot_lookup_failure_logs_and_still_packs(monkeypatch, caplog):
    monkeypatch.setattr(packer_mod, "_FALLBACKS_LOGGED", set())
    packer = packer_mod.RhenPacker(tokenizer=_CharTokenizer())

    with caplog.at_level(logging.WARNING, logger="rhen.rhen_packer"):
        packed = packer.pack(state={"a": 1}, questions_and_options=[("Pick one?", ["x", "y"])])

    lookup_warnings = [r for r in caplog.records if "latent slot lookup" in r.getMessage()]
    assert len(lookup_warnings) == 1
    assert packed.latent_slot_indices is None
    assert packed.input_ids.shape[1] == len(packed.depth_ids)


# --------------------------------------------------------------------------------------
# 3. Chunker token counting
# --------------------------------------------------------------------------------------


class _RaisingTokenizer:
    def encode(self, text, *args, **kwargs):
        raise RuntimeError("tokenizer backend exploded")


def test_chunker_count_tokens_falls_back_to_character_estimate_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(chunker_mod, "_TOKENIZER_FALLBACK_LOGGED", False)
    chunker = chunker_mod.SemanticBoundaryChunker(tokenizer=_RaisingTokenizer())
    text = "x" * 40
    expected = int(len(text) / chunker_mod._CHARS_PER_TOKEN)

    with caplog.at_level(logging.WARNING, logger="rhen.rhen_chunker"):
        assert chunker.count_tokens(text) == expected
        assert chunker.count_tokens(text) == expected  # second call: no second warning

    warnings_ = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_) == 1
    assert "character-count estimate" in warnings_[0].getMessage()
    assert "RuntimeError" in warnings_[0].getMessage()


# --------------------------------------------------------------------------------------
# 4. FusedSwiGLU diagnostics
# --------------------------------------------------------------------------------------


def test_fused_swiglu_compile_failure_warns_without_print(monkeypatch, capsys, caplog):
    def failing_compile(*args, **kwargs):
        raise RuntimeError("no compiler available")

    monkeypatch.setattr(torch, "compile", failing_compile)

    with caplog.at_level(logging.WARNING, logger="rhen.optim.fused_swiglu"):
        module = FusedSwiGLU(hidden_size=8, intermediate_size=16, use_torch_compile=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert module._compile_enabled is False

    warnings_ = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_) == 1
    assert "torch.compile failed" in warnings_[0].getMessage()
    assert "no compiler available" in warnings_[0].getMessage()


def test_fused_swiglu_compiled_forward_failure_logs_debug_and_falls_back(caplog):
    module = FusedSwiGLU(hidden_size=8, intermediate_size=16, use_torch_compile=False)

    def failing_forward(*args):
        raise RuntimeError("kernel launch failed")

    module._compile_enabled = True
    module._forward_fused = failing_forward

    gate_w = torch.randn(16, 8)
    up_w = torch.randn(16, 8)
    x = torch.randn(4, 8)

    with caplog.at_level(logging.DEBUG, logger="rhen.optim.fused_swiglu"):
        out = module(x, gate_w, None, up_w, None)

    assert out.shape == (4, 16)
    assert module._compile_enabled is False
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records
    assert debug_records[0].exc_info is not None


# --------------------------------------------------------------------------------------
# 5. Checkpoint round trip (optimizers.pt is read with weights_only=True)
# --------------------------------------------------------------------------------------


def _tiny_training_state():
    model = nn.Linear(4, 2)
    model.config = SimpleNamespace(to_dict=lambda: {"hidden": 4})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2)
    return model, opt, sched


def test_checkpoint_round_trip_restores_optimizer_scheduler_and_warns_on_rng_load(tmp_path, caplog):
    ctx = DistributedContext()
    model, opt, sched = _tiny_training_state()
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    sched.step()
    path = ckpt.save_checkpoint(
        str(tmp_path), 3, model, model, [opt], [sched], None, {"seed": 1}, ctx, keep_last=3,
    )

    model2, opt2, sched2 = _tiny_training_state()
    with caplog.at_level(logging.WARNING, logger="rhen.research.checkpointing"):
        trainer_state = ckpt.load_checkpoint(path, model2, model2, [opt2], [sched2], None, ctx)

    assert trainer_state["step"] == 3
    assert torch.equal(model2.weight, model.weight)
    assert torch.equal(
        opt2.state_dict()["state"][0]["exp_avg"], opt.state_dict()["state"][0]["exp_avg"],
    )
    assert sched2.last_epoch == sched.last_epoch
    assert any("weights_only=False" in r.getMessage() for r in caplog.records)
