"""Regression documentation for structural limits of the v1 decision path.

Each ``xfail(strict=True)`` test states a property that the README or the docstrings imply
and that the v1 code does not provide. They document findings A1, A2, A3 and A5 of
``docs/RHEN2_ARCHITECTURE_PLAN.md``; RHEN-2 (``rhen.v2``) provides these properties by
construction and tests them in ``tests/v2``. With ``strict=True`` an unexpected pass fails
the run, so a v1 fix has to remove the marker deliberately.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from rhen.config import EOSConfig
from rhen.model import EOSModel, MultiLatentAttention
from rhen.research.presets import preset_config
from rhen.tokenizer import get_tokenizer


def _tiny_model() -> EOSModel:
    torch.manual_seed(0)
    cfg = EOSConfig(**preset_config("tiny"))
    model = EOSModel(cfg)
    model.eval()
    return model


def _probs(model, tok, state, questions):
    out = model.evaluate_rhen(state, questions, tokenizer=tok, num_loops=2)
    return [r["probabilities"] for r in out]


class TestForwardRhenNeedsDecisionHead(unittest.TestCase):
    def test_enable_rhen_false_raises_instead_of_random_head(self):
        cfg = EOSConfig(**preset_config("tiny", enable_rhen=False))
        model = EOSModel(cfg).eval()
        with self.assertRaises(RuntimeError):
            model.evaluate_rhen("cpu 99%", [("Act?", ["yes", "no"])], tokenizer=get_tokenizer())


class TestKnownStructuralLimits(unittest.TestCase):
    @pytest.mark.xfail(strict=True, reason="A3: the slot channel mixes positions across branches")
    def test_other_question_options_do_not_change_an_answer(self):
        model, tok = _tiny_model(), get_tokenizer()
        state = "Node a1 load 0.97, queue depth 410, region east."
        q1 = ("Which action first?", ["throttle", "restart", "wait"])
        base = _probs(model, tok, state, [q1, ("Page the on-call?", ["yes", "no"])])[0]
        other = _probs(model, tok, state, [q1, ("Page the on-call?", ["certainly", "never", "maybe"])])[0]
        self.assertLess(max(abs(a - b) for a, b in zip(base, other)), 1e-5)

    @pytest.mark.xfail(strict=True, reason="A2: positions run through the packed sequence, so order matters")
    def test_question_order_does_not_change_an_answer(self):
        model, tok = _tiny_model(), get_tokenizer()
        state = "Node a1 load 0.97, queue depth 410, region east."
        x = ("Which action first?", ["throttle", "restart", "wait"])
        y = ("Page the on-call?", ["yes", "no"])
        first = _probs(model, tok, state, [x, y])[0]
        second = _probs(model, tok, state, [y, x])[1]
        self.assertLess(max(abs(a - b) for a, b in zip(first, second)), 1e-5)

    @pytest.mark.xfail(strict=True, reason="A5: option isolation is opt-in; default options attend causally")
    def test_option_order_does_not_change_probabilities_by_default(self):
        model, tok = _tiny_model(), get_tokenizer()
        state = "Node a1 load 0.97, queue depth 410, region east."
        opts = ["throttle", "restart", "wait", "failover"]
        fwd = _probs(model, tok, state, [("Which action first?", opts)])[0]
        rev = _probs(model, tok, state, [("Which action first?", opts[::-1])])[0][::-1]
        self.assertLess(max(abs(a - b) for a, b in zip(fwd, rev)), 1e-5)

    @pytest.mark.xfail(strict=True, reason="A1: odd layers use a causal sliding window over the state")
    def test_state_token_sees_later_state_tokens_in_odd_layers(self):
        torch.manual_seed(0)
        cfg = EOSConfig(**preset_config("tiny", swa_window_size=8, interleave_swa=True))
        layer = MultiLatentAttention(cfg, layer_idx=1).eval()
        length = 32
        rope = EOSModel(cfg).rope_emb
        x = torch.randn(1, length, cfg.hidden_size)
        cos, sin = rope(x, length)
        bidirectional = torch.zeros(1, 1, length, length)
        y1, _ = layer(x, cos[:length], sin[:length], attention_mask=bidirectional)
        x2 = x.clone()
        x2[:, -1] += 1.0
        y2, _ = layer(x2, cos[:length], sin[:length], attention_mask=bidirectional)
        self.assertGreater((y1[:, 0] - y2[:, 0]).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
