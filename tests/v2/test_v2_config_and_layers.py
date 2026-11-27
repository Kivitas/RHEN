"""RHEN-2 configuration accounting and layer behaviour (plan tasks T-101, T-102, T-201)."""

from __future__ import annotations

import unittest

import torch

from rhen.v2.config import PRESETS, Rhen2Config, count_parameters, preset
from rhen.v2.layers import Attention, EncoderBlock, Rotary
from rhen.v2.model import Rhen2Model
from rhen.v2.state_encoder import StateEncoder


class TestConfig(unittest.TestCase):
    def test_presets_construct_and_have_expected_ballpark(self):
        expected = {"r2-50m": 51.8e6, "r2-300m": 316.2e6, "r2-1b": 1007.6e6, "r2-8b": 7949.1e6}
        for name, target in expected.items():
            total = count_parameters(preset(name))["total"]
            self.assertLess(abs(total - target) / target, 0.005, name)
        for name in PRESETS:
            preset(name)

    def test_count_matches_instantiated_model_exactly(self):
        variants = [
            ("r2-tiny", {}),
            ("r2-tiny", {"use_set_layer": False, "use_numeric": False, "use_sandwich_norm": False}),
            ("r2-tiny", {"n_scratch": 0, "n_core_layers": 2}),
            ("r2-50m", {}),
        ]
        for name, kw in variants:
            cfg = preset(name, **kw)
            self.assertEqual(Rhen2Model(cfg).num_parameters(), count_parameters(cfg)["total"], (name, kw))

    def test_config_validation_and_roundtrip(self):
        with self.assertRaises(ValueError):
            Rhen2Config(d_model=100, n_heads=8)
        with self.assertRaises(ValueError):
            Rhen2Config(r_min=5, r_mean=3, r_max=8)
        with self.assertRaises(ValueError):
            Rhen2Config(max_options=300)
        cfg = preset("r2-tiny")
        self.assertEqual(Rhen2Config.from_dict(cfg.to_dict()), cfg)
        with self.assertRaises(ValueError):
            Rhen2Config.from_dict({"nonsense": 1})


class TestLayers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_padding_does_not_change_real_tokens(self):
        block = EncoderBlock(32, 4, 64).eval()
        rot = Rotary(8)
        x = torch.randn(1, 5, 32)
        padded = torch.cat([x, torch.randn(1, 3, 32)], dim=1)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        with torch.no_grad():
            a = block(x, key_mask=torch.ones(1, 5, dtype=torch.bool), rope=rot(5, x.device, x.dtype))
            b = block(padded, key_mask=mask, rope=rot(8, x.device, x.dtype))
        self.assertLess((a - b[:, :5]).abs().max().item(), 1e-5)

    def test_local_layers_do_not_see_beyond_their_window(self):
        cfg = preset("r2-tiny", n_state_layers=2, global_every=3, local_window=4)   # both layers local
        enc = StateEncoder(cfg).eval()
        x = torch.randn(1, 40, cfg.d_model)
        mask = torch.ones(1, 40, dtype=torch.bool)
        x2 = x.clone()
        x2[:, 30:] += 1.0
        with torch.no_grad():
            d = (enc(x, mask) - enc(x2, mask)).abs().amax(dim=(0, 2))
        self.assertEqual(d[:20].max().item(), 0.0)       # far tokens are untouched
        self.assertGreater(d[30:].max().item(), 0.0)

    def test_global_encoder_is_bidirectional(self):
        cfg = preset("r2-tiny", global_every=0)
        enc = StateEncoder(cfg).eval()
        x = torch.randn(1, 20, cfg.d_model)
        mask = torch.ones(1, 20, dtype=torch.bool)
        x2 = x.clone()
        x2[:, -1] += 1.0
        with torch.no_grad():
            d = (enc(x, mask) - enc(x2, mask)).abs()[0, 0].max().item()
        self.assertGreater(d, 1e-6)       # token 0 sees the last token: the A1 defect of v1 is absent

    def test_folded_cross_attention_matches_unfolded(self):
        attn = Attention(32, 4).eval()
        g, per, lq, lm = 2, 3, 5, 7
        x = torch.randn(g * per, lq, 32)
        mem = torch.randn(g, lm, 32)
        mask = torch.ones(g, lm, dtype=torch.bool)
        mask[1, 4:] = False
        kv = attn.project_kv(mem)
        with torch.no_grad():
            folded = attn(x, kv=kv, key_mask=mask, per_group=per)
            rows = []
            for i in range(g * per):
                j = i // per
                rows.append(attn(x[i:i + 1], kv=(kv[0][j:j + 1], kv[1][j:j + 1]), key_mask=mask[j:j + 1]))
            plain = torch.cat(rows)
        self.assertLess((folded - plain).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
