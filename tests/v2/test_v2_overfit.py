"""The model can learn at all (plan task T-208): a tiny model memorises 64 synthetic decisions."""

from __future__ import annotations

import unittest

import torch

from rhen.research.tasks import generate_split
from rhen.v2.config import preset
from rhen.v2.data.schema import from_v1_example
from rhen.v2.model import Rhen2Model
from rhen.v2.packing import Rhen2Packer
from rhen.v2.workspace import LoopPlan


class TestOverfit(unittest.TestCase):
    def test_tiny_model_reaches_high_accuracy_within_300_steps(self):
        torch.manual_seed(0)
        cfg = preset("r2-tiny")
        model, pk = Rhen2Model(cfg), Rhen2Packer(cfg)
        batch = pk.pack([from_v1_example(e) for e in generate_split("train", 64, seed=0)])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
        model.train()
        first = None
        for _ in range(300):
            out = model(batch)
            first = first if first is not None else float(out.loss.detach())
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        self.assertLess(float(out.loss.detach()), 0.2 * first)
        model.eval()
        n = batch.q_ids.shape[0] * batch.q_ids.shape[1]
        idx = batch.label_idx.reshape(n)
        labelled = idx >= 0
        multi = batch.has_multi.reshape(n)
        for plan in (LoopPlan.effort(cfg, "low"), LoopPlan.effort(cfg, "high")):
            with torch.no_grad():
                o = model(batch, plan)
            acc = float((o.probs.argmax(-1) == idx)[labelled].float().mean())
            macc = float(((o.multi_prob > 0.5).float() == batch.label_multi.reshape(n, -1))[multi].float().mean())
            self.assertGreater(acc, 0.95, plan)
            self.assertGreater(macc, 0.95, plan)


if __name__ == "__main__":
    unittest.main()
