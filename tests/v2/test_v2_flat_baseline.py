"""Flat baseline (plan 10.4 item 7): interface parity, learnability, parameter matching, FLOPs, order sensitivity."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rhen.v2 import eval as ev  # noqa: E402
from rhen.v2.config import count_parameters, preset  # noqa: E402
from rhen.v2.data import synth  # noqa: E402
from rhen.v2.flat_baseline import (  # noqa: E402
    FlatBaselineModel, count_flat_parameters, flops_estimate, matched_flat_config,
)
from rhen.v2.model import Rhen2Model  # noqa: E402
from rhen.v2.packing import NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer  # noqa: E402
from rhen.v2.workspace import LoopPlan  # noqa: E402

STATE = "Node a1 load 0.97, queue depth 410, region east, 1,200 units at 45% on 2026-03-01."


def mixed_examples():
    return [
        Rhen2Example(STATE, [
            QuestionSpec("choice", "Which region?", ["east", "west", "north"], label="east"),
            QuestionSpec("noul", "Is the load above 0.9?", label=True),
            QuestionSpec("score", "Severity?", ["low", "mid", "high"], label="high"),
            QuestionSpec("multi", "Which apply?", ["load", "queue", "units", "date"], label=[0, 1]),
            QuestionSpec("range", "How many units?", lo=0.0, hi=2000.0, label=1200.0),
            QuestionSpec("extract", "Which region?", label=NULL_LABEL),
        ]),
        Rhen2Example("short state 7", [QuestionSpec("choice", "pick", ["a", "b"], label=1)]),
    ]


def synth_examples(n, seed=0, families=("negation_choice", "noul_overdue", "score_pass_rate", "multi_mentions")):
    cfg = preset("r2-tiny")
    pk = Rhen2Packer(cfg)
    out = []
    for e in synth.generate_split("train", n * 3, seed, list(families)):
        try:
            pk.pack([e])
        except ValueError:
            continue
        out.append(e)
        if len(out) == n:
            break
    return out


class TestFlatBaseline(unittest.TestCase):
    def test_interface_parity_with_rhen2(self):
        torch.manual_seed(0)
        cfg = preset("r2-tiny")
        pk = Rhen2Packer(cfg)
        batch = pk.pack(mixed_examples())
        ref = Rhen2Model(cfg).eval()
        flat = FlatBaselineModel(matched_flat_config(cfg)).eval()
        with torch.no_grad():
            a = ref(batch, LoopPlan.effort(cfg, "low"))
            b = flat(batch, LoopPlan.effort(cfg, "low"))
            c = flat(batch)                                           # loops are optional and ignored
        for name in ("probs", "multi_prob", "quantiles", "span_start", "span_end", "q_vec", "loops_run"):
            self.assertEqual(tuple(getattr(a, name).shape), tuple(getattr(b, name).shape), name)
        self.assertEqual(tuple(a.final.logits.shape), tuple(b.final.logits.shape))
        self.assertEqual(len(b.per_loop), 1)
        self.assertEqual((b.batch_size, b.max_questions), (a.batch_size, a.max_questions))
        self.assertTrue(torch.equal(b.probs, c.probs))
        self.assertTrue(torch.isfinite(b.probs).all() and torch.isfinite(b.span_start).all())
        # probabilities of valid discrete questions sum to one (options plus null)
        self.assertTrue(torch.allclose(b.probs[0].sum(), torch.tensor(1.0), atol=1e-5))
        # training forward returns a finite loss with the same parts the workspace model logs
        flat.train()
        out = flat(batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertEqual(set(out.loss_parts), set(Rhen2Model(cfg)(batch).loss_parts))
        # and the evaluation code runs on it
        recs = ev.evaluate_model(flat, pk, mixed_examples(), "medium", 16)
        self.assertEqual(len(recs), 7)

    def test_overfits_32_synthetic_examples(self):
        torch.manual_seed(0)
        cfg = preset("r2-tiny")
        pk = Rhen2Packer(cfg)
        model = FlatBaselineModel(matched_flat_config(cfg))
        examples = synth_examples(32)
        self.assertEqual(len(examples), 32)
        batches = [pk.pack(examples[i:i + 16]) for i in range(0, 32, 16)]
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        model.train()
        first = None
        for step in range(70):
            out = model(batches[step % 2])
            if step < 2:
                first = float(out.loss.detach()) if first is None else first + float(out.loss.detach())
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            total = sum(float(model(bt).loss) for bt in batches)
        self.assertLess(total, 0.6 * first)            # loose: 70 steps on 32 examples

    def test_parameter_matching_within_three_percent(self):
        for name in ("r2-tiny", "r2-50m"):
            cfg = preset(name)
            target = count_parameters(cfg)["total"]
            flat_cfg = matched_flat_config(cfg)
            got = count_flat_parameters(flat_cfg)
            self.assertLessEqual(abs(got - target) / target, 0.03, (name, got, target))
        # the analytic count is exact for the instantiated model (checked at tiny; 50M is checked analytically)
        cfg = preset("r2-tiny")
        flat_cfg = matched_flat_config(cfg)
        self.assertEqual(FlatBaselineModel(flat_cfg).num_parameters(), count_flat_parameters(flat_cfg))
        self.assertEqual(Rhen2Model(cfg).num_parameters(), count_parameters(cfg)["total"])
        # an explicit target is honoured
        small = matched_flat_config(cfg, target_params=1_700_000)
        self.assertLessEqual(abs(count_flat_parameters(small) - 1_700_000) / 1_700_000, 0.03)

    def test_flops_estimate_monotone_in_sequence_length(self):
        cfg = preset("r2-tiny")
        pk = Rhen2Packer(cfg)
        ref = Rhen2Model(cfg)
        flat = FlatBaselineModel(matched_flat_config(cfg))
        values = {"flat": [], "workspace": []}
        for words in (5, 15, 40):
            ex = [Rhen2Example(" ".join(f"item{i} {i * 3}" for i in range(words)),
                               [QuestionSpec("choice", "Which is largest?", ["item1", "item2", "item3"], label=0)])]
            batch = pk.pack(ex)
            values["flat"].append(flops_estimate(flat, batch))
            values["workspace"].append(flops_estimate(ref, batch))
        for series in values.values():
            self.assertTrue(series[0] > 0 and series[0] < series[1] < series[2], series)
        # more questions on the same state cost the flat model proportionally more (state is re-encoded)
        q = QuestionSpec("choice", "Which is largest?", ["item1", "item2", "item3"], label=0)
        state = " ".join(f"item{i} {i * 3}" for i in range(40))
        one = flops_estimate(flat, pk.pack([Rhen2Example(state, [q])]))
        four = flops_estimate(flat, pk.pack([Rhen2Example(state, [q] * 4)]))
        self.assertAlmostEqual(four / one, 4.0, delta=0.05)
        w_one = flops_estimate(ref, pk.pack([Rhen2Example(state, [q])]))
        w_four = flops_estimate(ref, pk.pack([Rhen2Example(state, [q] * 4)]))
        self.assertLess(w_four / w_one, 4.0)            # the cached state is paid for once

    def test_option_order_invariance_workspace_yes_flat_no(self):
        torch.manual_seed(0)
        cfg = preset("r2-tiny", init_std=0.15)          # larger init so options are not near-symmetric
        pk = Rhen2Packer(cfg)
        rng = random.Random(0)
        examples = []
        for i in range(24):
            opts = [f"unit u{rng.randint(10, 99)} load {rng.randint(1, 99)}" for _ in range(5)]
            examples.append(Rhen2Example(STATE, [QuestionSpec("choice", f"Which unit is busiest {i}?", opts, label=0)]))
        ref = Rhen2Model(cfg).eval()
        flat = FlatBaselineModel(matched_flat_config(cfg)).eval()
        r_ref = ev.option_shuffle_flip_rate(ref, pk, examples, "medium", 64, seed=1)
        r_flat = ev.option_shuffle_flip_rate(flat, pk, examples, "medium", 64, seed=1)
        self.assertEqual(r_ref["flips"], 0)
        self.assertLess(r_ref["max_abs_prob_diff"], 1e-4)
        # Structural difference: the flat model sees option positions, so probabilities move.
        self.assertGreater(r_flat["max_abs_prob_diff"], 1e-4)
        print(f"\noption-shuffle flip rate on 24 random-init questions: workspace={r_ref['flip_rate']}, "
              f"flat={r_flat['flip_rate']} (max prob diff {r_ref['max_abs_prob_diff']:.2e} vs "
              f"{r_flat['max_abs_prob_diff']:.2e}); untrained models, not a result")


if __name__ == "__main__":
    unittest.main()
