"""Hardening tests: checkpoint io, EMA, state cache, bf16, checkpointing, optimizer groups, large K."""

from __future__ import annotations

import copy
import os
import random
import tempfile
import unittest

import torch

from rhen.v2.cache import StateCache
from rhen.v2.config import preset
from rhen.v2.ema import EMA
from rhen.v2.io import load_checkpoint, save_checkpoint
from rhen.v2.model import Rhen2Model
from rhen.v2.optim import build_param_groups, lr_at
from rhen.v2.packing import QuestionSpec, Rhen2Example, Rhen2Packer
from rhen.v2.workspace import Interventions, LoopPlan

STATE = "Node a1 load 0.97, queue depth 410, region east, 1,200 units at 45% on 2026-03-01."
PLAN = LoopPlan(n_loops=2, min_loops=2)


def _mk(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = preset("r2-tiny", **kw)
    return Rhen2Model(cfg).eval(), Rhen2Packer(cfg), cfg


def _examples():
    return [
        Rhen2Example(STATE, [QuestionSpec("choice", "Which?", ["a", "bb", "c"]), QuestionSpec("score", "Sev?", ["l", "m", "h"]),
                             QuestionSpec("range", "How many?", lo=0, hi=100), QuestionSpec("extract", "Where?")]),
        Rhen2Example("short", [QuestionSpec("noul", "ok?")]),
    ]


class TestIO(unittest.TestCase):
    def test_round_trip_bit_identical(self):
        model, pk, cfg = _mk()
        batch = pk.pack(_examples())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            save_checkpoint(path, model, cfg, extra={"step": 7, "tags": ["a"]})
            loaded, extra = load_checkpoint(path)
        self.assertEqual(extra, {"step": 7, "tags": ["a"]})
        with torch.no_grad():
            a, b = model(batch, PLAN), loaded(batch, PLAN)
        self.assertTrue(torch.equal(a.probs, b.probs))
        self.assertTrue(torch.equal(a.quantiles, b.quantiles))

    def test_rejects_unknown_version_cfg_key_and_bad_extra(self):
        model, _, cfg = _mk()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            with self.assertRaises(TypeError):
                save_checkpoint(path, model, cfg, extra={"x": object()})
            save_checkpoint(path, model, cfg)
            payload = torch.load(path, weights_only=True)
            bad = dict(payload, format=99)
            torch.save(bad, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
            import json
            c = json.loads(payload["cfg"])
            c["bogus"] = 1
            torch.save(dict(payload, cfg=json.dumps(c)), path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)


class TestEMA(unittest.TestCase):
    def test_warmup_swap_and_state(self):
        model, _, _ = _mk()
        ema = EMA(model, decay=0.999)
        self.assertAlmostEqual(ema.current_decay(), 0.1)
        w0 = model.mask_memory.detach().clone()
        with torch.no_grad():
            model.mask_memory.add_(1.0)
        ema.update(model)
        expected = w0 * 0.1 + (w0 + 1.0) * 0.9
        self.assertTrue(torch.allclose(ema.shadow["mask_memory"], expected, atol=1e-6))
        trained = model.mask_memory.detach().clone()
        with ema.swap_into(model):
            self.assertTrue(torch.allclose(model.mask_memory, expected, atol=1e-6))
        self.assertTrue(torch.equal(model.mask_memory, trained))
        e2 = EMA(model)
        e2.load_state_dict(ema.state_dict())
        self.assertEqual(e2.num_updates, 1)
        self.assertTrue(torch.equal(e2.shadow["mask_memory"], ema.shadow["mask_memory"]))
        ema.num_updates = 20000
        self.assertAlmostEqual(ema.current_decay(), 0.999)

    def test_swap_restores_on_error(self):
        model, _, _ = _mk()
        ema = EMA(model)
        with torch.no_grad():
            model.mask_memory.add_(2.0)
        before = model.mask_memory.detach().clone()
        with self.assertRaises(RuntimeError):
            with ema.swap_into(model):
                raise RuntimeError("boom")
        self.assertTrue(torch.equal(model.mask_memory, before))


class TestCache(unittest.TestCase):
    def setUp(self):
        self.model, self.pk, _ = _mk(init_std=0.1)

    def _enc(self, batch):
        with torch.no_grad():
            return self.model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)

    def test_hit_miss_equality_and_different_lengths(self):
        cache = StateCache(10_000_000)
        texts = [STATE, "short state 12", STATE + " plus a much longer tail of text about queues and nodes"]
        b1 = self.pk.pack([Rhen2Example(t, [QuestionSpec("noul", "ok?")]) for t in texts])
        m1 = cache.memory_for(self.model, b1)
        s = cache.stats()
        self.assertEqual((s["hits"], s["misses"]), (0, 3))
        fresh = self._enc(b1)
        self.assertEqual(m1.shape, fresh.shape)
        mask = b1.state_mask[:, :, None]
        self.assertLess(float(((m1 - fresh) * mask).abs().max()), 1e-5)
        # same rows, different batch composition and padded length
        b2 = self.pk.pack([Rhen2Example(texts[1], [QuestionSpec("noul", "ok?")]), Rhen2Example(texts[0], [QuestionSpec("noul", "ok?")])])
        m2 = cache.memory_for(self.model, b2)
        self.assertEqual(cache.stats()["hits"], 2)
        fresh2 = self._enc(b2)
        self.assertLess(float(((m2 - fresh2) * b2.state_mask[:, :, None]).abs().max()), 1e-5)
        # cached rows are bit-equal across repeated calls
        self.assertTrue(torch.equal(cache.memory_for(self.model, b2), m2))

    def test_no_collisions_and_weight_change_invalidates(self):
        cache = StateCache(10_000_000)
        b = self.pk.pack([Rhen2Example(STATE, [QuestionSpec("noul", "q")]), Rhen2Example(STATE.replace("410", "411"), [QuestionSpec("noul", "q")])])
        cache.memory_for(self.model, b)
        self.assertEqual(cache.stats()["misses"], 2)
        self.assertEqual(cache.stats()["entries"], 2)
        with torch.no_grad():
            self.model.mask_memory.add_(0.0)       # in-place write bumps the version counter
        cache.memory_for(self.model, b)
        self.assertEqual(cache.stats()["hits"], 0)

    def test_eviction_by_bytes(self):
        b = self.pk.pack([Rhen2Example(f"state number {i} " + "x " * 5, [QuestionSpec("noul", "q")]) for i in range(6)])
        one = self.pk.pack([Rhen2Example("state number 0 " + "x " * 5, [QuestionSpec("noul", "q")])])
        row_bytes = int(one.state_mask.sum()) * 64 * 4
        cache = StateCache(int(row_bytes * 2.5))
        cache.memory_for(self.model, b)
        s = cache.stats()
        self.assertLessEqual(s["bytes"], cache.max_bytes)
        self.assertGreaterEqual(s["evictions"], 3)


class TestMixedPrecision(unittest.TestCase):
    def test_autocast_and_bf16_close_to_fp32(self):
        model, pk, _ = _mk()
        batch = pk.pack(_examples())      # includes a row with fewer questions: padded questions and options
        plan = LoopPlan(n_loops=3, min_loops=3)
        with torch.no_grad():
            ref = model(batch, plan)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                auto = model(batch, plan)
            low = copy.deepcopy(model).to(torch.bfloat16)
            self.assertEqual(next(low.readout.parameters()).dtype, torch.float32)
            bf = low(batch, plan)
        for out in (auto, bf):
            for f in ("probs", "multi_prob", "quantiles", "evidence"):
                x = getattr(out, f).float()
                self.assertTrue(bool(torch.isfinite(x).all()), f)
                self.assertLess(float((x - getattr(ref, f).float()).abs().max()), 2e-2, f)
            self.assertTrue(bool(torch.isfinite(out.span_start).all()))

    def test_rope_is_float32(self):
        from rhen.v2.layers import Rotary
        cos, _ = Rotary(16)(10, torch.device("cpu"), torch.bfloat16)
        self.assertEqual(cos.dtype, torch.float32)


class TestCheckpointing(unittest.TestCase):
    def test_gradients_equal(self):
        grads = []
        for ckpt in (False, True):
            torch.manual_seed(0)
            cfg = preset("r2-tiny", gradient_checkpointing=ckpt, init_std=0.1)
            model, pk = Rhen2Model(cfg).train(), Rhen2Packer(cfg)
            qs = [QuestionSpec("choice", "Which?", ["a", "bb", "c"], label=1), QuestionSpec("range", "n?", lo=0, hi=10, label=3.0)]
            batch = pk.pack([Rhen2Example(STATE, qs)])
            out = model(batch, LoopPlan(n_loops=3, k_bptt=2, min_loops=3, train=True))
            out.loss.backward()
            grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        self.assertEqual(set(grads[0]), set(grads[1]))
        self.assertGreater(len(grads[0]), 30)
        for n in grads[0]:
            self.assertLess(float((grads[0][n] - grads[1][n]).abs().max()), 1e-5, n)


class TestOptim(unittest.TestCase):
    def test_groups_cover_everything_once(self):
        model, _, cfg = _mk()
        groups = build_param_groups(model, lr=1e-3, base_width=128)
        seen = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {id(p) for p in model.parameters()})
        by = {g["name"]: g for g in groups}
        self.assertEqual(by["no_decay"]["weight_decay"], 0.0)
        self.assertAlmostEqual(by["hidden_decay"]["lr"], 1e-3 * 128 / cfg.d_model)
        self.assertEqual(by["head_decay"]["lr"], 1e-3)
        names = {id(p): n for n, p in model.named_parameters()}
        nd = {names[id(p)] for p in by["no_decay"]["params"]}
        for must in ("embed.table.weight", "workspace.scratch", "workspace.rank_emb", "mask_memory",
                     "workspace.set_layer.null_vec", "readout.null_bias", "encoder.norm.weight", "qtype.weight"):
            self.assertIn(must, nd)
        torch.optim.AdamW(groups)

    def test_schedule_endpoints(self):
        total = 1000
        self.assertLess(lr_at(0, total, 1.0), 0.1)
        self.assertAlmostEqual(lr_at(19, total, 1.0), 1.0)           # end of 2% warmup
        self.assertAlmostEqual(lr_at(total, total, 1.0), 0.1, places=6)
        vals = [lr_at(s, total, 1.0) for s in range(20, total + 1)]
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])))


def _big_cfg(**kw):
    return dict(evidence_n=12, evidence_budget=50, init_std=0.1, max_state_tokens=256, **kw)


LONG = " ".join(f"item{i} value {i * 7 % 13}" for i in range(40))


class TestLargeKSubset(unittest.TestCase):
    def test_k255_subset_runs_with_gradients_and_mixed_batch(self):
        model, pk, cfg = _mk(**_big_cfg())
        opts = [f"opt{i}" for i in range(255)]
        qs = [QuestionSpec("choice", "Which?", opts, label=7), QuestionSpec("score", "Severity?", ["l", "m", "h"], label=1)]
        batch = pk.pack([Rhen2Example(LONG, qs)])
        self.assertGreater(batch.state_ids.shape[1], cfg.evidence_n)
        model.train()
        out = model(batch, LoopPlan(n_loops=2, k_bptt=1, min_loops=2, train=True))
        self.assertEqual(out.probs.shape, (2, 256))
        self.assertTrue(bool(torch.isfinite(out.probs).all()))
        self.assertAlmostEqual(float(out.probs[0].sum()), 1.0, places=4)
        self.assertAlmostEqual(float(out.probs[1].sum()), 1.0, places=4)
        self.assertEqual(float(out.probs[1, 3:255].abs().sum()), 0.0)
        out.loss.backward()
        total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        self.assertTrue(total > 0 and total == total)

    def test_subset_path_is_option_order_invariant_and_isolated(self):
        model, pk, _ = _mk(**_big_cfg())
        plan = LoopPlan(n_loops=2, min_loops=2)
        opts = [f"choice number {i} " + "w" * (i % 5) for i in range(20)]
        x = QuestionSpec("choice", "Which?", opts)
        y = QuestionSpec("noul", "Is it urgent?")

        def run(qs):
            with torch.no_grad():
                return model(pk.pack([Rhen2Example(LONG, qs)]), plan)

        base = run([x]).probs[0]
        perm = list(range(20))
        random.Random(1).shuffle(perm)
        got = run([QuestionSpec("choice", "Which?", [opts[i] for i in perm])]).probs[0]
        self.assertGreater(float(base[:20].max() - base[:20].min()), 1e-5)
        self.assertLess(float((got[:20] - base[:20][perm]).abs().max()), 1e-5)
        mixed = run([y, x, y]).probs[1]
        self.assertLess(float((mixed[:20] - base[:20]).abs().max()), 1e-5)

    def test_subset_zero_cross_question_gradient(self):
        model, pk, _ = _mk(**_big_cfg())
        batch = pk.pack([Rhen2Example(LONG, [QuestionSpec("choice", "Which?", ["a", "b", "c"]), QuestionSpec("noul", "urgent?")])])
        S = model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
        q_emb, *_ = model._question_inputs(batch)
        q_emb = q_emb.detach().requires_grad_(True)
        orig = model._question_inputs
        model._question_inputs = lambda b: (q_emb,) + tuple(orig(b)[1:])
        out = model.deliberate(S.detach(), batch, LoopPlan(n_loops=2, min_loops=2))
        model._question_inputs = orig
        out.probs[0, 0].backward()
        self.assertEqual(float(q_emb.grad[1].abs().max()), 0.0)
        self.assertGreater(float(q_emb.grad[0].abs().max()), 0.0)

    def test_stable_tie_break(self):
        # all-equal masses must select the lowest positions, whatever the padding
        score = torch.zeros(2, 10)
        score[:, 0] = float("inf")
        idx = score.sort(dim=-1, descending=True, stable=True).indices[:, :4]
        self.assertEqual(idx[0].tolist(), [0, 1, 2, 3])


class TestSharedStatePath(unittest.TestCase):
    def test_shared_matches_per_question_memory(self):
        """The copy-free shared-state path equals the per-question path (memory_mask all False)."""
        model, pk, _ = _mk(init_std=0.1)
        batch = pk.pack([Rhen2Example(STATE, [QuestionSpec("choice", "Which?", ["a", "b", "c"]), QuestionSpec("noul", "ok?")]),
                         Rhen2Example("short one", [QuestionSpec("score", "s", ["l", "h"])])])
        plan = LoopPlan(n_loops=2, min_loops=2)
        with torch.no_grad():
            S = model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
            a = model.deliberate(S, batch, plan)
            n, ls = batch.q_ids.shape[0] * batch.q_ids.shape[1], S.shape[1]
            b = model.deliberate(S, batch, plan, Interventions(memory_mask=torch.zeros(n, ls, dtype=torch.bool)))
        self.assertLess(float((a.probs - b.probs).abs().max()), 1e-5)

    def test_deterministic_and_empty_state(self):
        model, pk, _ = _mk()
        batch = pk.pack([Rhen2Example("", [QuestionSpec("choice", "Which?", ["a", "b"])])])
        with torch.no_grad():
            a, b = model(batch, PLAN), model(batch, PLAN)
        self.assertTrue(torch.equal(a.probs, b.probs))
        self.assertTrue(bool(torch.isfinite(a.probs).all()) and bool(torch.isfinite(a.span_start).all()))


if __name__ == "__main__":
    unittest.main()
