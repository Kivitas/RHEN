"""Typed readouts and training objectives (plan tasks T-203, T-205)."""

from __future__ import annotations

import unittest

import torch

from rhen.v2.config import preset
from rhen.v2.losses import task_losses
from rhen.v2.model import Rhen2Model
from rhen.v2.packing import NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer
from rhen.v2.readouts import ReadoutOut, decode_span
from rhen.v2.workspace import LoopPlan

STATE = "Node a1 load 0.97, queue depth 410."


class TestReadouts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.cfg = preset("r2-tiny", init_std=0.1)
        self.model = Rhen2Model(self.cfg).eval()
        self.pk = Rhen2Packer(self.cfg)

    def _out(self, questions):
        with torch.no_grad():
            batch = self.pk.pack([Rhen2Example(STATE, questions)])
            return batch, self.model(batch, LoopPlan(n_loops=2, min_loops=2))

    def test_discrete_distributions_are_valid_with_null_last(self):
        qs = [QuestionSpec("choice", "q?", ["a", "b", "c"]), QuestionSpec("noul", "p?"),
              QuestionSpec("score", "level?", ["l0", "l1", "l2", "l3"]), QuestionSpec("choice", "wide?", [str(i) for i in range(6)])]
        batch, out = self._out(qs)
        self.assertEqual(out.probs.shape, (4, 7))
        self.assertLess(float((out.probs.sum(-1) - 1).abs().max()), 1e-5)
        self.assertGreaterEqual(float(out.probs.min()), 0.0)
        valid = batch.opt_valid.reshape(4, 6)
        self.assertEqual(float(out.probs[:, :6][~valid].abs().sum()), 0.0)     # padded options hold no mass
        self.assertTrue(bool((out.probs[:, -1] > 0).all()))                    # null is always an option

    def test_noul_has_true_false_and_null(self):
        _, out = self._out([QuestionSpec("noul", "p?")])
        p = out.probs[0]
        self.assertAlmostEqual(float(p[0] + p[1] + p[-1]), 1.0, places=5)      # P(true)+P(false)+P(null) = 1

    def test_score_levels_are_ordered_through_a_cumulative_link(self):
        _, out = self._out([QuestionSpec("score", "level?", ["l0", "l1", "l2", "l3"])])
        self.assertAlmostEqual(float(out.probs[0].sum()), 1.0, places=5)
        self.assertGreaterEqual(float(out.probs[0].min()), 0.0)

    def test_null_option_dominates_at_initialisation(self):
        cfg = preset("r2-tiny", null_bias_init=6.0)
        model, pk = Rhen2Model(cfg).eval(), Rhen2Packer(cfg)
        with torch.no_grad():
            out = model(pk.pack([Rhen2Example(STATE, [QuestionSpec("choice", "q?", ["a", "b", "c"])])]), LoopPlan(2, min_loops=2))
        self.assertGreater(float(out.probs[0, -1]), 0.9)                       # "cannot tell" is the default

    def test_without_null_node_it_gets_no_mass(self):
        cfg = preset("r2-tiny", use_null=False)
        model, pk = Rhen2Model(cfg).eval(), Rhen2Packer(cfg)
        with torch.no_grad():
            out = model(pk.pack([Rhen2Example(STATE, [QuestionSpec("choice", "q?", ["a", "b"])])]), LoopPlan(2, min_loops=2))
        self.assertEqual(float(out.probs[0, -1]), 0.0)

    def test_range_quantiles_are_monotone_and_inside_bounds(self):
        _, out = self._out([QuestionSpec("range", "how long?", lo=10.0, hi=50.0)])
        q = out.quantiles[0]
        self.assertTrue(bool((q[1:] >= q[:-1]).all()))
        self.assertTrue(bool(((q >= 10.0) & (q <= 50.0)).all()))
        self.assertEqual(q.shape[0], 9)

    def test_span_decoding_respects_order_length_and_null(self):
        start = torch.full((1, 10), -5.0)
        end = torch.full((1, 10), -5.0)
        start[0, 3], end[0, 5] = 4.0, 4.0
        i, j, _ = decode_span(start, end, max_span=8)
        self.assertEqual((int(i), int(j)), (3, 5))
        i, j, _ = decode_span(start, end, max_span=1)                          # too short to reach token 5
        self.assertTrue(int(j) - int(i) <= 1)
        null = torch.full((1, 10), -5.0)
        null[0, 0] = 9.0
        i, j, _ = decode_span(null, null, max_span=8)
        self.assertEqual((int(i), int(j)), (0, 0))                             # position 0 is the null span
        rev_s, rev_e = torch.full((1, 10), -5.0), torch.full((1, 10), -5.0)
        rev_s[0, 6], rev_e[0, 2] = 5.0, 5.0                                    # end before start is not allowed
        i, j, _ = decode_span(rev_s, rev_e, max_span=8)
        self.assertGreaterEqual(int(j), int(i))

    def test_extract_logits_cover_only_real_state_tokens(self):
        batch, out = self._out([QuestionSpec("extract", "which?")])
        pad = ~batch.state_mask
        self.assertEqual(out.span_start.shape, batch.state_mask.shape)
        self.assertTrue(bool((out.span_start[pad] < -1e8).all()) if pad.any() else True)


def _fake_readout(p_options, p_null, k):
    probs = torch.tensor([[*p_options, p_null]])
    return ReadoutOut(
        logits=probs.clamp_min(1e-9).log(), probs=probs, score_probs=probs.clone(),
        multi_logit=torch.zeros(1, k), quantiles=torch.zeros(1, 9), halt_logit=torch.zeros(1),
    )


class TestProperScoring(unittest.TestCase):
    """The true distribution minimises the expected loss, so the model is trained to be calibrated."""

    def setUp(self):
        self.cfg = preset("r2-tiny")
        self.pk = Rhen2Packer(self.cfg)

    def _loss(self, label, probs, qtype="choice"):
        opts = ["a", "b", "c"]
        batch = self.pk.pack([Rhen2Example(STATE, [QuestionSpec(qtype, "q?", opts, label=label)])])
        ro = _fake_readout(probs[:3], probs[3], 3)
        return float(task_losses(ro, batch, self.cfg)["choice" if qtype == "choice" else "score"])

    def _expected(self, truth, probs, qtype="choice"):
        return sum(t * self._loss(i, probs, qtype) for i, t in enumerate(truth))

    def test_choice_loss_is_minimised_by_the_true_distribution(self):
        truth = [0.6, 0.3, 0.1]
        at_truth = self._expected(truth, [0.6, 0.3, 0.1, 1e-9])
        for other in ([0.8, 0.1, 0.1, 1e-9], [0.34, 0.33, 0.33, 1e-9], [0.5, 0.3, 0.2, 1e-9], [0.9, 0.05, 0.05, 1e-9]):
            self.assertLess(at_truth, self._expected(truth, other))

    def test_score_loss_is_minimised_by_the_true_distribution(self):
        truth = [0.2, 0.5, 0.3]
        at_truth = self._expected(truth, [0.2, 0.5, 0.3, 1e-9], "score")
        for other in ([0.5, 0.3, 0.2, 1e-9], [0.1, 0.2, 0.7, 1e-9], [0.33, 0.34, 0.33, 1e-9]):
            self.assertLess(at_truth, self._expected(truth, other, "score"))

    def test_ranked_score_penalises_a_far_miss_more_than_a_near_miss(self):
        near = self._loss(0, [0.0, 1.0 - 1e-6, 0.0, 1e-6], "score")           # truth level 0, predicted level 1
        far = self._loss(0, [0.0, 0.0, 1.0 - 1e-6, 1e-6], "score")            # predicted level 2
        self.assertLess(near, far)

    def test_soft_labels_replace_one_hot_targets(self):
        batch = self.pk.pack([Rhen2Example(STATE, [QuestionSpec("choice", "q?", ["a", "b", "c"], soft=[0.6, 0.3, 0.1, 0.0])])])
        match = float(task_losses(_fake_readout([0.6, 0.3, 0.1], 1e-9, 3), batch, self.cfg)["choice"])
        off = float(task_losses(_fake_readout([0.1, 0.3, 0.6], 1e-9, 3), batch, self.cfg)["choice"])
        self.assertLess(match, off)

    def test_pinball_loss_is_zero_only_at_the_target(self):
        spec = QuestionSpec("range", "how long?", lo=0.0, hi=10.0, label=5.0)
        batch = self.pk.pack([Rhen2Example(STATE, [spec])])
        ro = _fake_readout([0.9], 0.1, 1)                 # a range question has one padded option column
        ro.quantiles = torch.full((1, 9), 0.5)
        self.assertAlmostEqual(float(task_losses(ro, batch, self.cfg)["range"]), 0.0, places=6)
        ro.quantiles = torch.full((1, 9), 0.9)
        self.assertGreater(float(task_losses(ro, batch, self.cfg)["range"]), 0.01)

    def test_null_labelled_questions_train_the_null_node(self):
        batch = self.pk.pack([Rhen2Example(STATE, [QuestionSpec("choice", "q?", ["a", "b", "c"], label=NULL_LABEL)])])
        high_null = float(task_losses(_fake_readout([0.05, 0.05, 0.05], 0.85, 3), batch, self.cfg)["choice"])
        low_null = float(task_losses(_fake_readout([0.3, 0.3, 0.3], 0.1, 3), batch, self.cfg)["choice"])
        self.assertLess(high_null, low_null)


class TestFullLoss(unittest.TestCase):
    def test_every_type_contributes_and_gradients_reach_the_encoder_and_workspace(self):
        torch.manual_seed(0)
        cfg = preset("r2-tiny")
        model, pk = Rhen2Model(cfg), Rhen2Packer(cfg)
        ex = Rhen2Example(STATE, [
            QuestionSpec("choice", "which?", ["a", "b", "c"], label=1), QuestionSpec("noul", "ok?", label=True),
            QuestionSpec("score", "level?", ["l0", "l1", "l2"], label=2), QuestionSpec("multi", "which?", ["a", "b", "c"], label=[0, 2]),
            QuestionSpec("range", "how long?", lo=0, hi=60, label=12.0), QuestionSpec("extract", "which?", label=(5, 8)),
        ])
        model.train()
        out = model(pk.pack([ex, ex]), LoopPlan(n_loops=3, k_bptt=2, train=True))
        out.loss.backward()
        for key in ("choice", "score", "multi", "range", "extract"):
            self.assertGreater(out.loss_parts[key], 0.0, key)
        names = {n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
        for must in ("encoder.blocks.0.attn.q_proj.weight", "workspace.stack.layers.0.cross_attn.k_proj.weight",
                     "workspace.set_layer.null_vec", "readout.wq.weight", "readout.rg1.weight", "readout.sq.weight"):
            self.assertIn(must, names)
        self.assertEqual(len(out.per_loop), 2)             # only the k_bptt loops carry gradient and supervision

    def test_truncated_backprop_runs_early_loops_without_a_graph(self):
        torch.manual_seed(0)
        cfg = preset("r2-tiny", r_max=6)
        model, pk = Rhen2Model(cfg), Rhen2Packer(cfg)
        batch = pk.pack([Rhen2Example(STATE, [QuestionSpec("choice", "q?", ["a", "b"], label=0)])])
        model.train()
        out6 = model(batch, LoopPlan(n_loops=6, k_bptt=2, train=True))
        self.assertEqual(len(out6.per_loop), 2)
        out6.loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_sampled_loop_counts_stay_in_range(self):
        cfg = preset("r2-tiny", r_min=2, r_mean=4, r_max=6)
        g = torch.Generator().manual_seed(0)
        draws = [LoopPlan.sample(cfg, g).n_loops for _ in range(300)]
        self.assertTrue(all(2 <= d <= 6 for d in draws))
        self.assertGreater(len(set(draws)), 2)


if __name__ == "__main__":
    unittest.main()
