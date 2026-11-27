"""The structural guarantees of RHEN-2 (plan tasks T-206 and T-207).

Option order and question order cannot change an answer, nothing writes into the state
memory, and a question's loss has exactly zero gradient into any other question's tokens.
"""

from __future__ import annotations

import random
import threading
import unittest

import torch

from rhen.v2.config import preset
from rhen.v2.model import Rhen2Model
from rhen.v2.packing import QuestionSpec, Rhen2Example, Rhen2Packer
from rhen.v2.workspace import LoopPlan

STATE = "Node a1 load 0.97, queue depth 410, region east, 1,200 units at 45% on 2026-03-01."
PLAN = LoopPlan(n_loops=3, min_loops=3)


def _model(**kw):
    torch.manual_seed(0)
    cfg = preset("r2-tiny", init_std=0.15, **kw)     # larger init so options are not near-symmetric
    return Rhen2Model(cfg).eval(), Rhen2Packer(cfg)


def _run(model, packer, questions, state=STATE):
    with torch.no_grad():
        return model(packer.pack([Rhen2Example(state, questions)]), PLAN)


class TestOptionOrder(unittest.TestCase):
    def test_choice_probabilities_permute_exactly(self):
        model, pk = _model()
        opts = ["throttle", "restart the replica pod now", "wait", "fail over to the disaster recovery site", "x"]
        crit = ["slow it down", "reboot it", "do nothing", "switch region", "other"]
        base = _run(model, pk, [QuestionSpec("choice", "Which action?", opts, crit)]).probs[0]
        self.assertGreater(float(base[:5].max() - base[:5].min()), 1e-4)     # the check is not vacuous
        rng = random.Random(0)
        for _ in range(6):
            perm = list(range(5))
            rng.shuffle(perm)
            q = QuestionSpec("choice", "Which action?", [opts[i] for i in perm], [crit[i] for i in perm])
            got = _run(model, pk, [q]).probs[0]
            self.assertLess(float((got[:5] - base[:5][perm]).abs().max()), 1e-5)
            self.assertLess(abs(float(got[5] - base[5])), 1e-5)               # the null column does not move

    def test_multi_and_score_inputs_respect_option_order_where_they_should(self):
        model, pk = _model()
        opts = ["a policy", "b policy", "c policy", "d policy"]
        base = _run(model, pk, [QuestionSpec("multi", "Which apply?", opts)]).multi_prob[0, :4]
        perm = [2, 0, 3, 1]
        got = _run(model, pk, [QuestionSpec("multi", "Which apply?", [opts[i] for i in perm])]).multi_prob[0, :4]
        self.assertLess(float((got - base[perm]).abs().max()), 1e-5)
        # score levels are ordered, so reversing them is allowed to change the answer
        levels = ["low", "mid", "high"]
        a = _run(model, pk, [QuestionSpec("score", "Severity?", levels)]).probs[0, :3]
        b = _run(model, pk, [QuestionSpec("score", "Severity?", levels[::-1])]).probs[0, :3]
        self.assertGreater(float((a - b.flip(0)).abs().max()), 1e-6)


class TestQuestionIsolation(unittest.TestCase):
    def setUp(self):
        self.model, self.pk = _model()
        self.x = QuestionSpec("choice", "Which action first?", ["throttle", "restart", "wait"])
        self.y = QuestionSpec("noul", "Page the on-call?")
        self.z = QuestionSpec("choice", "Which region?", ["east", "west", "north", "south"])

    def test_answer_is_the_same_alone_with_others_and_in_any_order(self):
        alone = _run(self.model, self.pk, [self.x]).probs[0]          # [a, b, c, null]
        for others in ([self.y], [self.y, self.z], [self.z, self.y]):
            for first in (True, False):
                qs = [self.x] + others if first else others + [self.x]
                out = _run(self.model, self.pk, qs)
                row = 0 if first else len(others)
                got = out.probs[row]                                   # null sits at column K, which can shift
                self.assertLess(float((got[:3] - alone[:3]).abs().max()), 1e-5)
                self.assertLess(abs(float(got[-1] - alone[-1])), 1e-5)
                self.assertEqual(float(got[3:-1].abs().sum()), 0.0)    # padded option columns carry no mass

    def test_changing_another_question_changes_nothing(self):
        other = QuestionSpec("noul", "A completely different proposition about shipping?")
        a = _run(self.model, self.pk, [self.x, self.y]).probs[0]
        b = _run(self.model, self.pk, [self.x, other]).probs[0]
        self.assertLess(float((a - b).abs().max()), 1e-5)

    def test_state_memory_does_not_depend_on_questions(self):
        b1 = self.pk.pack([Rhen2Example(STATE, [self.x])])
        b2 = self.pk.pack([Rhen2Example(STATE, [self.y, self.z])])
        with torch.no_grad():
            s1 = self.model.encode_state(b1.state_ids, b1.state_mask, b1.state_num)
            s2 = self.model.encode_state(b2.state_ids, b2.state_mask, b2.state_num)
        self.assertTrue(torch.equal(s1, s2))

    def test_cached_state_gives_identical_answers(self):
        batch = self.pk.pack([Rhen2Example(STATE, [self.x, self.y])])
        with torch.no_grad():
            full = self.model(batch, PLAN)
            memory = self.model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
            cached = self.model.deliberate(memory, batch, PLAN)
        self.assertTrue(torch.equal(full.probs, cached.probs))

    def test_concurrent_calls_match_sequential_calls(self):
        batches = [self.pk.pack([Rhen2Example(STATE + f" run {i}", [self.x, self.y])]) for i in range(4)]
        with torch.no_grad():
            expected = [self.model(b, PLAN).probs.clone() for b in batches]
        results = [None] * 4

        def work(i):
            with torch.no_grad():
                for _ in range(5):
                    results[i] = self.model(batches[i], PLAN).probs.clone()

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        for got, want in zip(results, expected):
            self.assertLess(float((got - want).abs().max()), 1e-6)

    def test_gradient_into_other_questions_is_exactly_zero(self):
        batch = self.pk.pack([Rhen2Example(STATE, [self.x, self.y, self.z])])
        grads = []

        def keep(module, inputs, output):
            output.retain_grad()
            grads.append(output)

        handle = self.model.embed.register_forward_hook(keep)
        try:
            out = self.model(batch, PLAN)
            out.probs[0, 0].log().backward()           # only the first question's answer
        finally:
            handle.remove()
        state_out, q_out, o_out = grads                # embedding calls: state, questions, options
        q_grad, o_grad = q_out.grad, o_out.grad
        self.assertGreater(float(q_grad[0, 0].abs().sum()), 0.0)
        self.assertGreater(float(o_grad[0, 0].abs().sum()), 0.0)
        self.assertEqual(float(q_grad[0, 1:].abs().sum()), 0.0)
        self.assertEqual(float(o_grad[0, 1:].abs().sum()), 0.0)
        self.assertGreater(float(state_out.grad.abs().sum()), 0.0)     # the state does receive gradient


class TestHaltingAndEvidence(unittest.TestCase):
    def test_halting_stops_early_and_keeps_halting_loop_output(self):
        model, pk = _model()
        with torch.no_grad():
            model.readout.h2.bias.fill_(10.0)          # force "done" from the first allowed loop
        batch = pk.pack([Rhen2Example(STATE, [QuestionSpec("choice", "q?", ["a", "b"])])])
        with torch.no_grad():
            early = model(batch, LoopPlan(n_loops=3, min_loops=1, halting=True, halt_threshold=0.5))
            one = model(batch, LoopPlan(n_loops=1, min_loops=1))
        self.assertEqual(int(early.loops_run[0]), 1)
        self.assertLess(float((early.probs - one.probs).abs().max()), 1e-6)

    def test_evidence_is_a_distribution_over_real_state_tokens(self):
        model, pk = _model()
        batch = pk.pack([Rhen2Example(STATE, [QuestionSpec("noul", "ok?")]), Rhen2Example("short state", [QuestionSpec("noul", "ok?")])])
        out = _run_batch(model, batch)
        self.assertLess(float((out.evidence.sum(-1) - 1).abs().max()), 1e-5)
        pad = ~batch.state_mask
        self.assertEqual(float(out.evidence[pad].abs().max()), 0.0)


def _run_batch(model, batch):
    with torch.no_grad():
        return model(batch, PLAN)


class TestInterventions(unittest.TestCase):
    def test_memory_edits_change_answers_and_are_per_question(self):
        from rhen.v2.workspace import Interventions

        model, pk = _model()
        batch = pk.pack([Rhen2Example(STATE, [QuestionSpec("choice", "q?", ["a", "b", "c"]), QuestionSpec("noul", "ok?")])])
        with torch.no_grad():
            S = model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
            base = model.deliberate(S, batch, PLAN)
            same = model.deliberate(S, batch, PLAN, Interventions(memory_mask=torch.zeros(2, S.shape[1], dtype=torch.bool)))
            masked = torch.zeros(2, S.shape[1], dtype=torch.bool)
            masked[0, 1:] = True
            edited = model.deliberate(S, batch, PLAN, Interventions(memory_mask=masked))
            empty = model.deliberate(S, batch, PLAN, Interventions(empty_memory=True))
            raised = model.deliberate(S, batch, PLAN, Interventions(null_delta=1.0))
        self.assertLess(float((base.probs - same.probs).abs().max()), 1e-5)      # an empty edit is a no-op
        self.assertGreater(float((base.probs[0] - edited.probs[0]).abs().max()), 1e-6)
        self.assertLess(float((base.probs[1] - edited.probs[1]).abs().max()), 1e-5)  # other question untouched
        self.assertGreater(float((base.probs - empty.probs).abs().max()), 1e-6)
        self.assertTrue(bool((raised.probs[:, -1] > base.probs[:, -1]).all()))     # null_delta raises P(null)


if __name__ == "__main__":
    unittest.main()
