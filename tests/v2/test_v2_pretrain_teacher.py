"""Pretraining builders (T-306), benchmark blocklist (T-304) and teacher labelling (T-305)."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest

import torch

from rhen.v2.config import preset
from rhen.v2.data.blocklist import (
    BlockedExampleError, Blocklist, build_ngram_index, example_fingerprint, ngram_overlap,
)
from rhen.v2.data.pretrain import (
    IGNORE_INDEX, K_CHOICES, MLMHead, PretrainMixture, SpanChoiceBuilder, SpanLocateBuilder, TextCorpus,
    build_mlm_example, collate_mlm, mask_token_id, mlm_loss, structured_text,
)
from rhen.v2.data.schema import from_v1_example, read_jsonl
from rhen.v2.data.teacher import (
    EnsembleTeacher, FakeTeacher, HTTPTeacher, TeacherError, build_option_probability_prompt,
    main as teacher_main, parse_probability_response, run_labelling,
)
from rhen.v2.model import Rhen2Model
from rhen.v2.packing import NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer

CFG = preset("r2-tiny")
PACKER = Rhen2Packer(CFG)
PROSE = [
    "The harbour authority announced on Tuesday that the new cargo terminal will open early next spring, "
    "after months of delays caused by storms, supplier disputes and a shortage of qualified crane operators.",
    "Researchers measured the growth of the colony over fourteen days and found that temperature, rather than "
    "nutrient supply, explained most of the variation between the samples they collected in the northern basin.",
    "A careful reader will notice that every chapter of the book returns to the same small village, where the "
    "baker, the schoolteacher and the retired surveyor argue quietly about the price of coal and the weather.",
    "Several councils have asked for guidance on how to count households that share a single address, because "
    "the current form assumes that each front door belongs to exactly one family with one primary income.",
]


def _pool(n=12, seed=0):
    rng = random.Random(seed)
    return PROSE + [structured_text(rng) for _ in range(n)]


class TestMLM(unittest.TestCase):
    def test_masking_rate_and_untouched_positions(self):
        rng = random.Random(0)
        toks = list(range(100, 1100))
        inp, tgt = build_mlm_example(toks, 0.3, rng, mask_id=999, vocab_size=5000)
        rate = sum(t != IGNORE_INDEX for t in tgt) / len(toks)
        self.assertAlmostEqual(rate, 0.3, delta=0.01)
        for i, (a, b, t) in enumerate(zip(inp, tgt, toks)):
            if b == IGNORE_INDEX:
                self.assertEqual(a, t)
            else:
                self.assertEqual(b, t)
        self.assertGreater(sum(1 for a, b in zip(inp, tgt) if b != IGNORE_INDEX and a == 999), 0.7 * 300)
        _, tgt0 = build_mlm_example(toks, 0.9, rng, 999, keep_first=True)
        self.assertEqual(tgt0[0], IGNORE_INDEX)

    def test_collate_keeps_control_and_padding(self):
        mid = mask_token_id(PACKER)
        self.assertEqual(mid, PACKER.base + 10)
        self.assertLess(mid, CFG.vocab_size)
        self.assertNotIn(mid, PACKER.ctrl.values())
        lists = [PACKER.tok.encode(p, add_special_tokens=False)[:40] for p in PROSE[:3]]
        lists[1] = lists[1][:20]
        b = collate_mlm(lists, PACKER, 0.3, random.Random(1))
        self.assertTrue((b["state_ids"][:, 0] == PACKER.ctrl["state"]).all())
        self.assertTrue((b["targets"][:, 0] == IGNORE_INDEX).all())
        pad = ~b["state_mask"]
        self.assertTrue(pad.any())
        self.assertTrue((b["state_ids"][pad] == PACKER.pad_id).all())
        self.assertTrue((b["targets"][pad] == IGNORE_INDEX).all())
        self.assertTrue(((b["state_ids"] == mid).sum() > 0).item())

    def test_loss_decreases_on_repeated_passage(self):
        torch.manual_seed(0)
        model = Rhen2Model(CFG)
        head = MLMHead.for_model(model)
        self.assertGreater(head.count(), 0)
        self.assertFalse(any(p is model.embed.table.weight for p in head.parameters()))
        toks = PACKER.tok.encode(PROSE[0], add_special_tokens=False)[:48]
        rng = random.Random(0)
        params = list(model.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.0)
        losses = []
        for _ in range(60):
            b = collate_mlm([toks] * 4, PACKER, 0.3, rng)
            loss = mlm_loss(model, head, b["state_ids"], b["targets"], b["state_mask"], b["state_num"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        self.assertLess(sum(losses[-5:]) / 5, 0.6 * sum(losses[:5]) / 5)


class TestSpanBuilders(unittest.TestCase):
    def test_span_choice_invariants(self):
        builder = SpanChoiceBuilder(PACKER)
        pool = _pool()
        rng = random.Random(3)
        n, nulls, same, other = 2000, 0, 0, 0
        for i in range(n):
            ex, meta = builder.build_with_meta(rng.choice(pool), rng, pool)
            q = ex.questions[0]
            self.assertEqual(q.type, "choice")
            self.assertEqual(q.instructions, "which text fills the gap?")
            self.assertIn("[GAP]", ex.state)
            self.assertIn(len(q.options), K_CHOICES)
            self.assertEqual(len(set(q.options)), len(q.options))
            if q.label == NULL_LABEL:
                nulls += 1
                self.assertNotIn(meta["gold"], q.options)
                self.assertNotIn("true", meta["sources"])
            else:
                self.assertEqual(q.options[q.label], meta["gold"])
                self.assertEqual(meta["sources"][q.label], "true")
            same += meta["sources"].count("same")
            other += meta["sources"].count("other")
        self.assertAlmostEqual(nulls / n, 0.12, delta=0.03)
        self.assertGreater(same, 0)
        self.assertGreater(other, 0)
        batch = PACKER.pack([builder.build(pool[0], rng, pool) for _ in range(4)])
        self.assertEqual(batch.state_ids.shape[0], 4)

    def test_span_locate_offsets_exact(self):
        rng = random.Random(5)
        pool = _pool()
        for dict_rate in (0.0, 1.0):
            builder = SpanLocateBuilder(PACKER, dict_rate=dict_rate, perturb_rate=0.0)
            for _ in range(60):
                ex, meta = builder.build_with_meta(rng.choice(pool), rng)
                q = ex.questions[0]
                s, e = q.label
                full = PACKER.serialise_state(ex.state)
                self.assertEqual(isinstance(ex.state, dict), dict_rate == 1.0)
                self.assertEqual(full, meta["serialised"])
                got = full[s:e]
                if dict_rate == 1.0:
                    got = json.loads('"' + got + '"')
                self.assertEqual(got, meta["span"])
                self.assertEqual(" ".join(q.instructions.split()), "locate: " + " ".join(meta["span"].split()[:14]))
        PACKER.pack([builder.build(PROSE[0], rng) for _ in range(3)])   # labels are accepted by the packer

    def test_span_locate_perturbation(self):
        rng = random.Random(7)
        builder = SpanLocateBuilder(PACKER, perturb_rate=1.0, min_span=6, max_span=12)
        flags = [builder.build_with_meta(PROSE[1], rng)[1]["perturbed"] for _ in range(100)]
        self.assertGreater(sum(flags), 30)


class TestMixture(unittest.TestCase):
    def test_determinism_and_proportions(self):
        mix = PretrainMixture(PACKER, PROSE, seed=11)
        a, b = mix.sample_batch(200, step=3), PretrainMixture(PACKER, PROSE, seed=11).sample_batch(200, step=3)
        self.assertEqual(a.kinds, b.kinds)
        self.assertEqual(a.mlm, b.mlm)
        self.assertEqual([x.state for x in a.examples], [x.state for x in b.examples])
        self.assertNotEqual(a.kinds, mix.sample_batch(200, step=4).kinds)
        frac = {k: a.kinds.count(k) / 200 for k in mix.KINDS}
        self.assertAlmostEqual(frac["mlm"], 0.4, delta=0.1)
        self.assertAlmostEqual(frac["span_choice"], 0.4, delta=0.1)
        self.assertAlmostEqual(frac["span_locate"], 0.2, delta=0.1)
        self.assertEqual(len(a.mlm) + len(a.examples), 200)
        self.assertTrue(all(isinstance(t, int) for t in a.mlm[0]))

    def test_text_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.txt"), "w") as fh:
                fh.write(PROSE[0] + "\n\nshort\n\n" + PROSE[1])
            with open(os.path.join(d, "b.jsonl"), "w") as fh:
                fh.write(json.dumps({"text": PROSE[2]}) + "\n" + json.dumps(PROSE[3]) + "\n")
            self.assertEqual(TextCorpus(d).passages(), PROSE[:2] + PROSE[2:])
        self.assertEqual(structured_text(random.Random(1)), structured_text(random.Random(1)))


class TestBlocklist(unittest.TestCase):
    def ex(self, state="The quick brown fox", opts=("a", "b")):
        return Rhen2Example(state, [QuestionSpec("choice", "Pick one?", list(opts), label=0)])

    def test_default_file_loads_with_empty_fingerprints(self):
        bl = Blocklist.load()
        for name in ("banking77-test", "clinc150-test", "squad2-validation", "jev-evals", "microsoft-decision-1-suite"):
            self.assertIn(name, bl.names)
        self.assertEqual(bl.n_fingerprints, 0)
        self.assertFalse(bl.contains(self.ex()))

    def test_fingerprint_stability(self):
        a = example_fingerprint(self.ex())
        self.assertEqual(a, example_fingerprint(self.ex("  the QUICK   brown\nfox ")))
        self.assertNotEqual(a, example_fingerprint(self.ex(opts=("a", "c"))))
        self.assertEqual(example_fingerprint(self.ex({"x": 1, "y": 2})), example_fingerprint(self.ex({"y": 2, "x": 1})))

    def test_hit_filter_and_roundtrip(self):
        bl = Blocklist.load()
        bl.add_examples("banking77-test", [self.ex()])
        with self.assertRaises(BlockedExampleError):
            bl.check(self.ex("THE quick brown fox"))
        fresh = self.ex("another state")
        kept = list(bl.filter([self.ex(), fresh]))
        self.assertEqual(kept, [fresh])
        with self.assertRaises(BlockedExampleError):
            list(bl.filter([fresh, self.ex()], raise_on_hit=True))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.json")
            bl.save(p)
            bl2 = Blocklist.load(p)
            self.assertTrue(bl2.contains(self.ex()))
            self.assertFalse(bl2.contains(fresh))

    def test_ngram_overlap_detects_copy(self):
        words = " ".join(f"w{i}" for i in range(40))
        bench = self.ex(words)
        index = build_ngram_index([bench], 13)
        copied = self.ex("intro text " + " ".join(f"w{i}" for i in range(5, 30)) + " outro")
        self.assertGreater(ngram_overlap(copied, index, 13), 0.5)
        self.assertEqual(ngram_overlap(self.ex("completely different words here"), index, 13), 0.0)
        bl = Blocklist()
        bl.add_examples("x", [bench])
        self.assertTrue(bl.contains(copied, ngram_threshold=0.3))
        self.assertFalse(bl.contains(copied))


class TestTeacher(unittest.TestCase):
    def q(self):
        return QuestionSpec("choice", "Which?", ["a", "b", "c"])

    def test_fake_teacher_deterministic_and_valid(self):
        t = FakeTeacher(seed=1)
        p = t.label("state", self.q())
        self.assertEqual(p, FakeTeacher(seed=1).label("state", self.q()))
        self.assertNotEqual(p, FakeTeacher(seed=2).label("state", self.q()))
        self.assertEqual(len(p), 4)
        self.assertAlmostEqual(sum(p), 1.0, places=6)

    def test_ensemble_average_and_disagreement(self):
        class Fixed:
            def __init__(self, p):
                self.p = p

            def label(self, state, question):
                return list(self.p)

        ens = EnsembleTeacher([Fixed([1.0, 0.0, 0.0]), Fixed([0.0, 1.0, 0.0])])
        out = ens.label("s", QuestionSpec("choice", "q", ["a", "b"]))
        self.assertEqual(out, [0.5, 0.5, 0.0])
        self.assertGreater(ens.last_disagreement, 0.6)
        same = EnsembleTeacher([Fixed([0.2, 0.3, 0.5])] * 2)
        same.label("s", QuestionSpec("choice", "q", ["a", "b"]))
        self.assertAlmostEqual(same.last_disagreement, 0.0, places=9)

    def test_http_teacher_and_parsing(self):
        with self.assertRaises(TeacherError):
            HTTPTeacher().label("s", self.q())
        self.assertIn("exactly 4", build_option_probability_prompt("s", self.q()))
        t = HTTPTeacher(lambda prompt: "Sure: [0.5, 0.25, 0.15, 0.1]")
        self.assertEqual(t.label("s", self.q()), [0.5, 0.25, 0.15, 0.1])
        with self.assertRaises(TeacherError):
            parse_probability_response("[1, 2]", 3)

    def _examples(self, n=12):
        from rhen.research.tasks import generate_split
        return [from_v1_example(e) for e in generate_split("train", n, seed=0)]

    def test_run_labelling_soft_labels_and_resume(self):
        exs = self._examples()
        with tempfile.TemporaryDirectory() as d:
            full, part = os.path.join(d, "full.jsonl"), os.path.join(d, "part.jsonl")
            s = run_labelling(exs, EnsembleTeacher([FakeTeacher(1), FakeTeacher(2)]), full)
            self.assertEqual(s["written"], len({example_fingerprint(e) for e in exs}))
            run_labelling(exs[:5], EnsembleTeacher([FakeTeacher(1), FakeTeacher(2)]), part)
            s2 = run_labelling(exs, EnsembleTeacher([FakeTeacher(1), FakeTeacher(2)]), part)
            self.assertGreater(s2["skipped"], 0)
            with open(full) as a, open(part) as b:
                self.assertEqual(a.read(), b.read())
            s3 = run_labelling(exs, FakeTeacher(1), part)
            self.assertEqual(s3["written"], 0)
            for ex in read_jsonl(full):
                for q in ex.questions:
                    k = len(q.options) if q.type != "noul" else 2
                    self.assertEqual(len(q.soft), k + 1)
                    self.assertAlmostEqual(sum(q.soft), 1.0, places=6)
            recs = [json.loads(line) for line in open(full)]
            self.assertTrue(all("disagreement" in r["meta"] for r in recs))
            # soft labels pack with the null column last
            PACKER.pack(list(read_jsonl(full))[:4])

    def test_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "o.jsonl")
            self.assertEqual(teacher_main(["--dry-run", "--n", "20", "--out", out]), 0)
            self.assertEqual(sum(1 for _ in open(out)), 20)


if __name__ == "__main__":
    unittest.main()
