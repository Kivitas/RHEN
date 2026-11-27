"""RHEN-2 packing (plan task T-103)."""

from __future__ import annotations

import unittest

from rhen.v2.config import preset
from rhen.v2.data.schema import example_from_record, example_to_record
from rhen.v2.packing import (
    IGNORE, NULL_INDEX, NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer, k_bucket, make_batches,
)


class TestPacking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = preset("r2-tiny")
        cls.pk = Rhen2Packer(cls.cfg)

    def test_control_ids_are_above_the_tokenizer_and_cannot_come_from_text(self):
        ex = Rhen2Example("a <|state|> b <|decide|> c", [QuestionSpec("noul", "<|q|> ok?", label=True)])
        b = self.pk.pack([ex])
        base = self.pk.base
        self.assertEqual(int(b.state_ids[0, 0]), self.pk.ctrl["state"])
        self.assertFalse(bool((b.state_ids[0, 1:][b.state_mask[0, 1:]] >= base).any()))
        self.assertFalse(bool((b.q_ids[0, 0, 1:][b.q_mask[0, 0, 1:]] >= base).any()))
        self.assertGreaterEqual(self.pk.ctrl["state"], base)

    def test_numbers_dates_and_percents_get_features_but_identifiers_do_not(self):
        text = "Node a1 load 0.97, 1,200 units at 45% on 2026-03-01 id x7."
        b = self.pk.pack([Rhen2Example(text, [QuestionSpec("noul", "ok?")])])
        feats = b.state_num[0]
        hit = (feats.abs().sum(-1) > 0).nonzero().flatten().tolist()
        flags = sorted(feats[hit, 3].tolist())
        self.assertEqual(flags, [0.0, 0.0, 1.0, 2.0])           # two plain numbers, one percent, one date
        self.assertEqual(len(hit), 4)                          # a1 and x7 are identifiers, not numbers
        self.assertEqual(float(feats[0].abs().sum()), 0.0)     # the control token never carries a number

    def test_shapes_masks_and_labels(self):
        ex = Rhen2Example({"x": 1}, [
            QuestionSpec("choice", "which?", ["a", "bb"], label="bb"),
            QuestionSpec("noul", "ok?", label=False),
            QuestionSpec("score", "level?", ["lo", "mid", "hi"], label=NULL_LABEL),
            QuestionSpec("multi", "which?", ["a", "b", "c"], label=["a", "c"]),
            QuestionSpec("range", "how many?", lo=0, hi=10, label=2.5),
            QuestionSpec("extract", "which?", label=NULL_LABEL),
        ])
        b = self.pk.pack([ex, Rhen2Example("short", [QuestionSpec("choice", "q", ["x", "y"], label=0)])])
        self.assertEqual(b.q_ids.shape[:2], (2, 6))
        self.assertEqual(b.opt_ids.shape[2], 3)
        self.assertEqual(b.label_idx[0].tolist(), [1, 1, NULL_INDEX, IGNORE, IGNORE, IGNORE])
        self.assertTrue(bool(b.has_multi[0, 3]))
        self.assertEqual(b.label_multi[0, 3].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(float(b.label_range[0, 4]), 2.5)
        self.assertEqual((int(b.span_start[0, 5]), int(b.span_end[0, 5])), (0, 0))
        self.assertFalse(bool(b.q_valid[1, 1:].any()))
        self.assertEqual(b.opt_valid[0, 0].tolist(), [True, True, False])
        self.assertEqual(b.opt_valid[0, 1].tolist(), [True, True, False])   # noul has the implied true/false

    def test_extract_label_maps_chars_to_token_span(self):
        text = "alpha beta gamma"
        b = self.pk.pack([Rhen2Example(text, [QuestionSpec("extract", "which?", label=(6, 10))])])
        s, e = int(b.span_start[0, 0]), int(b.span_end[0, 0])
        a, z = b.state_offsets[0][s][0], b.state_offsets[0][e][1]
        self.assertEqual(text[a:z].strip(), "beta")

    def test_limits_raise_clear_errors(self):
        with self.assertRaises(ValueError):
            self.pk.pack([Rhen2Example("x " * 400, [QuestionSpec("noul", "q")])])
        with self.assertRaises(ValueError):
            QuestionSpec("choice", "q", []).validate(self.cfg)
        with self.assertRaises(ValueError):
            QuestionSpec("noul", "q", ["a", "b", "c"]).validate(self.cfg)
        with self.assertRaises(ValueError):
            QuestionSpec("range", "q", lo=1, hi=1).validate(self.cfg)
        with self.assertRaises(ValueError):
            QuestionSpec("choice", "q", [str(i) for i in range(256)]).validate(self.cfg)
        with self.assertRaises(ValueError):
            self.pk.pack([Rhen2Example("s", [QuestionSpec("choice", "q", ["a", "b"], label="zzz")])])

    def test_soft_labels_are_normalised_with_null_last(self):
        b = self.pk.pack([Rhen2Example("s", [QuestionSpec("choice", "q", ["a", "b"], soft=[2.0, 1.0, 1.0])])])
        self.assertAlmostEqual(float(b.label_soft[0, 0].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(b.label_soft[0, 0, -1]), 0.25, places=6)

    def test_record_roundtrip_and_bucketing(self):
        ex = Rhen2Example({"k": 1}, [QuestionSpec("choice", "q", ["a", "b"], label=1)])
        back = example_from_record(example_to_record(ex))
        self.assertEqual(back.questions[0].options, ["a", "b"])
        self.assertEqual([k_bucket(k) for k in (1, 2, 3, 9, 200, 255)], [2, 2, 4, 16, 255, 255])
        many = [Rhen2Example("s", [QuestionSpec("choice", "q", [str(i) for i in range(k)])]) for k in (2, 3, 40, 41)]
        groups = make_batches(many, max_questions=8)
        self.assertEqual(sorted(len(g) for g in groups), [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
