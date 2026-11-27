"""Tests for the second batch of dataset converters (NLI, QA, sentiment, tagging)."""

from __future__ import annotations

import unittest

from rhen.v2.config import QUESTION_TYPES, preset
from rhen.v2.data.converters import REGISTRY, load_all
from rhen.v2.packing import NULL_LABEL, Rhen2Packer

MINE = (
    "snli", "anli", "race", "commonsenseqa", "openbookqa", "arc_challenge", "arc_easy", "yelp",
    "amazon_reviews", "stsb", "goemotions", "reuters", "toxic_comments", "squad2", "conll_ner",
)

SQUAD_CTX = "The Eiffel Tower is in Paris. It was built in 1889."
CONLL_PAIR = ["EU", "rejects", "German", "call", "to", "boycott"]


def _mc(stem_key, stem, texts, labels, key):
    return {stem_key: stem, "choices": {"text": texts, "label": labels}, "answerKey": key}


# Fixture rows per converter: at least five, in the real schema, including bad rows.
FIXTURES = {
    "snli": [
        {"premise": "A man inside the hotel.", "hypothesis": "A man is inside.", "label": 0},
        {"premise": "A man inside the hotel.", "hypothesis": "A woman is outside.", "label": 2},
        {"premise": "A man inside the hotel.", "hypothesis": "The man is tall.", "label": 1},
        {"premise": "A man inside the hotel.", "hypothesis": "A man is inside.", "label": -1},
        {"premise": "", "hypothesis": "A man is inside.", "label": 0},
        {"premise": "A man inside the hotel.", "label": 0},
    ],
    "anli": [
        {"premise": "Ann wrote a book.", "hypothesis": "Ann is an author.", "label": 0, "uid": "a"},
        {"premise": "Ann wrote a book.", "hypothesis": "Ann is a chef.", "label": 2, "uid": "b"},
        {"premise": "Ann wrote a book.", "hypothesis": "Ann likes tea.", "label": 1, "uid": "c"},
        {"premise": "Ann wrote a book.", "hypothesis": "Ann is an author.", "label": -1, "uid": "d"},
        {"premise": None, "hypothesis": "Ann is an author.", "label": 0, "uid": "e"},
        {"premise": "Ann wrote a book.", "hypothesis": "Ann is an author.", "label": True, "uid": "f"},
    ],
    "race": [
        {"example_id": "x1", "article": "Tom has a cat.", "problem": "What does Tom have?",
         "options": ["a dog", "a cat", "a bird", "a fish"], "answer": "B"},
        {"example_id": "x2", "article": "Sue runs fast.", "problem": "How does Sue run?",
         "options": ["slowly", "quietly", "well", "fast"], "answer": "d"},
        {"example_id": "x3", "article": "Sue runs fast.", "problem": "Q?",
         "options": ["a", "b", "c", "d"], "answer": "E"},
        {"example_id": "x4", "article": "Sue runs fast.", "problem": "Q?",
         "options": ["a", "b", "c"], "answer": "A"},
        {"example_id": "x5", "problem": "Q?", "options": ["a", "b", "c", "d"], "answer": "A"},
        {"example_id": "x6", "article": "Sue runs fast.", "problem": "Q?", "options": ["a", "b", "c", "d"]},
    ],
    "commonsenseqa": [
        _mc("question", "Where do you keep a pen?", ["desk", "sky", "sea", "moon", "lake"],
            ["A", "B", "C", "D", "E"], "A"),
        _mc("question", "Where do you keep a pen?", ["desk", "sky", "sea", "moon", "lake"],
            ["A", "B", "C", "D", "E"], "C"),
        _mc("question", "Where do you keep a pen?", ["desk", "sky", "sea", "moon", "lake"],
            ["A", "B", "C", "D", "E"], ""),
        {"question": "Where do you keep a pen?", "answerKey": "A"},
        _mc("question", "Where do you keep a pen?", ["desk", "sky"], ["A", "B", "C"], "A"),
        _mc("question", "", ["desk", "sky"], ["A", "B"], "A"),
    ],
    "openbookqa": [
        _mc("question_stem", "A plant needs", ["sun", "rock", "glass", "wood"], ["A", "B", "C", "D"], "A"),
        _mc("question_stem", "A plant needs", ["sun", "rock", "glass", "wood"], ["A", "B", "C", "D"], "D"),
        _mc("question_stem", "A plant needs", ["sun", "rock", "glass", "wood"], ["A", "B", "C", "D"], "Z"),
        _mc("question_stem", "", ["sun", "rock"], ["A", "B"], "A"),
        _mc("question_stem", "A plant needs", ["", "rock"], ["A", "B"], "A"),
        _mc("question_stem", "A plant needs", ["sun", "rock"], ["A", "B"], None),
    ],
    "arc_challenge": [
        _mc("question", "Which is a gas?", ["ice", "steam", "rock", "sand"], ["1", "2", "3", "4"], "2"),
        _mc("question", "Which is a gas?", ["ice", "steam", "rock", "sand"], ["A", "B", "C", "D"], "B"),
        _mc("question", "Which is a gas?", ["ice", "steam", "rock", "sand"], ["1", "2", "3", "4"], "3"),
        _mc("question", "Which is a gas?", ["ice", "steam", "rock", "sand"], ["1", "2", "3", "4"], None),
        _mc("question", "", ["ice", "steam"], ["1", "2"], "1"),
        _mc("question", "Which is a gas?", ["ice", "steam"], ["1"], "1"),
    ],
    "arc_easy": [
        _mc("question", "What heats a room?", ["ice", "sun"], ["A", "B"], "B"),
        _mc("question", "What heats a room?", ["ice", "sun"], ["A", "B"], "A"),
        _mc("question", "What heats a room?", ["ice", "sun", "fan"], ["A", "B", "C"], "C"),
        _mc("question", "What heats a room?", ["ice", "sun"], ["A", "B"], ""),
        _mc("question", "What heats a room?", ["ice", "sun"], ["A", "B"], "Q"),
        {"question": "What heats a room?", "choices": {"text": ["ice"], "label": ["A"]}, "answerKey": "A"},
    ],
    "yelp": [
        {"text": "Great food!", "label": 4},
        {"text": "Terrible service.", "label": 0},
        {"text": "It was fine.", "label": 2},
        {"text": "Maybe.", "label": 5},
        {"text": "Maybe.", "label": -1},
        {"text": "", "label": 3},
        {"text": "Okay.", "label": True},
    ],
    "amazon_reviews": [
        {"review_body": "Works well.", "label": 2},
        {"text": "Broke in a day.", "stars": 1},
        {"text": "Solid.", "stars": 6},
        {"text": "No rating here."},
        {"text": "   "},
        {"review_body": "   ", "text": "Fallback text.", "label": 3},
    ],
    "stsb": [
        {"sentence1": "A cat sleeps.", "sentence2": "A cat is asleep.", "label": 4.6, "idx": 0},
        {"sentence1": "A dog runs.", "sentence2": "Stocks fell.", "label": 0.0, "idx": 1},
        {"sentence1": "A man eats.", "sentence2": "A man eats food.", "label": 2.4, "idx": 2},
        {"sentence1": "A man eats.", "sentence2": "A man eats food.", "label": 2.5, "idx": 3},
        {"sentence1": "A man eats.", "sentence2": "A man eats food.", "label": -1, "idx": 4},
        {"sentence1": "A man eats.", "sentence2": "A man eats food.", "label": None, "idx": 5},
        {"sentence1": "Same.", "sentence2": "Same.", "label": 5.0, "idx": 6},
    ],
    "goemotions": [
        {"text": "Thanks so much!", "labels": [15], "id": "a"},
        {"text": "Whatever.", "labels": [], "id": "b"},
        {"text": "Odd and sad.", "labels": [27, 0], "id": "c"},
        {"text": "Bad.", "labels": [28], "id": "d"},
        {"text": "Bad.", "id": "e"},
        {"text": "", "labels": [0], "id": "f"},
    ],
    "reuters": [
        {"text": "Oil prices rose.", "topics": ["crude", "earn"]},
        {"text": "Nothing here.", "topics": []},
        {"text": "No topics key."},
        {"title": "Title only", "text": "", "topics": ["acq"]},
        {"text": "Bad topic.", "topics": ["acq", ""]},
        {"text": "", "topics": ["acq"]},
    ],
    "toxic_comments": [
        {"comment_text": "you are awful", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0,
         "insult": 1, "identity_hate": 0},
        {"comment_text": "thanks for the edit", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0,
         "insult": 0, "identity_hate": 0},
        {"comment_text": "unlabelled", "toxic": -1, "severe_toxic": -1, "obscene": -1, "threat": -1,
         "insult": -1, "identity_hate": -1},
        {"comment_text": "missing a column", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0,
         "identity_hate": 0},
        {"comment_text": "", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0,
         "identity_hate": 0},
        {"comment_text": "bool flags", "toxic": True, "severe_toxic": False, "obscene": False,
         "threat": False, "insult": False, "identity_hate": False},
    ],
    "squad2": [
        {"id": "s1", "title": "Eiffel", "context": SQUAD_CTX, "question": "Where is the tower?",
         "answers": {"text": ["Paris"], "answer_start": [SQUAD_CTX.index("Paris")]}},
        {"id": "s2", "title": "Eiffel", "context": SQUAD_CTX, "question": "Who built it?",
         "answers": {"text": [], "answer_start": []}},
        {"id": "s3", "title": "Eiffel", "context": SQUAD_CTX, "question": "Where is the tower?",
         "answers": {"text": ["Paris"], "answer_start": [SQUAD_CTX.index("Paris") + 2]}},
        {"id": "s4", "title": "Eiffel", "context": "", "question": "Where?",
         "answers": {"text": ["Paris"], "answer_start": [0]}},
        {"id": "s5", "title": "Eiffel", "context": SQUAD_CTX, "question": "When?"},
        {"id": "s6", "title": "Eiffel", "context": SQUAD_CTX, "question": "When was it built?",
         "answers": {"text": ["1889"], "answer_start": [SQUAD_CTX.index("1889")]}},
    ],
    "conll_ner": [
        {"id": "0", "tokens": CONLL_PAIR, "ner_tags": [3, 0, 7, 0, 0, 0]},
        {"id": "1", "tokens": ["Peter", "Blackburn"], "ner_tags": [1, 2]},
        {"id": "2", "tokens": ["No", "entities", "here"], "ner_tags": [0, 0, 0]},
        {"id": "3", "tokens": ["Bad", "tag"], "ner_tags": [9, 0]},
        {"id": "4", "tokens": ["Short"], "ner_tags": [0, 0]},
        {"id": "5", "tokens": ["New", "York"], "ner_tags": [5, 6]},
    ],
}


def _by_type(ex):
    """Map entity type -> extract label, read from 'Extract the first <TYPE> entity ...'."""
    return {q.instructions.split()[3]: q.label for q in ex.questions}


def _span_text(state, span):
    s, e = span
    return Rhen2Packer.serialise_state(state)[s:e]


class TestConvertersB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_all()
        cls.cfg = preset("r2-50m")
        cls.packer = Rhen2Packer(cls.cfg)

    def conv(self, name, row):
        return REGISTRY[name].convert(row)

    # -- registration ------------------------------------------------------------
    def test_load_all_registers_every_name(self):
        for name in MINE:
            self.assertIn(name, REGISTRY)
            self.assertEqual(REGISTRY[name].name, name)

    def test_metadata_is_complete(self):
        for name in MINE:
            c = REGISTRY[name]
            self.assertTrue(c.licence.strip(), name)
            self.assertTrue(c.benchmark_test_splits, name)
            self.assertTrue(c.hf_path, name)
            self.assertTrue(set(c.question_types) <= set(QUESTION_TYPES), name)
            self.assertGreaterEqual(len(FIXTURES[name]), 5, name)

    def test_every_fixture_row_converts_or_is_skipped_and_packs(self):
        for name in MINE:
            for i, row in enumerate(FIXTURES[name]):
                ex = self.conv(name, row)
                if ex is None:
                    continue
                for q in ex.questions:
                    q.validate(self.cfg)
                batch = self.packer.pack([ex])
                self.assertEqual(batch.batch_size, 1, f"{name}[{i}]")

    def test_bad_rows_return_none(self):
        expected_none = {
            "snli": [3, 4, 5], "anli": [3, 4, 5], "race": [2, 3, 4, 5], "commonsenseqa": [2, 3, 4, 5],
            "openbookqa": [2, 3, 4, 5], "arc_challenge": [3, 4, 5], "arc_easy": [3, 4],
            "yelp": [3, 4, 5, 6], "amazon_reviews": [2, 3, 4], "stsb": [4, 5], "goemotions": [3, 4, 5],
            "reuters": [1, 2, 4, 5], "toxic_comments": [2, 3, 4], "squad2": [2, 3, 4], "conll_ner": [3, 4],
        }
        for name, idxs in expected_none.items():
            for i in idxs:
                self.assertIsNone(self.conv(name, FIXTURES[name][i]), f"{name}[{i}]")

    # -- SNLI / ANLI ---------------------------------------------------------------
    def test_snli(self):
        ex = self.conv("snli", FIXTURES["snli"][0])
        self.assertEqual(ex.state, "A man inside the hotel.")
        self.assertEqual(ex.questions[0].type, "noul")
        self.assertIs(ex.questions[0].label, True)
        self.assertEqual(ex.questions[0].options, [])
        self.assertIn("A man is inside.", ex.questions[0].instructions)
        self.assertIs(self.conv("snli", FIXTURES["snli"][1]).questions[0].label, False)
        self.assertIs(self.conv("snli", FIXTURES["snli"][2]).questions[0].label, False)

    def test_anli(self):
        self.assertIs(self.conv("anli", FIXTURES["anli"][0]).questions[0].label, True)
        self.assertIs(self.conv("anli", FIXTURES["anli"][1]).questions[0].label, False)
        self.assertIs(self.conv("anli", FIXTURES["anli"][2]).questions[0].label, False)

    # -- RACE / CSQA / OBQA / ARC -------------------------------------------------
    def test_race(self):
        ex = self.conv("race", FIXTURES["race"][0])
        q = ex.questions[0]
        self.assertEqual((q.type, q.label, len(q.options)), ("choice", 1, 4))
        self.assertEqual(q.options[1], "a cat")
        self.assertEqual(self.conv("race", FIXTURES["race"][1]).questions[0].label, 3)

    def test_commonsenseqa(self):
        q = self.conv("commonsenseqa", FIXTURES["commonsenseqa"][0]).questions[0]
        self.assertEqual((q.type, q.label, q.options[0]), ("choice", 0, "desk"))
        self.assertEqual(len(q.options), 5)
        self.assertEqual(self.conv("commonsenseqa", FIXTURES["commonsenseqa"][1]).questions[0].label, 2)

    def test_openbookqa(self):
        ex = self.conv("openbookqa", FIXTURES["openbookqa"][0])
        self.assertEqual(ex.state, "A plant needs")
        self.assertEqual(ex.questions[0].label, 0)
        self.assertEqual(self.conv("openbookqa", FIXTURES["openbookqa"][1]).questions[0].label, 3)

    def test_arc_digit_and_letter_labels(self):
        for name in ("arc_challenge", "arc_easy"):
            self.assertIn(name, REGISTRY)
        self.assertEqual(self.conv("arc_challenge", FIXTURES["arc_challenge"][0]).questions[0].label, 1)
        self.assertEqual(self.conv("arc_challenge", FIXTURES["arc_challenge"][1]).questions[0].label, 1)
        self.assertEqual(self.conv("arc_challenge", FIXTURES["arc_challenge"][2]).questions[0].label, 2)
        self.assertEqual(self.conv("arc_easy", FIXTURES["arc_easy"][0]).questions[0].label, 1)
        self.assertEqual(self.conv("arc_easy", FIXTURES["arc_easy"][2]).questions[0].label, 2)
        self.assertEqual(REGISTRY["arc_challenge"].hf_config, "ARC-Challenge")
        self.assertEqual(REGISTRY["arc_easy"].hf_config, "ARC-Easy")

    # -- sentiment and similarity ---------------------------------------------------
    def test_yelp_levels(self):
        q = self.conv("yelp", FIXTURES["yelp"][0]).questions[0]
        self.assertEqual(q.type, "score")
        self.assertEqual(q.options, ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"])
        self.assertEqual(q.label, 4)
        self.assertEqual(self.conv("yelp", FIXTURES["yelp"][1]).questions[0].label, 0)

    def test_amazon_reviews(self):
        q = self.conv("amazon_reviews", FIXTURES["amazon_reviews"][0]).questions[0]
        self.assertEqual((q.type, q.label, len(q.options)), ("score", 2, 5))
        self.assertEqual(self.conv("amazon_reviews", FIXTURES["amazon_reviews"][1]).questions[0].label, 0)
        ex = self.conv("amazon_reviews", FIXTURES["amazon_reviews"][5])
        self.assertEqual((ex.state, ex.questions[0].label), ("Fallback text.", 3))

    def test_stsb_rounding_and_skips(self):
        labels = [self.conv("stsb", row).questions[0].label for row in FIXTURES["stsb"][:4]]
        self.assertEqual(labels, [5, 0, 2, 3])          # 2.5 rounds half up
        self.assertIsNone(self.conv("stsb", FIXTURES["stsb"][4]))
        self.assertEqual(self.conv("stsb", FIXTURES["stsb"][6]).questions[0].label, 5)
        ex = self.conv("stsb", FIXTURES["stsb"][0])
        self.assertEqual(ex.state, {"sentence1": "A cat sleeps.", "sentence2": "A cat is asleep."})
        self.assertEqual(ex.questions[0].options, ["0", "1", "2", "3", "4", "5"])

    # -- multi-label ---------------------------------------------------------------
    def test_goemotions(self):
        q = self.conv("goemotions", FIXTURES["goemotions"][0]).questions[0]
        self.assertEqual(q.type, "multi")
        self.assertEqual(len(q.options), 28)
        self.assertEqual(q.options[27], "neutral")
        self.assertEqual(q.label, [15])
        self.assertEqual(self.conv("goemotions", FIXTURES["goemotions"][1]).questions[0].label, [])
        self.assertEqual(self.conv("goemotions", FIXTURES["goemotions"][2]).questions[0].label, [0, 27])

    def test_reuters(self):
        q = self.conv("reuters", FIXTURES["reuters"][0]).questions[0]
        self.assertEqual((q.type, q.options, q.label), ("multi", ["crude", "earn"], [0, 1]))
        # empty text falls back to the title, and the single topic is kept
        ex = self.conv("reuters", FIXTURES["reuters"][3])
        self.assertEqual(ex.state, "Title only")
        self.assertEqual((ex.questions[0].options, ex.questions[0].label), (["acq"], [0]))

    def test_toxic_comments(self):
        q = self.conv("toxic_comments", FIXTURES["toxic_comments"][0]).questions[0]
        self.assertEqual(q.type, "multi")
        self.assertEqual(q.options, ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
        self.assertEqual(q.label, [0, 4])
        self.assertEqual(self.conv("toxic_comments", FIXTURES["toxic_comments"][1]).questions[0].label, [])
        self.assertEqual(self.conv("toxic_comments", FIXTURES["toxic_comments"][5]).questions[0].label, [0])

    # -- extract -------------------------------------------------------------------
    def test_squad2_offsets_and_unanswerable(self):
        ex = self.conv("squad2", FIXTURES["squad2"][0])
        span = ex.questions[0].label
        self.assertEqual(ex.questions[0].type, "extract")
        self.assertEqual(_span_text(ex.state, span), "Paris")
        self.assertEqual(span, (SQUAD_CTX.index("Paris"), SQUAD_CTX.index("Paris") + 5))
        self.assertEqual(self.conv("squad2", FIXTURES["squad2"][1]).questions[0].label, NULL_LABEL)
        ex6 = self.conv("squad2", FIXTURES["squad2"][5])
        self.assertEqual(_span_text(ex6.state, ex6.questions[0].label), "1889")

    def test_squad2_misaligned_offsets_are_skipped(self):
        self.assertIsNone(self.conv("squad2", FIXTURES["squad2"][2]))

    def test_conll_spans_and_offsets(self):
        ex = self.conv("conll_ner", FIXTURES["conll_ner"][0])
        self.assertEqual(ex.state, "EU rejects German call to boycott")
        by_type = _by_type(ex)
        self.assertEqual(_span_text(ex.state, by_type["ORG"]), "EU")
        self.assertEqual(_span_text(ex.state, by_type["MISC"]), "German")
        self.assertEqual(by_type["PER"], NULL_LABEL)
        self.assertLessEqual(len(ex.questions), 3)

    def test_conll_bio_and_absent_types(self):
        ex = self.conv("conll_ner", FIXTURES["conll_ner"][1])
        self.assertEqual(ex.state, "Peter Blackburn")
        self.assertEqual(ex.questions[0].label, (0, 15))
        self.assertEqual(_span_text(ex.state, ex.questions[0].label), "Peter Blackburn")
        empty = self.conv("conll_ner", FIXTURES["conll_ner"][2])
        self.assertEqual([q.label for q in empty.questions], [NULL_LABEL] * 3)
        loc = self.conv("conll_ner", FIXTURES["conll_ner"][5])
        self.assertEqual(_span_text(loc.state, _by_type(loc)["LOC"]), "New York")
        self.assertEqual(_by_type(loc)["PER"], NULL_LABEL)


if __name__ == "__main__":
    unittest.main()
