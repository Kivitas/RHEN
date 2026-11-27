"""Dataset converters, batch A (boolq, mnli, qnli, fever, agnews, dbpedia, trec, banking77, clinc150, massive)."""

from __future__ import annotations

import contextlib
import importlib
import unittest
import warnings

from rhen.v2.config import preset
from rhen.v2.data.converters import REGISTRY, convert_rows, load_all
from rhen.v2.data.converters import banking77, clinc150, massive
from rhen.v2.packing import NULL_LABEL, Rhen2Example, Rhen2Packer

NAMES = ["boolq", "mnli", "qnli", "fever", "agnews", "dbpedia", "trec", "banking77", "clinc150", "massive"]

AGNEWS_OPTS = ["World", "Sports", "Business", "Sci/Tech"]
DBPEDIA_OPTS = [
    "Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation",
    "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
]
TREC_OPTS = ["abbreviation", "entity", "description", "human", "location", "numeric value"]
FEVER_OPTS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# Each case: (row, expected). expected is None for a row that must be skipped, otherwise a dict with
# state, qtype, options (list, or None to skip the check), label and optionally instruction.
CASES = {
    "boolq": [
        ({"question": "is france the largest country in europe", "passage": "France is in Western Europe. Russia is larger.", "answer": False},
         dict(state="France is in Western Europe. Russia is larger.", qtype="noul", options=[], label=False)),
        ({"question": "is ethanol fuel cheaper to produce than gasoline?", "passage": "Ethanol is produced from crops.", "answer": True},
         dict(state="Ethanol is produced from crops.", qtype="noul", options=[], label=True)),
        ({"question": "does the moon have water", "passage": "Ice was found in some lunar craters.", "answer": True},
         dict(state="Ice was found in some lunar craters.", qtype="noul", options=[], label=True)),
        ({"question": "is the sun a star", "passage": "The Sun is a G-type main-sequence star.", "answer": True},
         dict(state="The Sun is a G-type main-sequence star.", qtype="noul", options=[], label=True)),
        ({"question": "is a tomato a fruit", "passage": "Botanically a tomato is a fruit.", "answer": False},
         dict(state="Botanically a tomato is a fruit.", qtype="noul", options=[], label=False)),
        ({"question": "is this unlabelled", "passage": "Some passage.", "answer": None}, None),
        ({"question": "is this a string label", "passage": "Some passage.", "answer": "yes"}, None),
        ({"question": "empty passage", "passage": "   ", "answer": True}, None),
        ({"question": "   ", "passage": "Some passage.", "answer": True}, None),
        ({"passage": "no question key", "answer": True}, None),
    ],
    "mnli": [
        ({"premise": "A man is playing guitar on stage.", "hypothesis": "A person is making music.", "label": 0, "idx": 0},
         dict(state="A man is playing guitar on stage.", qtype="noul", options=[], label=True,
              instruction="Does the premise entail the following statement: A person is making music.")),
        ({"premise": "A woman reads a book.", "hypothesis": "The woman is asleep.", "label": 2, "idx": 1},
         dict(state="A woman reads a book.", qtype="noul", options=[], label=False)),
        ({"premise": "Two dogs run in a park.", "hypothesis": "The dogs are hungry.", "label": 1, "idx": 2},
         dict(state="Two dogs run in a park.", qtype="noul", options=[], label=False)),
        ({"premise": "It is raining.", "hypothesis": "Water falls from the sky.", "label": 0, "idx": 3},
         dict(state="It is raining.", qtype="noul", options=[], label=True)),
        ({"premise": "The store opens at nine.", "hypothesis": "The store is closed all day.", "label": -1, "idx": 4}, None),
        ({"premise": "", "hypothesis": "Something.", "label": 0, "idx": 5}, None),
        ({"premise": "Something.", "label": 0, "idx": 6}, None),
    ],
    "qnli": [
        ({"question": "What is the capital of France?", "sentence": "Paris is the capital and largest city of France.", "label": 0, "idx": 0},
         dict(state="Paris is the capital and largest city of France.", qtype="noul", options=[], label=True,
              instruction="Does the sentence answer the question: What is the capital of France?")),
        ({"question": "Who wrote Hamlet?", "sentence": "The Eiffel Tower is in Paris.", "label": 1, "idx": 1},
         dict(state="The Eiffel Tower is in Paris.", qtype="noul", options=[], label=False)),
        ({"question": "When did the war end?", "sentence": "The war ended in 1945.", "label": 0, "idx": 2},
         dict(state="The war ended in 1945.", qtype="noul", options=[], label=True)),
        ({"question": "How tall is Everest?", "sentence": "Mount Everest is the highest mountain.", "label": 1, "idx": 3},
         dict(state="Mount Everest is the highest mountain.", qtype="noul", options=[], label=False)),
        ({"question": "Unlabelled?", "sentence": "Some sentence.", "label": -1, "idx": 4}, None),
        ({"question": "Bad label?", "sentence": "Some sentence.", "label": 2, "idx": 5}, None),
        ({"question": "Q?", "sentence": "  ", "label": 0, "idx": 6}, None),
    ],
    "fever": [
        ({"id": 1, "label": "SUPPORTS", "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."},
         dict(state="Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", qtype="choice",
              options=FEVER_OPTS, label=0)),
        ({"id": 2, "label": "REFUTES", "claim": "Telemundo is a English-language television network."},
         dict(state="Telemundo is a English-language television network.", qtype="choice", options=FEVER_OPTS, label=1)),
        ({"id": 3, "label": "NOT ENOUGH INFO", "claim": "Mount Kilimanjaro is in Peru."},
         dict(state="Mount Kilimanjaro is in Peru.", qtype="choice", options=FEVER_OPTS, label=2)),
        ({"id": 4, "label": 2, "claim": "A claim given as a ClassLabel index."},
         dict(state="A claim given as a ClassLabel index.", qtype="choice", options=FEVER_OPTS, label=2)),
        ({"id": 5, "label": "SUPPORTS", "claim": "The claim has evidence text.", "evidence": "Some evidence sentence."},
         dict(state="Claim: The claim has evidence text.\nEvidence: Some evidence sentence.", qtype="choice",
              options=FEVER_OPTS, label=0)),
        ({"id": 6, "label": -1, "claim": "Invalid label."}, None),
        ({"id": 7, "label": "MAYBE", "claim": "Unknown label name."}, None),
        ({"id": 8, "label": "SUPPORTS", "claim": "   "}, None),
    ],
    "agnews": [
        ({"text": "Wall St. Bears Claw Back Into the Black (Reuters) Short-sellers are seeing green again.", "label": 2},
         dict(state="Wall St. Bears Claw Back Into the Black (Reuters) Short-sellers are seeing green again.",
              qtype="choice", options=AGNEWS_OPTS, label=2)),
        ({"text": "Afghan president meets with envoys in Kabul.", "label": 0}, dict(qtype="choice", options=AGNEWS_OPTS, label=0)),
        ({"text": "Phelps wins eighth gold medal at the Athens Games.", "label": 1}, dict(qtype="choice", options=AGNEWS_OPTS, label=1)),
        ({"text": "Microsoft unveils a new chip for data centres.", "label": 3}, dict(qtype="choice", options=AGNEWS_OPTS, label=3)),
        ({"text": "A label of minus one.", "label": -1}, None),
        ({"text": "   ", "label": 1}, None),
        ({"text": "Label out of range.", "label": 4}, None),
    ],
    "dbpedia": [
        ({"label": 0, "title": "Dunkin' Donuts", "content": "Dunkin' Donuts is an American multinational coffee and doughnut company."},
         dict(state="Dunkin' Donuts Dunkin' Donuts is an American multinational coffee and doughnut company.",
              qtype="choice", options=DBPEDIA_OPTS, label=0)),
        ({"label": 12, "title": "Solaris", "content": "Solaris is a 1972 Soviet science fiction film."},
         dict(qtype="choice", options=DBPEDIA_OPTS, label=12)),
        ({"label": 13, "title": "The Hobbit", "content": "The Hobbit is a fantasy novel by J. R. R. Tolkien."},
         dict(qtype="choice", options=DBPEDIA_OPTS, label=13)),
        ({"label": 2, "title": "Some Artist", "content": ""}, dict(state="Some Artist", qtype="choice", options=DBPEDIA_OPTS, label=2)),
        ({"label": -1, "title": "Bad", "content": "Label minus one."}, None),
        ({"label": 14, "title": "Bad", "content": "Label out of range."}, None),
        ({"label": 3, "title": "", "content": "   "}, None),
    ],
    "trec": [
        ({"text": "What is the abbreviation of NATO?", "coarse_label": 0, "fine_label": 1},
         dict(state="What is the abbreviation of NATO?", qtype="choice", options=TREC_OPTS, label=0)),
        ({"text": "What is an atom?", "coarse_label": 2, "fine_label": 9},
         dict(qtype="choice", options=TREC_OPTS, label=2)),
        ({"text": "Who was Galileo?", "coarse_label": 3, "fine_label": 30},
         dict(qtype="choice", options=TREC_OPTS, label=3)),
        ({"text": "Where is the Taj Mahal?", "coarse_label": 4, "fine_label": 37},
         dict(qtype="choice", options=TREC_OPTS, label=4)),
        ({"text": "How many moons does Mars have?", "coarse_label": 5, "fine_label": 42},
         dict(qtype="choice", options=TREC_OPTS, label=5)),
        ({"text": "Which coarse label is missing?", "coarse_label": -1, "fine_label": -1}, None),
        ({"text": "", "coarse_label": 1, "fine_label": 1}, None),
    ],
    "banking77": [
        ({"text": "I still have not received my new card, I ordered it over a week ago.", "label": 11},
         dict(state="I still have not received my new card, I ordered it over a week ago.", qtype="choice",
              options=banking77.INTENT_NAMES, label=11)),
        ({"text": "How do I activate my card?", "label": 0}, dict(qtype="choice", options=banking77.INTENT_NAMES, label=0)),
        ({"text": "Why was I charged a fee on my exchange?", "label": 76}, dict(qtype="choice", options=banking77.INTENT_NAMES, label=76)),
        ({"text": "Can I get a virtual card?", "label": 40}, dict(qtype="choice", options=banking77.INTENT_NAMES, label=40)),
        ({"text": "Top up failed, what happened?", "label": 59}, dict(qtype="choice", options=banking77.INTENT_NAMES, label=59)),
        ({"text": "Label -1 is invalid.", "label": -1}, None),
        ({"text": "", "label": 5}, None),
        ({"text": "Label 77 is one past the end.", "label": 77}, None),
    ],
    "clinc150": [
        ({"text": "how many calories are in an apple", "intent": 39},
         dict(state="how many calories are in an apple", qtype="choice", options=clinc150.INTENT_NAMES, label=39)),
        ({"text": "set an alarm for seven am", "intent": 2}, dict(qtype="choice", options=clinc150.INTENT_NAMES, label=2)),
        ({"text": "what is your name", "intent": 143}, dict(qtype="choice", options=clinc150.INTENT_NAMES, label=143)),
        ({"text": "tell me a joke", "intent": 121}, dict(qtype="choice", options=clinc150.INTENT_NAMES, label=121)),
        ({"text": "book me a table for two tonight", "intent": 92}, dict(qtype="choice", options=clinc150.INTENT_NAMES, label=92)),
        ({"text": "can you hold my cat for an hour, I have no idea what the weather will do", "intent": 150},
         dict(state="can you hold my cat for an hour, I have no idea what the weather will do", qtype="choice",
              options=clinc150.INTENT_NAMES, label=NULL_LABEL)),
        ({"text": "out of scope request", "intent": 150}, dict(qtype="choice", options=clinc150.INTENT_NAMES, label=NULL_LABEL)),
        ({"text": "intent -1", "intent": -1}, None),
        ({"text": "intent past out-of-scope", "intent": 151}, None),
        ({"text": "", "intent": 3}, None),
    ],
    "massive": [
        ({"utt": "wake me up at nine am on friday", "intent": 2, "id": "1", "locale": "en-US", "scenario": "alarm"},
         dict(state="wake me up at nine am on friday", qtype="choice", options=massive.INTENT_NAMES, label=2)),
        ({"utt": "turn the volume down", "intent": 3}, dict(qtype="choice", options=massive.INTENT_NAMES, label=3)),
        ({"utt": "play some jazz music", "intent": massive.INTENT_NAMES.index("play_music")},
         dict(qtype="choice", options=massive.INTENT_NAMES, label=massive.INTENT_NAMES.index("play_music"))),
        ({"utt": "what is the weather in paris", "intent": "weather_query"},
         dict(qtype="choice", options=massive.INTENT_NAMES, label=massive.INTENT_NAMES.index("weather_query"))),
        ({"utt": "how many ounces in a cup", "intent": massive.INTENT_NAMES.index("qa_maths")},
         dict(qtype="choice", options=massive.INTENT_NAMES, label=massive.INTENT_NAMES.index("qa_maths"))),
        ({"utt": "intent is missing", "intent": -1}, None),
        ({"utt": "intent name unknown", "intent": "not_an_intent"}, None),
        ({"utt": "   ", "intent": 4}, None),
        ({"utt": "intent past the end", "intent": 60}, None),
    ],
}

PACKED = {}


class TestConverterBatchA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_all()
        cls.packer = Rhen2Packer(preset("r2-50m"))

    def _convert(self, name, row):
        return REGISTRY[name].convert(row)

    def test_fixture_counts(self):
        for name in NAMES:
            good = [c for c in CASES[name] if c[1] is not None]
            bad = [c for c in CASES[name] if c[1] is None]
            self.assertGreaterEqual(len(CASES[name]), 5, name)
            self.assertGreaterEqual(len(good), 1, name)
            self.assertGreaterEqual(len(bad), 1, name)

    def test_converted_rows_have_expected_type_options_and_label(self):
        for name in NAMES:
            for row, expected in CASES[name]:
                if expected is None:
                    continue
                with self.subTest(name=name, row=row):
                    ex = self._convert(name, row)
                    self.assertIsInstance(ex, Rhen2Example)
                    self.assertEqual(len(ex.questions), 1)
                    spec = ex.questions[0]
                    self.assertEqual(spec.type, expected["qtype"])
                    self.assertEqual(spec.options, expected["options"])
                    self.assertEqual(spec.label, expected["label"])
                    if "state" in expected:
                        self.assertEqual(ex.state, expected["state"])
                    if "instruction" in expected:
                        self.assertEqual(spec.instructions, expected["instruction"])
                    self.assertIn(spec.type, REGISTRY[name].question_types)

    def test_bad_rows_return_none(self):
        for name in NAMES:
            for row, expected in CASES[name]:
                if expected is not None:
                    continue
                with self.subTest(name=name, row=row):
                    self.assertIsNone(self._convert(name, row))

    def test_option_counts_above_64(self):
        self.assertEqual(len(banking77.INTENT_NAMES), 77)
        self.assertEqual(len(clinc150.INTENT_NAMES), 150)
        ex = self._convert("banking77", CASES["banking77"][0][0])
        self.assertEqual(len(ex.questions[0].options), 77)
        ex = self._convert("clinc150", CASES["clinc150"][0][0])
        self.assertEqual(len(ex.questions[0].options), 150)

    def test_every_converted_example_packs(self):
        for name in NAMES:
            rows = [row for row, expected in CASES[name] if expected is not None]
            examples = convert_rows(name, rows)
            self.assertEqual(len(examples), len(rows), name)
            for ex in examples:
                with self.subTest(name=name):
                    batch = self.packer.pack([ex])
                    self.assertEqual(batch.batch_size, 1)
                    self.assertTrue(bool(batch.q_valid[0, 0]))

    def test_load_all_registers_every_name(self):
        registry = load_all()
        for name in NAMES:
            self.assertIn(name, registry)
            self.assertIs(registry[name], REGISTRY[name])

    def test_licence_and_benchmark_splits_are_set(self):
        registry = load_all()
        for name in NAMES:
            conv = registry[name]
            with self.subTest(name=name):
                self.assertTrue(conv.licence and conv.licence.strip())
                self.assertTrue(conv.benchmark_test_splits)
                self.assertTrue(all(isinstance(s, str) and s for s in conv.benchmark_test_splits))
                self.assertNotIn("train", conv.benchmark_test_splits)


LABEL_MODULES = {
    "banking77": ("INTENT_NAMES", "label", 77, 0),
    "clinc150": ("INTENT_NAMES", "intent", 150, 1),
    "massive": ("INTENT_NAMES", "intent", 60, 0),
    "fever": ("OPTIONS", "label", 3, 0),
    "agnews": ("OPTIONS", "label", 4, 0),
    "dbpedia": ("OPTIONS", "label", 14, 0),
    "trec": ("OPTIONS", "coarse_label", 6, 0),
}


class _ClassLabel:
    """Stand-in for datasets.ClassLabel: only ``.names`` is used."""

    def __init__(self, names):
        self.names = list(names)


class _Features(dict):
    pass


def _mod(name):
    return importlib.import_module(f"rhen.v2.data.converters.{name}")


@contextlib.contextmanager
def _isolated(name):
    """Give a converter module fresh label-name state and restore it afterwards."""
    mod = _mod(name)
    var = LABEL_MODULES[name][0]
    saved = (getattr(mod, var), dict(mod._STATE))
    try:
        mod._STATE.update(verified=False, warned=False)
        yield mod
    finally:
        setattr(mod, var, saved[0])
        mod._STATE.clear()
        mod._STATE.update(saved[1])


def _row_for(name, label):
    """A valid row for each module, using the given label id (or name for fever)."""
    rows = {
        "banking77": {"text": "where is my card", "label": label},
        "clinc150": {"text": "where is my card", "intent": label},
        "massive": {"utt": "where is my card", "intent": label},
        "fever": {"claim": "a claim", "label": label},
        "agnews": {"text": "some article", "label": label},
        "dbpedia": {"title": "a title", "content": "some content", "label": label},
        "trec": {"text": "who is it?", "coarse_label": label},
    }
    return rows[name]


def _reversed_names(name):
    base = list(getattr(_mod(name), "UNVERIFIED_FALLBACK"))
    names = list(reversed(base))
    if name == "clinc150":
        names = names + ["oos"]     # HF plus config: 151 entries, oos last
    return names


class TestLabelNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_all()

    def test_each_module_warns_once_until_names_are_set(self):
        for name in LABEL_MODULES:
            with self.subTest(name=name), _isolated(name) as mod:
                row = _row_for(name, 0)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    mod.convert(row)
                    mod.convert(row)
                user = [w for w in caught if issubclass(w.category, UserWarning)]
                self.assertEqual(len(user), 1, name)
                self.assertIn(name, str(user[0].message))
                self.assertIn("UNVERIFIED", str(user[0].message))

    def test_no_warning_after_set_label_names(self):
        for name in LABEL_MODULES:
            with self.subTest(name=name), _isolated(name) as mod:
                key = LABEL_MODULES[name][1]
                mod.apply_hf_features(_Features({key: _ClassLabel(_reversed_names(name))}))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    mod.convert(_row_for(name, 0 if name != "fever" else "SUPPORTS"))
                self.assertEqual([w for w in caught if issubclass(w.category, UserWarning)], [], name)

    def test_wrong_length_raises(self):
        for name in LABEL_MODULES:
            var, key, count, _ = LABEL_MODULES[name]
            with self.subTest(name=name), _isolated(name) as mod:
                names = list(reversed(list(getattr(mod, "UNVERIFIED_FALLBACK"))))
                with self.assertRaises(ValueError):
                    mod.set_label_names(names[:-1])
                with self.assertRaises(ValueError):
                    mod.set_label_names(names + ["extra"])
                with self.assertRaises(ValueError):
                    mod.apply_hf_features(_Features({key: _ClassLabel(names[:-1])}))

    def test_clinc_accepts_151_names_with_oos_last(self):
        with _isolated("clinc150") as mod:
            mod.set_label_names(_reversed_names("clinc150"))
            self.assertEqual(len(mod.INTENT_NAMES), 150)
            with self.assertRaises(ValueError):
                mod.set_label_names(_reversed_names("clinc150")[:-1] + ["not_oos"])

    def test_labels_follow_supplied_names_not_fallback(self):
        for name in LABEL_MODULES:
            var, key, count, _ = LABEL_MODULES[name]
            with self.subTest(name=name), _isolated(name) as mod:
                names = _reversed_names(name)
                mod.apply_hf_features(_Features({key: _ClassLabel(names)}))
                if name == "fever":
                    row = _row_for(name, "SUPPORTS")
                    want_index = names.index("SUPPORTS")
                else:
                    row = _row_for(name, 3)
                    want_index = 3
                ex = mod.convert(row)
                spec = ex.questions[0]
                self.assertEqual(spec.options, names[:150] if name == "clinc150" else names)
                self.assertEqual(spec.label, want_index)
                self.assertEqual(spec.options[spec.label], names[want_index])
                self.assertNotEqual(spec.options, list(mod.UNVERIFIED_FALLBACK))

    def test_notes_say_unverified_until_set_label_names(self):
        for name in LABEL_MODULES:
            self.assertIn("label names UNVERIFIED until set_label_names is called", load_all()[name].notes, name)



if __name__ == "__main__":
    unittest.main()
