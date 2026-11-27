"""Synthetic decision tasks (plan 7.1.3, task T-303)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import re
import tempfile
import unittest
from decimal import Decimal

from rhen.v2.config import preset
from rhen.v2.data import synth
from rhen.v2.data.schema import read_jsonl
from rhen.v2.data.synth import (
    FAMILIES, INJECTION_ID, INJECTION_OOD, family_names, fingerprint, generate_split, generate_tagged,
    with_unanswerable, write_benchmark,
)
from rhen.v2.packing import NULL_LABEL, Rhen2Packer


def ex_of(name, seed, ood=False):
    return FAMILIES[name](random.Random(seed), ood)


def parse_date(s):
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return dt.datetime.strptime(s, "%d %b %Y").date()


def readings(state):
    out = {}
    for line in state.splitlines():
        m = re.match(r"^(\S+)(?:: | => )(-?\d+(?:\.\d+)?) ", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def dated(state):
    out = {}
    for line in state.splitlines():
        m = re.match(r"^(\S+): (\d{4}-\d\d-\d\d|\d+ \w{3} \d{4})$", line)
        if m:
            out[m.group(1)] = parse_date(m.group(2))
    return out


class TestSynth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pk = Rhen2Packer(preset("r2-50m"))

    def test_family_count_and_types(self):
        self.assertGreaterEqual(len(family_names()), 40)
        types = set()
        for name in family_names():
            for ood in (False, True):
                types |= {q.type for q in ex_of(name, 1, ood).questions}
        self.assertEqual(types, {"choice", "noul", "score", "multi", "range", "extract"})

    def test_every_family_packs_in_both_modes(self):
        errors = []
        for name in family_names():
            for ood in (False, True):
                for seed in range(20):
                    try:
                        self.pk.pack([ex_of(name, seed, ood)])
                    except Exception as exc:  # noqa: BLE001
                        errors.append((name, ood, seed, repr(exc)))
        self.assertEqual(errors, [])

    def test_determinism(self):
        a = generate_split("train", 60, 7)
        b = generate_split("train", 60, 7)
        self.assertEqual([fingerprint(x) for x in a], [fingerprint(x) for x in b])
        self.assertEqual([fingerprint(x) for x in generate_split("train", 30, 7)], [fingerprint(x) for x in a[:30]])
        c = generate_split("train", 60, 8)
        self.assertNotEqual([fingerprint(x) for x in a], [fingerprint(x) for x in c])

    def test_splits_are_disjoint(self):
        fps = {s: {fingerprint(x) for x in generate_split(s, 300, 3)} for s in synth.SPLITS}
        for a in synth.SPLITS:
            for b in synth.SPLITS:
                if a < b:
                    self.assertFalse(fps[a] & fps[b], (a, b))

    def test_unanswerable_rate(self):
        exs = generate_split("train", 2000, 11)
        nul = 0
        for ex in exs:
            labels = [q.label for q in ex.questions]
            if any(lb == NULL_LABEL for lb in labels):
                nul += 1
                self.assertTrue(all(lb == NULL_LABEL for lb in labels))
        self.assertAlmostEqual(nul / len(exs), 0.12, delta=0.03)

    def test_with_unanswerable_variants(self):
        rng = random.Random(0)
        ex = ex_of("num_extreme", 5)
        while len(ex.questions[0].options) < 3:
            ex = ex_of("num_extreme", rng.randint(0, 999))
        out = with_unanswerable(ex, rng)
        self.assertEqual(out.questions[0].label, NULL_LABEL)
        self.assertEqual(len(out.questions[0].options), len(ex.questions[0].options) - 1)
        gold = ex.questions[0].options[ex.questions[0].label]
        self.assertNotIn(gold, out.questions[0].options)
        ex = ex_of("extract_id", 3)
        out = with_unanswerable(ex, rng)
        self.assertEqual(out.questions[0].label, NULL_LABEL)
        self.pk.pack([out])
        self.assertIsNone(with_unanswerable(ex_of("range_mean", 1), rng))

    def test_ood_differs_from_id(self):
        def vocab(ood):
            words, big = set(), 0.0
            for name in family_names():
                for seed in range(8):
                    ex = ex_of(name, seed, ood)
                    text = json.dumps(ex.state) + " ".join(q.instructions + " ".join(q.options) for q in ex.questions)
                    words |= set(re.findall(r"[A-Za-z]{3,}", text))
                    nums = [abs(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", str(ex.state))[:50]]
                    big = max([big] + nums)
            return words, big
        wi, bi = vocab(False)
        wo, bo = vocab(True)
        self.assertGreater(len(wo - wi) / len(wo), 0.1)
        self.assertGreater(bo, bi * 3)
        ks = lambda ood: {len(ex_of("wide_lookup", s, ood).questions[0].options) for s in range(60)}  # noqa: E731
        self.assertTrue(ks(True) - ks(False))
        self.assertGreaterEqual(max(ks(True)), 48)
        self.assertEqual(min(ks(False)), 2)

    def test_gold_position_roughly_uniform(self):
        choice = [n for n in family_names() if ex_of(n, 0).questions[0].type == "choice"]
        exs = generate_split("train", 1500, 5, families=choice, unanswerable_rate=0.0)
        for k in (2, 3, 4):
            pos = [0] * k
            for ex in exs:
                q = ex.questions[0]
                if len(q.options) == k:
                    pos[q.label] += 1
            tot = sum(pos)
            self.assertGreater(tot, 40, (k, pos))
            chi = sum((c - tot / k) ** 2 / (tot / k) for c in pos)
            self.assertLess(chi, 20 + 5 * k, (k, pos))

    # ---------------------------------------------------------------- independent ground truth
    def test_num_extreme(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("num_extreme", seed, ood)
                q = ex.questions[0]
                r = readings(ex.state)
                fn = max if "highest" in q.instructions else min
                self.assertEqual(q.options[q.label], fn(r, key=r.get))

    def test_date_extreme(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("date_extreme", seed, ood)
                q = ex.questions[0]
                d = dated(ex.state)
                fn = min if "earliest" in q.instructions else max
                self.assertEqual(q.options[q.label], fn(d, key=d.get))

    def test_multihop_json(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("multihop_json", seed, ood)
                q, st = ex.questions[0], ex.state
                oid = re.search(r"'([^']+)'", q.instructions).group(1)
                cust = st["customers"][st["orders"][oid]["customer"]]
                city = st["branches"][cust["branch"]]["city"] if "branch" in cust else cust["city"]
                self.assertEqual(q.options[q.label], city)

    def test_count_matching(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("count_matching", seed, ood)
                q = ex.questions[0]
                col = re.search(r"'([^']+)'", q.instructions).group(1)
                n = sum(1 for ln in ex.state.splitlines() if ln.endswith(": " + col))
                self.assertEqual(q.options[q.label], str(n))

    def test_sum_choice(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("sum_choice", seed, ood)
                q = ex.questions[0]
                tot = sum(Decimal(m) for m in re.findall(r": (\d+\.\d{2})$", ex.state, re.M))
                self.assertEqual(Decimal(q.options[q.label]), tot)

    def test_threshold_band(self):
        for ood in (False, True):
            for seed in range(20):
                ex = ex_of("threshold_band", seed, ood)
                q = ex.questions[0]
                m = re.search(r"below (\d+) is (\w+), from \d+ up to but excluding (\d+) is (\w+), \d+ and above is (\w+)",
                              ex.state)
                t1, b0, t2, b1, b2 = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), m.group(5)
                v = int(re.search(r"Observed score: (-?\d+)", ex.state).group(1))
                exp = b0 if v < t1 else (b1 if v < t2 else b2)
                self.assertEqual(q.options[q.label], exp)

    def test_tolerance(self):
        for ood in (False, True):
            for seed in range(20):
                ex = ex_of("tolerance", seed, ood)
                q = ex.questions[0]
                t, tol = map(float, re.search(r"Spec: (\S+) \+/- (\S+)", ex.state).groups())
                v = float(re.search(r"Measured: (\S+)", ex.state).group(1))
                exp = "too low" if v < t - tol else ("too high" if v > t + tol else "within tolerance")
                self.assertEqual(q.options[q.label], exp)

    def test_policy_exceptions(self):
        for ood in (False, True):
            names = ["pay out", "reject", "refer upward"] if ood else ["approve", "deny", "escalate"]
            for seed in range(30):
                ex = ex_of("policy_exceptions", seed, ood)
                q, st = ex.questions[0], ex.state
                lim = int(re.search(r"(?:up to|ceiling:) (\d+)", st["policy"]).group(1))
                tier = re.search(r"'(\w+)' tier", st["policy"]).group(1)
                c = st["case"]
                if c["fraud_flag"]:
                    exp = names[1]
                elif c["amount"] <= lim or (c["tier"] == tier and c["amount"] <= 2 * lim):
                    exp = names[0]
                else:
                    exp = names[2]
                self.assertEqual(q.options[q.label], exp)

    def test_noisy_one_line(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("noisy_one_line", seed, ood)
                q = ex.questions[0]
                flag = "FATAL" if ood else "CRITICAL"
                hits = [ln.split()[1] for ln in ex.state.splitlines() if f" {flag} " in ln]
                self.assertEqual(len(hits), 1)
                self.assertEqual(q.options[q.label], hits[0])

    def test_multi_policies(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("multi_policies", seed, ood)
                q, st = ex.questions[0], ex.state
                pol = {p["id"]: p for p in st["policies"]}
                bad = set()
                for p in pol.values():
                    v = st["record"][p["field"]]
                    if not (v <= p["value"] if p["op"] == "<=" else v >= p["value"]):
                        bad.add(p["id"])
                got = {re.match(r"(P\d+):", q.options[i]).group(1) for i in q.label}
                self.assertEqual(got, bad)
                self.assertTrue(bad)

    def test_multi_over_limit(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("multi_over_limit", seed, ood)
                q = ex.questions[0]
                over = set()
                for ln in ex.state.splitlines():
                    m = re.match(r"^(\S+): (\d+) \(limit (\d+)\)", ln)
                    if m and int(m.group(2)) > int(m.group(3)):
                        over.add(m.group(1))
                self.assertEqual({q.options[i] for i in q.label}, over)

    def test_multi_members(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("multi_members", seed, ood)
                q = ex.questions[0]
                x, y = re.findall(r"'([^']+)'", q.instructions)
                both = set()
                for ln in ex.state.splitlines()[1:]:
                    who, rest = ln.rsplit(": ", 1)
                    items = rest.split(", ")
                    if x in items and y in items:
                        both.add(who)
                self.assertEqual({q.options[i] for i in q.label}, both)

    def test_range_table_sum(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("range_table_sum", seed, ood)
                q = ex.questions[0]
                rows = [ln.split(",") for ln in ex.state.splitlines()[1:]]
                if "total quantity" in q.instructions:
                    exp = sum(int(r[1]) for r in rows)
                else:
                    exp = sum(int(r[1]) * int(r[2]) for r in rows)
                self.assertEqual(q.label, exp)
                self.assertLess(q.lo, q.hi)
                self.assertLessEqual(q.label, q.hi)

    def test_range_date_diff_and_percent(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("range_date_diff", seed, ood)
                d = {k: parse_date(v) for k, v in re.findall(r"^(\w+): (.+)$", ex.state, re.M)}
                self.assertEqual(ex.questions[0].label, (d["Closed"] - d["Opened"]).days)
                ex = ex_of("range_percent_change", seed, ood)
                a, b = map(int, re.findall(r": (\d+)$", ex.state, re.M))
                self.assertAlmostEqual(ex.questions[0].label, (b - a) / a * 100, places=2)

    def test_range_mean_and_lookup(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("range_mean", seed, ood)
                vs = [float(x) for x in re.findall(r"^- (\S+)$", ex.state, re.M)]
                self.assertAlmostEqual(ex.questions[0].label, sum(vs) / len(vs), places=6)
                ex = ex_of("range_lookup_product", seed, ood)
                t = re.search(r"'([^']+)'", ex.questions[0].instructions).group(1)
                o = ex.state["orders"][t]
                cost = ex.state["products"][o["product"]]["price"] * o["qty"] * (1 - o.get("discount_pct", 0) / 100)
                self.assertAlmostEqual(ex.questions[0].label, cost)

    def test_extract_families_offsets(self):
        for ood in (False, True):
            for seed in range(15):
                ex = ex_of("extract_id", seed, ood)
                s, e = ex.questions[0].label
                flag = "QUARANTINED" if ood else "FLAGGED"
                line = [ln for ln in ex.state.splitlines() if ln.endswith(flag)][0]
                self.assertEqual(ex.state[s:e], line.split()[1])
                ex = ex_of("extract_json_value", seed, ood)
                s, e = ex.questions[0].label
                rid = re.search(r"'([^']+)'", ex.questions[0].instructions).group(1)
                rec = [r for r in json.loads(ex.state)["records"] if r["id"] == rid][0]
                self.assertEqual(ex.state[s:e], rec["tracking"])
                ex = ex_of("extract_date", seed, ood)
                s, e = ex.questions[0].label
                evt = ex.questions[0].instructions.split(": ")[1]
                sent = [x for x in ex.state.split(". ") if x.startswith(f"The {evt} ")][0].rstrip(".")
                self.assertEqual(ex.state[s:e], sent.rsplit(" on ", 1)[1])
                ex = ex_of("extract_noisy", seed, ood)
                s, e = ex.questions[0].label
                word = "chargeback" if ood else "refund"
                line = [ln for ln in ex.state.splitlines() if word in ln][0]
                self.assertEqual(ex.state[s:e], line.rsplit(" ", 1)[1].rstrip("."))
                for name in ("extract_name", "extract_amount"):
                    ex = ex_of(name, seed, ood)
                    s, e = ex.questions[0].label
                    self.assertTrue(0 <= s < e <= len(ex.state))
                    self.assertTrue(ex.state[s:e].strip())

    # ----------------------------------------------------------------------- pairs, adversarial
    def test_noul_pairs_complementary_and_correct(self):
        pairs = [n for n in family_names() if n.startswith("pair_")]
        self.assertGreaterEqual(len(pairs), 6)
        for name in pairs:
            for ood in (False, True):
                for seed in range(20):
                    ex = ex_of(name, seed, ood)
                    self.assertEqual(len(ex.questions), 2)
                    a, b = ex.questions
                    self.assertEqual((a.type, b.type), ("noul", "noul"))
                    self.assertNotEqual(a.label, b.label, (name, seed))
        for ood in (False, True):  # independent truth for numeric and date pairs
            for seed in range(20):
                ex = ex_of("pair_numeric", seed, ood)
                r = readings(ex.state)
                for q in ex.questions:
                    m = re.search(r"(\S+) has a (higher|lower) reading than (\S+?)\.?$", q.instructions)
                    t = (r[m.group(1).lower()] > r[m.group(3)]) == (m.group(2) == "higher")
                    neg = "not the case" in q.instructions or "not true" in q.instructions or "false" in q.instructions
                    self.assertEqual(q.label, t != neg)
                ex = ex_of("pair_date", seed, ood)
                d = dated(ex.state)
                for q in ex.questions:
                    m = re.search(r"(\S+) happened (before|after) (\S+?)\.?$", q.instructions)
                    t = (d[m.group(1).lower()] < d[m.group(3)]) == (m.group(2) == "before")
                    neg = "not the case" in q.instructions or "not true" in q.instructions or "false" in q.instructions
                    self.assertEqual(q.label, t != neg)

    def test_adversarial_families_inject_and_label_ignores(self):
        for ood in (False, True):
            tpls = INJECTION_OOD if ood else INJECTION_ID
            for seed in range(25):
                ex = ex_of("adversarial_choice", seed, ood)
                q = ex.questions[0]
                gold = q.options[q.label]
                found = [o for o in q.options for t in tpls if t.replace("{opt}", o) in ex.state]
                self.assertEqual(len(found), 1)
                self.assertNotEqual(found[0], gold)
                r = readings(ex.state)
                self.assertEqual(gold, (max if "highest" in q.instructions else min)(r, key=r.get))

                ex = ex_of("adversarial_noul", seed, ood)
                q = ex.questions[0]
                wrong = "false" if q.label else "true"
                self.assertTrue(any(t.replace("{opt}", wrong) in ex.state for t in tpls))

                ex = ex_of("adversarial_extract", seed, ood)
                s, e = ex.questions[0].label
                gold = ex.state[s:e]
                inj = [ln for ln in ex.state.splitlines() if any(ln == t.replace("{opt}", ln.split()[-1].strip(".,")) or
                                                                  t.split("{opt}")[0] in ln for t in tpls)]
                self.assertTrue(inj)
                self.assertNotIn(gold, inj[0])

    # ---------------------------------------------------------------------- benchmark writer
    def test_write_benchmark(self):
        sizes = {"train": 40, "val": 10, "test": 10, "test_ood": 10}
        with tempfile.TemporaryDirectory() as d:
            man = write_benchmark(d, sizes, seed=2)
            with open(os.path.join(d, "manifest.json")) as fh:
                disk = json.load(fh)
            self.assertEqual(disk["counts"], sizes)
            self.assertEqual(disk["families"], family_names())
            for split, n in sizes.items():
                path = os.path.join(d, f"{split}.jsonl")
                with open(path, "rb") as fh:
                    self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), man["files"][f"{split}.jsonl"]["sha256"])
                self.assertEqual(len(list(read_jsonl(path))), n)
            man2 = write_benchmark(os.path.join(d, "again"), sizes, seed=2)
            self.assertEqual(man["files"], man2["files"])

    def test_tagged_matches_split(self):
        tagged = generate_tagged("val", 20, 4)
        self.assertEqual([fingerprint(e) for _, e in tagged], [fingerprint(e) for e in generate_split("val", 20, 4)])
        self.assertTrue(all(f in FAMILIES for f, _ in tagged))


if __name__ == "__main__":
    unittest.main()
