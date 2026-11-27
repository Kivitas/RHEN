"""Teacher labelling (plan 7.1.4, 7.5).  No network code and no keys live here.

``Teacher.label(state, question)`` returns probabilities over the options then null (null last).  Real
providers are plugged in through ``HTTPTeacher``, which only wraps a caller-supplied ``prompt -> text``
callable.  ``EnsembleTeacher`` averages distributions and records their disagreement (generalised
Jensen-Shannon divergence, nats) so it survives as genuine uncertainty in the soft label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from rhen.v2.data.blocklist import example_fingerprint
from rhen.v2.data.schema import example_to_record, from_v1_example
from rhen.v2.packing import QuestionSpec, Rhen2Example

log = logging.getLogger("rhen.v2.teacher")
SOFT_TYPES = ("choice", "score", "multi", "noul")


class TeacherError(RuntimeError):
    pass


def n_options(q: QuestionSpec) -> int:
    if q.type == "noul":
        return 2
    if q.type in ("choice", "score", "multi"):
        return len(q.options)
    raise ValueError(f"no option distribution for question type {q.type!r}")


def _normalise(p: Sequence[float]) -> List[float]:
    vals = [max(0.0, float(x)) for x in p]
    total = sum(vals)
    if total <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [v / total for v in vals]


def _entropy(p: Sequence[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def js_divergence(dists: Sequence[Sequence[float]]) -> float:
    """Generalised JSD: H(mean) - mean H(p_i); 0 when all distributions agree."""
    k = len(dists)
    mean = [sum(d[i] for d in dists) / k for i in range(len(dists[0]))]
    return max(0.0, _entropy(mean) - sum(_entropy(d) for d in dists) / k)


@runtime_checkable
class Teacher(Protocol):
    def label(self, state: Any, question: QuestionSpec) -> List[float]: ...


def _state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)


class FakeTeacher:
    """Deterministic from a hash of (seed, state, question); for dry runs and tests."""

    name = "fake"

    def __init__(self, seed: int = 0, sharpness: float = 2.0, null_mass: float = 0.05):
        self.seed, self.sharpness, self.null_mass = seed, sharpness, null_mass
        self.calls = 0
        self.prompt_chars = 0

    def label(self, state: Any, question: QuestionSpec) -> List[float]:
        k = n_options(question)
        key = "\x1f".join([str(self.seed), _state_text(state), question.type, question.instructions, *question.options])
        self.calls += 1
        self.prompt_chars += len(key)
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        logits = []
        for i in range(k):
            h = hashlib.sha256(digest + i.to_bytes(4, "big")).digest()
            logits.append((int.from_bytes(h[:4], "big") / 2**32) * self.sharpness * 2)
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        z = sum(exps)
        return [e / z * (1 - self.null_mass) for e in exps] + [self.null_mass]


class EnsembleTeacher:
    name = "ensemble"

    def __init__(self, teachers: Sequence[Teacher]):
        if not teachers:
            raise ValueError("ensemble needs at least one teacher")
        self.teachers = list(teachers)
        self.last_disagreement = 0.0
        self.disagreements: List[float] = []

    @property
    def calls(self) -> int:
        return sum(getattr(t, "calls", 0) for t in self.teachers)

    @property
    def prompt_chars(self) -> int:
        return sum(getattr(t, "prompt_chars", 0) for t in self.teachers)

    def label(self, state: Any, question: QuestionSpec) -> List[float]:
        dists = [_normalise(t.label(state, question)) for t in self.teachers]
        if len({len(d) for d in dists}) != 1:
            raise TeacherError("teachers returned distributions of different lengths")
        mean = [sum(d[i] for d in dists) / len(dists) for i in range(len(dists[0]))]
        self.last_disagreement = js_divergence(dists) if len(dists) > 1 else 0.0
        self.disagreements.append(self.last_disagreement)
        return mean

    @property
    def mean_disagreement(self) -> float:
        return sum(self.disagreements) / len(self.disagreements) if self.disagreements else 0.0


# --------------------------------------------------------------------------- prompt builders (pure)
def build_state_generation_prompt(domain: str, n_states: int = 1, seed_topic: str = "") -> str:
    return (
        f"You write realistic operational records for the domain: {domain}.\n"
        f"Write {n_states} distinct record(s) as a JSON list. Each record is a JSON object or a short text "
        "(ticket, log excerpt, email, table) containing the facts needed to make one decision, plus some "
        "irrelevant detail. Do not state the decision.\n"
        + (f"Topic hint: {seed_topic}\n" if seed_topic else "")
        + "Return only the JSON list."
    )


def build_question_generation_prompt(state: Any, qtype: str = "choice", n_questions: int = 3, k_options: int = 4) -> str:
    return (
        f"Here is a record:\n{_state_text(state)}\n\n"
        f"Write {n_questions} decision question(s) of type '{qtype}' about it, each with {k_options} plausible "
        "options (omit options for 'noul', 'range' and 'extract' types). Some questions should be unanswerable "
        "from the record. Return JSON: a list of {\"instructions\": str, \"options\": [str]}."
    )


def build_option_probability_prompt(state: Any, question: QuestionSpec) -> str:
    k = n_options(question)
    opts = ["true", "false"] if question.type == "noul" else list(question.options)
    listing = "\n".join(f"{i}: {o}" for i, o in enumerate(opts))
    return (
        f"Record:\n{_state_text(state)}\n\nQuestion ({question.type}): {question.instructions}\nOptions:\n{listing}\n\n"
        f"Give the probability that each option is correct, then the probability that the record does not "
        f"contain enough information to answer (the null option, last). Return a JSON list of exactly {k + 1} "
        "numbers that sums to 1 and nothing else."
    )


def parse_probability_response(text: str, k: int) -> List[float]:
    """Parse a JSON list of k+1 numbers from a model reply (first bracketed list), normalised."""
    m = re.search(r"\[[^\[\]]*\]", text)
    if not m:
        raise TeacherError("no list found in teacher reply")
    try:
        vals = [float(x) for x in json.loads(m.group(0))]
    except (ValueError, TypeError) as exc:
        raise TeacherError(f"unparseable teacher reply: {exc}") from exc
    if len(vals) != k + 1 or any(v < 0 or not math.isfinite(v) for v in vals):
        raise TeacherError(f"expected {k + 1} non-negative numbers, got {vals}")
    if sum(vals) <= 0:
        raise TeacherError("teacher reply has no probability mass")
    return _normalise(vals)


class HTTPTeacher:
    """Wraps a caller-supplied ``complete(prompt: str) -> str``.  This module never makes a request."""

    def __init__(self, complete: Optional[Callable[[str], str]] = None, name: str = "http"):
        self.complete = complete
        self.name = name
        self.calls = 0
        self.prompt_chars = 0

    def label(self, state: Any, question: QuestionSpec) -> List[float]:
        if self.complete is None:
            raise TeacherError(
                "HTTPTeacher has no completion callable. Pass complete=<function prompt -> text> that talks to "
                "your provider; no network client or API key is bundled with the repository."
            )
        prompt = build_option_probability_prompt(state, question)
        self.calls += 1
        self.prompt_chars += len(prompt)
        return parse_probability_response(self.complete(prompt), n_options(question))


# --------------------------------------------------------------------------- labelling run
def _done_ids(path: str) -> set:
    ids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["meta"]["id"])
                    except (ValueError, KeyError):
                        break           # truncated last line from an interrupted run
    return ids


def run_labelling(examples: Iterable[Rhen2Example], teacher: Teacher, out_path: str, resume: bool = True) -> Dict[str, Any]:
    """Write one JSONL record per example with ``soft`` labels (options then null, normalised).  Records are
    identified by the example fingerprint; with ``resume`` ids already in ``out_path`` are skipped.  Unsupported
    question types (range, extract) keep their hard labels and get no soft label."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = _done_ids(out_path) if resume else set()
    mode = "a" if resume and os.path.exists(out_path) else "w"
    if mode == "a":                         # drop a truncated trailing line so appends stay valid JSONL
        with open(out_path, "rb") as fh:
            data = fh.read()
        if data and not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
            with open(out_path, "wb") as fh:
                fh.write(data)
    t0 = time.time()
    written = skipped = n_q = 0
    calls0 = getattr(teacher, "calls", 0)
    chars0 = getattr(teacher, "prompt_chars", 0)
    with open(out_path, mode, encoding="utf-8") as fh:
        for ex in examples:
            rid = example_fingerprint(ex)[:16]
            if rid in done:
                skipped += 1
                continue
            new = copy.deepcopy(ex)
            dis = []
            for q in new.questions:
                if q.type not in SOFT_TYPES:
                    continue
                q.soft = _normalise(teacher.label(new.state, q))
                n_q += 1
                if hasattr(teacher, "last_disagreement"):
                    dis.append(teacher.last_disagreement)
            meta: Dict[str, Any] = {"id": rid, "teacher": getattr(teacher, "name", type(teacher).__name__)}
            if dis:
                meta["disagreement"] = round(sum(dis) / len(dis), 6)
            fh.write(json.dumps(example_to_record(new, meta), ensure_ascii=False) + "\n")
            fh.flush()
            done.add(rid)
            written += 1
    summary = {
        "written": written, "skipped": skipped, "questions": n_q,
        "teacher_calls": getattr(teacher, "calls", 0) - calls0,
        "prompt_chars": getattr(teacher, "prompt_chars", 0) - chars0,
        "seconds": round(time.time() - t0, 3), "out": out_path,
    }
    log.info("labelling done: %s", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Teacher soft-labelling for RHEN-2 (dry run uses a fake teacher).")
    ap.add_argument("--dry-run", action="store_true", help="use FakeTeacher over synthetic v1 tasks")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.dry_run:
        ap.error("only --dry-run is available from the CLI; wrap a provider with HTTPTeacher in Python")
    from rhen.research.tasks import generate_split

    examples = [from_v1_example(e) for e in generate_split("train", args.n, seed=args.seed)]
    summary = run_labelling(examples, FakeTeacher(seed=args.seed), args.out, resume=not args.no_resume)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
