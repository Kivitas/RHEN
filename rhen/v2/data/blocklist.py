"""Benchmark firewall (plan 7.1, T-304).

``blocklist.json`` names every benchmark split that must never be trained on and holds, per name, a map
of example fingerprints (empty until the benchmark data is registered with ``add_examples``).  Loaders
call ``Blocklist.filter`` / ``check``; a hit raises ``BlockedExampleError`` or is dropped.  Fingerprints
catch exact (normalised) copies; ``ngram_overlap`` catches partial leakage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from rhen.v2.packing import Rhen2Example

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocklist.json")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


class BlockedExampleError(ValueError):
    """Raised when an example matches a blocklisted benchmark."""


def _norm(text: str) -> str:
    return _WS.sub(" ", str(text)).strip().lower()


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)


def example_fingerprint(example: Rhen2Example) -> str:
    """sha256 over the normalised state text, question texts and options (labels excluded)."""
    parts = [_norm(_state_text(example.state))]
    for q in example.questions:
        parts.append(_norm(q.type))
        parts.append(_norm(q.instructions))
        parts.extend(_norm(o) for o in q.options)
        parts.append("|")
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def example_text(example: Rhen2Example) -> str:
    parts = [_state_text(example.state)]
    for q in example.questions:
        parts.append(q.instructions)
        parts.extend(q.options)
    return "\n".join(parts)


def _ngrams(text: str, n: int) -> Iterator[int]:
    words = _WORD.findall(text.lower())
    for i in range(len(words) - n + 1):
        yield int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode("utf-8"), digest_size=8).digest(), "big")


def build_ngram_index(examples: Iterable[Rhen2Example], n: int = 13) -> Set[int]:
    index: Set[int] = set()
    for ex in examples:
        index.update(_ngrams(example_text(ex), n))
    return index


def ngram_overlap(example: Rhen2Example, index: Set[int], n: int = 13) -> float:
    """Fraction of the example's word n-grams that occur in ``index`` (0.0 if it has no n-gram)."""
    grams = list(_ngrams(example_text(example), n))
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in index) / len(grams)


class Blocklist:
    def __init__(self, benchmarks: Optional[Dict[str, Dict[str, Any]]] = None, ngram_n: int = 13):
        self.benchmarks: Dict[str, Dict[str, Any]] = benchmarks or {}
        self.ngram_n = ngram_n
        self._index: Set[int] = set()
        self._fp: Dict[str, str] = {}
        for name, entry in self.benchmarks.items():
            entry.setdefault("fingerprints", {})
            for fp in entry["fingerprints"]:
                self._fp[fp] = name
            self._index.update(int(g) for g in entry.get("ngrams", []))

    # -- persistence
    @classmethod
    def load(cls, path: Optional[str] = None) -> "Blocklist":
        with open(path or DEFAULT_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        fps = raw.get("benchmarks", {})
        for entry in fps.values():          # accept a list as well as a map
            if isinstance(entry.get("fingerprints"), list):
                entry["fingerprints"] = {fp: 1 for fp in entry["fingerprints"]}
        return cls(fps, ngram_n=int(raw.get("ngram_n", 13)))

    def save(self, path: Optional[str] = None) -> None:
        out = {"version": 1, "ngram_n": self.ngram_n, "benchmarks": {}}
        for name, entry in self.benchmarks.items():
            e = dict(entry)
            if not e.get("ngrams"):
                e.pop("ngrams", None)
            out["benchmarks"][name] = e
        with open(path or DEFAULT_PATH, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=False)
            fh.write("\n")

    @property
    def names(self) -> List[str]:
        return list(self.benchmarks)

    @property
    def n_fingerprints(self) -> int:
        return len(self._fp)

    @property
    def ngram_index(self) -> Set[int]:
        return self._index

    # -- registration
    def add_examples(self, name: str, examples: Iterable[Rhen2Example], with_ngrams: bool = True) -> int:
        entry = self.benchmarks.setdefault(name, {"note": "", "fingerprints": {}})
        grams = set(entry.get("ngrams", []))
        count = 0
        for ex in examples:
            fp = example_fingerprint(ex)
            entry["fingerprints"][fp] = 1
            self._fp[fp] = name
            if with_ngrams:
                new = set(_ngrams(example_text(ex), self.ngram_n))
                grams |= new
                self._index |= new
            count += 1
        if with_ngrams:
            entry["ngrams"] = sorted(grams)
        return count

    # -- checks
    def match(self, example: Rhen2Example, ngram_threshold: Optional[float] = None) -> Optional[str]:
        """Name of the blocklisted benchmark the example matches, else None."""
        name = self._fp.get(example_fingerprint(example))
        if name:
            return name
        if ngram_threshold is not None and self._index and ngram_overlap(example, self._index, self.ngram_n) >= ngram_threshold:
            return "ngram-overlap"
        return None

    def contains(self, example: Rhen2Example, ngram_threshold: Optional[float] = None) -> bool:
        return self.match(example, ngram_threshold) is not None

    def check(self, example: Rhen2Example, ngram_threshold: Optional[float] = None) -> None:
        name = self.match(example, ngram_threshold)
        if name:
            raise BlockedExampleError(f"example matches blocklisted benchmark {name!r}")

    def filter(self, examples: Iterable[Rhen2Example], raise_on_hit: bool = False,
               ngram_threshold: Optional[float] = None) -> Iterator[Rhen2Example]:
        for ex in examples:
            if raise_on_hit:
                self.check(ex, ngram_threshold)
                yield ex
            elif not self.contains(ex, ngram_threshold):
                yield ex
