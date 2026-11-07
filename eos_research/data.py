"""Data for language-model and decision training.

Language modelling uses a flat token file (``train.bin``/``val.bin``) read through a memory
map, so corpora larger than RAM work. Training windows are drawn from a generator seeded by
``(seed, rank, step, micro_step)``: the batch for a given step is a pure function of those
numbers, so a resumed run sees exactly the batches an uninterrupted run would, with no data
iterator state to checkpoint.

Decision examples are JSON lines. Each line holds a shared ``state`` and one or more
questions, each with ``question``, ``options``, ``primitive`` and ``label``::

    {"state": {"amount": 12000, "flagged": false},
     "questions": [{"question": "Routing tier", "options": ["Tier 1", "Tier 2", "Tier 3"],
                    "primitive": "choice", "label": 1}]}

A line with a single top-level ``question`` is accepted as a one-question example.
Labels follow ``rhen_loss.compute_rhen_multi_task_loss``: ``bool`` for noul, ``int`` for
choice, ``float`` (0 .. levels-1) for score and a list of indices for multichoice.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

META_FILE = "meta.json"
PRIMITIVES = ("noul", "choice", "score", "multichoice")


# ---------------------------------------------------------------------------
# Token corpora
# ---------------------------------------------------------------------------
def _token_dtype(vocab_size: int):
    return np.uint16 if vocab_size <= np.iinfo(np.uint16).max + 1 else np.uint32


def prepare_token_corpus(
    texts: Iterable[str],
    encode,
    out_dir: str,
    vocab_size: int,
    eos_token_id: Optional[int] = None,
    val_fraction: float = 0.01,
    source_note: str = "",
) -> Dict[str, Any]:
    """Tokenize documents and write ``train.bin``, ``val.bin`` and ``meta.json``.

    ``encode`` maps text to token ids. Each document ends with ``eos_token_id`` when given.
    Documents are assigned to the validation split by a hash of their text, so the split is
    stable when documents are added or reordered.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    os.makedirs(out_dir, exist_ok=True)
    dtype = _token_dtype(vocab_size)
    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    digest = hashlib.sha256()
    paths = {split: os.path.join(out_dir, f"{split}.bin") for split in counts}
    handles = {split: open(path, "wb") for split, path in paths.items()}
    try:
        for text in texts:
            text_digest = hashlib.sha256(text.encode("utf-8")).digest()
            digest.update(text_digest)
            bucket = int.from_bytes(text_digest[:8], "big") / float(1 << 64)
            split = "val" if bucket < val_fraction else "train"
            ids = list(encode(text))
            if eos_token_id is not None:
                ids.append(int(eos_token_id))
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.int64)
            if arr.min() < 0 or arr.max() >= vocab_size:
                raise ValueError(f"token id out of range for vocab_size={vocab_size}")
            handles[split].write(arr.astype(dtype).tobytes())
            counts[split] += len(ids)
            docs[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    meta = {
        "vocab_size": int(vocab_size),
        "dtype": np.dtype(dtype).name,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "train_documents": docs["train"],
        "val_documents": docs["val"],
        "eos_token_id": eos_token_id,
        "val_fraction": val_fraction,
        "content_sha256": digest.hexdigest(),
        "source": source_note,
    }
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def read_meta(data_dir: str) -> Dict[str, Any]:
    with open(os.path.join(data_dir, META_FILE), "r", encoding="utf-8") as fh:
        return json.load(fh)


class TokenWindows:
    """Fixed-length training windows from one split of a prepared corpus."""

    def __init__(self, data_dir: str, split: str, seq_len: int):
        meta = read_meta(data_dir)
        self.path = os.path.join(data_dir, f"{split}.bin")
        self.dtype = np.dtype(meta["dtype"])
        self.vocab_size = int(meta["vocab_size"])
        self.seq_len = int(seq_len)
        n_tokens = os.path.getsize(self.path) // self.dtype.itemsize
        self.num_tokens = n_tokens
        if n_tokens < self.seq_len + 1:
            raise ValueError(
                f"{self.path} holds {n_tokens} tokens, fewer than seq_len + 1 = {self.seq_len + 1}"
            )
        self._data = np.memmap(self.path, dtype=self.dtype, mode="r", shape=(n_tokens,))

    @property
    def num_windows(self) -> int:
        return (self.num_tokens - 1) // self.seq_len

    def _window(self, start: int) -> np.ndarray:
        return np.asarray(self._data[start:start + self.seq_len + 1], dtype=np.int64)

    def sample(self, batch_size: int, seed: int, rank: int, step: int, micro_step: int = 0) -> torch.Tensor:
        """``[batch_size, seq_len + 1]`` random windows, fixed by (seed, rank, step, micro_step)."""
        gen = np.random.default_rng([int(seed), int(rank), int(step), int(micro_step)])
        starts = gen.integers(0, self.num_tokens - self.seq_len - 1, size=batch_size, endpoint=True)
        return torch.from_numpy(np.stack([self._window(int(s)) for s in starts]))

    def sequential(self, batch_size: int, max_batches: Optional[int] = None) -> Iterator[torch.Tensor]:
        """Non-overlapping windows in file order, for deterministic evaluation."""
        produced = 0
        batch: List[np.ndarray] = []
        for w in range(self.num_windows):
            batch.append(self._window(w * self.seq_len))
            if len(batch) == batch_size:
                yield torch.from_numpy(np.stack(batch))
                batch = []
                produced += 1
                if max_batches is not None and produced >= max_batches:
                    return
        if batch:
            yield torch.from_numpy(np.stack(batch))


def lm_inputs(window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``EOSModel`` shifts labels internally, so inputs and labels are the same tokens."""
    tokens = window[:, :-1].contiguous()
    return tokens, tokens.clone()


# ---------------------------------------------------------------------------
# Decision examples
# ---------------------------------------------------------------------------
@dataclass
class DecisionQuestion:
    question: str
    options: List[str]
    primitive: str
    label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"primitive must be one of {PRIMITIVES}, got {self.primitive!r}")
        if self.primitive == "noul":
            if not isinstance(self.label, bool):
                raise ValueError("noul labels are booleans")
        elif self.primitive == "choice":
            if isinstance(self.label, bool) or not isinstance(self.label, int):
                raise ValueError("choice labels are integer option indices")
            if not 0 <= self.label < len(self.options):
                raise ValueError(f"choice label {self.label} outside {len(self.options)} options")
        elif self.primitive == "score":
            self.label = float(self.label)
            if not 0.0 <= self.label <= len(self.options) - 1:
                raise ValueError(f"score label {self.label} outside 0..{len(self.options) - 1}")
        elif self.primitive == "multichoice":
            if not isinstance(self.label, list) or any(not 0 <= int(i) < len(self.options) for i in self.label):
                raise ValueError("multichoice labels are lists of option indices")
            self.label = [int(i) for i in self.label]

    @property
    def num_classes(self) -> int:
        return 2 if self.primitive == "noul" else len(self.options)

    def class_index(self) -> Optional[int]:
        """Index of the correct class for single-label primitives (noul: 0 = yes, 1 = no)."""
        if self.primitive == "noul":
            return 0 if self.label else 1
        if self.primitive == "choice":
            return int(self.label)
        if self.primitive == "score":
            return int(round(self.label))
        return None

    def packer_item(self) -> Tuple[str, List[str], str, Optional[Dict[str, Any]]]:
        options = self.options if self.primitive != "noul" else (self.options or ["Yes", "No"])
        return (self.question, list(options), self.primitive, dict(self.metadata) or None)

    def to_dict(self) -> Dict[str, Any]:
        out = {"question": self.question, "options": self.options, "primitive": self.primitive, "label": self.label}
        if self.metadata:
            out["metadata"] = self.metadata
        return out


@dataclass
class DecisionExample:
    state: Any
    questions: List[DecisionQuestion]
    split: str = "train"
    task: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "questions": [q.to_dict() for q in self.questions],
            "split": self.split,
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "DecisionExample":
        raw_questions = record.get("questions")
        if raw_questions is None:
            raw_questions = [record]
        questions = [
            DecisionQuestion(
                question=str(q["question"]),
                options=[str(o) for o in q.get("options", [])],
                primitive=str(q.get("primitive", "choice")).lower(),
                label=q["label"],
                metadata=dict(q.get("metadata") or {}),
            )
            for q in raw_questions
        ]
        if not questions:
            raise ValueError("a decision example needs at least one question")
        return cls(
            state=record["state"],
            questions=questions,
            split=str(record.get("split", "train")),
            task=str(record.get("task", "")),
        )


def write_decision_jsonl(examples: Sequence[DecisionExample], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), sort_keys=True) + "\n")


def load_decision_jsonl(path: str) -> List[DecisionExample]:
    examples: List[DecisionExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(DecisionExample.from_dict(json.loads(line)))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return examples


def sample_decision_batch(
    examples: Sequence[DecisionExample], batch_size: int, seed: int, rank: int, step: int, micro_step: int = 0
) -> List[DecisionExample]:
    """Deterministic decision minibatch, fixed by (seed, rank, step, micro_step) like the LM windows."""
    gen = np.random.default_rng([int(seed), int(rank), int(step), int(micro_step), 7])
    idx = gen.integers(0, len(examples), size=batch_size)
    return [examples[int(i)] for i in idx]
