"""Packing for RHEN-2: states, questions and options become padded tensors.

Layout (no state and no other question appears in a question or option sequence):

* state    ``[CTRL_STATE] state tokens``                 position 0 doubles as the null span
* question ``[CTRL_TYPE_x] instruction tokens``
* option   ``label tokens [CTRL_CRITERIA] criteria tokens``

Control ids sit above the tokenizer vocabulary and have no text form, so state or question
text can never produce them. Delimiter look-alikes in the text are neutralised first.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import torch

from rhen.rhen_packer import sanitize_delimiter_text
from rhen.v2.config import QUESTION_TYPES, TYPE_INDEX, Rhen2Config

NULL_LABEL = "__null__"
IGNORE = -1       # no label for this question
NULL_INDEX = -2   # gold answer is "cannot tell"

CTRL_NAMES = (
    "state", "null", "type_choice", "type_noul", "type_score", "type_multi", "type_range",
    "type_extract", "criteria", "mask",
)
K_BUCKETS = (2, 4, 8, 16, 32, 64, 128, 255)

_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?%?")
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_EPOCH = _dt.date(2000, 1, 1)


def k_bucket(k: int) -> int:
    for edge in K_BUCKETS:
        if k <= edge:
            return edge
    return K_BUCKETS[-1]


@dataclass
class QuestionSpec:
    type: str
    instructions: str
    options: List[str] = field(default_factory=list)
    criteria: List[str] = field(default_factory=list)
    lo: Optional[float] = None
    hi: Optional[float] = None
    # Optional supervision. choice/score: option index, option label or NULL_LABEL;
    # noul: bool or NULL_LABEL; multi: list of indices or labels; range: float;
    # extract: (start_char, end_char) in the serialised state, or NULL_LABEL.
    label: Any = None
    soft: Optional[List[float]] = None   # distribution over options + null (null last)

    def validate(self, cfg: Rhen2Config) -> None:
        if self.type not in TYPE_INDEX:
            raise ValueError(f"unknown question type {self.type!r}; choose from {QUESTION_TYPES}")
        if self.type == "noul" and self.options not in ([], ["true", "false"]):
            raise ValueError("noul questions use the implied options true/false; leave options empty")
        if self.type in ("range", "extract") and self.options:
            raise ValueError(f"{self.type} questions take no options")
        if self.type in ("choice", "score", "multi") and not self.options:
            raise ValueError(f"{self.type} questions need options")
        if len(self.options) > cfg.max_options:
            raise ValueError(f"{len(self.options)} options exceed max_options={cfg.max_options}")
        if self.type == "score" and len(self.options) > cfg.max_levels:
            raise ValueError(f"score questions allow at most {cfg.max_levels} levels")
        if self.criteria and len(self.criteria) != len(self.options):
            raise ValueError("criteria must have one entry per option")
        if self.type == "range" and (self.lo is None or self.hi is None or not self.hi > self.lo):
            raise ValueError("range questions need lo < hi")


@dataclass
class Rhen2Example:
    state: Any
    questions: List[QuestionSpec]


@dataclass
class Rhen2Batch:
    state_ids: torch.Tensor      # [B, Ls]
    state_mask: torch.Tensor     # [B, Ls] bool
    state_num: torch.Tensor      # [B, Ls, 4]
    q_ids: torch.Tensor          # [B, M, Lq]
    q_mask: torch.Tensor         # [B, M, Lq] bool
    q_num: torch.Tensor          # [B, M, Lq, 4]
    q_type: torch.Tensor         # [B, M] long
    q_valid: torch.Tensor        # [B, M] bool
    opt_ids: torch.Tensor        # [B, M, K, Lo]
    opt_mask: torch.Tensor       # [B, M, K, Lo] bool
    opt_valid: torch.Tensor      # [B, M, K] bool
    range_lo: torch.Tensor       # [B, M]
    range_hi: torch.Tensor       # [B, M]
    # supervision (all optional)
    label_idx: Optional[torch.Tensor] = None     # [B, M] long: >=0 option, NULL_INDEX, IGNORE
    label_soft: Optional[torch.Tensor] = None    # [B, M, K+1] float
    has_soft: Optional[torch.Tensor] = None      # [B, M] bool
    label_multi: Optional[torch.Tensor] = None   # [B, M, K] float
    has_multi: Optional[torch.Tensor] = None     # [B, M] bool
    label_range: Optional[torch.Tensor] = None   # [B, M] float
    has_range: Optional[torch.Tensor] = None     # [B, M] bool
    span_start: Optional[torch.Tensor] = None    # [B, M] long (0 = null span, IGNORE = none)
    span_end: Optional[torch.Tensor] = None      # [B, M] long
    # python-side metadata for decoding results
    state_texts: List[str] = field(default_factory=list)
    state_offsets: List[List[Tuple[int, int]]] = field(default_factory=list)
    specs: List[List[QuestionSpec]] = field(default_factory=list)

    _TENSORS = (
        "state_ids", "state_mask", "state_num", "q_ids", "q_mask", "q_num", "q_type", "q_valid",
        "opt_ids", "opt_mask", "opt_valid", "range_lo", "range_hi", "label_idx", "label_soft",
        "has_soft", "label_multi", "has_multi", "label_range", "has_range", "span_start", "span_end",
    )

    @property
    def batch_size(self) -> int:
        return int(self.state_ids.shape[0])

    @property
    def max_questions(self) -> int:
        return int(self.q_ids.shape[1])

    @property
    def max_options(self) -> int:
        return int(self.opt_ids.shape[2])

    def to(self, device) -> "Rhen2Batch":
        for name in self._TENSORS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        return self


class Rhen2Packer:
    def __init__(self, cfg: Rhen2Config, tokenizer=None):
        if tokenizer is None:
            from rhen.tokenizer import get_tokenizer

            tokenizer = get_tokenizer()
        self.cfg = cfg
        self.tok = tokenizer
        self.base = int(tokenizer.vocab_size)
        if self.base + len(CTRL_NAMES) > cfg.vocab_size:
            raise ValueError(
                f"model vocab_size={cfg.vocab_size} leaves no room for {len(CTRL_NAMES)} control ids "
                f"above the tokenizer's {self.base} entries"
            )
        self.ctrl = {name: self.base + i for i, name in enumerate(CTRL_NAMES)}
        self.pad_id = int(tokenizer.pad_token_id)

    # -- text ----------------------------------------------------------------
    @staticmethod
    def serialise_state(state: Any) -> str:
        if isinstance(state, str):
            text = state
        else:
            text = json.dumps(state, ensure_ascii=False, default=str)
        return sanitize_delimiter_text(text)

    def _encode(self, text: str, with_numbers: bool) -> Tuple[List[int], List[Tuple[int, int]], List[List[float]]]:
        enc = self.tok.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        feats = [[0.0, 0.0, 0.0, 0.0] for _ in ids]
        if with_numbers and ids:
            self._numeric_features(text, offsets, feats)
        return ids, offsets, feats

    @staticmethod
    def _feature(value: float, flag: float) -> List[float]:
        mag = math.log10(abs(value) + 1.0)
        return [1.0 if value < 0 else 0.0, min(mag / 12.0, 2.0), mag - math.floor(mag), flag]

    def _numeric_features(self, text: str, offsets, feats) -> None:
        claimed: List[Tuple[int, int]] = []

        def put(start: int, end: int, value: float, flag: float) -> None:
            for i, (a, b) in enumerate(offsets):
                if b > start and a < end:          # first token overlapping the match
                    feats[i] = self._feature(value, flag)
                    break
            claimed.append((start, end))

        for m in _DATE.finditer(text):
            try:
                days = (_dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) - _EPOCH).days
            except ValueError:
                continue
            put(m.start(), m.end(), float(days), 2.0)
        for m in _NUMBER.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in claimed):
                continue
            raw = m.group(0)
            percent = raw.endswith("%")
            try:
                value = float(raw.rstrip("%").replace(",", ""))
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            put(m.start(), m.end(), value, 1.0 if percent else 0.0)

    # -- labels --------------------------------------------------------------
    @staticmethod
    def _option_index(spec: QuestionSpec, label: Any) -> int:
        if isinstance(label, int) and not isinstance(label, bool):
            if not 0 <= label < len(spec.options):
                raise ValueError(f"label index {label} out of range for {len(spec.options)} options")
            return label
        if label not in spec.options:
            raise ValueError(f"label {label!r} is not one of the options")
        return spec.options.index(label)

    # -- packing -------------------------------------------------------------
    def pack(self, examples: Sequence[Rhen2Example]) -> Rhen2Batch:
        cfg = self.cfg
        if not examples:
            raise ValueError("pack needs at least one example")
        bsz = len(examples)
        m_max = max(1, max(len(e.questions) for e in examples))

        states: List[Tuple[List[int], List[List[float]], List[Tuple[int, int]], str]] = []
        for ex in examples:
            text = self.serialise_state(ex.state)
            ids, offs, feats = self._encode(text, cfg.use_numeric)
            if len(ids) + 1 > cfg.max_state_tokens:
                raise ValueError(f"state has {len(ids) + 1} tokens; max_state_tokens={cfg.max_state_tokens}")
            states.append(([self.ctrl["state"]] + ids, [[0.0] * 4] + feats, [(0, 0)] + offs, text))

        enc_q: List[List[Optional[Tuple[List[int], List[List[float]]]]]] = []
        enc_o: List[List[List[List[int]]]] = []
        k_max = 1
        for ex in examples:
            qrow, orow = [], []
            for spec in ex.questions:
                spec.validate(cfg)
                text = sanitize_delimiter_text(spec.instructions)
                ids, _, feats = self._encode(text, cfg.use_numeric)
                if len(ids) + 1 > cfg.max_q_tokens:
                    raise ValueError(f"question has {len(ids) + 1} tokens; max_q_tokens={cfg.max_q_tokens}")
                qrow.append(([self.ctrl["type_" + spec.type]] + ids, [[0.0] * 4] + feats))
                labels = ["true", "false"] if spec.type == "noul" else list(spec.options)
                opts = []
                for i, label in enumerate(labels):
                    oid, _, _ = self._encode(sanitize_delimiter_text(str(label)), False)
                    if spec.criteria:
                        cid, _, _ = self._encode(sanitize_delimiter_text(str(spec.criteria[i])), False)
                        oid = oid + [self.ctrl["criteria"]] + cid
                    if not oid:
                        oid = [self.pad_id]
                    if len(oid) > cfg.max_opt_tokens:
                        raise ValueError(f"option {i} has {len(oid)} tokens; max_opt_tokens={cfg.max_opt_tokens}")
                    opts.append(oid)
                orow.append(opts)
                k_max = max(k_max, len(opts))
            enc_q.append(qrow)
            enc_o.append(orow)

        ls = max(len(s[0]) for s in states)
        lq = max([len(q[0]) for row in enc_q for q in row] + [1])
        lo = max([len(o) for row in enc_o for opts in row for o in opts] + [1])

        pad = self.pad_id
        state_ids = torch.full((bsz, ls), pad, dtype=torch.long)
        state_mask = torch.zeros(bsz, ls, dtype=torch.bool)
        state_num = torch.zeros(bsz, ls, 4)
        q_ids = torch.full((bsz, m_max, lq), pad, dtype=torch.long)
        q_mask = torch.zeros(bsz, m_max, lq, dtype=torch.bool)
        q_num = torch.zeros(bsz, m_max, lq, 4)
        q_type = torch.zeros(bsz, m_max, dtype=torch.long)
        q_valid = torch.zeros(bsz, m_max, dtype=torch.bool)
        opt_ids = torch.full((bsz, m_max, k_max, lo), pad, dtype=torch.long)
        opt_mask = torch.zeros(bsz, m_max, k_max, lo, dtype=torch.bool)
        opt_valid = torch.zeros(bsz, m_max, k_max, dtype=torch.bool)
        range_lo = torch.zeros(bsz, m_max)
        range_hi = torch.ones(bsz, m_max)

        label_idx = torch.full((bsz, m_max), IGNORE, dtype=torch.long)
        label_soft = torch.zeros(bsz, m_max, k_max + 1)
        has_soft = torch.zeros(bsz, m_max, dtype=torch.bool)
        label_multi = torch.zeros(bsz, m_max, k_max)
        has_multi = torch.zeros(bsz, m_max, dtype=torch.bool)
        label_range = torch.zeros(bsz, m_max)
        has_range = torch.zeros(bsz, m_max, dtype=torch.bool)
        span_start = torch.full((bsz, m_max), IGNORE, dtype=torch.long)
        span_end = torch.full((bsz, m_max), IGNORE, dtype=torch.long)

        for b, (ids, feats, offs, _text) in enumerate(states):
            n = len(ids)
            state_ids[b, :n] = torch.tensor(ids)
            state_mask[b, :n] = True
            state_num[b, :n] = torch.tensor(feats)
            for m, spec in enumerate(examples[b].questions):
                qids, qfeats = enc_q[b][m]
                n = len(qids)
                q_ids[b, m, :n] = torch.tensor(qids)
                q_mask[b, m, :n] = True
                q_num[b, m, :n] = torch.tensor(qfeats)
                q_type[b, m] = TYPE_INDEX[spec.type]
                q_valid[b, m] = True
                for k, oid in enumerate(enc_o[b][m]):
                    opt_ids[b, m, k, : len(oid)] = torch.tensor(oid)
                    opt_mask[b, m, k, : len(oid)] = True
                    opt_valid[b, m, k] = True
                if spec.type == "range":
                    range_lo[b, m], range_hi[b, m] = float(spec.lo), float(spec.hi)
                self._fill_labels(
                    spec, b, m, offs, label_idx, label_soft, has_soft, label_multi, has_multi,
                    label_range, has_range, span_start, span_end, len(enc_o[b][m]),
                )

        return Rhen2Batch(
            state_ids, state_mask, state_num, q_ids, q_mask, q_num, q_type, q_valid,
            opt_ids, opt_mask, opt_valid, range_lo, range_hi,
            label_idx, label_soft, has_soft, label_multi, has_multi, label_range, has_range,
            span_start, span_end,
            state_texts=[s[3] for s in states], state_offsets=[s[2] for s in states],
            specs=[list(e.questions) for e in examples],
        )

    def _fill_labels(self, spec, b, m, offs, label_idx, label_soft, has_soft, label_multi, has_multi,
                     label_range, has_range, span_start, span_end, k_real) -> None:
        if spec.soft is not None:
            if len(spec.soft) != k_real + 1:
                raise ValueError(f"soft label needs {k_real + 1} entries (options then null), got {len(spec.soft)}")
            total = float(sum(spec.soft))
            if total <= 0:
                raise ValueError("soft label must have positive mass")
            label_soft[b, m, :k_real] = torch.tensor(spec.soft[:k_real]) / total
            label_soft[b, m, -1] = float(spec.soft[-1]) / total     # null column is always last
            has_soft[b, m] = True
        if spec.label is None:
            return
        if spec.type in ("choice", "score"):
            label_idx[b, m] = NULL_INDEX if spec.label == NULL_LABEL else self._option_index(spec, spec.label)
        elif spec.type == "noul":
            if spec.label == NULL_LABEL:
                label_idx[b, m] = NULL_INDEX
            else:
                label_idx[b, m] = 0 if bool(spec.label) else 1        # options are [true, false]
        elif spec.type == "multi":
            for item in spec.label:
                label_multi[b, m, self._option_index(spec, item)] = 1.0
            has_multi[b, m] = True
        elif spec.type == "range":
            label_range[b, m] = float(spec.label)
            has_range[b, m] = True
        elif spec.type == "extract":
            if spec.label == NULL_LABEL:
                span_start[b, m] = span_end[b, m] = 0
            else:
                c0, c1 = spec.label
                first = last = None
                for i, (a, z) in enumerate(offs):
                    if i == 0:
                        continue
                    if first is None and z > c0:
                        first = i
                    if a < c1:
                        last = i
                if first is None or last is None or last < first:
                    raise ValueError(f"extract label {spec.label} does not cover any state token")
                span_start[b, m], span_end[b, m] = first, last


def make_batches(examples: Sequence[Rhen2Example], max_questions: int = 64) -> List[List[Rhen2Example]]:
    """Group examples so that option counts and state lengths are similar inside a batch."""
    def key(ex: Rhen2Example):
        k = max([len(q.options) or 2 for q in ex.questions] + [1])
        return (k_bucket(k), len(str(ex.state)))

    ordered = sorted(examples, key=key)
    batches: List[List[Rhen2Example]] = []
    current: List[Rhen2Example] = []
    count = 0
    bucket = None
    for ex in ordered:
        b = key(ex)[0]
        n = max(1, len(ex.questions))
        if current and (count + n > max_questions or b != bucket):
            batches.append(current)
            current, count = [], 0
        current.append(ex)
        count += n
        bucket = b
    if current:
        batches.append(current)
    return batches
