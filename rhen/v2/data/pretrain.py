"""Stage-1 pretraining builders for RHEN-2 (plan 7.2).

* ``TextCorpus`` / ``structured_text``: passages of prose and synthetic operational text.
* ``build_mlm_example`` / ``collate_mlm`` / ``MLMHead`` / ``mlm_loss``: masked token modelling on the
  state encoder.  The head lives outside ``Rhen2Model`` and ties its output matrix to ``model.embed.table``.
* ``SpanChoiceBuilder``: cut a span out of a passage, ask which text fills the "[GAP]".
* ``SpanLocateBuilder``: copy a span into the question, label is its char span in the serialised state.
* ``PretrainMixture``: deterministic 40/40/20 mix of the three objectives.

Texts are passed through ``sanitize_delimiter_text`` first so the char offsets we compute are the ones the
packer sees ("<|" look-alikes are rewritten by the packer).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhen.rhen_packer import sanitize_delimiter_text
from rhen.v2.layers import RMSNorm
from rhen.v2.packing import NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer

IGNORE_INDEX = -100
GAP_MARKER = "[GAP]"
SPAN_CHOICE_QUESTION = "which text fills the gap?"
K_CHOICES = (2, 4, 8, 16, 32)
NULL_RATE = 0.12


# --------------------------------------------------------------------------- corpus
class TextCorpus:
    """Passages from ``.txt`` (blank-line separated paragraphs) and ``.jsonl`` (``text`` field or bare
    strings) files.  ``paths`` is a directory or a list of files/directories; order is sorted, so
    iteration is deterministic."""

    def __init__(self, paths, min_chars: int = 40):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        files: List[str] = []
        for p in paths:
            p = os.fspath(p)
            if os.path.isdir(p):
                for root, _dirs, names in sorted(os.walk(p)):
                    for n in sorted(names):
                        if n.endswith((".txt", ".jsonl")):
                            files.append(os.path.join(root, n))
            elif os.path.exists(p):
                files.append(p)
            else:
                raise FileNotFoundError(p)
        self.files = files
        self.min_chars = min_chars

    def __iter__(self) -> Iterator[str]:
        for path in self.files:
            with open(path, "r", encoding="utf-8") as fh:
                if path.endswith(".jsonl"):
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        text = obj if isinstance(obj, str) else obj.get("text")
                        if isinstance(text, str) and len(text.strip()) >= self.min_chars:
                            yield text.strip()
                else:
                    for para in fh.read().split("\n\n"):
                        para = para.strip()
                        if len(para) >= self.min_chars:
                            yield para

    def passages(self) -> List[str]:
        return list(self)


_NAMES = ["alice", "bob", "carol", "dmitri", "elena", "farid", "grace", "hiro", "ines", "jamal"]
_SERVICES = ["auth", "billing", "search", "gateway", "ingest", "export", "notifier", "scheduler"]
_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
_STATUS = ["open", "pending", "resolved", "escalated", "closed"]
_REGIONS = ["north", "south", "east", "west", "central"]
_TOPICS = ["refund request", "login failure", "invoice mismatch", "shipping delay", "data export", "api timeout"]


def structured_text(rng: random.Random) -> str:
    """One synthetic JSON / log / table / ticket / email / config text (deterministic given ``rng``)."""
    kind = rng.choice(["json", "log", "table", "ticket", "email", "config"])
    if kind == "json":
        obj = {
            "order_id": rng.randint(1000, 99999), "customer": rng.choice(_NAMES),
            "region": rng.choice(_REGIONS), "total": round(rng.uniform(5, 900), 2),
            "items": [{"sku": f"SKU-{rng.randint(100, 999)}", "qty": rng.randint(1, 9)} for _ in range(rng.randint(1, 5))],
            "flags": {"expedite": rng.random() < 0.3, "gift": rng.random() < 0.2},
            "status": rng.choice(_STATUS),
        }
        return json.dumps(obj, ensure_ascii=False)
    if kind == "log":
        lines = []
        for i in range(rng.randint(6, 14)):
            lines.append(
                f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} "
                f"{rng.choice(_LEVELS)} {rng.choice(_SERVICES)} request {rng.randint(10000, 99999)} took {rng.randint(3, 2500)}ms status={rng.choice([200, 200, 404, 500, 503])}"
            )
        return "\n".join(lines)
    if kind == "table":
        rows = ["id | name | region | amount | status"]
        for i in range(rng.randint(4, 9)):
            rows.append(f"{rng.randint(1, 999)} | {rng.choice(_NAMES)} | {rng.choice(_REGIONS)} | {rng.uniform(1, 5000):.2f} | {rng.choice(_STATUS)}")
        return "\n".join(rows)
    if kind == "ticket":
        return (
            f"Ticket #{rng.randint(1000, 9999)}\nPriority: {rng.choice(['low', 'medium', 'high', 'urgent'])}\n"
            f"Reporter: {rng.choice(_NAMES)}\nTopic: {rng.choice(_TOPICS)}\nStatus: {rng.choice(_STATUS)}\n"
            f"Description: The customer in the {rng.choice(_REGIONS)} region reports a {rng.choice(_TOPICS)} "
            f"affecting {rng.randint(1, 400)} accounts since {rng.randint(1, 23):02d}:00. Last action by {rng.choice(_NAMES)}."
        )
    if kind == "email":
        return (
            f"From: {rng.choice(_NAMES)}@example.org\nTo: support@example.org\nSubject: {rng.choice(_TOPICS)} ({rng.randint(100, 999)})\n\n"
            f"Hello team, order {rng.randint(1000, 99999)} was placed on day {rng.randint(1, 28)} and has not arrived. "
            f"Please advise whether a {rng.choice(['refund', 'replacement', 'credit'])} of {rng.randint(5, 300)} dollars applies. Thanks, {rng.choice(_NAMES)}"
        )
    lines = [f"[{rng.choice(_SERVICES)}]"]
    for _ in range(rng.randint(4, 9)):
        lines.append(f"{rng.choice(['timeout', 'retries', 'port', 'workers', 'limit', 'ttl'])}_{rng.randint(1, 9)} = {rng.randint(1, 9000)}")
    lines.append(f"enabled = {rng.choice(['true', 'false'])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- MLM
def build_mlm_example(
    token_ids: Sequence[int], mask_rate: float = 0.3, rng: Optional[random.Random] = None,
    mask_id: int = 0, *, vocab_size: Optional[int] = None, mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1, keep_first: bool = False,
) -> Tuple[List[int], List[int]]:
    """Mask ``mask_rate`` of the tokens.  Returns ``(inputs, targets)``; ``targets`` is IGNORE_INDEX (-100)
    at unmasked positions and the original id at selected positions.  Selected positions are replaced by
    ``mask_id`` with prob ``mask_token_prob``, by a random id (< vocab_size) with prob ``random_token_prob``
    (only when ``vocab_size`` is given), else kept.  ``keep_first`` protects position 0."""
    rng = rng or random.Random(0)
    inputs, targets = list(token_ids), [IGNORE_INDEX] * len(token_ids)
    start = 1 if keep_first else 0
    cand = list(range(start, len(inputs)))
    if not cand:
        return inputs, targets
    n_mask = min(len(cand), max(1, round(mask_rate * len(cand))))
    for pos in rng.sample(cand, n_mask):
        targets[pos] = inputs[pos]
        r = rng.random()
        if r < mask_token_prob or (vocab_size is None and r < mask_token_prob + random_token_prob):
            inputs[pos] = mask_id
        elif r < mask_token_prob + random_token_prob and vocab_size is not None:
            inputs[pos] = rng.randrange(vocab_size)
    return inputs, targets


def mask_token_id(packer: Rhen2Packer) -> int:
    """Dedicated MLM mask id: first id above the packer's control ids."""
    mid = packer.base + len(packer.ctrl)
    assert mid < packer.cfg.vocab_size, f"mask id {mid} does not fit vocab_size={packer.cfg.vocab_size}"
    assert mid not in packer.ctrl.values()
    return mid


def collate_mlm(
    token_lists: Sequence[Sequence[int]], packer: Rhen2Packer, mask_rate: float = 0.3,
    rng: Optional[random.Random] = None, max_tokens: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Mask raw token-id lists and pack them as state tensors: ``[CTRL_STATE] + tokens`` with position 0 never
    masked, padding after.  Returns ids, targets, mask (True = real token) and a zero numeric tensor."""
    rng = rng or random.Random(0)
    mid = mask_token_id(packer)
    limit = (max_tokens or packer.cfg.max_state_tokens - 1)
    rows = []
    for toks in token_lists:
        toks = list(toks)[:limit]
        inp, tgt = build_mlm_example(toks, mask_rate, rng, mid, vocab_size=packer.base)
        rows.append(([packer.ctrl["state"]] + inp, [IGNORE_INDEX] + tgt))
    ln = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), ln), packer.pad_id, dtype=torch.long)
    targets = torch.full((len(rows), ln), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(len(rows), ln, dtype=torch.bool)
    for i, (a, t) in enumerate(rows):
        ids[i, : len(a)] = torch.tensor(a)
        targets[i, : len(t)] = torch.tensor(t)
        mask[i, : len(a)] = True
    return {"state_ids": ids, "targets": targets, "state_mask": mask, "state_num": torch.zeros(len(rows), ln, 4)}


class MLMHead(nn.Module):
    """``Linear(d, d_embed)`` -> GELU -> RMSNorm, logits against the tied embedding table plus a bias.
    The tied matrix belongs to the model (passed in at call time), so only the transform and the bias are
    parameters of the head."""

    def __init__(self, d_model: int, d_embed: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_embed)
        self.norm = RMSNorm(d_embed)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        h = self.norm(F.gelu(self.dense(hidden)))
        return h @ table.t() + self.bias

    def count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def for_model(cls, model) -> "MLMHead":
        cfg = model.cfg
        return cls(cfg.d_model, model.embed.table.embedding_dim, model.embed.table.num_embeddings)


def mlm_loss(
    model, head: MLMHead, batch_state_ids: torch.Tensor, targets: torch.Tensor,
    state_mask: Optional[torch.Tensor] = None, state_num: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy on masked positions only.  Position 0 (state control token) and padding are always
    excluded even if ``targets`` has an entry there."""
    if state_mask is None:
        state_mask = torch.ones_like(batch_state_ids, dtype=torch.bool)
    if state_num is None:
        state_num = torch.zeros(*batch_state_ids.shape, 4, device=batch_state_ids.device)
    S = model.encode_state(batch_state_ids, state_mask, state_num)
    sel = (targets != IGNORE_INDEX) & state_mask
    sel[:, 0] = False
    if not bool(sel.any()):
        return S.sum() * 0.0
    logits = head(S[sel], model.embed.table.weight)
    return F.cross_entropy(logits.float(), targets[sel])


# --------------------------------------------------------------------------- span helpers
def _tokenise(packer: Rhen2Packer, text: str) -> List[Tuple[int, int]]:
    enc = packer.tok.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(a), int(b)) for a, b in enc["offset_mapping"]]


def _clean(text: str) -> str:
    return sanitize_delimiter_text(text)


def _crop(packer: Rhen2Packer, text: str, max_tokens: int) -> Tuple[str, List[Tuple[int, int]]]:
    offs = _tokenise(packer, text)
    if len(offs) > max_tokens:
        text = text[: offs[max_tokens - 1][1]]
        offs = offs[:max_tokens]
    return text, offs


def _cut(text: str, offs: Sequence[Tuple[int, int]], i0: int, i1: int) -> Tuple[int, int]:
    """Char range covering tokens [i0, i1), trimmed of whitespace."""
    a, b = offs[i0][0], offs[i1 - 1][1]
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


@dataclass
class SpanChoiceBuilder:
    packer: Rhen2Packer
    max_passage_tokens: int = 200
    min_span: int = 3
    max_span: int = 30
    max_option_tokens: int = 22
    ks: Tuple[int, ...] = K_CHOICES
    null_rate: float = NULL_RATE
    max_options: int = 255

    def _span_len(self, rng: random.Random, n_tokens: int) -> int:
        hi = min(self.max_span, self.max_option_tokens, max(1, n_tokens // 3))
        lo = min(self.min_span, hi)
        return rng.randint(lo, hi)

    def _random_span(self, rng: random.Random, text: str, offs, length: int, forbid: Optional[Tuple[int, int]] = None):
        n = len(offs)
        length = min(length, n)
        for _ in range(12):
            i0 = rng.randint(0, n - length)
            a, b = _cut(text, offs, i0, i0 + length)
            if forbid and a < forbid[1] and b > forbid[0]:
                continue
            if b > a:
                return text[a:b]
        return None

    def build(self, passage: str, rng: random.Random, others: Sequence[str] = ()) -> Rhen2Example:
        return self.build_with_meta(passage, rng, others)[0]

    def build_with_meta(self, passage: str, rng: random.Random, others: Sequence[str] = ()):
        text, offs = _crop(self.packer, _clean(passage), self.max_passage_tokens)
        if len(offs) < self.min_span + 2:
            raise ValueError("passage too short for span choice")
        length = self._span_len(rng, len(offs))
        i0 = rng.randint(0, len(offs) - length)
        a, b = _cut(text, offs, i0, i0 + length)
        gold = text[a:b]
        state = text[:a] + GAP_MARKER + text[b:]
        is_null = rng.random() < self.null_rate
        k = rng.choice(self.ks)
        n_distract = k if is_null else k - 1
        want_same = n_distract // 2
        chosen: Dict[str, str] = {}
        seen = {gold}
        others_tok = [_crop(self.packer, _clean(o), self.max_passage_tokens) for o in others]
        others_tok = [(t, o) for t, o in others_tok if len(o) >= self.min_span]

        def add(source: str, span: Optional[str]) -> bool:
            if span and span not in seen:
                seen.add(span)
                chosen[span] = source
                return True
            return False

        tries = 0
        while sum(1 for s in chosen.values() if s == "same") < want_same and tries < 20 * max(1, want_same):
            tries += 1
            add("same", self._random_span(rng, text, offs, self._span_len(rng, len(offs)), forbid=(a, b)))
        tries = 0
        while len(chosen) < n_distract and others_tok and tries < 30 * n_distract:
            tries += 1
            t, o = rng.choice(others_tok)
            add("other", self._random_span(rng, t, o, self._span_len(rng, len(o))))
        tries = 0
        while len(chosen) < n_distract and tries < 30 * n_distract:   # fall back to the same document
            tries += 1
            add("same", self._random_span(rng, text, offs, self._span_len(rng, len(offs)), forbid=(a, b)))
        n_have = len(chosen) + (0 if is_null else 1)
        k_final = max([x for x in self.ks if x <= n_have] or [min(self.ks)])
        items = list(chosen.items())
        if len(items) > (k_final if is_null else k_final - 1):
            same = [it for it in items if it[1] == "same"]
            other = [it for it in items if it[1] == "other"]
            need = k_final if is_null else k_final - 1
            take_same = min(len(same), need // 2)
            items = same[:take_same] + other[: need - take_same]
            if len(items) < need:
                items += [it for it in same[take_same:] + other[need - take_same:]][: need - len(items)]
        options = [(s, src) for s, src in items]
        if not is_null:
            options.append((gold, "true"))
        rng.shuffle(options)
        texts = [o[0] for o in options]
        label = NULL_LABEL if is_null else texts.index(gold)
        q = QuestionSpec("choice", SPAN_CHOICE_QUESTION, texts, label=label)
        meta = {"k": len(texts), "null": is_null, "sources": [o[1] for o in options], "gold": gold}
        return Rhen2Example(state, [q]), meta


@dataclass
class SpanLocateBuilder:
    packer: Rhen2Packer
    max_passage_tokens: int = 200
    min_span: int = 3
    max_span: int = 12
    perturb_rate: float = 0.15
    dict_rate: float = 0.0
    max_question_words: int = 14

    def build(self, passage: str, rng: random.Random, others: Sequence[str] = ()) -> Rhen2Example:
        return self.build_with_meta(passage, rng)[0]

    def build_with_meta(self, passage: str, rng: random.Random, others: Sequence[str] = ()):
        text, offs = _crop(self.packer, _clean(passage), self.max_passage_tokens)
        if len(offs) < self.min_span + 1:
            raise ValueError("passage too short for span locate")
        hi = min(self.max_span, max(1, len(offs) // 2))
        length = rng.randint(min(self.min_span, hi), hi)
        i0 = rng.randint(0, len(offs) - length)
        a, b = _cut(text, offs, i0, i0 + length)
        span = text[a:b]
        as_dict = rng.random() < self.dict_rate
        if as_dict:
            state: Any = {"id": rng.randint(1, 9999), "text": text}
            full = self.packer.serialise_state(state)
            enc_all = json.dumps(text, ensure_ascii=False)[1:-1]
            base = full.find(enc_all)
            s0 = base + len(json.dumps(text[:a], ensure_ascii=False)[1:-1])
            s1 = s0 + len(json.dumps(span, ensure_ascii=False)[1:-1])
        else:
            state = text
            full = self.packer.serialise_state(state)
            s0, s1 = a, b
        perturbed = False
        shown = span
        if rng.random() < self.perturb_rate:
            words = span.split()
            if len(words) >= 2:
                keep = [w for w in words if rng.random() > 0.25]
                if not keep:
                    keep = [rng.choice(words)]
                if keep != words:
                    shown, perturbed = " ".join(keep), True
        shown = " ".join(shown.split()[: self.max_question_words]) if len(shown.split()) > self.max_question_words else shown
        q = QuestionSpec("extract", f"locate: {shown}", label=(s0, s1))
        return Rhen2Example(state, [q]), {"span": span, "perturbed": perturbed, "serialised": full}


# --------------------------------------------------------------------------- mixture
@dataclass
class MixtureBatch:
    kinds: List[str]                  # per slot: "mlm" | "span_choice" | "span_locate"
    mlm: List[List[int]]              # raw token ids for mlm_loss / collate_mlm
    examples: List[Rhen2Example]      # decision examples (span choice / locate), in slot order
    example_kinds: List[str] = field(default_factory=list)


class PretrainMixture:
    """Deterministic sampler: batch ``i`` depends only on ``(seed, i)``.  Passages are prose from
    ``passages`` and synthetic structured text (``structured_frac`` of the draws)."""

    KINDS = ("mlm", "span_choice", "span_locate")

    def __init__(
        self, packer: Rhen2Packer, passages: Sequence[str] = (), weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        seed: int = 0, structured_frac: float = 0.5, max_passage_tokens: int = 200,
        locate_dict_rate: float = 0.2, max_option_tokens: int = 22,
    ):
        if len(weights) != 3 or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("weights must be three non-negative numbers")
        self.packer = packer
        self.passages = list(passages)
        total = float(sum(weights))
        self.weights = tuple(w / total for w in weights)
        self.seed = seed
        self.structured_frac = 1.0 if not self.passages else structured_frac
        self.max_passage_tokens = max_passage_tokens
        self.choice = SpanChoiceBuilder(packer, max_passage_tokens=max_passage_tokens, max_option_tokens=max_option_tokens)
        self.locate = SpanLocateBuilder(packer, max_passage_tokens=max_passage_tokens, dict_rate=locate_dict_rate)

    def _passage(self, rng: random.Random) -> str:
        if rng.random() < self.structured_frac:
            return structured_text(rng)
        return rng.choice(self.passages)

    def sample_batch(self, batch_size: int, step: int = 0) -> MixtureBatch:
        rng = random.Random(f"rhen2-pretrain:{self.seed}:{step}")
        kinds = rng.choices(self.KINDS, weights=self.weights, k=batch_size)
        pool = [self._passage(rng) for _ in range(max(8, batch_size))]
        out = MixtureBatch(kinds=kinds, mlm=[], examples=[])
        for kind in kinds:
            passage = self._passage(rng)
            if kind == "mlm":
                toks = self.packer.tok.encode(_clean(passage), add_special_tokens=False)
                out.mlm.append([int(t) for t in toks[: self.max_passage_tokens]])
                continue
            builder = self.choice if kind == "span_choice" else self.locate
            for _ in range(5):
                try:
                    out.examples.append(builder.build(passage, rng, pool))
                    out.example_kinds.append(kind)
                    break
                except ValueError:
                    passage = structured_text(rng)
            else:  # pragma: no cover - structured text is always long enough
                raise RuntimeError("could not build an example")
        return out
