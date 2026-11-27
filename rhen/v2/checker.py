"""The Checker: RHEN-2's built-in reasoner (plan section 6).

It measures instead of trusting a self-report. Stages:

* C1 ``CorrectnessHead``: probability that the decision is right, from features of the model's
  own output (probabilities, depth behaviour, evidence concentration).
* C2 evidence spans from the model's state attribution, as character offsets.
* C3 interventions that re-run only the workspace on the cached state memory ``S``:
  a evidence deletion (UNGROUNDED), b empty memory (PRIOR_DRIVEN), c depth consistency
  (UNSTABLE), d null bias +1.0 (MARGINAL).
* C4 ``RuleChecker``: deterministic checks that can override the model (RULE_CONFLICT).
* ``decide_verdict``: the pure verdict rules of plan 6.5.

Choices where the plan is ambiguous (all documented here, none hidden):

* Without a calibration object the Checker is neutral (T=1, no conformal set,
  ``calibrated=False``). It never returns CONFIRM above tier ``low`` in that state. Because
  every threshold must come from calibration, an uncalibrated ``low``-tier decision is CONFIRM
  when no flag fired and the answer is not null; no ``p_correct`` threshold is applied.
* A final guard in ``decide_verdict`` turns any CONFIRM into ESCALATE when the calibration is
  absent or ``calibrated`` is False and the tier is not ``low``. This also covers rule 1
  (a deterministic override): fail-closed (plan 9.2) wins over rule 1 above ``low``.
* ``delta_ev`` missing means 0.0: C3a/C3b then flag only when the intervention did not move
  the probability at all in the expected direction. C3b flags when the answer is unchanged
  and the null probability rose by less than ``delta_ev``.
* The model exposes only the final loop, so depth features come from ``depth_trace``, which
  re-runs the workspace with 1..R loops (R passes). Loop-wise "answer" is the type-specific
  decoded key (option index / chosen set / span), not the argmax of ``probs``.
* The chosen option vector for C1 is the masked mean of the chosen option's token embeddings
  (the workspace's pooled option vectors are not exposed). Evidence concentration threshold
  is ``4 / (number of real state tokens)``, not ``4 / padded Ls``.
* Without a head ``p_correct`` falls back to a raw confidence: top-1 probability (choice, noul,
  score), min_k max(p_k, 1-p_k) (multi), span probability (extract), 0.0 (range). Always
  passed through ``calibration.isotonic``.
* C3 for medium effort runs when ``p_correct < tau_review``; with no ``tau_review`` it runs.
* Range answers are the median; the (q0.1, q0.9) interval is in ``extras["interval"]``. Score
  expected level, noul P(true) etc. are in ``extras``. ``CheckedDecision.extras`` is an addition
  to the plan's dataclass (defaulted).
* Extract answers that are not an exact substring of the state raise the flag
  ``RULE_VIOLATION`` (answer None, ESCALATE). Critical-tier exact C3a (mask input tokens and
  re-run the state encoder) is NOT implemented.
* The REVISE re-run reruns the whole batch at the next effort level once and keeps only the
  revised rows.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhen.v2.config import QUANTILE_LEVELS, QUESTION_TYPES, TYPE_INDEX
from rhen.v2.packing import Rhen2Batch, k_bucket
from rhen.v2.readouts import decode_span
from rhen.v2.workspace import Interventions, LoopPlan

NULL = "__null__"
TIERS = ("low", "medium", "high", "critical")
EFFORTS = ("low", "medium", "high")
FLAGS = ("UNGROUNDED", "PRIOR_DRIVEN", "UNSTABLE", "MARGINAL", "RULE_CONFLICT", "OOD", "RULE_VIOLATION")
N_SCALARS = 9


# ---------------------------------------------------------------------------- calibration


class CalibrationLike(Protocol):
    calibrated: bool

    def temperature(self, qtype: str, k: int) -> float: ...
    def conformal_q(self, qtype: str, k: int) -> Optional[float]: ...
    def tau_act(self, tier: str) -> Optional[float]: ...
    @property
    def tau_review(self) -> Optional[float]: ...
    @property
    def delta_ev(self) -> Optional[float]: ...
    @property
    def halt_threshold(self) -> Optional[float]: ...
    def isotonic(self, qtype: str, p: float) -> float: ...
    def ood_flags(self, qtype: str, k: int, energy: float, p_null: float, q_vec: Any) -> List[str]: ...


class NeutralCalibration:
    """Used when the caller supplies no calibration: identity maps, no thresholds, uncalibrated."""

    calibrated = False
    tau_review = None
    delta_ev = None
    halt_threshold = None

    def temperature(self, qtype: str, k: int) -> float:
        return 1.0

    def conformal_q(self, qtype: str, k: int) -> Optional[float]:
        return None

    def tau_act(self, tier: str) -> Optional[float]:
        return None

    def isotonic(self, qtype: str, p: float) -> float:
        return float(p)

    def ood_flags(self, qtype: str, k: int, energy: float, p_null: float, q_vec: Any) -> List[str]:
        return []


# ---------------------------------------------------------------------------- verdict


def decide_verdict_ex(
    flags: Sequence[str],
    p_correct: float,
    top_is_null: bool,
    conformal_empty: bool,
    tier: str,
    calibration: Optional[CalibrationLike],
    can_revise: bool,
) -> Tuple[str, int]:
    """Plan 6.5 verdict rules in order. Returns (verdict, rule number that decided it).

    Verdicts: CONFIRM, ESCALATE, REVISE. Rule 0 is the fail-closed guard (no calibration above
    the ``low`` tier), rule 6 is the default ESCALATE.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    cal = calibration if calibration is not None else NeutralCalibration()
    calibrated = bool(getattr(cal, "calibrated", False))
    flagset = set(flags)

    def guard(verdict: str, rule: int) -> Tuple[str, int]:
        if verdict == "CONFIRM" and not calibrated and tier != "low":
            return "ESCALATE", 0
        return verdict, rule

    if "RULE_CONFLICT" in flagset:                                   # rule 1
        return guard("CONFIRM", 1)
    if top_is_null or conformal_empty or "RULE_VIOLATION" in flagset:  # rule 2
        return "ESCALATE", 2
    if flagset & {"UNGROUNDED", "PRIOR_DRIVEN"}:                     # rule 3
        return "ESCALATE", 3
    if "OOD" in flagset and tier != "low":                           # rule 3 (plan 8.3)
        return "ESCALATE", 3
    if flagset & {"UNSTABLE", "MARGINAL"}:                           # rule 4
        return ("REVISE", 4) if can_revise else ("ESCALATE", 4)
    if not calibrated:                                               # rule 5 needs a threshold
        return guard("CONFIRM", 5) if tier == "low" else ("ESCALATE", 0)
    tau = cal.tau_act(tier)
    if tau is not None and p_correct >= tau:                         # rule 5
        return guard("CONFIRM", 5)
    return "ESCALATE", 6                                             # rule 6


def decide_verdict(flags, p_correct, top_is_null, conformal_empty, tier, calibration, can_revise) -> str:
    return decide_verdict_ex(flags, p_correct, top_is_null, conformal_empty, tier, calibration, can_revise)[0]


# ---------------------------------------------------------------------------- result object


@dataclass
class CheckedDecision:
    answer: Any
    probabilities: Dict[str, float]
    confidence: float
    conformal_set: List[str]
    verdict: str
    flags: List[str]
    evidence_spans: List[Tuple[int, int, float]]
    loops_run: int
    effort: str
    calibrated: bool
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------- decoding helpers


def _types(batch: Rhen2Batch) -> List[str]:
    return [QUESTION_TYPES[int(t)] for t in batch.q_type.reshape(-1)]


def decode_keys(out, batch: Rhen2Batch, max_span: int) -> List[Any]:
    """Per-question comparable decision key: option index (-1 null), chosen-set tuple, span, or None."""
    n = out.probs.shape[0]
    kmax = out.probs.shape[1] - 1
    si, sj, _ = decode_span(out.span_start, out.span_end, max_span)
    am = out.probs.argmax(-1)
    valid = batch.opt_valid.reshape(n, -1)
    keys: List[Any] = []
    for i, t in enumerate(_types(batch)):
        if t in ("choice", "noul", "score"):
            a = int(am[i])
            keys.append(-1 if a == kmax else a)
        elif t == "multi":
            keys.append(tuple(int(j) for j in torch.nonzero((out.multi_prob[i] > 0.5) & valid[i]).flatten()))
        elif t == "extract":
            keys.append((int(si[i]), int(sj[i])))
        else:
            keys.append(None)
    return keys


def _span_prob(out, n: int, key: Tuple[int, int]) -> float:
    return float(out.span_start[n].softmax(-1)[key[0]] * out.span_end[n].softmax(-1)[key[1]])


def chosen_prob(out, n: int, qtype: str, key: Any) -> Optional[float]:
    """Probability the run ``out`` gives to the decision ``key`` (None when not applicable)."""
    if qtype in ("choice", "noul", "score"):
        return float(out.probs[n, out.probs.shape[1] - 1 if key == -1 else key])
    if qtype == "multi":
        return float(out.multi_prob[n, list(key)].mean()) if key else None
    if qtype == "extract":
        return _span_prob(out, n, key)
    return None


def null_prob(out, n: int, qtype: str) -> Optional[float]:
    if qtype in ("choice", "noul", "score", "multi"):
        return float(out.probs[n, -1])
    if qtype == "extract":
        return _span_prob(out, n, (0, 0))
    return None


@dataclass
class DepthTrace:
    keys: List[List[Any]]          # [R][N] decoded key after r loops
    q_vec: torch.Tensor            # [R, N, d]
    probs: torch.Tensor            # [R, N, K+1]
    loops: List[int]


@torch.no_grad()
def depth_trace(model, S: torch.Tensor, batch: Rhen2Batch, max_loops: int) -> DepthTrace:
    """Run the workspace with 1..max_loops loops (no halting) and keep every loop's decision."""
    keys, qv, pr = [], [], []
    for r in range(1, max(1, int(max_loops)) + 1):
        out = model.deliberate(S, batch, LoopPlan(n_loops=r, min_loops=r))
        keys.append(decode_keys(out, batch, model.cfg.max_span))
        qv.append(out.q_vec)
        pr.append(out.probs)
    return DepthTrace(keys, torch.stack(qv), torch.stack(pr), list(range(1, len(keys) + 1)))


# ---------------------------------------------------------------------------- C1 head


@torch.no_grad()
def correctness_features(model, batch: Rhen2Batch, out, trace: DepthTrace, keys: Optional[List[Any]] = None) -> torch.Tensor:
    """[N, 2*d + N_SCALARS + n_types] features for ``CorrectnessHead`` (plan 6.1)."""
    n, kk = out.probs.shape
    d = out.q_vec.shape[-1]
    keys = keys if keys is not None else decode_keys(out, batch, model.cfg.max_span)
    types = _types(batch)
    probs = out.probs.clamp_min(1e-9)
    top2 = probs.topk(min(2, kk), dim=-1).values
    top1 = top2[:, 0]
    margin = top1 - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top1))
    p_null = out.probs[:, -1]
    n_valid = batch.opt_valid.reshape(n, -1).sum(-1).clamp_min(1).float()
    ent = -(out.probs * probs.log()).sum(-1) / (n_valid + 1).log().clamp_min(1e-6)

    # chosen option vector: masked mean of option token embeddings (0 for null / non-option answers)
    opt_ids = batch.opt_ids.reshape(n, batch.opt_ids.shape[2], -1)
    opt_mask = batch.opt_mask.reshape(n, batch.opt_ids.shape[2], -1)
    u = torch.zeros(n, d)
    for i, key in enumerate(keys):
        if isinstance(key, int) and key >= 0:
            idx = [key]
        elif isinstance(key, tuple) and types[i] == "multi" and key:
            idx = list(key)
        else:
            continue
        emb = model.embed(opt_ids[i, idx])
        msk = opt_mask[i, idx].unsqueeze(-1).to(emb.dtype)
        u[i] = ((emb * msk).sum((0, 1)) / msk.sum().clamp_min(1.0)).to(u.dtype) if emb.shape[-1] == d else 0.0

    final_key = keys
    agree = torch.ones(n)
    for i in range(n):
        agree[i] = sum(1.0 for step in trace.keys if step[i] == final_key[i]) / len(trace.keys)
    qdist = (trace.q_vec[-1] - trace.q_vec[-2]).norm(dim=-1) if trace.q_vec.shape[0] > 1 else torch.zeros(n)

    ev = out.evidence if out.evidence is not None else torch.full((n, 1), 1.0)
    if out.evidence is not None:
        real = batch.state_mask.sum(-1).float()
        real = real.repeat_interleave(batch.max_questions).clamp_min(2.0)
        ev_ent = -(ev * ev.clamp_min(1e-9).log()).sum(-1) / real.log()
    else:
        ev_ent = torch.ones(n)
    log_k = n_valid.log()
    log_ls = torch.full((n,), math.log(float(batch.state_mask.shape[1])))
    onehot = F.one_hot(batch.q_type.reshape(n), len(TYPE_INDEX)).float()
    scalars = torch.stack([top1, margin, p_null, ent, agree, qdist, ev_ent, log_k, log_ls], dim=-1)
    return torch.cat([out.q_vec.float(), u, scalars, onehot], dim=-1)


class CorrectnessHead(nn.Module):
    """C1: MLP over [proj(q_vec), proj(u_chosen), scalars, type one-hot] -> sigmoid p_correct."""

    def __init__(self, d_model: int, proj: int = 128, hidden: Tuple[int, int] = (256, 64)):
        super().__init__()
        self.d_model = d_model
        self.proj_q = nn.Linear(d_model, proj)
        self.proj_u = nn.Linear(d_model, proj)
        self.mlp = nn.Sequential(
            nn.Linear(2 * proj + N_SCALARS + len(TYPE_INDEX), hidden[0]), nn.GELU(),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(), nn.Linear(hidden[1], 1),
        )

    def logit(self, feats: torch.Tensor) -> torch.Tensor:
        d = self.d_model
        q, u, rest = feats[:, :d], feats[:, d:2 * d], feats[:, 2 * d:]
        return self.mlp(torch.cat([self.proj_q(q), self.proj_u(u), rest], dim=-1)).squeeze(-1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(feats))


def fit_correctness_head(head: CorrectnessHead, features: torch.Tensor, labels: torch.Tensor,
                         epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4) -> List[float]:
    """Full-batch BCE fit. ``features`` must come from cross-fitted predictions (plan 7.6)."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    y = labels.float().reshape(-1)
    losses: List[float] = []
    head.train()
    for _ in range(int(epochs)):
        loss = F.binary_cross_entropy_with_logits(head.logit(features.detach()), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    head.eval()
    return losses


# ---------------------------------------------------------------------------- C4 rules


_UP = re.compile(r"\b(larger|greater|higher|more|bigger|largest|greatest|highest|biggest|most)\b", re.I)
_DOWN = re.compile(r"\b(smaller|lower|less|fewer|smallest|lowest|least)\b", re.I)


@dataclass
class RuleResult:
    flag: Optional[str] = None       # RULE_CONFLICT | RULE_VIOLATION | None
    answer: Any = None               # deterministic answer (option label) when flag == RULE_CONFLICT
    note: str = ""


class RuleChecker:
    """Deterministic checks (plan 6.4). Returns an empty result when no rule applies."""

    @staticmethod
    def numeric_comparison(instructions: str, options: Sequence[str], state_text: str) -> Optional[str]:
        up, down = bool(_UP.search(instructions)), bool(_DOWN.search(instructions))
        if up == down or len(options) < 2:
            return None
        try:
            state = json.loads(state_text)
        except (ValueError, TypeError):
            return None
        if not isinstance(state, dict):
            return None
        vals = []
        for label in options:
            v = state.get(label)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return None
            vals.append(float(v))
        best = max(vals) if up else min(vals)
        if vals.count(best) != 1:
            return None
        return options[vals.index(best)]

    @classmethod
    def check(cls, spec, state_text: str, answer: Any) -> RuleResult:
        if spec.type == "choice":
            det = cls.numeric_comparison(spec.instructions, spec.options, state_text)
            if det is not None and det != answer:
                return RuleResult("RULE_CONFLICT", det, f"numeric comparison gives {det!r}")
        elif spec.type == "extract":
            if answer is not None and answer not in state_text:
                return RuleResult("RULE_VIOLATION", None, "extract answer is not a substring of the state")
        return RuleResult()


# ---------------------------------------------------------------------------- evidence


def evidence_spans(evidence: torch.Tensor, state_mask: torch.Tensor, offsets: Sequence[Tuple[int, int]],
                   max_spans: int = 5) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int]]]:
    """Merge adjacent tokens above 4/n_real into spans. Returns (char spans, token ranges).

    If no token clears the threshold the single heaviest real token is used.
    """
    real = [i for i in range(1, int(state_mask.shape[0])) if bool(state_mask[i])]
    if not real:
        return [], []
    thr = 4.0 / len(real)
    ev = evidence.tolist()
    hot = [i for i in real if ev[i] > thr]
    if not hot:
        hot = [max(real, key=lambda i: ev[i])]
    runs: List[List[int]] = []
    for i in hot:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    runs.sort(key=lambda r: -sum(ev[i] for i in r))
    runs = runs[:max_spans]
    chars = [(int(offsets[r[0]][0]), int(offsets[r[-1]][1]), float(sum(ev[i] for i in r))) for r in runs]
    return chars, [(r[0], r[-1]) for r in runs]


# ---------------------------------------------------------------------------- the checker


@dataclass
class _Draft:
    decision: CheckedDecision
    verdict: str
    rule: int


class Checker:
    def __init__(self, model, calibration: Optional[CalibrationLike] = None, head: Optional[CorrectnessHead] = None):
        self.model = model
        self.cal: CalibrationLike = calibration if calibration is not None else NeutralCalibration()
        self.head = head

    # -- public ---------------------------------------------------------------------------
    def check(self, batch: Rhen2Batch, effort: str = "medium", risk_tier: str = "low") -> List[CheckedDecision]:
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}")
        if risk_tier not in TIERS:
            raise ValueError(f"risk_tier must be one of {TIERS}")
        if risk_tier in ("high", "critical"):
            effort = "high"
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                S = self.model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
                drafts = self._pass(batch, S, effort, risk_tier, can_revise=effort != "high")
                redo = [i for i, d in enumerate(drafts) if d is not None and d.verdict == "REVISE"]
                if redo:
                    nxt = EFFORTS[EFFORTS.index(effort) + 1]
                    second = self._pass(batch, S, nxt, risk_tier, can_revise=False)
                    for i in redo:
                        second[i].decision.extras["revised_from"] = effort
                        drafts[i] = second[i]
        finally:
            self.model.train(was_training)
        return [d.decision for d in drafts if d is not None]

    # -- one full pass at an effort level ------------------------------------------------
    def _pass(self, batch: Rhen2Batch, S: torch.Tensor, effort: str, tier: str, can_revise: bool) -> List[Optional[_Draft]]:
        model, cal, cfg = self.model, self.cal, self.model.cfg
        halt = cal.halt_threshold if cal.halt_threshold is not None else 0.9
        plan = LoopPlan.effort(cfg, effort, halt_threshold=float(halt))
        out = model.deliberate(S, batch, plan)
        trace = depth_trace(model, S, batch, plan.n_loops)
        keys = decode_keys(out, batch, cfg.max_span)
        types = _types(batch)
        n, m = out.probs.shape[0], batch.max_questions
        valid = batch.q_valid.reshape(n).tolist()

        feats = correctness_features(model, batch, out, trace, keys)
        if self.head is not None:
            p_raw = self.head.eval()(feats).tolist()
        else:
            p_raw = [self._fallback_confidence(out, i, types[i], keys[i]) for i in range(n)]

        # evidence spans
        spans: List[Any] = [None] * n
        mask = torch.zeros(n, batch.state_mask.shape[1], dtype=torch.bool)
        for i in range(n):
            if valid[i] and out.evidence is not None:
                b = i // m
                chars, toks = evidence_spans(out.evidence[i], batch.state_mask[b], batch.state_offsets[b])
                spans[i] = chars
                for a, z in toks:
                    mask[i, a:z + 1] = True

        delta = float(cal.delta_ev) if cal.delta_ev is not None else 0.0
        run_c3 = effort == "high" or (
            effort == "medium" and (cal.tau_review is None or any(
                valid[i] and cal.isotonic(types[i], p_raw[i]) < cal.tau_review for i in range(n))))
        c3: Dict[str, Any] = {}
        if run_c3:
            c3["masked"] = model.deliberate(S, batch, plan, Interventions(memory_mask=mask))
            c3["empty"] = model.deliberate(S, batch, plan, Interventions(empty_memory=True))
            c3["null"] = model.deliberate(S, batch, plan, Interventions(null_delta=1.0))
            c3["maxloop"] = model.deliberate(S, batch, LoopPlan(n_loops=plan.n_loops, min_loops=plan.n_loops))

        drafts: List[Optional[_Draft]] = []
        for i in range(n):
            if not valid[i]:
                drafts.append(None)
                continue
            b, mi = divmod(i, m)
            spec = batch.specs[b][mi]
            flags: List[str] = []
            qtype, key = types[i], keys[i]
            if c3:
                flags += self._c3_flags(c3, out, i, qtype, key, delta, trace)
            answer, probs, conf_set, extras, top_null, k = self._decode(batch, out, i, spec, qtype, key, b)
            # C4
            rule = RuleChecker.check(spec, batch.state_texts[b], answer)
            if rule.flag == "RULE_CONFLICT":
                flags.append("RULE_CONFLICT")
                extras["model_answer"] = answer
                answer, extras["rule_note"] = rule.answer, rule.note
            elif rule.flag == "RULE_VIOLATION":
                flags.append("RULE_VIOLATION")
                extras["rule_note"] = rule.note
                answer = None
            # calibration: conformal and OOD
            conformal_empty = conf_set is not None and len(conf_set) == 0
            lg = out.final.logits[i]
            temp = float(cal.temperature(qtype, k))
            energy = float(-temp * torch.logsumexp(lg / temp, 0) + temp * math.log(k + 1))
            flags += [f for f in cal.ood_flags(qtype, k, energy, float(out.probs[i, -1]), out.q_vec[i]) if f not in flags]
            confidence = float(cal.isotonic(qtype, float(p_raw[i])))
            verdict, rule_no = decide_verdict_ex(flags, confidence, top_null, conformal_empty, tier, cal, can_revise)
            extras.update(rule=rule_no, p_correct_raw=float(p_raw[i]), qtype=qtype)
            final_answer = None if (top_null and "RULE_CONFLICT" not in flags) else answer
            dec = CheckedDecision(
                answer=final_answer, probabilities=probs, confidence=confidence,
                conformal_set=conf_set or [], verdict="ESCALATE" if verdict == "REVISE" else verdict,
                flags=flags, evidence_spans=spans[i] or [], loops_run=int(out.loops_run[i]),
                effort=effort, calibrated=bool(cal.calibrated), extras=extras,
            )
            drafts.append(_Draft(dec, verdict, rule_no))
        return drafts

    # -- C3 ----------------------------------------------------------------------------
    def _c3_flags(self, c3, out, i: int, qtype: str, key: Any, delta: float, trace: DepthTrace) -> List[str]:
        flags: List[str] = []
        if qtype == "range":
            return flags
        p0 = chosen_prob(out, i, qtype, key)
        if p0 is not None:
            p_masked = chosen_prob(c3["masked"], i, qtype, key)
            if p_masked is not None and (p0 - p_masked) < delta:
                flags.append("UNGROUNDED")
            empty = c3["empty"]
            pn0, pn1 = null_prob(out, i, qtype), null_prob(empty, i, qtype)
            same = self._key_of(empty, i, qtype) == key
            if key not in (-1, (0, 0)) and same and pn0 is not None and (pn1 - pn0) < delta:
                flags.append("PRIOR_DRIVEN")
        # C3c: halting-loop decision vs decision at the maximum loop
        if self._key_of(c3["maxloop"], i, qtype) != key:
            flags.append("UNSTABLE")
        # C3d: null bias +1.0
        kn = self._key_of(c3["null"], i, qtype)
        if qtype in ("choice", "noul", "score") and kn == -1 and key != -1:
            flags.append("MARGINAL")
        elif qtype == "multi" and key and not set(key) <= set(kn):
            flags.append("MARGINAL")
        return flags

    def _key_of(self, out, i: int, qtype: str) -> Any:
        kmax = out.probs.shape[1] - 1
        if qtype in ("choice", "noul", "score"):
            a = int(out.probs[i].argmax())
            return -1 if a == kmax else a
        if qtype == "multi":
            # padded options have multi_logit at -1e9, so they never exceed 0.5
            return tuple(int(j) for j in torch.nonzero(out.multi_prob[i] > 0.5).flatten())
        if qtype == "extract":
            a, b, _ = decode_span(out.span_start[i:i + 1], out.span_end[i:i + 1], self.model.cfg.max_span)
            return (int(a[0]), int(b[0]))
        return None

    # -- decoding ------------------------------------------------------------------------
    def _fallback_confidence(self, out, i: int, qtype: str, key: Any) -> float:
        if qtype in ("choice", "noul", "score"):
            return float(out.probs[i].max())
        if qtype == "multi":
            p = out.multi_prob[i]
            valid = out.final.multi_logit[i] > -1e8
            p = p[valid]
            return float(torch.maximum(p, 1 - p).min()) if p.numel() else 1.0
        if qtype == "extract":
            return _span_prob(out, i, key)
        return 0.0

    def _decode(self, batch, out, i: int, spec, qtype: str, key: Any, b: int):
        """-> answer, probabilities, conformal_set (None if no conformal), extras, top_is_null, k."""
        cal = self.cal
        extras: Dict[str, Any] = {}
        labels = ["true", "false"] if qtype == "noul" else list(spec.options)
        k = len(labels)
        top_null = False
        conf: Optional[List[str]] = None
        if qtype in ("choice", "noul", "score"):
            temp = float(cal.temperature(qtype, k))
            logits = out.final.logits[i, list(range(k)) + [out.probs.shape[1] - 1]]
            p = (logits / temp).softmax(-1)
            if qtype == "score":
                lv = out.final.score_probs[i, :k]
                lv = lv / lv.sum().clamp_min(1e-9) * (1 - p[-1])
                p = torch.cat([lv, p[-1:]])
            probs = {lab: float(p[j]) for j, lab in enumerate(labels)}
            probs[NULL] = float(p[-1])
            top_null = key == -1
            answer = None
            if not top_null:
                answer = labels[key]
                if qtype == "noul":
                    answer = key == 0
                    extras["p_true"] = probs["true"]
            if qtype == "score":
                extras["expected_level"] = float(sum(j * float(p[j]) for j in range(k)))
            q = cal.conformal_q(qtype, k)
            if q is not None:
                names = labels + [NULL]
                conf = [nm for j, nm in enumerate(names) if 1.0 - float(p[j]) <= q]
            return answer, probs, conf, extras, top_null, k
        if qtype == "multi":
            mp = out.multi_prob[i]
            probs = {lab: float(mp[j]) for j, lab in enumerate(labels)}
            probs[NULL] = float(out.probs[i, -1])
            return [labels[j] for j in key], probs, None, extras, False, k
        if qtype == "range":
            qs = out.quantiles[i].tolist()
            lo = QUANTILE_LEVELS.index(min(QUANTILE_LEVELS, key=lambda x: abs(x - 0.1)))
            hi = QUANTILE_LEVELS.index(min(QUANTILE_LEVELS, key=lambda x: abs(x - 0.9)))
            mid = QUANTILE_LEVELS.index(min(QUANTILE_LEVELS, key=lambda x: abs(x - 0.5)))
            extras["interval"] = (qs[lo], qs[hi])
            return qs[mid], {NULL: 0.0}, None, extras, False, k
        # extract
        top_null = key == (0, 0)
        probs = {NULL: _span_prob(out, i, (0, 0))}
        answer = None
        if not top_null:
            offs = batch.state_offsets[b]
            c0, c1 = offs[key[0]][0], offs[key[1]][1]
            answer = batch.state_texts[b][c0:c1]
            extras["span_chars"] = (int(c0), int(c1))
            probs["__span__"] = _span_prob(out, i, key)
        return answer, probs, None, extras, top_null, k


__all__ = [
    "CalibrationLike", "NeutralCalibration", "CheckedDecision", "Checker", "CorrectnessHead", "DepthTrace",
    "RuleChecker", "RuleResult", "correctness_features", "decide_verdict", "decide_verdict_ex", "decode_keys",
    "depth_trace", "evidence_spans", "fit_correctness_head", "k_bucket", "NULL", "FLAGS",
]
