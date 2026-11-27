"""Unified decision pipeline for RHEN-2 (plan section 9).

Ten stages in fixed order, each appending a typed record to a ``DecisionTrace``::

    INGEST PLAN ENCODE DELIBERATE CALIBRATE CHECK POLICY STAGE AUDIT FEEDBACK

Behaviour that the plan leaves open, and what this file does:

* ``Checker.check`` encodes and deliberates internally.  The pipeline encodes through the state cache
  and runs the Checker's own pass on that cached memory (``Checker._pass``, the same code as
  ``Checker.check``, with at most one REVISE); it falls back to the public ``Checker.check`` if that
  private method is absent.  The DELIBERATE stage therefore runs one extra workspace pass at the
  requested effort to expose raw logits for the feedback ledger and the trace.
* Calibration: a file whose ``model_hash`` is empty or differs from the hash of the loaded weights
  (``DecisionPipeline.model_hash``) is a mismatch.  A mismatch, a missing file, or a Calibration that is
  not ``calibrated`` all give ``calibrated=False`` for every decision, the Checker runs with the neutral
  calibration, and nothing above tier ``low`` can EXECUTE.  A drift alarm on a group (feedback ledger)
  downgrades that group's decisions the same way.
* Any exception in ENCODE..CHECK gives ABSTAIN for every question of the batch, the error is recorded in
  the trace, and no other scorer is tried.
* An untrained model (``training_run_id is None``) never stages and ``execute`` refuses its decisions.
* Questions with more than ``cfg.max_options`` options are routed (``routing.py``) in the DELIBERATE stage.
  Their confidence is the product of the two stage probabilities, labelled ``uncalibrated_joint``, and
  ``calibrated`` is always False (there is no whole-workflow calibration here).
* The audit record never contains raw state, option text or answers: only hashes, counts, flags,
  probabilities as numbers, tiers and versions.  Exception messages are replaced by a type name and a hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from rhen.decision_pipeline import DecisionGateError
from rhen.rhen_packer import sanitize_delimiter_text
from rhen.v2.checker import EFFORTS, TIERS, CheckedDecision, Checker, NeutralCalibration
from rhen.v2.feedback import DecisionEntry, OutcomeLedger
from rhen.v2.packing import QuestionSpec, Rhen2Example, Rhen2Packer
from rhen.v2.policy import NULL, CostSpec, DecisionPolicy2, PolicyOutcome2, PolicyStatus
from rhen.v2.routing import route_large_option_set
from rhen.v2.workspace import LoopPlan

STAGES = ("INGEST", "PLAN", "ENCODE", "DELIBERATE", "CALIBRATE", "CHECK", "POLICY", "STAGE", "AUDIT", "FEEDBACK")
_SEVERITY = {PolicyStatus.EXECUTE: 0, PolicyStatus.NEEDS_APPROVAL: 1, PolicyStatus.ABSTAIN: 2, PolicyStatus.BLOCKED: 3}


def _sha(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def weights_hash(model: torch.nn.Module) -> str:
    """sha256 over the config and every parameter/buffer (float32 bytes), in sorted key order."""
    h = hashlib.sha256()
    h.update(json.dumps(model.cfg.to_dict(), sort_keys=True).encode())
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------- trace


@dataclass
class StageRecord:
    stage: str
    duration_s: float
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    trace_id: str = field(default_factory=lambda: f"tr-{uuid.uuid4().hex}")
    records: List[StageRecord] = field(default_factory=list)

    def append(self, stage: str, duration_s: float, summary: Dict[str, Any]) -> StageRecord:
        rec = StageRecord(stage, float(duration_s), dict(summary))
        self.records.append(rec)
        return rec

    def stage_names(self) -> List[str]:
        return [r.stage for r in self.records]

    def get(self, stage: str) -> StageRecord:
        for r in self.records:
            if r.stage == stage:
                return r
        raise KeyError(stage)

    def to_dict(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id,
                "stages": [{"stage": r.stage, "duration_s": r.duration_s, "summary": r.summary} for r in self.records]}

    def to_audit_dict(self) -> Dict[str, Any]:
        """Like ``to_dict`` but an ``error`` message is replaced by its type and a hash (it may quote input)."""
        stages = []
        for r in self.records:
            s = dict(r.summary)
            if "error" in s:
                s["error"] = {"type": s.get("error_type", "Exception"),
                              "message_sha256": _sha(str(s["error"]))}
                s.pop("error_type", None)
            stages.append({"stage": r.stage, "duration_s": round(r.duration_s, 6), "summary": s})
        return {"trace_id": self.trace_id, "stages": stages}


# ---------------------------------------------------------------------------- results


@dataclass
class QuestionDecision:
    decision_id: str
    item: int
    index: int
    spec: QuestionSpec
    checked: Optional[CheckedDecision]
    outcome: PolicyOutcome2
    calibrated: bool = False
    routed: bool = False
    card_id: Optional[str] = None
    error: Optional[str] = None
    logits: Optional[List[float]] = None
    executed_at: Optional[float] = None

    @property
    def status(self) -> PolicyStatus:
        return self.outcome.status

    @property
    def answer(self) -> Any:
        return None if self.checked is None else self.checked.answer


@dataclass
class PipelineResult:
    decisions: List[QuestionDecision]
    trace: DecisionTrace
    state_hash: str
    model_hash: str
    calibration_version: str
    calibrated: bool
    effort: str
    risk_tier: str
    error: Optional[str] = None

    @property
    def status(self) -> PolicyStatus:
        if not self.decisions:
            return PolicyStatus.ABSTAIN
        return max((d.status for d in self.decisions), key=lambda s: _SEVERITY[s])

    @property
    def decision(self) -> QuestionDecision:
        if len(self.decisions) != 1:
            raise ValueError(f"result has {len(self.decisions)} decisions; index .decisions")
        return self.decisions[0]


# ---------------------------------------------------------------------------- fallback state cache


class _FallbackCache:
    """Small LRU of per-example state memory keyed by a hash of the tokens, bounded by bytes."""

    def __init__(self, max_bytes: int = 64 << 20) -> None:
        self.max_bytes = max_bytes
        self.items: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    @staticmethod
    def _tag(model) -> str:
        return f"{id(model)}:{sum(int(p._version) for p in model.parameters())}"

    @torch.no_grad()
    def memory_for(self, model, batch) -> torch.Tensor:
        ids, mask, num = batch.state_ids, batch.state_mask, batch.state_num
        b, ls = ids.shape
        tag = self._tag(model)
        lens = [int(mask[i].nonzero().max().item()) + 1 if bool(mask[i].any()) else 1 for i in range(b)]
        keys = [_sha(tag + "|" + _sha(ids[i, :lens[i]].cpu().numpy().tobytes())
                     + _sha(num[i, :lens[i]].float().cpu().numpy().tobytes())) for i in range(b)]
        rows: List[Optional[torch.Tensor]] = []
        for k in keys:
            v = self.items.get(k)
            if v is not None:
                self.items.move_to_end(k)
                self.hits += 1
            else:
                self.misses += 1
            rows.append(v)
        todo = [i for i, r in enumerate(rows) if r is None]
        if todo:
            lm = max(lens[i] for i in todo)
            enc = model.encode_state(ids[todo][:, :lm], mask[todo][:, :lm], num[todo][:, :lm])
            for j, i in enumerate(todo):
                v = enc[j, :lens[i]].detach().clone()
                rows[i] = v
                size = v.numel() * v.element_size()
                if size <= self.max_bytes and keys[i] not in self.items:
                    self.items[keys[i]] = v
                    self.bytes += size
                    while self.bytes > self.max_bytes and self.items:
                        _, old = self.items.popitem(last=False)
                        self.bytes -= old.numel() * old.element_size()
        d = rows[0].shape[-1]
        out = torch.zeros(b, ls, d, dtype=rows[0].dtype)
        for i, r in enumerate(rows):
            out[i, :r.shape[0]] = r
        return out


# ---------------------------------------------------------------------------- pipeline


@dataclass
class _Slot:
    item: int
    index: int
    spec: QuestionSpec
    routed: bool
    qtype: str
    k: int


class DecisionPipeline:
    def __init__(self, model, packer: Rhen2Packer, calibration: Any = None, checker: Optional[Checker] = None,
                 policy: Optional[DecisionPolicy2] = None, staging: Any = None, audit: Any = None,
                 state_cache: Any = None, training_run_id: Optional[str] = None,
                 feedback: Optional[OutcomeLedger] = None, model_hash: Optional[str] = None,
                 max_registry: int = 10000) -> None:
        self.model, self.packer = model, packer
        self.calibration = calibration
        self._checker = checker
        self.policy = policy if policy is not None else DecisionPolicy2()
        self.staging, self.audit = staging, audit
        self.training_run_id = training_run_id
        self.feedback = feedback
        self.model_hash = model_hash if model_hash is not None else weights_hash(model)
        self._cache = state_cache
        self._cache_name = "none"
        self._cache_unusable = False
        self._fallback = _FallbackCache()
        self._max_registry = int(max_registry)
        self._decisions: "OrderedDict[str, QuestionDecision]" = OrderedDict()
        self._sealed: Dict[str, PolicyStatus] = {}
        self._approvals: Dict[str, Dict[str, Any]] = {}
        self.alarms: "deque[Dict[str, Any]]" = deque(maxlen=1000)

    # ------------------------------------------------------------------ calibration state
    def refresh_model_hash(self) -> str:
        self.model_hash = weights_hash(self.model)
        return self.model_hash

    def _cal_state(self) -> Tuple[Any, bool, str]:
        cal = self.calibration
        if cal is None:
            return NeutralCalibration(), False, "no_calibration"
        h = getattr(cal, "model_hash", None)
        if not h or h != self.model_hash:
            return NeutralCalibration(), False, "model_hash_mismatch"
        return cal, bool(getattr(cal, "calibrated", False)), "ok"

    def calibration_version(self) -> str:
        cal = self.calibration
        if cal is None:
            return "none"
        return ":".join(str(getattr(cal, a, "") or "")[:16] for a in ("model_hash", "data_hash", "created")) or "unknown"

    def _effective_checker(self, cal: Any, reason: str) -> Checker:
        if self._checker is None:
            return Checker(self.model, cal)
        if reason == "ok":
            return self._checker
        ck = copy.copy(self._checker)
        ck.cal = cal
        return ck

    def _group_alarmed(self, qtype: str, k: int) -> bool:
        return self.feedback is not None and not self.feedback.group_calibrated(qtype, k)

    # ------------------------------------------------------------------ cache
    def _memory(self, batch) -> Tuple[torch.Tensor, str]:
        cache = None if self._cache_unusable else self._cache
        if cache is None and not self._cache_unusable:
            try:
                from rhen.v2.cache import StateCache

                cache = self._cache = StateCache(max_bytes=256 << 20)
            except Exception:
                cache = None
        if cache is not None:
            try:
                S = cache.memory_for(self.model, batch)
                self._cache_name = type(cache).__name__
                return S, self._cache_name
            except (AttributeError, TypeError):   # API differs from memory_for(model, batch)
                self._cache_unusable = True
        self._cache_name = "fallback_lru"
        return self._fallback.memory_for(self.model, batch), "fallback_lru"

    # ------------------------------------------------------------------ public API
    def decide(self, state: Any, questions: Sequence[QuestionSpec], *, effort: str = "medium", risk_tier: str = "low",
               costs: Any = None, constraints: Optional[Sequence[Tuple[str, Callable]]] = None,
               approval: Union[None, bool, Sequence[str]] = None,
               cost_fn: Optional[Callable[[Mapping[str, Any]], CostSpec]] = None,
               features: Optional[Mapping[str, Any]] = None) -> PipelineResult:
        """``cost_fn(features) -> CostSpec`` (plan 9.5) supplies the cost of every question whose ``costs`` entry
        is None; ``features`` (caller data such as forecast breach bounds) is merged into the features that the
        hook and the constraints see.  Without ``cost_fn`` nothing changes."""
        return self.decide_batch([(state, questions)], effort=effort, risk_tier=risk_tier,
                                 costs=[costs], constraints=constraints, approval=approval,
                                 cost_fn=cost_fn, features=[features])[0]

    def decide_batch(self, items: Sequence[Tuple[Any, Sequence[QuestionSpec]]], *, effort: str = "medium",
                     risk_tier: str = "low", costs: Optional[Sequence[Any]] = None,
                     constraints: Optional[Sequence[Tuple[str, Callable]]] = None,
                     approval: Union[None, bool, Sequence[str]] = None,
                     cost_fn: Optional[Callable[[Mapping[str, Any]], CostSpec]] = None,
                     features: Optional[Sequence[Optional[Mapping[str, Any]]]] = None) -> List[PipelineResult]:
        """``costs`` is aligned with ``items``; each entry is None, one CostSpec, or one per question.
        ``features`` is aligned with ``items`` too; ``cost_fn`` is described on ``decide``."""
        cfg = self.model.cfg
        n_items = len(items)
        if n_items == 0:
            raise ValueError("decide_batch needs at least one item")
        traces = [DecisionTrace() for _ in items]

        def rec(name: str, t0: float, summary: Any) -> None:
            dt = time.perf_counter() - t0
            for i, tr in enumerate(traces):
                s = summary[i] if isinstance(summary, list) else summary
                tr.append(name, dt, s)

        # 1 INGEST -------------------------------------------------------------------
        t0 = time.perf_counter()
        texts = [self.packer.serialise_state(s) for s, _ in items]
        state_hashes = [_sha(t) for t in texts]
        qlists = [list(q) for _, q in items]
        rec("INGEST", t0, [{"state_hash": state_hashes[i], "state_chars": len(texts[i]), "n_questions": len(qlists[i]),
                            "delimiters_neutralised": sanitize_delimiter_text(texts[i]) != texts[i]}
                           for i in range(n_items)])

        # 2 PLAN ---------------------------------------------------------------------
        t0 = time.perf_counter()
        if risk_tier not in TIERS:
            raise ValueError(f"risk_tier must be one of {TIERS}, got {risk_tier!r}")
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
        eff = "high" if risk_tier in ("high", "critical") else effort
        costs_in = list(costs) if costs is not None else [None] * n_items
        if len(costs_in) != n_items:
            raise ValueError("costs must have one entry per item")
        item_feats: List[Mapping[str, Any]] = [dict(f) if f else {} for f in (features or [None] * n_items)]
        if len(item_feats) != n_items:
            raise ValueError("features must have one entry per item")
        slots: List[_Slot] = []
        item_costs: List[List[Optional[CostSpec]]] = []
        for i, qs in enumerate(qlists):
            c = costs_in[i]
            if c is None or isinstance(c, CostSpec):
                per_q = [c] * len(qs)
            else:
                per_q = list(c)
                if len(per_q) != len(qs):
                    raise ValueError("per-question costs must match the number of questions")
            item_costs.append(per_q)
            for j, q in enumerate(qs):
                if not isinstance(q, QuestionSpec):
                    raise TypeError("questions must be QuestionSpec objects")
                routed = q.type == "choice" and len(q.options) > cfg.max_options
                if not routed:
                    q.validate(cfg)
                elif len(set(q.options)) != len(q.options):
                    raise ValueError("options must be unique")
                k = 2 if q.type == "noul" else len(q.options)
                slots.append(_Slot(i, j, q, routed, q.type, k))
                if cost_fn is not None and per_q[j] is None:
                    hook = cost_fn({**item_feats[i], "risk_tier": risk_tier, "qtype": q.type, "question_index": j,
                                    "item": i})
                    if not isinstance(hook, CostSpec):
                        raise TypeError("cost_fn must return a CostSpec")
                    per_q[j] = hook
        rec("PLAN", t0, [{"effort": eff, "risk_tier": risk_tier, "n_questions": len(qlists[i]),
                          "routed": sum(1 for s in slots if s.item == i and s.routed)} for i in range(n_items)])

        cal, cal_ok, cal_reason = self._cal_state()
        checker = self._effective_checker(cal, cal_reason)
        failure: Optional[Tuple[str, BaseException]] = None
        batch = S = None
        out = None
        checked: Dict[Tuple[int, int], CheckedDecision] = {}
        logits_of: Dict[Tuple[int, int], List[float]] = {}
        packed_slots = [s for s in slots if not s.routed]
        routed_slots = [s for s in slots if s.routed]

        def run(name: str, fn: Callable[[], Any]) -> None:
            nonlocal failure
            t0 = time.perf_counter()
            if failure is not None:
                rec(name, t0, {"skipped": True, "reason": f"error in {failure[0]}"})
                return
            try:
                summary = fn()
            except Exception as exc:                       # stages 3 to 6: record, ABSTAIN, no other scorer
                failure = (name, exc)
                rec(name, t0, {"error": str(exc)[:500], "error_type": type(exc).__name__})
                return
            rec(name, t0, summary)

        # 3 ENCODE -------------------------------------------------------------------
        def encode() -> Dict[str, Any]:
            nonlocal batch, S
            packed_items = sorted({s.item for s in packed_slots})
            if not packed_items:
                return {"batch_size": 0, "cache": "not_needed"}
            exs = [Rhen2Example(items[i][0], [s.spec for s in packed_slots if s.item == i]) for i in packed_items]
            batch = self.packer.pack(exs)
            self._pos = {i: b for b, i in enumerate(packed_items)}
            was = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    S, name = self._memory(batch)
            finally:
                self.model.train(was)
            return {"batch_size": len(packed_items), "cache": name, "state_tokens": int(batch.state_mask.sum())}

        run("ENCODE", encode)

        # 4 DELIBERATE ---------------------------------------------------------------
        def deliberate() -> Dict[str, Any]:
            nonlocal out
            summ: Dict[str, Any] = {"effort": eff, "loops_run": [], "routed_calls": 0}
            if batch is not None:
                halt = getattr(cal, "halt_threshold", None)
                plan = LoopPlan.effort(cfg, eff, halt_threshold=float(halt) if halt is not None else 0.9)
                was = self.model.training
                self.model.eval()
                try:
                    with torch.no_grad():
                        out = self.model.deliberate(S, batch, plan)
                finally:
                    self.model.train(was)
                m = batch.max_questions
                last = out.probs.shape[1] - 1
                per_item: Dict[int, int] = {}
                for s in packed_slots:
                    b = self._pos[s.item]
                    qi = per_item.get(s.item, 0)
                    per_item[s.item] = qi + 1
                    row = b * m + qi
                    summ["loops_run"].append(int(out.loops_run[row]))
                    if s.qtype in ("choice", "noul", "score"):
                        logits_of[(s.item, s.index)] = [float(x) for x in
                                                        out.final.logits[row, list(range(s.k)) + [last]]]
            for s in routed_slots:
                res = route_large_option_set(self.model, self.packer, items[s.item][0], s.spec, s.spec.options,
                                             effort=eff)
                summ["routed_calls"] += res.report.model_calls
                p = res.joint_confidence
                ans = res.option
                checked[(s.item, s.index)] = CheckedDecision(
                    answer=ans,
                    probabilities=({ans: p, NULL: 1.0 - p} if ans is not None else {NULL: 1.0}),
                    confidence=p, conformal_set=[], verdict="ESCALATE" if res.abstained else "CONFIRM",
                    flags=["ROUTED"], evidence_spans=[], loops_run=0, effort=eff, calibrated=False,
                    extras={"routing": res.report.to_dict(), "joint_label": res.joint_label})
            return summ

        run("DELIBERATE", deliberate)

        # 5 CALIBRATE ----------------------------------------------------------------
        def calibrate() -> Dict[str, Any]:
            groups = [{"qtype": s.qtype, "k": s.k, "temperature": float(cal.temperature(s.qtype, s.k)),
                       "alarm": self._group_alarmed(s.qtype, s.k)} for s in packed_slots]
            return {"calibration_ok": cal_ok, "reason": cal_reason, "calibration_version": self.calibration_version(),
                    "model_hash": self.model_hash[:16], "groups": groups}

        run("CALIBRATE", calibrate)

        # 6 CHECK --------------------------------------------------------------------
        def check() -> Dict[str, Any]:
            passes = 1
            if batch is None:
                return {"passes": 0, "revised": 0, "verdicts": [], "flags": []}
            if hasattr(checker, "_pass"):
                was = checker.model.training
                checker.model.eval()
                try:
                    with torch.no_grad():
                        drafts = checker._pass(batch, S, eff, risk_tier, can_revise=eff != "high")
                        redo = [i for i, d in enumerate(drafts) if d is not None and d.verdict == "REVISE"]
                        if redo:
                            passes = 2
                            second = checker._pass(batch, S, EFFORTS[EFFORTS.index(eff) + 1], risk_tier, can_revise=False)
                            for i in redo:
                                second[i].decision.extras["revised_from"] = eff
                                drafts[i] = second[i]
                finally:
                    checker.model.train(was)
                decs = [d.decision for d in drafts if d is not None]
            else:
                decs = checker.check(batch, eff, risk_tier)
            if len(decs) != len(packed_slots):
                raise RuntimeError(f"Checker returned {len(decs)} decisions for {len(packed_slots)} questions")
            for s, d in zip(packed_slots, decs):
                checked[(s.item, s.index)] = d
            return {"passes": passes, "revised": sum(1 for d in decs if "revised_from" in d.extras),
                    "verdicts": [d.verdict for d in decs], "flags": [list(d.flags) for d in decs]}

        run("CHECK", check)

        # 7 POLICY -------------------------------------------------------------------
        t0 = time.perf_counter()
        decisions_by_item: List[List[QuestionDecision]] = [[] for _ in items]
        err = None if failure is None else f"{failure[0]}: {type(failure[1]).__name__}"
        for s in slots:
            did = f"d2-{uuid.uuid4().hex}"
            key = (s.item, s.index)
            cd = checked.get(key)
            if failure is not None or cd is None:
                outcome = PolicyOutcome2(PolicyStatus.ABSTAIN, None, None, None, risk_tier,
                                         [f"error in {failure[0]}: {type(failure[1]).__name__}" if failure
                                          else "no checked decision"])
                qd = QuestionDecision(did, s.item, s.index, s.spec, None, outcome, False, s.routed, error=err)
            else:
                eff_cal = bool(cd.calibrated) and cal_ok and not s.routed and not self._group_alarmed(s.qtype, s.k)
                outcome = self._policy(s, cd, risk_tier, eff_cal, item_costs[s.item][s.index], constraints, approval,
                                       item_feats[s.item])
                if cal_reason != "ok":
                    outcome.reasons.append(f"calibration: {cal_reason}")
                if self.calibration is not None and cal_reason == "ok" and not cal_ok:
                    outcome.reasons.append("calibration file is not marked calibrated")
                if self._group_alarmed(s.qtype, s.k):
                    outcome.reasons.append("drift alarm active for this group: calibrated=False")
                if self.training_run_id is None:
                    outcome.reasons.append("untrained model: staging and execution disabled")
                qd = QuestionDecision(did, s.item, s.index, s.spec, cd, outcome, eff_cal, s.routed,
                                      logits=logits_of.get(key))
            decisions_by_item[s.item].append(qd)
            self._register(qd)
        rec("POLICY", t0, [{"statuses": [d.status.value for d in decisions_by_item[i]],
                            "risk": [None if d.outcome.risk_act is None else round(d.outcome.risk_act, 6)
                                     for d in decisions_by_item[i]]} for i in range(n_items)])

        # 8 STAGE --------------------------------------------------------------------
        t0 = time.perf_counter()
        stage_summ: List[Dict[str, Any]] = []
        for i in range(n_items):
            staged, errors = [], []
            for qd in decisions_by_item[i]:
                if qd.status not in (PolicyStatus.EXECUTE, PolicyStatus.NEEDS_APPROVAL):
                    continue
                if self.training_run_id is None or self.staging is None:
                    continue
                try:
                    card = self._stage(qd, risk_tier, state_hashes[i])
                    qd.card_id = card.id
                    staged.append(card.id)
                except Exception as exc:
                    self._downgrade(qd, f"staging failed: {type(exc).__name__}")
                    errors.append(type(exc).__name__)
            reason = ("untrained model" if self.training_run_id is None else
                      "no staging engine" if self.staging is None else "")
            stage_summ.append({"staged": staged, "errors": errors, "skipped_reason": reason})
        rec("STAGE", t0, stage_summ)

        # 9 AUDIT --------------------------------------------------------------------
        t0 = time.perf_counter()
        results: List[PipelineResult] = []
        audit_summ: List[Dict[str, Any]] = []
        for i in range(n_items):
            res = PipelineResult(decisions_by_item[i], traces[i], state_hashes[i], self.model_hash,
                                 self.calibration_version(), cal_ok, eff, risk_tier, err)
            results.append(res)
            if self.audit is None:
                audit_summ.append({"recorded": False, "reason": "no audit ledger"})
                continue
            try:
                entry = self.audit.record("decision_v2", self._audit_payload(res, qlists[i]))
                audit_summ.append({"recorded": True, "sequence": getattr(entry, "sequence", None),
                                   "entry_hash": getattr(entry, "entry_hash", None)})
            except Exception as exc:     # audit failure means the action is not authorised
                audit_summ.append({"recorded": False, "error": str(exc)[:200], "error_type": type(exc).__name__})
                for qd in res.decisions:
                    self._reject_card(qd)
                    self._downgrade(qd, f"audit failed: {type(exc).__name__}")
        rec("AUDIT", t0, audit_summ)

        # 10 FEEDBACK ----------------------------------------------------------------
        t0 = time.perf_counter()
        fb: List[Dict[str, Any]] = []
        for i in range(n_items):
            n = 0
            if self.feedback is not None:
                for qd in decisions_by_item[i]:
                    if qd.checked is None:
                        continue
                    pred = None
                    if qd.logits is not None:
                        pred = max(range(len(qd.logits) - 1), key=lambda j: qd.logits[j])
                    self.feedback.record_decision(DecisionEntry(
                        decision_id=qd.decision_id, qtype=qd.spec.type,
                        k=2 if qd.spec.type == "noul" else len(qd.spec.options), status=qd.status.value,
                        answer=None, predicted_idx=pred, confidence=float(qd.checked.confidence),
                        calibrated=qd.calibrated, ood="OOD" in qd.checked.flags, logits=qd.logits,
                        meta={"risk_tier": risk_tier, "state_hash": state_hashes[i]}))
                    n += 1
            fb.append({"recorded": n, "ledger": self.feedback is not None})
        rec("FEEDBACK", t0, fb)
        return results

    # ------------------------------------------------------------------ policy / staging helpers
    def _policy(self, s: _Slot, cd: CheckedDecision, tier: str, calibrated: bool, costs: Optional[CostSpec],
                constraints, approval, extra: Optional[Mapping[str, Any]] = None) -> PolicyOutcome2:
        conf = float(min(max(cd.confidence, 0.0), 1.0))
        if s.qtype in ("choice", "noul", "score") and cd.answer is not None:
            top = ("true" if cd.answer else "false") if s.qtype == "noul" else str(cd.answer)
            probs = {str(k): float(v) for k, v in cd.probabilities.items()}
            if top not in probs:
                top = str(cd.answer)
        else:
            top = str(cd.answer) if cd.answer is not None else "__answer__"
            probs = {top: conf, NULL: 1.0 - conf}
        feats = dict(extra or {})
        feats.update({"risk_tier": tier, "qtype": s.qtype, "question_index": s.index, "item": s.item,
                      "flags": list(cd.flags)})
        return self.policy.evaluate(probs, verdict=cd.verdict, risk_tier=tier, calibrated=calibrated, costs=costs,
                                    constraints=constraints or (), features=feats, p_correct=conf, top_label=top,
                                    approval=approval)

    def _stage(self, qd: QuestionDecision, tier: str, state_hash: str):
        cd = qd.checked
        payload = {"decision_id": qd.decision_id, "option": qd.outcome.option, "qtype": qd.spec.type,
                   "state_hash": state_hash}
        meta = {"conformal_set": {"is_singleton": len(cd.conformal_set) == 1, "size": len(cd.conformal_set)},
                "model_hash": self.model_hash, "training_run_id": self.training_run_id,
                "calibration_version": self.calibration_version(), "calibrated": qd.calibrated,
                "decision_id": qd.decision_id}
        return self.staging.stage_action({"confidence": float(cd.confidence)}, action_type="execute", payload=payload,
                                         title=f"RHEN-2 decision {qd.decision_id}", persona="rhen-v2-pipeline",
                                         risk_tier=tier, metadata=meta)

    def _reject_card(self, qd: QuestionDecision) -> None:
        if qd.card_id and self.staging is not None:
            try:
                self.staging.reject_action(qd.card_id, reason="audit record failed")
            except Exception:
                pass

    def _register(self, qd: QuestionDecision) -> None:
        self._decisions[qd.decision_id] = qd
        self._sealed[qd.decision_id] = qd.outcome.status
        while len(self._decisions) > self._max_registry:
            old, _ = self._decisions.popitem(last=False)
            self._sealed.pop(old, None)
            self._approvals.pop(old, None)

    def _downgrade(self, qd: QuestionDecision, reason: str) -> None:
        qd.outcome.status = PolicyStatus.ABSTAIN
        qd.outcome.reasons.append(reason)
        self._sealed[qd.decision_id] = PolicyStatus.ABSTAIN

    # ------------------------------------------------------------------ audit
    def _audit_payload(self, res: PipelineResult, questions: Sequence[QuestionSpec]) -> Dict[str, Any]:
        qhash = _sha(json.dumps([[q.type, q.instructions, list(q.options)] for q in questions], sort_keys=True))
        qs = []
        for d in res.decisions:
            c = d.checked
            probs = sorted((float(v) for v in c.probabilities.values()), reverse=True)[:3] if c else []
            qs.append({
                "decision_id": d.decision_id, "qtype": d.spec.type, "status": d.status.value,
                "tier": d.outcome.risk_tier, "option_hash": _sha(str(d.outcome.option)) if d.outcome.option else None,
                "answer_hash": _sha(json.dumps(c.answer, sort_keys=True, default=str)) if c and c.answer is not None else None,
                "confidence": None if c is None else round(float(c.confidence), 6), "top_probs": [round(p, 6) for p in probs],
                "verdict": None if c is None else c.verdict, "flags": [] if c is None else list(c.flags),
                "conformal_set_size": None if c is None else len(c.conformal_set), "calibrated": d.calibrated,
                "loops_run": None if c is None else c.loops_run, "risk_act": d.outcome.risk_act,
                "risk_abstain": d.outcome.risk_abstain, "card_id": d.card_id, "routed": d.routed,
                "error": None if d.error is None else d.error,
            })
        return {"trace_id": res.trace.trace_id, "state_hash": res.state_hash, "questions_hash": qhash,
                "model_hash": res.model_hash, "calibration_version": res.calibration_version,
                "calibrated": res.calibrated, "training_run_id": self.training_run_id, "effort": res.effort,
                "risk_tier": res.risk_tier, "status": res.status.value, "questions": qs,
                "trace": res.trace.to_audit_dict()}

    # ------------------------------------------------------------------ approval / execution
    def approve(self, decision_id: str, approver: str) -> Dict[str, Any]:
        self._lookup(decision_id)
        if not isinstance(approver, str) or not approver.strip():
            raise DecisionGateError("approver must be a non-empty name")
        if self._sealed.get(decision_id) is not PolicyStatus.NEEDS_APPROVAL:
            raise DecisionGateError(f"decision {decision_id} is not NEEDS_APPROVAL; only those can be approved")
        if decision_id in self._approvals:
            raise DecisionGateError(f"decision {decision_id} is already approved")
        entry = {"approver": approver.strip(), "approved_at": time.time()}
        if self.audit is not None:
            self.audit.record("approval_v2", {"decision_id": decision_id, "approver": entry["approver"]})
        self._approvals[decision_id] = entry
        return dict(entry)

    def _lookup(self, decision_id: str) -> QuestionDecision:
        qd = self._decisions.get(decision_id)
        if qd is None:
            raise DecisionGateError(f"unknown decision_id {decision_id!r}; only decisions made by this pipeline")
        return qd

    def execute(self, result: Union[PipelineResult, QuestionDecision],
                executor: Callable[[str, QuestionDecision], Any]) -> Any:
        """Run ``executor(option, decision)`` only for EXECUTE, or NEEDS_APPROVAL with a registered approval.

        The status is read from a private table written by the pipeline, not from the decision object.
        """
        qd = result.decision if isinstance(result, PipelineResult) else result
        if not isinstance(qd, QuestionDecision):
            raise TypeError("execute needs a PipelineResult or QuestionDecision from this pipeline")
        if not callable(executor):
            raise TypeError("executor must be callable as executor(option, decision)")
        if self._decisions.get(qd.decision_id) is not qd:
            raise DecisionGateError(f"decision {qd.decision_id} was not made by this pipeline")
        if self.training_run_id is None:
            raise DecisionGateError("model has no training_run_id: its decisions cannot be executed")
        if qd.executed_at is not None:
            raise DecisionGateError(f"decision {qd.decision_id} has already been executed")
        status = self._sealed.get(qd.decision_id)
        if status is PolicyStatus.NEEDS_APPROVAL and qd.decision_id not in self._approvals:
            raise DecisionGateError(f"decision {qd.decision_id} needs human approval before execution")
        if status not in (PolicyStatus.EXECUTE, PolicyStatus.NEEDS_APPROVAL):
            raise DecisionGateError(f"decision {qd.decision_id} has status "
                                    f"{status.value if status else None} and cannot be executed")
        option = qd.outcome.option
        if option is None:
            raise DecisionGateError(f"decision {qd.decision_id} has no option to execute")
        if qd.card_id is not None and self.staging is not None:
            report = self.staging.preflight_verification(qd.card_id)
            if not report["allowed"]:
                raise DecisionGateError(f"staging preflight refused card {qd.card_id}: {report['errors']}")
            self.staging.approve_action(qd.card_id, self._approvals.get(qd.decision_id, {}).get("approver", "pipeline"))
            qd.executed_at = time.time()
            box: Dict[str, Any] = {}

            def fn(payload: Dict[str, Any]) -> Dict[str, Any]:
                box["value"] = executor(option, qd)
                return {"status": "success"}

            card = self.staging.execute_action(qd.card_id, fn)
            if card.status != "executed":
                raise DecisionGateError(f"staged execution of {qd.card_id} failed: {card.execution_result}")
            return box["value"]
        qd.executed_at = time.time()
        return executor(option, qd)

    # ------------------------------------------------------------------ feedback
    def record_outcome(self, decision_id: str, decision_correct: Optional[bool] = None,
                       label_idx: Optional[int] = None) -> List[str]:
        """Forward an outcome to the feedback ledger; alarms are kept in ``self.alarms`` and downgrade the group."""
        if self.feedback is None:
            raise RuntimeError("no feedback ledger configured")
        alarms = self.feedback.record_outcome(decision_id, decision_correct, label_idx)
        for a in alarms:
            self.alarms.append({"decision_id": decision_id, "alarm": a})
        return alarms


__all__ = ["DecisionPipeline", "DecisionTrace", "StageRecord", "PipelineResult", "QuestionDecision", "STAGES",
           "weights_hash", "DecisionGateError"]
