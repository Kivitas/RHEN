"""v1-compatible adapter over the RHEN-2 decision pipeline (plan task T-603).

``V2Backend`` wraps a ``DecisionPipeline`` and exposes the primitive methods of the v1 ``RhenEngine``
(``noul``, ``choice``, ``score``, ``multichoice``, ``range``, ``extract``, ``evaluate``, ``cascade``).  The
methods take the v1 primitives (``Noul``, ``Choice`` ...) and return the v1 result dataclasses, filled from the
v2 ``CheckedDecision`` and the policy outcome of the pipeline.

Mapping of v1 primitives to v2 question types
---------------------------------------------
* ``Noul(proposition)`` -> v2 ``noul`` (options ``true``/``false``).  v1 labels ``Yes``/``No`` map to
  ``True``/``False``; ``NoulResult.p_true`` is the v2 probability of ``true``.
* ``Choice`` -> v2 ``choice`` (K above ``cfg.max_options`` is routed hierarchically by the pipeline).
* ``Score(levels, min_score, max_score)`` -> v2 ``score`` with the levels as ordered options when
  ``2 <= len(levels) <= cfg.max_levels``; the score is ``min + E[level] / (K - 1) * (max - min)``.  Otherwise
  (one level, or more levels than the model has) -> v2 ``range`` on ``[min_score, max_score]``: the score is the
  median and ``level_probabilities`` is empty (``meta_reasoning_context['score_via'] == 'range'``).
* ``MultiChoice`` -> v2 ``multi``; an option is selected when its calibrated probability is above ``threshold``.
* ``Range(min_val, max_val)`` -> v2 ``range``; ``expected_value`` is the median, the interval is the 10%..90%
  quantile pair (``confidence_interval = 0.8``), both clamped into ``[min_val, max_val]``.
* ``Extract(slot_name, query)`` -> v2 ``extract`` (instructions ``query`` or ``slot_name``).
* ``Cascade`` / callable stages are run stage by stage like v1.  The cascade confidence is the product of the
  stage confidences, labelled ``uncalibrated_joint`` in ``meta_reasoning_context`` (plan 9.4).

What goes where in the result
-----------------------------
* ``confidence`` is the Checker's calibrated confidence; ``temperature`` is the calibration temperature for the
  group (1.0 without calibration).
* ``probabilities`` / ``level_probabilities`` hold the option probabilities only.  The null ("cannot tell")
  probability goes into ``meta_reasoning_context['null_probability']`` and decides ``abstention_verdict``.
* ``conformal_set`` is the v1 dict (``prediction_set``, ``set_size``, ``is_singleton``, ``is_ambiguous``,
  ``is_empty``, ``includes_null``, ``calibrated``); ``None`` when the calibration has no conformal quantile.
* ``abstention_verdict`` (and so ``should_abstain``): ``Answer`` only when the pipeline policy says EXECUTE.
  ABSTAIN and BLOCKED give ``Abstain``; NEEDS_APPROVAL gives ``NeedClarification`` (not final without a human).
  The reasons carry the policy reasons, the Checker flags and the calibration state.  An untrained or
  uncalibrated backend still runs, but the pipeline's fail-closed rule (plan 9.2) makes every decision above
  tier ``low`` an abstention; at tier ``low`` it may answer, and ``meta_reasoning_context['calibrated']`` /
  ``['untrained']`` say so.
* ``meta_reasoning_context`` (a dict) also carries ``backend='rhen2'``, ``verdict``, ``flags``, ``status``,
  ``risk_tier``, ``effort``, ``loops_run``, ``decision_id`` and the Checker ``extras``.

Not supported on this backend (raise ``NotImplementedError``): ``delta_state``/``is_delta``, ``latent_tensors``,
``num_loops`` other than None (v2 effort sets the loops), and the v1 latent-search methods.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from rhen.decision_pipeline import PolicyStatus
from rhen.rhen import (
    ABSTAIN, ANSWER, NEED_CLARIFICATION,
    AbstentionVerdict, Cascade, CascadeResult, Choice, ChoiceResult, Extract, ExtractResult, MultiChoice,
    MultiChoiceResult, Noul, NoulResult, Range, RangeResult, Score, ScoreResult,
)
from rhen.v2.packing import QuestionSpec, Rhen2Packer
from rhen.v2.pipeline import DecisionPipeline, PipelineResult, QuestionDecision
from rhen.v2.policy import NULL, CostSpec, cascade_confidence

Primitive = Union[Noul, Choice, Score, MultiChoice, Range, Extract]


def _clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def _as_primitive(q: Any) -> Any:
    """Raw v1 tuples ``(question, options[, kind, meta])`` become primitives; primitives pass through."""
    if isinstance(q, (Noul, Choice, Score, MultiChoice, Range, Extract, Cascade)):
        return q
    if isinstance(q, tuple) and len(q) >= 2:
        question, options = q[0], q[1]
        kind = q[2] if len(q) > 2 else "choice"
        meta = q[3] if len(q) > 3 and isinstance(q[3], dict) else {}
        if kind == "multichoice":
            return MultiChoice(question=question, options=list(options), threshold=meta.get("threshold", 0.5))
        if kind == "range":
            return Range(question=question, min_val=meta.get("min_val", 0.0), max_val=meta.get("max_val", 100.0))
        if kind == "extract":
            return Extract(slot_name=meta.get("slot_name", question), query=question)
        if kind == "choice":
            if list(options) == ["Yes", "No"]:
                return Noul(proposition=question)
            return Choice(question=question, options=list(options))
    raise TypeError(f"Unsupported question specification: {type(q)}")


class V2Backend:
    """RHEN-1 style primitive API on top of a RHEN-2 ``DecisionPipeline``."""

    def __init__(self, pipeline: DecisionPipeline, *, risk_tier: str = "low", effort: str = "medium") -> None:
        self.pipeline = pipeline
        self.risk_tier = risk_tier
        self.effort = effort

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_model(cls, model: Any, calibration: Any = None, *, packer: Optional[Rhen2Packer] = None,
                   training_run_id: Optional[str] = None, risk_tier: str = "low", effort: str = "medium",
                   **pipeline_kwargs: Any) -> "V2Backend":
        """Wrap an already built ``Rhen2Model``.  ``calibration`` is a ``Calibration`` object or None."""
        pk = packer if packer is not None else Rhen2Packer(model.cfg)
        pipe = DecisionPipeline(model, pk, calibration=calibration, training_run_id=training_run_id,
                                **pipeline_kwargs)
        return cls(pipe, risk_tier=risk_tier, effort=effort)

    @classmethod
    def from_checkpoint(cls, checkpoint: Optional[str] = None, calibration: Optional[str] = None, *,
                        risk_tier: str = "low", effort: str = "medium", **pipeline_kwargs: Any) -> "V2Backend":
        """Load ``checkpoint`` (``io.load_checkpoint``) and an optional ``calibration.json`` path.

        ``checkpoint=None`` builds an untrained ``r2-tiny`` model for tests and wiring; it has no
        ``training_run_id``, so it never stages and every decision above tier ``low`` abstains.
        A calibration file whose ``model_hash`` does not match the loaded weights is kept as is: the pipeline
        reports the mismatch and treats the decisions as uncalibrated.
        """
        from rhen.v2.calibrate import Calibration

        run_id = None
        if checkpoint is None:
            from rhen.v2.config import preset
            from rhen.v2.model import Rhen2Model

            model = Rhen2Model(preset("r2-tiny"))
            model.eval()
            warnings.warn("V2Backend: no checkpoint given; using an untrained r2-tiny model (abstains above tier 'low')",
                          RuntimeWarning, stacklevel=2)
        else:
            from rhen.v2.io import load_checkpoint

            model, extra = load_checkpoint(checkpoint)
            run_id = (extra or {}).get("training_run_id") or None
        cal = Calibration.load(calibration) if calibration is not None else None
        return cls.from_model(model, cal, training_run_id=run_id, risk_tier=risk_tier, effort=effort,
                              **pipeline_kwargs)

    @property
    def model(self) -> Any:
        return self.pipeline.model

    @property
    def packer(self) -> Rhen2Packer:
        return self.pipeline.packer

    # ------------------------------------------------------------------ primitive methods
    def noul(self, state: Any, proposition: str, effort: Optional[str] = None, num_loops: Optional[int] = None,
             **kw: Any) -> NoulResult:
        return self.evaluate(state, [Noul(proposition=proposition)], effort=effort, num_loops=num_loops, **kw)[0]

    def choice(self, state: Any, question: str, options: List[str], effort: Optional[str] = None,
               num_loops: Optional[int] = None, **kw: Any) -> ChoiceResult:
        return self.evaluate(state, [Choice(question=question, options=options)], effort=effort,
                             num_loops=num_loops, **kw)[0]

    def score(self, state: Any, question: str, levels: Optional[List[str]] = None, min_score: float = 0.0,
              max_score: float = 10.0, effort: Optional[str] = None, num_loops: Optional[int] = None,
              **kw: Any) -> ScoreResult:
        lv = levels if levels is not None else ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"]
        return self.evaluate(state, [Score(question=question, levels=lv, min_score=min_score, max_score=max_score)],
                             effort=effort, num_loops=num_loops, **kw)[0]

    def multichoice(self, state: Any, question: str, options: List[str], threshold: float = 0.5,
                    effort: Optional[str] = None, num_loops: Optional[int] = None, **kw: Any) -> MultiChoiceResult:
        return self.evaluate(state, [MultiChoice(question=question, options=options, threshold=threshold)],
                             effort=effort, num_loops=num_loops, **kw)[0]

    def range(self, state: Any, question: str, min_val: float = 0.0, max_val: float = 100.0,
              effort: Optional[str] = None, num_loops: Optional[int] = None, **kw: Any) -> RangeResult:
        return self.evaluate(state, [Range(question=question, min_val=min_val, max_val=max_val)], effort=effort,
                             num_loops=num_loops, **kw)[0]

    def extract(self, state: Any, slot_name: str, query: str = "", effort: Optional[str] = None,
                num_loops: Optional[int] = None, **kw: Any) -> ExtractResult:
        return self.evaluate(state, [Extract(slot_name=slot_name, query=query)], effort=effort,
                             num_loops=num_loops, **kw)[0]

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, state: Any, questions: Sequence[Any], delta_state: Any = None, is_delta: bool = False,
                 latent_tensors: Any = None, effort: Optional[str] = None, num_loops: Optional[int] = None,
                 **kw: Any) -> List[Any]:
        """Typed results for ``questions`` (primitives, raw tuples and ``Cascade`` objects), in order.

        Non-cascade questions share one pipeline call (one state encode).  Extra keywords:
        ``risk_tier``, ``costs``, ``constraints``, ``approval``, ``cost_fn``, ``features`` (see
        ``DecisionPipeline.decide``).
        """
        prims = [_as_primitive(q) for q in questions]
        flat = [(i, q) for i, q in enumerate(prims) if not isinstance(q, Cascade)]
        out: List[Any] = [None] * len(prims)
        if flat:
            results, _ = self.evaluate_detailed(state, [q for _, q in flat], delta_state=delta_state,
                                                is_delta=is_delta, latent_tensors=latent_tensors, effort=effort,
                                                num_loops=num_loops, **kw)
            for (i, _), r in zip(flat, results):
                out[i] = r
        for i, q in enumerate(prims):
            if isinstance(q, Cascade):
                out[i] = self.cascade(state, q, effort=effort, **kw)
        return out

    def evaluate_detailed(self, state: Any, questions: Sequence[Any], delta_state: Any = None,
                          is_delta: bool = False, latent_tensors: Any = None, effort: Optional[str] = None,
                          num_loops: Optional[int] = None, risk_tier: Optional[str] = None, costs: Any = None,
                          constraints: Optional[Sequence[Tuple[str, Callable]]] = None,
                          approval: Union[None, bool, Sequence[str]] = None,
                          cost_fn: Optional[Callable[[Mapping[str, Any]], CostSpec]] = None,
                          features: Optional[Mapping[str, Any]] = None) -> Tuple[List[Any], PipelineResult]:
        """Like ``evaluate`` for primitives only; also returns the ``PipelineResult`` (trace, policy outcomes)."""
        if delta_state is not None or is_delta:
            raise NotImplementedError("the v2 backend does not support delta states")
        if latent_tensors is not None:
            raise NotImplementedError("the v2 backend does not take latent_tensors")
        if num_loops is not None:
            raise NotImplementedError("the v2 backend sets loops by effort; num_loops is not supported")
        prims = [_as_primitive(q) for q in questions]
        if any(isinstance(q, Cascade) for q in prims):
            raise TypeError("evaluate_detailed takes primitives only; use cascade() for Cascade objects")
        specs = [self._spec(q) for q in prims]
        tier = risk_tier if risk_tier is not None else self.risk_tier
        res = self.pipeline.decide(state, specs, effort=effort or self.effort, risk_tier=tier, costs=costs,
                                   constraints=constraints, approval=approval, cost_fn=cost_fn, features=features)
        return [self._convert(q, qd, res) for q, qd in zip(prims, res.decisions)], res

    # ------------------------------------------------------------------ cascade
    def cascade(self, state: Any, stages: Any, effort: Optional[str] = None, **kw: Any) -> Any:
        """Stage-by-stage cascade like v1.  Callable stages return the list of stage results."""
        if isinstance(stages, Cascade):
            return self._run_cascade(stages, state, effort, kw)
        if callable(stages[0]):
            context: Dict[str, Any] = {}
            results = []
            for fn in stages:
                res = self.evaluate(state, [fn(context)], effort=effort, **kw)[0]
                results.append(res)
                self._update_context(context, res)
            return results
        return self._run_cascade(Cascade(stages=list(stages)), state, effort, kw)

    @staticmethod
    def _update_context(context: Dict[str, Any], res: Any) -> None:
        if isinstance(res, NoulResult):
            context[res.proposition] = res.decision
        elif isinstance(res, ChoiceResult):
            context[res.question] = res.choice_label
        elif isinstance(res, ScoreResult):
            context[res.question] = res.score
        elif isinstance(res, MultiChoiceResult):
            context[res.question] = res.selected_options
        elif isinstance(res, RangeResult):
            context[res.question] = res.expected_value
        elif isinstance(res, ExtractResult):
            context[res.slot_name] = res.extracted_value

    def _run_cascade(self, cascade: Cascade, state: Any, effort: Optional[str], kw: Dict[str, Any]) -> CascadeResult:
        context: Dict[str, Any] = {}
        stage_results: List[Any] = []
        for stage in cascade.stages:
            res = self.evaluate(state, [stage], effort=effort, **kw)[0]
            stage_results.append(res)
            self._update_context(context, res)
        confs = [_clamp01(getattr(r, "confidence", 0.0)) for r in stage_results]
        report = cascade_confidence(confs) if confs else None
        temps = [r.temperature for r in stage_results if hasattr(r, "temperature")]
        conformals = [r.conformal_set for r in stage_results if getattr(r, "conformal_set", None) is not None]
        conformal = None
        if conformals:
            amb = any(c.get("is_ambiguous", False) for c in conformals)
            empty = any(c.get("is_empty", False) for c in conformals)
            conformal = {"is_ambiguous": amb, "is_singleton": not amb and not empty, "is_empty": empty,
                         "stage_conformal_sets": conformals}
        abstained = any(getattr(r, "should_abstain", False) for r in stage_results)
        reasons = [f"stage {i}: {r}" for i, s in enumerate(stage_results)
                   if getattr(s, "abstention_verdict", None) for r in s.abstention_verdict.reasons]
        verdict = AbstentionVerdict(decision=ABSTAIN if abstained else ANSWER, reasons=reasons)
        meta = {"backend": "rhen2", "confidence_label": report.label if report else None,
                "stage_confidences": confs}
        return CascadeResult(id=cascade.id, stages=stage_results, decisions=context,
                             confidence=report.value if report else 0.0,
                             temperature=sum(temps) / len(temps) if temps else 0.0,
                             meta_reasoning_context=meta, conformal_set=conformal, abstention_verdict=verdict)

    # ------------------------------------------------------------------ spec building
    def _spec(self, q: Primitive) -> QuestionSpec:
        cfg = self.model.cfg
        if isinstance(q, Noul):
            return QuestionSpec("noul", q.proposition)
        if isinstance(q, Choice):
            return QuestionSpec("choice", q.question, list(q.options))
        if isinstance(q, Score):
            if 2 <= len(q.levels) <= cfg.max_levels:
                return QuestionSpec("score", q.question, list(q.levels))
            return QuestionSpec("range", q.question, lo=float(q.min_score), hi=float(q.max_score))
        if isinstance(q, MultiChoice):
            return QuestionSpec("multi", q.question, list(q.options))
        if isinstance(q, Range):
            return QuestionSpec("range", q.question, lo=float(q.min_val), hi=float(q.max_val))
        if isinstance(q, Extract):
            return QuestionSpec("extract", f"{q.slot_name}: {q.query}" if q.query else q.slot_name)
        raise TypeError(f"Unsupported question specification: {type(q)}")

    # ------------------------------------------------------------------ result conversion
    def _temperature(self, qd: QuestionDecision) -> float:
        try:
            cal, _, _ = self.pipeline._cal_state()
            k = 2 if qd.spec.type == "noul" else max(len(qd.spec.options), 1)
            return float(cal.temperature(qd.spec.type, k))
        except Exception:
            return 1.0

    @staticmethod
    def _conformal_dict(qd: QuestionDecision, noul: bool) -> Optional[Dict[str, Any]]:
        cd = qd.checked
        if cd is None or cd.conformal_set is None:
            return None
        names = list(cd.conformal_set)
        labels = [("Yes" if n == "true" else "No") if noul else n for n in names if n != NULL]
        return {"prediction_set": labels, "set_size": len(names), "is_singleton": len(names) == 1,
                "is_ambiguous": len(names) > 1, "is_empty": len(names) == 0, "includes_null": NULL in names,
                "calibrated": bool(qd.calibrated), "source": "rhen2"}

    def _verdict(self, qd: QuestionDecision, null_p: float, top_p: Optional[float]) -> AbstentionVerdict:
        st = qd.status
        cd = qd.checked
        decision = ANSWER if st is PolicyStatus.EXECUTE else (
            NEED_CLARIFICATION if st is PolicyStatus.NEEDS_APPROVAL else ABSTAIN)
        reasons: List[str] = []
        if decision != ANSWER:
            reasons.append(f"policy status {st.value}")
            reasons.extend(qd.outcome.reasons)
            if cd is not None:
                reasons.extend(f"flag {f}" for f in cd.flags)
                reasons.append(f"checker verdict {cd.verdict}")
                if cd.answer is None:
                    reasons.append("null option is the top choice (cannot tell)")
            if not qd.calibrated:
                reasons.append("not calibrated")
            if qd.error:
                reasons.append(f"error {qd.error}")
        size = None if cd is None or cd.conformal_set is None else len(cd.conformal_set)
        return AbstentionVerdict(decision=decision, reasons=reasons, free_energy=None, conformal_size=size,
                                 max_probability=top_p)

    def _context(self, qd: QuestionDecision, res: PipelineResult, null_p: float, extra: Mapping[str, Any]) -> Dict:
        cd = qd.checked
        ctx = {"backend": "rhen2", "decision_id": qd.decision_id, "status": qd.status.value,
               "risk_tier": res.risk_tier, "effort": res.effort, "calibrated": bool(qd.calibrated),
               "untrained": self.pipeline.training_run_id is None, "null_probability": null_p,
               "verdict": None if cd is None else cd.verdict, "flags": [] if cd is None else list(cd.flags),
               "loops_run": None if cd is None else cd.loops_run, "routed": qd.routed,
               "policy": qd.outcome.to_dict()}
        if cd is not None:
            ctx["checker_extras"] = dict(cd.extras)
        ctx.update(extra)
        return ctx

    def _convert(self, q: Primitive, qd: QuestionDecision, res: PipelineResult) -> Any:
        cd = qd.checked
        probs: Dict[str, float] = {} if cd is None else {str(k): float(v) for k, v in cd.probabilities.items()}
        null_p = float(probs.get(NULL, 0.0 if cd is not None and cd.answer is not None else 1.0))
        conf = 0.0 if cd is None else _clamp01(cd.confidence)
        temp = self._temperature(qd)
        common = dict(confidence=conf, temperature=temp, risk_assessment=f"tier={res.risk_tier}; status={qd.status.value}",
                      free_energy=None)

        if isinstance(q, Noul):
            p_true = float(probs.get("true", 0.0))
            ctx = self._context(qd, res, null_p, {"p_false": float(probs.get("false", 0.0))})
            return NoulResult(proposition=q.proposition, p_true=p_true,
                              decision=bool(cd is not None and cd.answer is True),
                              meta_reasoning_context=ctx, conformal_set=self._conformal_dict(qd, True),
                              abstention_verdict=self._verdict(qd, null_p, max(p_true, 1.0 - p_true - null_p)),
                              **common)
        if isinstance(q, Choice):
            raw = [float(probs.get(o, 0.0)) for o in q.options]
            if cd is not None and cd.answer in q.options:
                idx = q.options.index(cd.answer)
            else:
                idx = max(range(len(raw)), key=lambda j: raw[j])
            alts = [{"option": o, "probability": raw[j]} for j, o in sorted(
                enumerate(q.options), key=lambda t: -raw[t[0]]) if j != idx]
            ctx = self._context(qd, res, null_p, {})
            return ChoiceResult(question=q.question, choice_index=idx, choice_label=q.options[idx],
                                probabilities=dict(zip(q.options, raw)), raw_probabilities=raw,
                                meta_reasoning_context=ctx, alternative_hypotheses=alts,
                                conformal_set=self._conformal_dict(qd, False),
                                abstention_verdict=self._verdict(qd, null_p, raw[idx]), **common)
        if isinstance(q, Score):
            via_levels = qd.spec.type == "score"
            span = float(q.max_score) - float(q.min_score)
            if via_levels:
                raw = [float(probs.get(o, 0.0)) for o in q.levels]
                exp_level = float((cd.extras.get("expected_level", 0.0)) if cd is not None else 0.0)
                score = float(q.min_score) + exp_level / max(len(q.levels) - 1, 1) * span
                ctx = self._context(qd, res, null_p, {"score_via": "levels", "expected_level": exp_level})
                top = max(raw) if raw else None
                conformal = self._conformal_dict(qd, False)
            else:
                raw = []
                score = float(cd.answer) if cd is not None and cd.answer is not None else float(q.min_score)
                score = min(max(score, float(q.min_score)), float(q.max_score))
                ctx = self._context(qd, res, null_p, {"score_via": "range"})
                top, conformal = None, None
            return ScoreResult(question=q.question, score=score, min_score=float(q.min_score),
                               max_score=float(q.max_score), level_probabilities=raw, meta_reasoning_context=ctx,
                               conformal_set=conformal, abstention_verdict=self._verdict(qd, null_p, top), **common)
        if isinstance(q, MultiChoice):
            raw = [float(probs.get(o, 0.0)) for o in q.options]
            sel = [j for j, p in enumerate(raw) if p > q.threshold]
            ctx = self._context(qd, res, null_p, {})
            return MultiChoiceResult(question=q.question, selected_indices=sel,
                                     selected_options=[q.options[j] for j in sel],
                                     probabilities=dict(zip(q.options, raw)), raw_probabilities=raw,
                                     threshold=q.threshold, meta_reasoning_context=ctx,
                                     abstention_verdict=self._verdict(qd, null_p, max(raw) if raw else None),
                                     **common)
        if isinstance(q, Range):
            lo_b, hi_b = float(q.min_val), float(q.max_val)

            def clamp(v: float) -> float:
                return min(max(float(v), lo_b), hi_b)

            ev = clamp(cd.answer) if cd is not None and cd.answer is not None else lo_b
            lo, hi = (clamp(x) for x in (cd.extras.get("interval", (ev, ev)) if cd is not None else (ev, ev)))
            ctx = self._context(qd, res, null_p, {"clamped_to": [lo_b, hi_b]})
            return RangeResult(question=q.question, expected_value=ev, interval_low=min(lo, ev),
                               interval_high=max(hi, ev), confidence_interval=0.8, min_val=lo_b, max_val=hi_b,
                               meta_reasoning_context=ctx, abstention_verdict=self._verdict(qd, null_p, None),
                               **common)
        if isinstance(q, Extract):
            span = cd.extras.get("span_chars") if cd is not None else None
            start, end = (int(span[0]), int(span[1])) if span else (0, 0)
            ctx = self._context(qd, res, null_p, {})
            return ExtractResult(slot_name=q.slot_name,
                                 extracted_value="" if cd is None or cd.answer is None else str(cd.answer),
                                 start_idx=start, end_idx=end, meta_reasoning_context=ctx,
                                 abstention_verdict=self._verdict(qd, null_p, probs.get("__span__")), **common)
        raise TypeError(f"Unsupported question specification: {type(q)}")


__all__ = ["V2Backend"]
