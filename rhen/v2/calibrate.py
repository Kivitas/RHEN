"""RHEN-2 calibration layer (plan section 8, tasks T-501; drift pieces of 9.6).

Everything here is post-hoc and numpy only (no scipy).  Records are dicts
``{type, k, logits (K+1 floats, null LAST), label_idx, q_vec (optional)}``;
``label_idx`` follows the packer (>=0 option, NULL_INDEX=-2 null, IGNORE=-1 skipped).

Design decisions / deviations from the plan (all documented, none invented numbers):
* Dataclass field names that would clash with the query methods carry a ``_table``
  suffix (``temperature_table``, ``isotonic_table``, ``conformal_table``,
  ``tau_act_table``); the JSON keys are the plan names.  k buckets are stored as
  string keys; the type-level fallback entry uses the key ``"all"``.
* One extra dict ``extras`` holds what drift detection and the Mahalanobis signal need:
  ``alpha``, ``ece[type][kb]`` (offline ECE), ``ood_rate`` (flag rate on the calibration
  data), ``ood_stats[type]`` (q_vec mean and precision matrix).
* Clopper-Pearson bound is ONE-SIDED 95 % (0 errors of 100 -> 0.0295).
* tau_act search is evaluated on at most ``max_candidates`` thresholds (quantile grid of
  the acted set) so the exact binomial bound stays cheap; the smallest qualifying one wins.
* OnlineRecalibrator: temperature may move within [T_off/2, 2*T_off] (larger T is the
  cautious direction; leaving the band clamps and alarms); the conformal quantile may drop
  freely and rise to 2*q_off.  Rolling-ECE alarm uses the offline temperature on the window
  and compares with ``max(2*ece_offline, noise_floor)``; the noise floor is the 99th percentile
  of the ECE of a perfectly calibrated model on a window of that size (parametric bootstrap),
  because equal-mass ECE of a few hundred points is ~0.03-0.06 even when calibrated.  A miss streak of 3 fires
  about once per 1/alpha^3 outcomes on stationary data, so use a small alpha when online.
* ``tau_review``, ``delta_ev`` and ``halt_threshold`` are not fitted here (no fitting rule in
  the plan); ``fit_calibration`` stores whatever the caller passes (default None).
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rhen.v2.config import QUESTION_TYPES
from rhen.v2.packing import IGNORE, NULL_INDEX, k_bucket

MIN_TEMP_N = 500
MIN_CONFORMAL_N = 100
DEFAULT_TOLERANCES = {"low": 0.10, "medium": 0.05, "high": 0.01, "critical": 0.001}
OOD_SIGNALS = ("energy", "p_null", "mahalanobis")


class CalibrationMismatch(RuntimeError):
    pass


# --------------------------------------------------------------------------- metrics
def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def expected_calibration_error(probs, labels, bins: int = 15) -> float:
    """Equal-mass ECE of the top-1 prediction.  probs [n,K] with integer labels [n], or
    probs [n] confidences with labels [n] in {0,1} correctness."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels)
    if p.ndim == 2:
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == y).astype(np.float64)
    else:
        conf, correct = p, y.astype(np.float64)
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(conf, kind="stable")
    ece = 0.0
    for idx in np.array_split(order, min(bins, n)):
        if len(idx):
            ece += len(idx) / n * abs(conf[idx].mean() - correct[idx].mean())
    return float(ece)


def brier(probs, labels) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def nll(probs, labels) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


# --------------------------------------------------------------------------- Clopper-Pearson
_LF: List[float] = [0.0]


def _logfact(n: int) -> np.ndarray:
    while len(_LF) <= n:
        _LF.append(math.lgamma(len(_LF) + 1))
    return np.asarray(_LF[: n + 1])


def _binom_cdf(e: int, n: int, p: float) -> float:
    """P(X <= e), X ~ Binomial(n, p), via lgamma-based log pmf."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if e >= n else 0.0
    lf = _logfact(n)
    i = np.arange(e + 1)
    logpmf = lf[n] - lf[i] - lf[n - i] + i * math.log(p) + (n - i) * math.log1p(-p)
    m = logpmf.max()
    return float(min(1.0, math.exp(m) * np.exp(logpmf - m).sum()))


def clopper_pearson_upper(errors: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) binomial upper bound at ``confidence`` (default 95 %,
    i.e. alpha = 0.05 in ONE tail, not a two-sided 95 % interval), found by bisection on the
    lgamma-based binomial cdf.  0 errors of 100 -> 0.0295; 5 of 100 -> 0.1023."""
    if n <= 0 or errors >= n:
        return 1.0
    target = 1.0 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(errors, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------- record helpers
def _prep(records: Iterable[dict]) -> List[dict]:
    out = []
    for r in records:
        if "label" in r:  # already prepared
            out.append(r)
            continue
        lg = np.asarray(r["logits"], dtype=np.float64)
        k = len(lg) - 1
        lab = int(r["label_idx"])
        if lab == IGNORE:
            continue
        if lab == NULL_INDEX:
            lab = k
        qv = r.get("q_vec")
        out.append({"type": r["type"], "k": k, "kb": str(k_bucket(k)), "logits": lg, "label": lab,
                    "q_vec": None if qv is None else np.asarray(qv, dtype=np.float64)})
    return out


def _matrix(recs: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    w = max(r["k"] for r in recs)
    z = np.full((len(recs), w + 1), -1e30)
    y = np.zeros(len(recs), dtype=np.int64)
    for i, r in enumerate(recs):
        k = r["k"]
        z[i, :k] = r["logits"][:k]
        z[i, w] = r["logits"][k]
        y[i] = w if r["label"] == k else r["label"]
    return z, y


def _nll_at(z: np.ndarray, y: np.ndarray, t: float) -> float:
    s = z / t
    s = s - s.max(axis=1, keepdims=True)
    lse = np.log(np.exp(s).sum(axis=1))
    return float((lse - s[np.arange(len(y)), y]).mean())


def _fit_t(z: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Golden-section search on log T (NLL is unimodal in log T)."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _nll_at(z, y, math.exp(c)), _nll_at(z, y, math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _nll_at(z, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _nll_at(z, y, math.exp(d))
    return float(math.exp(0.5 * (a + b)))


def _groups(recs: Sequence[dict]) -> Dict[Tuple[str, str], List[dict]]:
    g: Dict[Tuple[str, str], List[dict]] = {}
    for r in recs:
        g.setdefault((r["type"], r["kb"]), []).append(r)
    return g


def _by_type(recs: Sequence[dict]) -> Dict[str, List[dict]]:
    g: Dict[str, List[dict]] = {}
    for r in recs:
        g.setdefault(r["type"], []).append(r)
    return g


def _lookup(table: dict, n_table: dict, qtype: str, kb: str, min_n: int = MIN_TEMP_N) -> float:
    t = table.get(qtype, {})
    if n_table.get(qtype, {}).get(kb, 0) >= min_n and kb in t:
        return float(t[kb])
    if "all" in t:
        return float(t["all"])
    return 1.0


# --------------------------------------------------------------------------- fitting
def fit_temperatures(records: Iterable[dict], min_group: int = 10) -> Tuple[dict, dict]:
    """Returns (temperature[type][kb|'all'], n_calibration[type][kb|'all'])."""
    recs = _prep(records)
    temps: Dict[str, Dict[str, float]] = {}
    ns: Dict[str, Dict[str, int]] = {}
    for qtype, rs in _by_type(recs).items():
        ns[qtype] = {"all": len(rs)}
        temps[qtype] = {}
        if len(rs) >= min_group:
            temps[qtype]["all"] = _fit_t(*_matrix(rs))
    for (qtype, kb), rs in _groups(recs).items():
        ns[qtype][kb] = len(rs)
        if len(rs) >= min_group:
            temps[qtype][kb] = _fit_t(*_matrix(rs))
    return temps, ns


def _p_correct_arrays(recs: Sequence[dict], temps: dict, ns: dict) -> Tuple[np.ndarray, np.ndarray]:
    conf = np.zeros(len(recs))
    corr = np.zeros(len(recs))
    for i, r in enumerate(recs):
        t = _lookup(temps, ns, r["type"], r["kb"])
        p = _softmax(r["logits"] / t)
        conf[i] = p.max()
        corr[i] = float(p.argmax() == r["label"])
    return conf, corr


def _pav(conf: np.ndarray, correct: np.ndarray) -> Dict[str, List[float]]:
    order = np.argsort(conf, kind="stable")
    x, y = conf[order], correct[order].astype(np.float64)
    blocks: List[List[float]] = []  # [sum, weight, max_x]
    for xi, yi in zip(x, y):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            s, w, mx = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += w
            blocks[-1][2] = mx
    return {"x": [b[2] for b in blocks], "y": [b[0] / b[1] for b in blocks]}


def fit_isotonic(records: Iterable[dict], temps: dict, ns: dict) -> dict:
    """Per type: monotone step map from top-1 calibrated confidence to P(correct) (pure PAV)."""
    recs = _prep(records)
    out = {}
    for qtype, rs in _by_type(recs).items():
        conf, corr = _p_correct_arrays(rs, temps, ns)
        out[qtype] = _pav(conf, corr)
    return out


def _isotonic_apply(table: Optional[dict], p: float) -> float:
    if not table:
        return float(p)
    x, y = table["x"], table["y"]
    i = int(np.searchsorted(x, p, side="left"))
    return float(y[min(i, len(y) - 1)])


def _conf_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return 1.0  # cannot certify: every label enters the set
    return float(np.sort(scores)[k - 1])


def fit_group_conformal(records: Iterable[dict], temps: dict, ns: dict, alpha: float = 0.1,
                        min_n: int = MIN_CONFORMAL_N) -> Dict[str, Dict[str, Optional[float]]]:
    recs = _prep(records)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for (qtype, kb), rs in _groups(recs).items():
        if len(rs) < min_n:
            out.setdefault(qtype, {})[kb] = None
            continue
        t = _lookup(temps, ns, qtype, kb)
        sc = np.array([1.0 - _softmax(r["logits"] / t)[r["label"]] for r in rs])
        out.setdefault(qtype, {})[kb] = _conf_quantile(sc, alpha)
    return out


def fit_tau_act(p_correct, correct, tolerance: float, confidence: float = 0.95,
                max_candidates: int = 400) -> Optional[float]:
    """Smallest threshold t on p_correct such that the Clopper-Pearson upper bound of the
    error rate among decisions with p_correct >= t is <= tolerance.  None if none qualifies."""
    p = np.asarray(p_correct, dtype=np.float64)
    c = np.asarray(correct, dtype=np.float64)
    if len(p) == 0:
        return None
    cands = np.unique(p)
    if len(cands) > max_candidates:
        cands = np.unique(np.quantile(p, np.linspace(0, 1, max_candidates)))
    order = np.argsort(-p, kind="stable")
    ps, errs = p[order], np.cumsum(1.0 - c[order])
    best = None
    for t in cands[::-1]:  # high to low; keep the smallest qualifying
        n = int(np.searchsorted(-ps, -t, side="right"))
        if n == 0:
            continue
        e = int(round(errs[n - 1]))
        if e / n > tolerance:
            continue
        if clopper_pearson_upper(e, n, confidence) <= tolerance:
            best = float(t)
    return best


def ood_signals(logits, temperature: float) -> Tuple[float, float]:
    """(normalised energy, P(null)); logits include null last.  E = -T*logsumexp(options/T)."""
    lg = np.asarray(logits, dtype=np.float64)
    k = len(lg) - 1
    s = lg[:k] / temperature
    m = s.max()
    energy = -temperature * (m + math.log(np.exp(s - m).sum())) + temperature * math.log(k + 1)
    return float(energy), float(_softmax(lg / temperature)[k])


def _maha_stats(qs: np.ndarray, ridge: float = 1e-3) -> dict:
    mu = qs.mean(axis=0)
    xc = qs - mu
    cov = xc.T @ xc / max(len(qs) - 1, 1)
    cov = cov + (ridge * np.trace(cov) / cov.shape[0] + 1e-8) * np.eye(cov.shape[0])
    return {"mean": mu.tolist(), "precision": np.linalg.inv(cov).tolist()}


def _maha(stats: dict, q) -> float:
    d = np.asarray(q, dtype=np.float64) - np.asarray(stats["mean"])
    return float(d @ np.asarray(stats["precision"]) @ d)


def fit_ood(records: Iterable[dict], temps: dict, ns: dict, quantile: float = 0.99,
            min_n: int = MIN_CONFORMAL_N) -> Tuple[dict, dict]:
    """Returns (ood[signal][type][kb|'all'] thresholds, ood_stats[type] Mahalanobis stats)."""
    recs = _prep(records)
    ood: Dict[str, Dict[str, Dict[str, float]]] = {s: {} for s in OOD_SIGNALS}
    stats: Dict[str, dict] = {}
    for qtype, rs in _by_type(recs).items():
        qs = [r["q_vec"] for r in rs if r["q_vec"] is not None]
        if len(qs) >= min_n:
            stats[qtype] = _maha_stats(np.stack(qs))
        subsets = [("all", rs)] + [(kb, g) for (t, kb), g in _groups(rs).items()]
        for kb, g in subsets:
            if len(g) < min_n:
                continue
            e, pn, md = [], [], []
            for r in g:
                t = _lookup(temps, ns, qtype, r["kb"])
                a, b = ood_signals(r["logits"], t)
                e.append(a)
                pn.append(b)
                if qtype in stats and r["q_vec"] is not None:
                    md.append(_maha(stats[qtype], r["q_vec"]))
            ood["energy"].setdefault(qtype, {})[kb] = float(np.quantile(e, quantile))
            ood["p_null"].setdefault(qtype, {})[kb] = float(np.quantile(pn, quantile))
            if md:
                ood["mahalanobis"].setdefault(qtype, {})[kb] = float(np.quantile(md, quantile))
    return ood, stats


# --------------------------------------------------------------------------- Calibration
@dataclass
class Calibration:
    model_hash: str = ""
    data_hash: str = ""
    created: str = ""
    temperature_table: Dict[str, Dict[str, float]] = field(default_factory=dict)
    isotonic_table: Dict[str, dict] = field(default_factory=dict)
    conformal_table: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    tau_act_table: Dict[str, Optional[float]] = field(default_factory=dict)
    tau_review: Optional[float] = None
    halt_threshold: Optional[float] = None
    delta_ev: Optional[float] = None
    ood: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    n_calibration: Dict[str, Dict[str, int]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    _KEYS = {"temperature_table": "temperature", "isotonic_table": "isotonic",
             "conformal_table": "conformal_q", "tau_act_table": "tau_act"}

    # ---- io
    def to_dict(self) -> dict:
        d = {self._KEYS.get(k, k): getattr(self, k) for k in self.__dataclass_fields__}
        return d

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        inv = {v: k for k, v in cls._KEYS.items()}
        return cls(**{inv.get(k, k): v for k, v in d.items()})

    @classmethod
    def load(cls, path, expected_model_hash: Optional[str] = None) -> "Calibration":
        with open(path) as f:
            cal = cls.from_dict(json.load(f))
        if expected_model_hash is not None and cal.model_hash != expected_model_hash:
            raise CalibrationMismatch(
                f"calibration model_hash {cal.model_hash!r} != checkpoint {expected_model_hash!r}")
        return cal

    # ---- queries
    def n(self, qtype: str, k: int) -> int:
        return int(self.n_calibration.get(qtype, {}).get(str(k_bucket(k)), 0))

    def temperature(self, qtype: str, k: int) -> float:
        return _lookup(self.temperature_table, self.n_calibration, qtype, str(k_bucket(k)))

    def conformal_q(self, qtype: str, k: int) -> Optional[float]:
        if self.n(qtype, k) < MIN_CONFORMAL_N:
            return None
        return self.conformal_table.get(qtype, {}).get(str(k_bucket(k)))

    def tau_act(self, tier: str) -> Optional[float]:
        return self.tau_act_table.get(tier)

    def isotonic(self, qtype: str, p: float) -> float:
        return _isotonic_apply(self.isotonic_table.get(qtype), p)

    def apply(self, qtype: str, k: int, logits) -> np.ndarray:
        return _softmax(np.asarray(logits, dtype=np.float64) / self.temperature(qtype, k))

    def prediction_set(self, qtype: str, k: int, probs) -> Optional[List[int]]:
        q = self.conformal_q(qtype, k)
        if q is None:
            return None
        p = np.asarray(probs, dtype=np.float64)
        s = [int(i) for i in np.nonzero(1.0 - p <= q)[0]]
        return s or [int(p.argmax())]

    def ood_threshold(self, signal: str, qtype: str, k: int) -> Optional[float]:
        t = self.ood.get(signal, {}).get(qtype, {})
        kb = str(k_bucket(k))
        if kb in t:
            return float(t[kb])
        return float(t["all"]) if "all" in t else None

    def ood_flags(self, qtype: str, k: int, energy: float, p_null: float, q_vec=None) -> List[str]:
        hit = []
        vals = {"energy": energy, "p_null": p_null}
        stats = self.extras.get("ood_stats", {}).get(qtype)
        if q_vec is not None and stats is not None:
            vals["mahalanobis"] = _maha(stats, q_vec)
        for s in OOD_SIGNALS:
            thr = self.ood_threshold(s, qtype, k)
            if thr is not None and s in vals and vals[s] > thr:
                hit.append(s)
        return (["OOD"] + hit) if hit else []

    def group_calibrated(self, qtype: str, k: int) -> bool:
        return (self.n(qtype, k) >= MIN_CONFORMAL_N and self.conformal_q(qtype, k) is not None
                and qtype in self.temperature_table)

    @property
    def calibrated(self) -> bool:
        if not (self.temperature_table and self.isotonic_table and self.conformal_table
                and self.ood and self.n_calibration):
            return False
        return any(n >= MIN_CONFORMAL_N for t in self.n_calibration.values()
                   for kb, n in t.items() if kb != "all")


def fit_calibration(records: Iterable[dict], model_hash: str = "", data_hash: str = "",
                    alpha: float = 0.1, tolerances: Optional[Dict[str, float]] = None,
                    confidence: float = 0.95, tau_review: Optional[float] = None,
                    halt_threshold: Optional[float] = None,
                    delta_ev: Optional[float] = None) -> Calibration:
    recs = _prep(records)
    tol = dict(DEFAULT_TOLERANCES, **(tolerances or {}))
    temps, ns = fit_temperatures(recs)
    iso = fit_isotonic(recs, temps, ns)
    conf_q = fit_group_conformal(recs, temps, ns, alpha)
    ood, stats = fit_ood(recs, temps, ns)
    # p_correct for risk control: isotonic of the calibrated top-1 confidence
    pc, corr = [], []
    for r in recs:
        c, y = _p_correct_arrays([r], temps, ns)
        pc.append(_isotonic_apply(iso.get(r["type"]), c[0]))
        corr.append(y[0])
    tau = {tier: fit_tau_act(pc, corr, t, confidence) for tier, t in tol.items()}
    # offline ECE per group and OOD flag rate on the calibration data
    ece: Dict[str, Dict[str, float]] = {}
    for (qtype, kb), rs in _groups(recs).items():
        t = _lookup(temps, ns, qtype, kb)
        z, y = _matrix(rs)
        ece.setdefault(qtype, {})[kb] = expected_calibration_error(_softmax(z / t), y)
    cal = Calibration(model_hash=model_hash, data_hash=data_hash,
                      created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      temperature_table=temps, isotonic_table=iso, conformal_table=conf_q,
                      tau_act_table=tau, tau_review=tau_review, halt_threshold=halt_threshold,
                      delta_ev=delta_ev, ood=ood, n_calibration=ns,
                      extras={"alpha": alpha, "ece": ece, "ood_stats": stats})
    flagged = 0
    for r in recs:
        e, pn = ood_signals(r["logits"], cal.temperature(r["type"], r["k"]))
        flagged += bool(cal.ood_flags(r["type"], r["k"], e, pn, r["q_vec"]))
    cal.extras["ood_rate"] = flagged / max(len(recs), 1)
    return cal


def collect_records(model, batches: Iterable, plan=None,
                    types: Sequence[str] = ("choice", "noul")) -> List[dict]:
    """Run the model without grad and build calibration records.  Unlabelled and padded
    questions are skipped; only ``types`` are kept (their logits are softmax-style)."""
    import torch

    was_training = model.training
    model.eval()
    recs: List[dict] = []
    try:
        with torch.no_grad():
            for batch in batches:
                if batch.label_idx is None:
                    continue
                out = model(batch, plan)
                logits = out.final.logits.float()
                kmax = logits.shape[1] - 1
                qt = batch.q_type.reshape(-1)
                qvalid = batch.q_valid.reshape(-1)
                lab = batch.label_idx.reshape(-1)
                ov = batch.opt_valid.reshape(-1, batch.opt_valid.shape[-1])
                for n in range(logits.shape[0]):
                    if not bool(qvalid[n]) or int(lab[n]) == IGNORE:
                        continue
                    tname = QUESTION_TYPES[int(qt[n])]
                    if tname not in types:
                        continue
                    valid = ov[n]
                    opts = logits[n, :kmax][valid]
                    k = int(valid.sum())
                    li = int(lab[n])
                    if li >= k:
                        continue
                    rec = {"type": tname, "k": k,
                           "logits": opts.tolist() + [float(logits[n, kmax])],
                           "label_idx": li, "q_vec": out.q_vec[n].float().tolist()}
                    recs.append(rec)
    finally:
        model.train(was_training)
    return recs


# --------------------------------------------------------------------------- online drift
class OnlineRecalibrator:
    """Plan 9.6.  observe() one labelled outcome at a time; returns alarms raised by it."""

    def __init__(self, cal: Calibration, window: int = 2000, refit_every: int = 500,
                 miss_streak: int = 3, ece_boot: int = 100, min_window: int = 200,
                 ece_min_window: int = 1000):
        self.cal = cal
        self.window, self.refit_every = window, refit_every
        self.miss_streak, self.ece_boot, self.min_window = miss_streak, ece_boot, min_window
        self.ece_min_window = ece_min_window  # ECE of fewer points is too noisy to alarm on
        self._buf: Dict[Tuple[str, str], deque] = {}
        self._flags: Dict[Tuple[str, str], deque] = {}
        self._since: Dict[Tuple[str, str], int] = {}
        self._streak: Dict[Tuple[str, str], int] = {}
        self._T: Dict[Tuple[str, str], float] = {}
        self._Q: Dict[Tuple[str, str], Optional[float]] = {}
        self.alarms: Dict[Tuple[str, str], List[str]] = {}

    def _key(self, qtype: str, k: int) -> Tuple[str, str]:
        return (qtype, str(k_bucket(k)))

    def temperature(self, qtype: str, k: int) -> float:
        return self._T.get(self._key(qtype, k), self.cal.temperature(qtype, k))

    def conformal_q(self, qtype: str, k: int) -> Optional[float]:
        key = self._key(qtype, k)
        return self._Q[key] if key in self._Q else self.cal.conformal_q(qtype, k)

    def group_calibrated(self, qtype: str, k: int) -> bool:
        return self._key(qtype, k) not in self.alarms and self.cal.group_calibrated(qtype, k)

    def reset_alarm(self, qtype: str, k: int) -> None:
        self.alarms.pop(self._key(qtype, k), None)
        self._streak[self._key(qtype, k)] = 0

    def _alarm(self, key, reason: str, out: List[str]) -> None:
        lst = self.alarms.setdefault(key, [])
        if reason not in lst:
            lst.append(reason)
        out.append(reason)

    def observe(self, qtype: str, k: int, logits, label_idx: int, ood_flag: bool = False) -> List[str]:
        key = self._key(qtype, k)
        lg = np.asarray(logits, dtype=np.float64)
        lab = k if label_idx == NULL_INDEX else int(label_idx)
        raised: List[str] = []
        buf = self._buf.setdefault(key, deque(maxlen=self.window))
        flg = self._flags.setdefault(key, deque(maxlen=self.window))
        buf.append((lg, lab))
        flg.append(bool(ood_flag))
        # adaptive conformal miss streak with the currently deployed set
        q = self.conformal_q(qtype, k)
        if q is not None:
            p = _softmax(lg / self.temperature(qtype, k))
            miss = (1.0 - p[lab]) > q
            self._streak[key] = self._streak.get(key, 0) + 1 if miss else 0
            if self._streak[key] >= self.miss_streak:
                self._alarm(key, "miss_streak", raised)
        # OOD flag rate
        base = float(self.cal.extras.get("ood_rate", 0.0))
        if len(flg) >= self.min_window and sum(flg) / len(flg) > 3.0 * max(base, 1e-3):
            self._alarm(key, "ood_rate", raised)
        self._since[key] = self._since.get(key, 0) + 1
        if self._since[key] >= self.refit_every:
            self._since[key] = 0
            self._refit(key, raised)
        return raised

    def _refit(self, key: Tuple[str, str], raised: List[str]) -> None:
        qtype, kb = key
        recs = [{"type": qtype, "k": len(lg) - 1, "kb": kb, "logits": lg, "label": lab, "q_vec": None}
                for lg, lab in self._buf[key]]
        if len(recs) < self.min_window:
            return
        k0 = recs[0]["k"]
        t_off = self.cal.temperature(qtype, k0)
        # rolling ECE of the OFFLINE calibration on the window
        z, y = _matrix(recs)
        probs = _softmax(z / t_off)
        ece = expected_calibration_error(probs, y)
        ece_off = float(self.cal.extras.get("ece", {}).get(qtype, {}).get(kb, 0.0))
        # sampling-noise floor: 99th percentile of the ECE of a PERFECTLY calibrated model on a
        # window of this size (labels redrawn from the model's own probabilities)
        rng = np.random.RandomState(len(self._buf[key]))
        conf = probs.max(axis=1)
        boots = [expected_calibration_error(conf, (rng.rand(len(conf)) < conf).astype(int))
                 for _ in range(self.ece_boot)]
        floor = float(np.quantile(boots, 0.99))
        if len(recs) >= self.ece_min_window and ece > max(2.0 * ece_off, floor):
            self._alarm(key, "ece", raised)
        # temperature refit, clamped to [T_off/2, 2 T_off]
        t_new = _fit_t(z, y)
        t_cl = min(max(t_new, t_off / 2.0), 2.0 * t_off)
        if t_cl != t_new:
            self._alarm(key, "temperature_drift", raised)
        self._T[key] = t_cl
        # conformal quantile refit at the offline alpha; tighten freely, loosen up to 2x
        q_off = self.cal.conformal_q(qtype, k0)
        if q_off is not None and len(recs) >= MIN_CONFORMAL_N:
            alpha = float(self.cal.extras.get("alpha", 0.1))
            sc = np.array([1.0 - _softmax(r["logits"] / t_cl)[r["label"]] for r in recs])
            q_new = _conf_quantile(sc, alpha)
            if q_new > 2.0 * q_off:
                self._alarm(key, "conformal_drift", raised)
            self._Q[key] = min(q_new, 2.0 * q_off)
