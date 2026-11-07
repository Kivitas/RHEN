"""Post-hoc calibration for Rhen decisions: temperature scaling and split conformal prediction.

Both are fitted on a held-out calibration split (never on the test split) and saved as a
small JSON artifact next to the checkpoint, so every reported number names the calibration
it used. Temperature scaling rescales log-probabilities (``softmax(log p / T)``), which keeps
the argmax and therefore accuracy unchanged. The conformal threshold gives prediction sets
with marginal coverage of at least ``1 - alpha`` on exchangeable data.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


def _pad_log_probs(prob_lists: Sequence[Sequence[float]]) -> torch.Tensor:
    width = max(len(p) for p in prob_lists)
    out = torch.full((len(prob_lists), width), float("-inf"), dtype=torch.float64)
    for i, probs in enumerate(prob_lists):
        p = torch.tensor([max(float(x), 1e-12) for x in probs], dtype=torch.float64)
        out[i, : len(probs)] = torch.log(p / p.sum())
    return out


def apply_temperature(probs: Sequence[float], temperature: float) -> List[float]:
    if temperature == 1.0:
        return [float(p) for p in probs]
    logp = _pad_log_probs([probs])[0]
    return torch.softmax(logp / temperature, dim=-1).tolist()


MIN_TEMPERATURE, MAX_TEMPERATURE = 0.05, 20.0


def fit_temperature(prob_lists: Sequence[Sequence[float]], labels: Sequence[int], iterations: int = 80) -> float:
    """The temperature in [0.05, 20] that minimises the NLL of ``labels``.

    The NLL of ``softmax(beta * log p)`` is convex in the inverse temperature ``beta`` (a
    log-sum-exp minus a linear term), so a golden-section search over ``beta`` finds the
    minimum deterministically, including at a boundary (a model whose ranking carries no
    information drives the optimum to the hottest temperature allowed).
    """
    if not prob_lists:
        raise ValueError("cannot fit a temperature on an empty calibration set")
    logp = _pad_log_probs(prob_lists)
    y = torch.tensor([int(v) for v in labels], dtype=torch.long)

    def nll(beta: float) -> float:
        return float(torch.nn.functional.cross_entropy(logp * beta, y).item())

    lo, hi = 1.0 / MAX_TEMPERATURE, 1.0 / MIN_TEMPERATURE
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(iterations):
        if fa <= fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(b)
    candidates = [(nll(beta), beta) for beta in (lo, (lo + hi) / 2.0, hi, 1.0 / MAX_TEMPERATURE, 1.0 / MIN_TEMPERATURE)]
    return 1.0 / min(candidates)[1]


def fit_conformal_threshold(prob_lists: Sequence[Sequence[float]], labels: Sequence[int], alpha: float) -> float:
    """Split-conformal ``qhat`` for the score ``1 - p(true)``, with the finite-sample correction."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = sorted(1.0 - float(p[int(y)]) for p, y in zip(prob_lists, labels))
    n = len(scores)
    if n == 0:
        raise ValueError("cannot fit a conformal threshold on an empty calibration set")
    k = math.ceil((n + 1) * (1.0 - alpha))
    return 1.0 if k > n else scores[k - 1]


@dataclass
class DecisionCalibration:
    temperature: float = 1.0
    conformal_alpha: Optional[float] = None
    conformal_qhat: Optional[float] = None
    n_calibration: int = 0
    fitted_on: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def apply(self, probs: Sequence[float]) -> List[float]:
        return apply_temperature(probs, self.temperature)

    def prediction_set(self, probs: Sequence[float]) -> Optional[List[int]]:
        if self.conformal_qhat is None:
            return None
        calibrated = self.apply(probs)
        return [k for k, p in enumerate(calibrated) if 1.0 - p <= self.conformal_qhat]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "DecisionCalibration":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


def fit_decision_calibration(
    records: Sequence[Dict[str, Any]], alpha: Optional[float] = 0.1, fitted_on: str = ""
) -> DecisionCalibration:
    """Fit temperature (then conformal on the temperature-scaled probabilities) on single-label records."""
    usable = [r for r in records if r.get("label_index") is not None]
    probs = [r["probs"] for r in usable]
    labels = [int(r["label_index"]) for r in usable]
    temperature = fit_temperature(probs, labels)
    scaled = [apply_temperature(p, temperature) for p in probs]
    qhat = fit_conformal_threshold(scaled, labels, alpha) if alpha is not None else None
    return DecisionCalibration(
        temperature=temperature,
        conformal_alpha=alpha,
        conformal_qhat=qhat,
        n_calibration=len(usable),
        fitted_on=fitted_on,
    )


def calibrate_records(records: Sequence[Dict[str, Any]], calibration: DecisionCalibration) -> List[Dict[str, Any]]:
    """Copies of single-label records with temperature-scaled probabilities."""
    out = []
    for r in records:
        r = dict(r)
        if r.get("label_index") is not None or r.get("primitive") in ("noul", "choice", "score"):
            r["probs"] = calibration.apply(r["probs"])
        out.append(r)
    return out
