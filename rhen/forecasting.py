"""
Predictive layer for RHEN: forecast containers, forecasters and forecast error metrics.

Module-level imports are standard library and NumPy only. TimesFM is imported lazily,
inside ``TimesFMForecaster.__init__``, so the baseline path never needs torch or timesfm.
(Importing this module as ``rhen.forecasting`` still runs ``rhen/__init__.py``, which
imports the torch-backed decision engine; this file itself has no torch import.)

Forecasts carry nine quantiles at levels 0.1, 0.2, ..., 0.9 per horizon step. Interval
and exceedance queries are answered from that quantile function by linear interpolation.

Licensing: the TimesFM 3.0 *source* is Apache-2.0, but the 3.0 pretrained weights are
released under ``timesfm-non-commercial-license-v1.0`` (non-commercial, non-production
use only). ``TimesFMForecaster`` therefore refuses to load them unless the caller passes
``license_acknowledged=True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

#: Quantile levels stored in every forecast row, in order.
QUANTILE_LEVELS: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: Index of the 0.5 level in ``QUANTILE_LEVELS`` (the median).
MEDIAN_INDEX: int = 4

_LEVELS_ARR = np.asarray(QUANTILE_LEVELS, dtype=np.float64)
_TIMESFM_INSTALL_HINT = 'pip install "timesfm[torch]>=3.0.2"'
_TIMESFM_LICENSE_MESSAGE = (
    "TimesFM 3.0 pretrained weights (google/timesfm-3.0-pytorch) are released under the "
    "timesfm-non-commercial-license-v1.0, which permits only non-commercial, non-production use. "
    "RHEN is MIT-licensed and is documented for production use, so TimesFMForecaster refuses to "
    "load these weights unless license_acknowledged=True is passed explicitly. Pass it only if "
    "your use of the weights is non-commercial and non-production."
)


def _check_horizon(horizon: Any) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError(f"horizon must be an integer, got {type(horizon).__name__}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return int(horizon)


def _as_series(values: Any, min_len: int, label: str = "series") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional sequence of numbers")
    if arr.size < min_len:
        raise ValueError(f"{label} needs at least {min_len} observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values")
    return arr


def _exceed_at(row: np.ndarray, threshold: float) -> float:
    """P(value > threshold) for one step, from its nine quantiles.

    The CDF F(v) = P(value <= v) is the right-continuous piecewise-linear interpolation
    through the points (q_k, level_k). Ties are resolved to the highest tied level, so a
    threshold that equals a tied quantile takes the CDF value at the top of the tie. Cases:

    - Threshold equal to a tied lower group. Row [5, 5, 5, 6, 7, 8, 9, 10, 11] with threshold
      5 gives F(5) = 0.3, the highest tied level, so P(value > 5) = 0.70, not 0.90.
    - Threshold inside an interior tie, such as row [1, 2, 3, 3, 3, 4, 5, 6, 7] with threshold
      3: F(3) = 0.5, so P(value > 3) = 0.50.
    - Threshold equal to a tied upper group. Row [1, 2, 3, 4, 5, 6, 9, 9, 9] with threshold 9:
      F(9) = 0.9, so P(value > 9) = 0.10. The mass above the top tie is the 0.1 tail.
    - Threshold strictly between two quantiles: linear interpolation between their levels.
    - Threshold below the lowest quantile: F is below 0.1 there, so the CDF is extrapolated
      linearly downward from (q_0, 0.1), using the slope of the lowest segment with positive
      width, and clamped to [0, 1].
    - Threshold above the highest quantile: F is at least 0.9 there, so the CDF is
      extrapolated upward from (q_last, 0.9), using the slope of the highest segment with
      positive width, and clamped to [0, 1].
    - Every quantile equal (a zero-spread row): a point mass. P(value > threshold) is 1
      below the point and 0 at or above it. No segment has width to extrapolate from, so no
      tail mass is added above the point.

    The true tail mass beyond the 0.1 and 0.9 quantiles is unknown, so the extrapolated
    values are approximations.
    """
    q = row
    p = _LEVELS_ARR
    positive = np.nonzero(np.diff(q) > 0)[0]
    if positive.size == 0:
        return 1.0 if threshold < q[0] else 0.0
    # Strict inequality: a threshold equal to q[0] belongs to the interpolation branch below,
    # which takes the highest tied level. Only values strictly below q[0] are extrapolated.
    if threshold < q[0]:
        k = int(positive[0])
        slope = (p[k + 1] - p[k]) / (q[k + 1] - q[k])
        cdf = p[0] + slope * (threshold - q[0])
    elif threshold >= q[-1]:
        k = int(positive[-1])
        slope = (p[k + 1] - p[k]) / (q[k + 1] - q[k])
        cdf = p[-1] + slope * (threshold - q[-1])
    else:
        # q[k] <= threshold < q[k + 1], and q[k + 1] > q[k] because threshold lies between them.
        k = int(np.searchsorted(q, threshold, side="right")) - 1
        cdf = p[k] + (threshold - q[k]) * (p[k + 1] - p[k]) / (q[k + 1] - q[k])
    cdf = min(1.0, max(0.0, float(cdf)))
    return 1.0 - cdf


@dataclass
class Forecast:
    """A forecast for one named series over ``horizon`` steps.

    Attributes:
        name: Series name.
        horizon: Number of future steps.
        mean: Point forecast per step. For ``TimesFMForecaster`` this is the median path
            (quantile 0.5). For ``NaiveQuantileForecaster`` it is the last observed value.
        quantiles: Optional ``horizon`` rows of nine values, one per level in
            ``QUANTILE_LEVELS`` (0.1 ... 0.9). Each row must be non-decreasing.
    """

    name: str
    horizon: int
    mean: List[float]
    quantiles: Optional[List[List[float]]] = None

    def __post_init__(self) -> None:
        self.horizon = _check_horizon(self.horizon)
        mean = _as_series(self.mean, min_len=self.horizon, label=f"mean of {self.name!r}")
        if mean.size != self.horizon:
            raise ValueError(f"mean of {self.name!r} must have {self.horizon} values, got {mean.size}")
        self.mean = [float(v) for v in mean]
        if self.quantiles is not None:
            rows = np.asarray(self.quantiles, dtype=np.float64)
            if rows.shape != (self.horizon, len(QUANTILE_LEVELS)):
                raise ValueError(
                    f"quantiles of {self.name!r} must have shape ({self.horizon}, {len(QUANTILE_LEVELS)}), "
                    f"got {rows.shape}"
                )
            if not np.all(np.isfinite(rows)):
                raise ValueError(f"quantiles of {self.name!r} contain non-finite values")
            if np.any(np.diff(rows, axis=1) < -1e-9):
                raise ValueError(f"quantiles of {self.name!r} must be non-decreasing across levels")
            self.quantiles = [[float(v) for v in row] for row in rows]

    # ------------------------------------------------------------------ queries

    def _require_quantiles(self) -> np.ndarray:
        if self.quantiles is None:
            raise ValueError(f"forecast {self.name!r} has no quantiles")
        return np.asarray(self.quantiles, dtype=np.float64)

    def interval(self, level: float) -> Tuple[List[float], List[float]]:
        """Central prediction interval per step, as ``(lower, upper)`` lists.

        ``level=0.8`` returns the 0.1 and 0.9 quantiles; ``level=0.5`` returns the 0.25 and
        0.75 quantiles (linearly interpolated between stored levels).

        Supported levels are ``0 < level <= 0.8``. The interval at level L reads the quantile
        levels ``(1 - L) / 2`` and ``(1 + L) / 2``, which must lie in the stored range
        [0.1, 0.9]. Larger levels need tails the forecast does not carry and raise ``ValueError``.
        """
        rows = self._require_quantiles()
        level = float(level)
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        p_lo = round((1.0 - level) / 2.0, 12)
        p_hi = round(1.0 - p_lo, 12)
        if p_lo < QUANTILE_LEVELS[0] - 1e-12 or p_hi > QUANTILE_LEVELS[-1] + 1e-12:
            raise ValueError(
                f"level {level} needs quantile levels outside the stored range [0.1, 0.9]; "
                "supported central interval levels are 0 < level <= 0.8"
            )
        lower = [float(np.interp(p_lo, _LEVELS_ARR, row)) for row in rows]
        upper = [float(np.interp(p_hi, _LEVELS_ARR, row)) for row in rows]
        return lower, upper

    def step_exceedance_probs(self, threshold: float) -> List[float]:
        """P(value > threshold) at each step of the horizon, in step order.

        Each entry is ``prob_exceeds(threshold, step=i)``. See ``prob_exceeds`` for the tie
        and extrapolation rules.
        """
        rows = self._require_quantiles()
        thr = float(threshold)
        return [_exceed_at(row, thr) for row in rows]

    def prob_exceeds(self, threshold: float, step: Optional[int] = None) -> float:
        """Probability that the value exceeds ``threshold``.

        With ``step`` (zero-based index into the horizon), returns P(value > threshold) at
        that step. It is estimated from the quantile function by linear interpolation over
        the nine levels, extrapolated and clamped to [0, 1]. Ties resolve to the highest tied
        level: for row [5, 5, 5, 6, ..., 11] and threshold 5 the result is 0.70.

        With ``step=None``, returns the LOWER BOUND ``max_t P(value_t > threshold)``. This is
        a lower bound on P(the path exceeds the threshold at some step), because the union
        is at least the maximum of its members. The upper bound is
        ``min(1, sum_t P(value_t > threshold))`` (the union bound), available from
        ``step_exceedance_probs``. The joint law of the steps is not modelled, so the true
        probability lies between the two bounds.
        """
        thr = float(threshold)
        if step is None:
            return float(max(self.step_exceedance_probs(thr)))
        rows = self._require_quantiles()
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer index or None")
        if not 0 <= step < self.horizon:
            raise IndexError(f"step {step} is outside the horizon of {self.horizon}")
        return _exceed_at(rows[int(step)], thr)

    def uncertainty(self) -> float:
        """Mean relative width of the 80% band across the horizon.

        Each step contributes ``(upper - lower) / max(|median|, 1e-12)``, so the value is
        scale-dependent: series near zero produce large values. Returns ``math.inf`` when
        the forecast carries no quantiles, so that unknown uncertainty fails closed.
        """
        if self.quantiles is None:
            return math.inf
        lower, upper = self.interval(0.8)
        rel = [(hi - lo) / max(abs(m), 1e-12) for lo, hi, m in zip(lower, upper, self.mean)]
        return float(sum(rel) / len(rel))

    # ------------------------------------------------------------ serialization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "mean": list(self.mean),
            "quantiles": None if self.quantiles is None else [list(r) for r in self.quantiles],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Forecast":
        return cls(
            name=str(data["name"]),
            horizon=int(data["horizon"]),
            mean=list(data["mean"]),
            quantiles=None if data.get("quantiles") is None else [list(r) for r in data["quantiles"]],
        )


@runtime_checkable
class Forecaster(Protocol):
    """Anything that turns histories into forecasts.

    ``predict`` returns one ``Forecast`` per input series, in the same order.
    """

    def predict(self, series_list: Sequence[Sequence[float]], horizon: int) -> List[Forecast]:
        ...


class NaiveQuantileForecaster:
    """Dependency-free random-walk baseline with empirical quantile bands.

    The median path is flat at the last observed value. For horizon step ``h`` the band is
    ``last + sqrt(h) * (Q_d(p) - Q_d(0.5))``, where ``Q_d`` is the empirical quantile
    function of the historical one-step differences ``d``. Centring on the difference
    median keeps the median path exactly flat. Bands widen with the square root of the
    horizon, as they do for a random walk with i.i.d. increments.

    This is the default forecaster and the test oracle. It is a baseline, not a
    calibrated model of any particular business series.
    """

    name = "naive_random_walk"

    def predict(self, series_list: Sequence[Sequence[float]], horizon: int) -> List[Forecast]:
        horizon = _check_horizon(horizon)
        out: List[Forecast] = []
        steps = np.sqrt(np.arange(1, horizon + 1, dtype=np.float64))[:, None]
        for i, raw in enumerate(series_list):
            x = _as_series(raw, min_len=2, label=f"series[{i}]")
            diffs = np.diff(x)
            qd = np.quantile(diffs, _LEVELS_ARR)
            centred = qd - qd[MEDIAN_INDEX]
            rows = x[-1] + steps * centred[None, :]
            out.append(
                Forecast(
                    name=f"series[{i}]",
                    horizon=horizon,
                    mean=rows[:, MEDIAN_INDEX].tolist(),
                    quantiles=rows.tolist(),
                )
            )
        return out


class TimesFMForecaster:
    """Adapter for Google TimesFM 3.0 (``timesfm3`` package, PyTorch backend).

    Licence: the 3.0 weights are under ``timesfm-non-commercial-license-v1.0`` and are
    restricted to non-commercial, non-production use. Construction raises ``ValueError``
    unless ``license_acknowledged=True`` is passed. The check runs before any import, so
    an unacknowledged call fails the same way whether or not timesfm is installed.

    Requires ``pip install "timesfm[torch]>=3.0.2"``. The import of ``timesfm3`` is lazy:
    a missing install raises ``ImportError`` with that hint from the constructor.
    """

    name = "timesfm-3.0"

    def __init__(
        self,
        checkpoint_path: str = "google/timesfm-3.0-pytorch",
        device: str = "cpu",
        per_core_batch_size: int = 32,
        license_acknowledged: bool = False,
    ) -> None:
        if license_acknowledged is not True:
            raise ValueError(_TIMESFM_LICENSE_MESSAGE)
        try:
            from timesfm3 import ModelConfig, TimesFM3Evaluator
        except ImportError as exc:
            raise ImportError(
                f"TimesFMForecaster needs the timesfm 3.x package with its torch backend. "
                f"Install it with: {_TIMESFM_INSTALL_HINT}"
            ) from exc
        config = ModelConfig(
            checkpoint_path=checkpoint_path,
            per_core_batch_size=per_core_batch_size,
            device=device,
        )
        self._evaluator = TimesFM3Evaluator(config)

    def predict(self, series_list: Sequence[Sequence[float]], horizon: int) -> List[Forecast]:
        horizon = _check_horizon(horizon)
        contexts = [
            _as_series(raw, min_len=1, label=f"series[{i}]").astype(np.float32)
            for i, raw in enumerate(series_list)
        ]
        if not contexts:
            return []
        outputs = list(
            self._evaluator.predict_batch(
                contexts,
                horizon=horizon,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )
        if len(outputs) != len(contexts):
            raise RuntimeError(f"TimesFM returned {len(outputs)} forecasts for {len(contexts)} series")
        result: List[Forecast] = []
        for i, out in enumerate(outputs):
            if out.forecast is None or out.quantiles is None:
                raise RuntimeError("TimesFM returned no forecast or quantiles; return_quantiles must be True")
            point = np.asarray(out.forecast, dtype=np.float64).reshape(-1)
            quant = np.asarray(out.quantiles, dtype=np.float64)
            if point.shape != (horizon,):
                raise RuntimeError(f"TimesFM point forecast has shape {point.shape}, expected ({horizon},)")
            if quant.shape != (horizon, len(QUANTILE_LEVELS)):
                raise RuntimeError(
                    f"TimesFM quantiles have shape {quant.shape}, expected ({horizon}, {len(QUANTILE_LEVELS)})"
                )
            result.append(
                Forecast(
                    name=f"series[{i}]",
                    horizon=horizon,
                    mean=point.tolist(),
                    quantiles=quant.tolist(),
                )
            )
        return result


# ---------------------------------------------------------------- error metrics


def pinball_loss(
    quantiles: Sequence[Sequence[float]],
    realized: Sequence[float],
    levels: Sequence[float] = QUANTILE_LEVELS,
) -> float:
    """Mean quantile (pinball) loss over steps and levels.

    ``quantiles`` holds one row per step, each with one value per level. For level tau and
    residual d = y - q, the loss is max(tau * d, (tau - 1) * d).
    """
    q = np.asarray(quantiles, dtype=np.float64)
    y = np.asarray(realized, dtype=np.float64).reshape(-1)
    tau = np.asarray(levels, dtype=np.float64).reshape(-1)
    if q.ndim != 2 or q.shape != (y.size, tau.size):
        raise ValueError(f"quantiles must have shape ({y.size}, {tau.size}), got {q.shape}")
    diff = y[:, None] - q
    loss = np.maximum(tau[None, :] * diff, (tau[None, :] - 1.0) * diff)
    return float(loss.mean())


def interval_coverage(forecast: Forecast, realized: Sequence[float], level: float) -> float:
    """Fraction of steps whose realized value lies within the central ``level`` interval."""
    y = np.asarray(realized, dtype=np.float64).reshape(-1)
    if y.size != forecast.horizon:
        raise ValueError(f"realized has {y.size} values, forecast horizon is {forecast.horizon}")
    lower, upper = forecast.interval(level)
    lo = np.asarray(lower)
    hi = np.asarray(upper)
    return float(np.mean((y >= lo) & (y <= hi)))


def mase(forecast_values: Sequence[float], realized: Sequence[float], history: Sequence[float]) -> float:
    """Mean absolute scaled error against the in-sample one-step naive forecast.

    The scale is the mean absolute one-step difference of ``history``. A constant history
    has zero scale and raises ``ValueError``.
    """
    f = np.asarray(forecast_values, dtype=np.float64).reshape(-1)
    y = np.asarray(realized, dtype=np.float64).reshape(-1)
    h = _as_series(history, min_len=2, label="history")
    if f.size != y.size:
        raise ValueError(f"forecast_values ({f.size}) and realized ({y.size}) must have equal length")
    scale = float(np.mean(np.abs(np.diff(h))))
    if scale == 0.0:
        raise ValueError("history has zero in-sample naive error; MASE is undefined")
    return float(np.mean(np.abs(y - f)) / scale)


def brier_score(probability: float, occurred: bool) -> float:
    """Squared error of a probability forecast for a binary outcome."""
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    return (p - (1.0 if occurred else 0.0)) ** 2
