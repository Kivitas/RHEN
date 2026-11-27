"""RHEN-2 selective decision tools (plan 7.6): cross-fitting, selective-reward fine-tuning of
the C1 head, and risk-coverage metrics.

Conventions: ``confidence`` higher = more willing to act.  Ties in the risk-coverage curve are
broken by a stable sort (input order), so the curve is exact for distinct confidences.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- cross-fitting
def cross_fit_predictions(examples: Sequence, fit_fn: Callable, predict_fn: Callable,
                          folds: int = 5, seed: int = 0) -> List[Tuple]:
    """K-fold cross-fitting.  For each fold, ``model = fit_fn(train_examples)`` on the other
    folds and ``predict_fn(model, held_out)`` -> list of (features, correct), one per held-out
    example in order.  Returns a list aligned with ``examples``."""
    n = len(examples)
    if folds < 2 or n < folds:
        raise ValueError("need folds >= 2 and at least `folds` examples")
    perm = np.random.RandomState(seed).permutation(n)
    parts = np.array_split(perm, folds)
    out: List[Optional[Tuple]] = [None] * n
    for f in range(folds):
        held = sorted(int(i) for i in parts[f])
        train = sorted(int(i) for fo in range(folds) if fo != f for i in parts[fo])
        model = fit_fn([examples[i] for i in train])
        preds = list(predict_fn(model, [examples[i] for i in held]))
        if len(preds) != len(held):
            raise ValueError("predict_fn must return one (features, correct) per held-out example")
        for i, pr in zip(held, preds):
            out[i] = tuple(pr)
    return out  # type: ignore[return-value]


# --------------------------------------------------------------------------- selective loss
def selective_reward_loss(p_correct: torch.Tensor, correct: torch.Tensor, cost: float,
                          temperature: float = 0.1) -> torch.Tensor:
    """Negative expected reward.  Policy: act iff p_correct > cost/(1+cost), relaxed with a
    sigmoid of width ``temperature``.  Reward: +1 correct act, -cost wrong act, 0 abstain."""
    thr = cost / (1.0 + cost)
    act = torch.sigmoid((p_correct - thr) / temperature)
    c = correct.to(p_correct.dtype)
    reward = act * (c - cost * (1.0 - c))
    return -reward.mean()


def selective_finetune(head: torch.nn.Module, features: torch.Tensor, correct: torch.Tensor,
                       costs: Sequence[float] = (1, 3, 9, 19), steps: int = 300, lr: float = 1e-2,
                       temperature: float = 0.1, batch_size: Optional[int] = None,
                       seed: int = 0) -> List[float]:
    """Train ``head`` (features -> logit of p_correct) with the selective reward; the cost is
    sampled uniformly from ``costs`` each step.  Returns the loss history."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = features.shape[0]
    hist: List[float] = []
    head.train()
    for _ in range(steps):
        cost = float(costs[int(torch.randint(len(costs), (1,), generator=gen))])
        if batch_size and batch_size < n:
            idx = torch.randint(n, (batch_size,), generator=gen)
            x, y = features[idx], correct[idx]
        else:
            x, y = features, correct
        p = torch.sigmoid(head(x).reshape(-1))
        loss = selective_reward_loss(p, y, cost, temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    return hist


# --------------------------------------------------------------------------- risk-coverage
def risk_coverage_curve(confidence, correct) -> Tuple[np.ndarray, np.ndarray]:
    """(coverage[k], risk[k]) for accepting the top-k most confident, k=1..n."""
    c = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-c, kind="stable")
    errs = np.cumsum(1.0 - ok[order])
    k = np.arange(1, len(c) + 1)
    return k / len(c), errs / k


def aurc(confidence, correct) -> float:
    """Area under the risk-coverage curve (mean risk over k=1..n; lower is better)."""
    return float(risk_coverage_curve(confidence, correct)[1].mean())


def accuracy_at_coverage(confidence, correct, coverage: float) -> float:
    cov, risk = risk_coverage_curve(confidence, correct)
    k = max(1, int(math.ceil(coverage * len(cov) - 1e-9)))
    return float(1.0 - risk[k - 1])


def coverage_at_error(confidence, correct, max_error: float = 0.01) -> float:
    """Largest coverage whose risk is <= max_error (0.0 if no prefix qualifies)."""
    cov, risk = risk_coverage_curve(confidence, correct)
    ok = np.nonzero(risk <= max_error)[0]
    return float(cov[ok.max()]) if len(ok) else 0.0
