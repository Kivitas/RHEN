"""Decision-quality metrics, written out explicitly so every number is auditable.

A *record* describes one evaluated question::

    {"task": "routing", "primitive": "choice", "split": "test",
     "probs": [0.1, 0.7, 0.2], "label_index": 1,      # single-label primitives
     "label_set": [0, 2],                               # multichoice only
     "free_energy": -3.2, "consistency_tv": 0.04}       # optional

Single-label primitives (noul, choice, score) get accuracy, NLL, Brier, top-label ECE and
selective-prediction (risk-coverage) metrics. Multichoice gets exact-match and Hamming
accuracy plus a per-option Brier score. Score questions additionally get the mean absolute
level error.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

SINGLE_LABEL = ("noul", "choice", "score")
EPS = 1e-12


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def top_label_ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> float:
    """Expected calibration error of the top label with equal-width confidence bins."""
    if not confidences:
        return float("nan")
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        bins[min(n_bins - 1, int(c * n_bins))].append(i)
    n = len(confidences)
    ece = 0.0
    for members in bins:
        if members:
            acc = sum(1.0 for i in members if correct[i]) / len(members)
            conf = sum(confidences[i] for i in members) / len(members)
            ece += len(members) / n * abs(acc - conf)
    return ece


def risk_coverage(confidences: Sequence[float], correct: Sequence[bool]) -> Dict[str, float]:
    """Area under the risk-coverage curve (lower is better) and selective accuracy.

    Predictions are accepted in order of decreasing confidence; at coverage k/n the risk is
    the error rate of the k most confident predictions. ``aurc`` averages that risk over all
    coverages; ``excess_aurc`` subtracts the AURC of an oracle that ranks every correct
    prediction first, isolating the quality of the confidence ranking.
    """
    n = len(confidences)
    if n == 0:
        return {}
    order = sorted(range(n), key=lambda i: -confidences[i])
    errors, risks = 0, []
    for k, i in enumerate(order, start=1):
        errors += 0 if correct[i] else 1
        risks.append(errors / k)
    aurc = _mean(risks)
    n_err = sum(1 for c in correct if not c)
    oracle = [max(0, k - (n - n_err)) / k for k in range(1, n + 1)]
    out = {"aurc": aurc, "excess_aurc": aurc - _mean(oracle)}
    for cov in (0.5, 0.8, 0.9):
        k = max(1, int(round(cov * n)))
        out[f"selective_acc@{int(cov * 100)}"] = 1.0 - risks[k - 1]
    return out


def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """P(score of a random positive > score of a random negative), ties counted as one half."""
    if not positive_scores or not negative_scores:
        return float("nan")
    pooled = sorted([(s, 1) for s in positive_scores] + [(s, 0) for s in negative_scores])
    rank_sum, i = 0.0, 0
    while i < len(pooled):
        j = i
        while j < len(pooled) and pooled[j][0] == pooled[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # ranks are 1-based
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pooled[k][1] == 1)
        i = j
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def conformal_set(probs: Sequence[float], qhat: float) -> List[int]:
    """Split-conformal prediction set: every class whose non-conformity 1 - p is at most qhat."""
    return [k for k, p in enumerate(probs) if 1.0 - p <= qhat]


def single_label_metrics(records: Sequence[Dict[str, Any]], n_bins: int = 15, qhat: Optional[float] = None) -> Dict[str, Any]:
    recs = [r for r in records if r.get("label_index") is not None]
    if not recs:
        return {}
    conf, correct, nll, brier = [], [], [], []
    for r in recs:
        probs, y = r["probs"], int(r["label_index"])
        pred = max(range(len(probs)), key=lambda k: probs[k])
        conf.append(float(probs[pred]))
        correct.append(pred == y)
        nll.append(-math.log(max(float(probs[y]), EPS)))
        brier.append(sum((float(p) - (1.0 if k == y else 0.0)) ** 2 for k, p in enumerate(probs)))
    out: Dict[str, Any] = {
        "n": len(recs),
        "accuracy": _mean([float(c) for c in correct]),
        "nll": _mean(nll),
        "brier": _mean(brier),
        "ece": top_label_ece(conf, correct, n_bins),
        "mean_confidence": _mean(conf),
    }
    out.update(risk_coverage(conf, correct))
    scores = [r for r in recs if r.get("primitive") == "score"]
    if scores:
        out["score_mae_levels"] = _mean([
            abs(max(range(len(r["probs"])), key=lambda k: r["probs"][k]) - float(r.get("label_value", r["label_index"])))
            for r in scores
        ])
    if qhat is not None:
        sets = [conformal_set(r["probs"], qhat) for r in recs]
        out["conformal_coverage"] = _mean([float(int(r["label_index"]) in s) for r, s in zip(recs, sets)])
        out["conformal_mean_set_size"] = _mean([float(len(s)) for s in sets])
        out["conformal_singleton_rate"] = _mean([float(len(s) == 1) for s in sets])
    return out


def multichoice_metrics(records: Sequence[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    recs = [r for r in records if r.get("label_set") is not None]
    if not recs:
        return {}
    exact, hamming, brier = [], [], []
    for r in recs:
        truth = set(int(i) for i in r["label_set"])
        probs = [float(p) for p in r["probs"]]
        pred = {k for k, p in enumerate(probs) if p >= threshold}
        exact.append(float(pred == truth))
        hamming.append(_mean([float((k in pred) == (k in truth)) for k in range(len(probs))]))
        brier.append(_mean([(p - (1.0 if k in truth else 0.0)) ** 2 for k, p in enumerate(probs)]))
    return {"n": len(recs), "exact_match": _mean(exact), "hamming_accuracy": _mean(hamming), "brier": _mean(brier)}


def _group(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r.get(key, ""))].append(r)
    return dict(groups)


def summarize(records: Sequence[Dict[str, Any]], qhat: Optional[float] = None, n_bins: int = 15) -> Dict[str, Any]:
    """Overall, per-primitive and per-task metrics for a list of records."""
    single = [r for r in records if r.get("primitive") in SINGLE_LABEL]
    multi = [r for r in records if r.get("primitive") == "multichoice"]
    out: Dict[str, Any] = {
        "n_questions": len(records),
        "single_label": single_label_metrics(single, n_bins=n_bins, qhat=qhat),
        "multichoice": multichoice_metrics(multi),
        "by_primitive": {},
        "by_task": {},
    }
    for name, group in _group(records, "primitive").items():
        out["by_primitive"][name] = (
            multichoice_metrics(group) if name == "multichoice" else single_label_metrics(group, n_bins=n_bins, qhat=qhat)
        )
    for name, group in _group(records, "task").items():
        out["by_task"][name] = {
            "single_label": single_label_metrics([r for r in group if r.get("primitive") in SINGLE_LABEL], n_bins=n_bins),
            "multichoice": multichoice_metrics([r for r in group if r.get("primitive") == "multichoice"]),
        }
    tv = [r["consistency_tv"] for r in records if r.get("consistency_tv") is not None]
    if tv:
        flips = [r["consistency_flip"] for r in records if r.get("consistency_flip") is not None]
        out["permutation_consistency"] = {
            "n": len(tv),
            "mean_total_variation": _mean(tv),
            "max_total_variation": max(tv),
            "argmax_flip_rate": _mean([float(f) for f in flips]),
        }
    return out


def ood_detection(id_records: Sequence[Dict[str, Any]], ood_records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """How well free energy and (1 - confidence) separate OOD from ID questions (AUROC)."""
    out: Dict[str, float] = {}
    id_fe = [r["free_energy"] for r in id_records if r.get("free_energy") is not None]
    ood_fe = [r["free_energy"] for r in ood_records if r.get("free_energy") is not None]
    if id_fe and ood_fe:
        out["auroc_free_energy"] = auroc(ood_fe, id_fe)  # higher energy = more OOD
    id_unc = [1.0 - max(r["probs"]) for r in id_records if r.get("label_index") is not None]
    ood_unc = [1.0 - max(r["probs"]) for r in ood_records if r.get("label_index") is not None]
    if id_unc and ood_unc:
        out["auroc_max_prob"] = auroc(ood_unc, id_unc)
    return out
