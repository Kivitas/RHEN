"""Hierarchical routing for option sets larger than 255 (plan 9.3, finding C8).

The model's option vectors (the output of the workspace's option pooling, from an option pass with an
EMPTY question) are clustered with seeded numpy k-means into at most 255 groups.  The model then
chooses a group (each group is presented as its three most central labels joined with `` | ``) and
then chooses inside the group.  There is no string similarity anywhere.

Model calls (``RoutingReport``): one state encode, one option pass per chunk of at most
``cfg.max_options`` options (K > 255 does not fit one pass, so ``ceil(K / 255)`` passes), one group
choice, one inside choice (more when a group still exceeds ``max_options`` and is routed again).

Option vectors are captured with a forward hook on ``model.workspace.o_pool``, so ``model.py`` needs no
change.  Probabilities of the two stages are reported separately; their product is labelled
``uncalibrated_joint`` (plan 9.4).  The accuracy comparison with a flat split-and-merge baseline
(plan 10.2) is not done here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from rhen.v2.packing import QuestionSpec, Rhen2Example, Rhen2Packer
from rhen.v2.workspace import LoopPlan

MAX_GROUPS = 255


@dataclass
class RoutingReport:
    n_options: int = 0
    n_groups: int = 0
    group_sizes: List[int] = field(default_factory=list)
    encode_calls: int = 0
    option_pass_calls: int = 0
    group_choice_calls: int = 0
    inner_choice_calls: int = 0
    depth: int = 1
    seed: int = 0

    @property
    def model_calls(self) -> int:
        return self.encode_calls + self.option_pass_calls + self.group_choice_calls + self.inner_choice_calls

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model_calls"] = self.model_calls
        return d


@dataclass
class RoutingResult:
    option: Optional[str]            # None when the model abstained (null won a stage)
    index: Optional[int]             # index into the caller's option list
    group: int
    p_group: float
    p_inside: float
    p_null_group: float
    p_null_inside: float
    abstained: bool
    joint_confidence: float          # p_group * p_inside, uncalibrated
    joint_label: str
    report: RoutingReport


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """Seeded spherical k-means (k-means++ init, Lloyd steps).  Returns labels in ``0..k'-1`` with no empty cluster."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    k = max(1, min(int(k), n))
    rng = np.random.RandomState(seed)
    centers = [x[rng.randint(n)]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        tot = d2.sum()
        idx = int(np.argmax(d2)) if tot <= 0 else int(rng.choice(n, p=d2 / tot))
        centers.append(x[idx])
        d2 = np.minimum(d2, ((x - x[idx]) ** 2).sum(1))
    c = np.stack(centers)
    labels = np.full(n, -1)
    for _ in range(iters):
        dist = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        new = dist.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(c.shape[0]):
            m = labels == j
            if m.any():
                c[j] = x[m].mean(0)
            else:                                   # re-seed an empty cluster at the worst-fit point
                c[j] = x[int(dist.min(1).argmax())]
    _, labels = np.unique(labels, return_inverse=True)   # drop empty clusters, relabel 0..g-1
    return labels


def _join_labels(labels: Sequence[str], packer: Rhen2Packer, max_tokens: int) -> str:
    """Join the most central labels with `` | ``, dropping trailing ones until the text fits one option slot."""
    for n in range(len(labels), 0, -1):
        lab = " | ".join(labels[:n])
        if len(packer._encode(lab, False)[0]) <= max_tokens or n == 1:
            return lab
    return labels[0]


def _pack(packer: Rhen2Packer, state: Any, instructions: str, options: Sequence[str]):
    return packer.pack([Rhen2Example(state, [QuestionSpec("choice", instructions, list(options))])])


def _choose(model, S, batch, effort: str):
    out = model.deliberate(S, batch, LoopPlan.effort(model.cfg, effort))
    p = out.probs[0]
    k = int(batch.opt_valid[0, 0].sum())
    return p[:k].double().cpu().numpy(), float(p[-1])


def route_large_option_set(
    model,
    packer: Rhen2Packer,
    state: Any,
    question: Union[str, QuestionSpec],
    options: Sequence[str],
    *,
    n_groups: Optional[int] = None,
    seed: int = 0,
    effort: str = "medium",
    _depth: int = 1,
    _report: Optional[RoutingReport] = None,
) -> RoutingResult:
    options = [str(o) for o in options]
    cfg = model.cfg
    kmax = cfg.max_options
    if len(options) <= kmax:
        raise ValueError(f"routing is for more than {kmax} options; use a flat choice question for {len(options)}")
    if len(set(options)) != len(options):
        raise ValueError("options must be unique")
    if _depth > 4:
        raise RuntimeError("routing recursion too deep")
    instructions = question if isinstance(question, str) else question.instructions
    report = _report if _report is not None else RoutingReport(n_options=len(options), seed=seed)
    report.depth = max(report.depth, _depth)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # --- one option pass per chunk, empty question, model's own option vectors
            vecs: List[np.ndarray] = []
            S = None
            for lo in range(0, len(options), kmax):
                chunk = options[lo:lo + kmax]
                batch = _pack(packer, state, "", chunk)
                if S is None:
                    S = model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
                    report.encode_calls += 1
                captured: List[torch.Tensor] = []
                h = model.workspace.o_pool.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
                try:
                    model.deliberate(S, batch, LoopPlan.effort(cfg, "low"))
                finally:
                    h.remove()
                report.option_pass_calls += 1
                v = captured[-1].reshape(batch.opt_valid.shape[2], -1)[: len(chunk)]
                vecs.append(v.float().cpu().numpy())
            vec = np.concatenate(vecs, 0)

            # --- cluster
            g = n_groups if n_groups is not None else int(np.ceil(np.sqrt(len(options))))
            g = max(2, min(int(g), MAX_GROUPS, len(options)))
            labels = kmeans(vec, g, seed=seed)
            n_g = int(labels.max()) + 1
            members = [np.nonzero(labels == j)[0] for j in range(n_g)]
            if _depth == 1:
                report.n_groups = n_g
                report.group_sizes = [len(m) for m in members]
            unit = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
            group_labels: List[str] = []
            for j, mem in enumerate(members):
                cen = unit[mem].mean(0)
                order = mem[np.argsort(-(unit[mem] @ cen), kind="stable")][:3]
                lab = _join_labels([options[i] for i in order], packer, cfg.max_opt_tokens)
                group_labels.append(lab if lab not in group_labels else f"{lab} #{j}")

            # --- choose a group
            gbatch = _pack(packer, state, instructions, group_labels)
            gp, gnull = _choose(model, S, gbatch, effort)
            report.group_choice_calls += 1
            top_g = int(gp.argmax())
            if gnull > gp[top_g]:
                return RoutingResult(None, None, -1, float(gp[top_g]), 0.0, gnull, 0.0, True, 0.0,
                                     "uncalibrated_joint", report)
            mem = members[top_g]
            sub = [options[i] for i in mem]

            # --- choose inside the group
            if len(sub) > kmax:
                inner = route_large_option_set(model, packer, state, instructions, sub, n_groups=n_groups, seed=seed,
                                               effort=effort, _depth=_depth + 1, _report=report)
                idx = None if inner.index is None else int(mem[inner.index])
                return RoutingResult(inner.option, idx, top_g, float(gp[top_g]), inner.joint_confidence, gnull,
                                     inner.p_null_inside, inner.abstained,
                                     float(gp[top_g]) * inner.joint_confidence, "uncalibrated_joint", report)
            ibatch = _pack(packer, state, instructions, sub)
            ip, inull = _choose(model, S, ibatch, effort)
            report.inner_choice_calls += 1
            top_i = int(ip.argmax())
            if inull > ip[top_i]:
                return RoutingResult(None, None, top_g, float(gp[top_g]), float(ip[top_i]), gnull, inull, True, 0.0,
                                     "uncalibrated_joint", report)
            return RoutingResult(sub[top_i], int(mem[top_i]), top_g, float(gp[top_g]), float(ip[top_i]), gnull, inull,
                                 False, float(gp[top_g] * ip[top_i]), "uncalibrated_joint", report)
    finally:
        model.train(was_training)


__all__ = ["RoutingReport", "RoutingResult", "kmeans", "route_large_option_set", "MAX_GROUPS"]
