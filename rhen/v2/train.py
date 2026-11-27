"""RHEN-2 trainer (plan section 7, tasks T-401 and T-403).

Stages
------
``pretrain``  MLM + span-choice + span-locate (plan 7.2, 40/40/20) through the same forward path as
              decisions. The MLM head is trained and checkpointed too.
``decision``  supervised mix (plan 7.3: 45% teacher-labelled, 35% converted public, 20% programmatic
              synthetic, plus 10% span-choice retained against forgetting). Soft labels are used when present.
``distill``   plan 7.5: a teacher checkpoint labels every question with a temperature-2 distribution; the
              student target is a 50/50 blend of that distribution and the original label.
``adapter``   plan 5.1 option (b); needs a pretrained backbone and is not implemented (NotImplementedError).

Determinism and resume
----------------------
Batch ``(step, micro, rank)`` is a pure function of ``(seed, step, micro, rank)`` (string-seeded
``random.Random``), so no data-loader state exists. The checkpoint holds model, optimizer, EMA, MLM head,
loop-sampling generator, global torch/python RNG state and the step as plain tensors, ints and floats; it loads
with ``weights_only=True``. A run stopped at step 3 and resumed gives the same weights and metrics as an
uninterrupted run (tested to 1e-6).

Limits that are stated, not hidden: examples that do not fit the model's token limits are redrawn (counted in
the metrics); FSDP selection and wrapping exist but have not been run in this repository (no multi-process
test); the ``p_correct`` target of plan 7.5 is not distilled here (the Checker lives in ``checker.py``).
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from rhen.research import device as devmod
from rhen.research import distributed as distmod
from rhen.research.checkpointing import checkpoint_dir, latest_checkpoint
from rhen.research.manifest import build_manifest, canonical_hash, write_json
from rhen.v2.config import PRESETS, Rhen2Config, count_parameters, preset
from rhen.v2.data.blocklist import Blocklist, BlockedExampleError
from rhen.v2.data.pretrain import MLMHead, PretrainMixture, TextCorpus, collate_mlm, mlm_loss
from rhen.v2.data.schema import example_from_record
from rhen.v2.data.synth import family_names, generate_split
from rhen.v2.data.teacher import SOFT_TYPES, n_options
from rhen.v2.ema import EMA
from rhen.v2.io import load_checkpoint, save_checkpoint
from rhen.v2.model import Rhen2Model
from rhen.v2.optim import build_param_groups, lr_at, set_lr
from rhen.v2.packing import NULL_INDEX, NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer, make_batches
from rhen.v2.workspace import LoopPlan

log = logging.getLogger("rhen.v2.train")

STAGES = ("pretrain", "decision", "distill", "adapter")
BUCKETS = ("teacher", "converted", "synthetic")
PRECISIONS = ("fp32", "bf16", "auto")
STRATEGIES = ("auto", "none", "ddp", "fsdp")
TRAINER_STATE_FORMAT = 1
SMOKE_NOTICE = ("r2-tiny run: this proves the training plumbing only (data, loss, optimiser, EMA, checkpoints, "
                "resume). Its numbers are not evidence about model quality.")


# --------------------------------------------------------------------------- config
@dataclass
class TrainConfig:
    # run
    run_dir: str = "runs/v2"
    stage: str = "decision"
    seed: int = 0
    notes: str = ""
    # model
    preset: str = "r2-tiny"
    model_overrides: Dict[str, Any] = field(default_factory=dict)
    gradient_checkpointing: Optional[bool] = None     # None: leave the preset / override value
    init_from: Optional[str] = None                   # model file, step dir or run dir to start weights from
    # distillation (stage "distill")
    teacher_checkpoint: Optional[str] = None
    distill_temperature: float = 2.0
    distill_teacher_weight: float = 0.5               # target = w * teacher(T) + (1 - w) * original label
    teacher_effort: str = "medium"
    # data
    decision_jsonl: List[str] = field(default_factory=list)    # generic labelled JSONL, counted in "converted"
    teacher_jsonl: List[str] = field(default_factory=list)     # soft-labelled by teachers (plan 7.1.4)
    converted_jsonl: List[str] = field(default_factory=list)   # output of converters (plan 7.1.2)
    synthetic: List[Dict[str, Any]] = field(default_factory=list)  # {"n", "seed", "families", "unanswerable_rate"}
    pretrain_corpus_dir: Optional[str] = None
    pretrain_weights: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])   # mlm / span choice / span locate
    pretrain_structured_frac: float = 0.5
    max_passage_tokens: int = 200
    decision_weights: Dict[str, float] = field(
        default_factory=lambda: {"teacher": 0.45, "converted": 0.35, "synthetic": 0.20})
    span_choice_frac: float = 0.10
    state_len_curriculum: List[List[float]] = field(default_factory=list)   # [[fraction of steps, max state tokens], ...]
    blocklist_path: Optional[str] = None              # None: the default rhen/v2/data/blocklist.json
    blocklist_action: str = "raise"                   # "raise" (BlockedExampleError) or "drop"
    blocklist_ngram_threshold: Optional[float] = None
    eval_jsonl: List[str] = field(default_factory=list)
    eval_synthetic: Optional[Dict[str, Any]] = None   # {"split", "n", "seed", "families"}; default val split
    eval_batches: int = 2                             # pretrain eval: number of fixed mixture batches
    eval_raw: bool = True
    # schedule and optimiser
    steps: int = 100
    batch_questions: int = 32                         # questions per micro-batch (pretrain: slots)
    grad_accum: int = 1
    lr: float = 3e-4
    base_width: int = 512                             # muP base width (rhen/v2/optim.py)
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_frac: float = 0.02
    min_lr_ratio: float = 0.1
    precision: str = "fp32"
    device: str = "auto"
    ema_decay: float = 0.999
    loop_sampling: bool = True
    fixed_loops: Optional[int] = None
    dist_strategy: str = "auto"
    # cadence
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 100
    keep_last: int = 3
    resume: bool = True
    tensorboard: bool = False
    # adapter stage (plan 5.1 option b); keys only, the trainer raises NotImplementedError
    adapter_backbone: Optional[str] = None
    adapter_freeze_backbone_steps: int = 0
    adapter_bidirectional_adaptation_steps: int = 0
    adapter_layers_trainable: int = 0

    def __post_init__(self) -> None:
        self.validate()

    # -- validation
    def model_config(self) -> Rhen2Config:
        overrides = dict(self.model_overrides)
        if self.gradient_checkpointing is not None:
            overrides["gradient_checkpointing"] = bool(self.gradient_checkpointing)
        return preset(self.preset, **overrides)

    def validate(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        need(self.stage in STAGES, f"stage must be one of {STAGES}, got {self.stage!r}")
        need(self.preset in PRESETS, f"unknown preset {self.preset!r}; choose from {sorted(PRESETS)}")
        try:
            self.model_config()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid model config: {exc}") from exc
        for name in ("steps", "batch_questions", "grad_accum", "log_every", "eval_every", "save_every", "keep_last",
                     "max_passage_tokens", "base_width", "eval_batches"):
            need(isinstance(getattr(self, name), int) and getattr(self, name) >= 1, f"{name} must be an integer >= 1")
        need(self.lr > 0 and self.weight_decay >= 0 and self.grad_clip > 0 and self.eps > 0, "lr, grad_clip, eps must be > 0")
        need(0 <= self.beta1 < 1 and 0 <= self.beta2 < 1, "betas must be in [0, 1)")
        need(0 <= self.warmup_frac < 1 and 0 <= self.min_lr_ratio <= 1, "warmup_frac in [0,1), min_lr_ratio in [0,1]")
        need(self.precision in PRECISIONS, f"precision must be one of {PRECISIONS}")
        need(self.dist_strategy in STRATEGIES, f"dist_strategy must be one of {STRATEGIES}")
        need(0.0 <= self.ema_decay < 1.0, "ema_decay must be in [0, 1)")
        need(self.blocklist_action in ("raise", "drop"), "blocklist_action must be 'raise' or 'drop'")
        need(self.blocklist_ngram_threshold is None or 0 < self.blocklist_ngram_threshold <= 1, "blocklist_ngram_threshold in (0,1]")
        need(len(self.pretrain_weights) == 3 and min(self.pretrain_weights) >= 0 and sum(self.pretrain_weights) > 0,
             "pretrain_weights needs three non-negative numbers (mlm, span choice, span locate)")
        need(0.0 <= self.pretrain_structured_frac <= 1.0, "pretrain_structured_frac must be in [0, 1]")
        need(set(self.decision_weights) <= set(BUCKETS), f"decision_weights keys must be among {BUCKETS}")
        need(all(w >= 0 for w in self.decision_weights.values()), "decision_weights must be non-negative")
        need(0.0 <= self.span_choice_frac < 1.0, "span_choice_frac must be in [0, 1)")
        need(self.distill_temperature > 0 and 0.0 <= self.distill_teacher_weight <= 1.0, "bad distillation settings")
        need(self.teacher_effort in ("low", "medium", "high"), "teacher_effort must be low, medium or high")
        for name in ("decision_jsonl", "teacher_jsonl", "converted_jsonl", "eval_jsonl"):
            v = getattr(self, name)
            need(isinstance(v, list) and all(isinstance(p, str) for p in v), f"{name} must be a list of paths")
        known = set(family_names())
        for i, spec in enumerate(self.synthetic):
            need(isinstance(spec, dict) and set(spec) <= {"n", "seed", "families", "unanswerable_rate"},
                 f"synthetic[{i}] keys must be among n, seed, families, unanswerable_rate")
            need(int(spec.get("n", 0)) >= 1, f"synthetic[{i}].n must be >= 1")
            bad = set(spec.get("families") or []) - known
            need(not bad, f"synthetic[{i}] unknown families {sorted(bad)}")
        if self.eval_synthetic is not None:
            need(set(self.eval_synthetic) <= {"split", "n", "seed", "families"}, "eval_synthetic keys: split, n, seed, families")
            need(self.eval_synthetic.get("split", "val") in ("val", "test", "test_ood", "train"), "bad eval_synthetic.split")
        prev = -1.0
        for entry in self.state_len_curriculum:
            need(isinstance(entry, list) and len(entry) == 2, "state_len_curriculum entries are [fraction, max_state_tokens]")
            need(0.0 <= entry[0] < 1.0 and entry[0] > prev and entry[1] >= 8, "state_len_curriculum: increasing fractions in [0,1), tokens >= 8")
            prev = entry[0]
        mc = self.model_config()
        need(self.fixed_loops is None or mc.r_min <= self.fixed_loops <= mc.r_max, "fixed_loops must be within [r_min, r_max]")
        if self.stage == "distill":
            need(bool(self.teacher_checkpoint), "stage 'distill' needs teacher_checkpoint")
        if self.stage == "adapter":
            need(bool(self.adapter_backbone), "stage 'adapter' needs adapter_backbone (a pretrained open LLM, plan 5.1 option b)")

    # -- (de)serialisation
    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str, overrides: Sequence[str] = ()) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(apply_overrides(data, overrides))

    def save(self, path: str) -> None:
        write_json(path, self.to_dict())

    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def limit_at(self, step: int) -> int:
        """State-token limit from the sequence-length curriculum at ``step`` (the model limit when there is none)."""
        top = self.model_config().max_state_tokens
        cur = top
        for frac, tokens in self.state_len_curriculum:
            if step >= frac * self.steps:
                cur = int(tokens)
        return min(top, cur)


_NOT_RESUME_CRITICAL = {"run_dir", "resume", "log_every", "eval_every", "save_every", "keep_last", "tensorboard", "notes",
                        "eval_jsonl", "eval_synthetic", "eval_batches", "eval_raw", "device", "dist_strategy"}


def apply_overrides(data: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``key=value`` strings (JSON values; dotted keys reach into dicts and lists) to a config dict."""
    out = json.loads(json.dumps(data))
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        parts = key.split(".")
        node: Any = out
        for p in parts[:-1]:
            if isinstance(node, list):
                node = node[int(p)]
            elif isinstance(node, dict):
                if not isinstance(node.get(p), (dict, list)):
                    node[p] = {}
                node = node[p]
            else:
                raise ValueError(f"cannot set {key!r}: {p!r} is not a container")
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return out


# --------------------------------------------------------------------------- distributed selection
def choose_strategy(requested: str, world_size: int, model_cfg: Rhen2Config) -> str:
    """'none' on one process; otherwise the request, where 'auto' picks FSDP from d_model >= 1536 (1B and up,
    plan 5.1) and DDP below."""
    if requested not in STRATEGIES:
        raise ValueError(f"dist_strategy must be one of {STRATEGIES}, got {requested!r}")
    if world_size <= 1 or requested == "none":
        return "none"
    if requested == "auto":
        return "fsdp" if model_cfg.d_model >= 1536 else "ddp"
    return requested


def wrap_for_training(module: nn.Module, ctx: "distmod.DistributedContext", strategy: str) -> nn.Module:
    """DDP through the v1 helper; FSDP with a v2 auto-wrap policy (the v1 helper wraps v1 layer classes)."""
    if strategy in ("none", "ddp") or not ctx.enabled:
        return distmod.wrap_model(module, ctx, "none" if strategy == "none" else "ddp")
    if strategy != "fsdp":
        raise ValueError(f"unknown strategy {strategy!r}")
    import functools

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from rhen.v2.layers import EncoderBlock
    from rhen.v2.workspace import CoreBlock

    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={EncoderBlock, CoreBlock})
    return FSDP(module, auto_wrap_policy=policy, device_id=ctx.device, use_orig_params=True)


# --------------------------------------------------------------------------- data sources
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Screen:
    """Blocklist enforcement shared by every training source."""

    def __init__(self, blocklist: Blocklist, action: str, threshold: Optional[float]):
        self.blocklist, self.action, self.threshold = blocklist, action, threshold
        self.dropped = 0

    def ok(self, ex: Rhen2Example, where: str) -> bool:
        hit = self.blocklist.match(ex, self.threshold)
        if not hit:
            return True
        if self.action == "raise":
            raise BlockedExampleError(f"{where}: example matches blocklisted benchmark {hit!r}")
        self.dropped += 1
        return False


class ListSource:
    def __init__(self, name: str, examples: Sequence[Rhen2Example], screen: _Screen, meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.meta = meta or {}
        self.examples = [ex for i, ex in enumerate(examples) if screen.ok(ex, f"{name}[{i}]")]
        self.n_questions = sum(len(e.questions) for e in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def get(self, i: int) -> Rhen2Example:
        return self.examples[i]

    def describe(self) -> Dict[str, Any]:
        d = {"name": self.name, "examples": len(self), "questions": self.n_questions}
        d.update(self.meta)
        return d


class JsonlSource:
    """Random-access JSONL: one pass indexes line offsets and screens every record against the blocklist."""

    def __init__(self, path: str, screen: _Screen):
        if not os.path.exists(path):
            raise FileNotFoundError(f"data file not found: {path}")
        self.path = path
        self.name = path
        self.sha256 = file_sha256(path)
        self.offsets: List[int] = []
        self.n_questions = 0
        self._fh = None
        with open(path, "rb") as fh:
            pos = 0
            for lineno, line in enumerate(fh, 1):
                start, pos = pos, pos + len(line)
                if not line.strip():
                    continue
                ex = example_from_record(json.loads(line))
                if screen.ok(ex, f"{path}:{lineno}"):
                    self.offsets.append(start)
                    self.n_questions += len(ex.questions)

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, i: int) -> Rhen2Example:
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(self.offsets[i])
        return example_from_record(json.loads(self._fh.readline()))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.path, "sha256": self.sha256, "examples": len(self), "questions": self.n_questions}


class Pool:
    def __init__(self, sources: Sequence[Any]):
        self.sources = [s for s in sources if len(s) > 0]
        self.cum: List[int] = []
        total = 0
        for s in self.sources:
            total += len(s)
            self.cum.append(total)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def sample(self, rng: random.Random) -> Tuple[Any, int]:
        r = rng.randrange(self.total)
        k = bisect.bisect_right(self.cum, r)
        base = self.cum[k - 1] if k else 0
        return self.sources[k], r - base


def make_blocklist(cfg: TrainConfig) -> Tuple[Blocklist, Dict[str, Any]]:
    from rhen.v2.data.blocklist import DEFAULT_PATH

    path = cfg.blocklist_path or DEFAULT_PATH
    return Blocklist.load(path), {"path": path, "sha256": file_sha256(path)}


class DecisionData:
    """Decision-stage sampler. ``draw`` is deterministic in its ``rng`` argument."""

    def __init__(self, cfg: TrainConfig, packer: Rhen2Packer, model_cfg: Rhen2Config, screen: _Screen, passages: Sequence[str]):
        self.cfg, self.packer, self.model_cfg = cfg, packer, model_cfg
        self.rejected = 0
        self._fit: Dict[Tuple[int, int, int], bool] = {}
        sources: Dict[str, List[Any]] = {b: [] for b in BUCKETS}
        for p in cfg.teacher_jsonl:
            sources["teacher"].append(JsonlSource(p, screen))
        for p in list(cfg.converted_jsonl) + list(cfg.decision_jsonl):
            sources["converted"].append(JsonlSource(p, screen))
        for i, spec in enumerate(cfg.synthetic):
            exs = generate_split("train", int(spec["n"]), int(spec.get("seed", 0)), spec.get("families") or None,
                                 float(spec.get("unanswerable_rate", 0.12)))
            sources["synthetic"].append(ListSource(
                f"synthetic[{i}]", exs, screen,
                {"n": int(spec["n"]), "seed": int(spec.get("seed", 0)), "families": spec.get("families") or "all"}))
        self.sources = sources
        self.pools = {b: Pool(s) for b, s in sources.items()}
        weights: Dict[str, float] = {b: float(cfg.decision_weights.get(b, 0.0)) for b in BUCKETS if len(self.pools[b])}
        self.dropped_weight = [b for b in BUCKETS if cfg.decision_weights.get(b, 0.0) > 0 and not len(self.pools[b])]
        total = sum(weights.values())
        self.scp: Optional[PretrainMixture] = None
        self.names: List[str] = []
        self.probs: List[float] = []
        scp_frac = cfg.span_choice_frac
        if total > 0:
            for b, w in weights.items():
                if w > 0:
                    self.names.append(b)
                    self.probs.append((1.0 - scp_frac) * w / total)
        else:
            scp_frac = 1.0 if scp_frac > 0 else 0.0
        if scp_frac > 0:
            self.names.append("scp")
            self.probs.append(scp_frac)
            self.scp = PretrainMixture(
                packer, passages, (0.0, 1.0, 0.0), seed=cfg.seed * 100003 + 17,
                structured_frac=cfg.pretrain_structured_frac,
                max_passage_tokens=min(cfg.max_passage_tokens, model_cfg.max_state_tokens - 48))
        if not self.names:
            raise ValueError("no training data: give teacher_jsonl, converted_jsonl, decision_jsonl or synthetic sources "
                             "(or a positive span_choice_frac)")
        norm = sum(self.probs)
        self.probs = [p / norm for p in self.probs]

    def effective_weights(self) -> Dict[str, float]:
        return {n: round(p, 6) for n, p in zip(self.names, self.probs)}

    def describe(self) -> Dict[str, Any]:
        return {b: [s.describe() for s in self.sources[b]] for b in BUCKETS}

    def fits(self, ex: Rhen2Example, limit: int, key: Tuple[int, int, int]) -> bool:
        hit = self._fit.get(key + (limit,))
        if hit is None:
            try:
                n_state = int(self.packer.pack([ex]).state_mask.sum())
                hit = n_state <= limit
            except ValueError:
                hit = False
            if len(self._fit) > 500_000:
                self._fit.clear()
            self._fit[key + (limit,)] = hit
        return hit

    def draw(self, rng: random.Random, n_questions: int, limit: int, scp_key: int) -> List[Rhen2Example]:
        out: List[Rhen2Example] = []
        nq = scp = attempts = 0
        buckets = list(self.pools)
        while nq < n_questions:
            attempts += 1
            if attempts > 500 * n_questions:
                raise RuntimeError("too many sampled examples do not fit the model's token limits; "
                                   "raise max_state_tokens or filter the data")
            name = rng.choices(self.names, weights=self.probs)[0]
            if name == "scp":
                scp += 1
                nq += 1
                continue
            src, idx = self.pools[name].sample(rng)
            ex = src.get(idx)
            if not self.fits(ex, limit, (buckets.index(name), id(src) & 0xFFFFFF, idx)):
                self.rejected += 1
                continue
            if nq > 0 and nq + len(ex.questions) > n_questions:
                break
            out.append(ex)
            nq += len(ex.questions)
        if scp:
            mix = self.scp.sample_batch(scp, step=scp_key)
            out.extend(e for e in mix.examples if self.fits(e, limit, (-1, scp_key, id(e) & 0xFFFF)))
        return out


# --------------------------------------------------------------------------- work items and loss module
@dataclass
class WorkItem:
    kind: str                                   # "decision" | "mlm"
    units: int                                  # weight of this chunk in the optimizer step
    examples: Optional[List[Rhen2Example]] = None
    mlm: Optional[Dict[str, torch.Tensor]] = None


class LossModule(nn.Module):
    """One module whose ``forward`` computes the loss, so DDP/FSDP see every parameter use."""

    def __init__(self, model: Rhen2Model, head: Optional[MLMHead]):
        super().__init__()
        self.model = model
        self.head = head

    def forward(self, kind: str, payload: Any, plan: Optional[LoopPlan] = None):
        if kind == "decision":
            out = self.model(payload, plan)
            return out.loss, dict(out.loss_parts), float(out.loops_run.float().mean())
        if kind == "mlm":
            loss = mlm_loss(self.model, self.head, payload["state_ids"], payload["targets"],
                            payload["state_mask"], payload["state_num"])
            return loss, {"mlm": float(loss.detach())}, 0.0
        raise ValueError(kind)


class TeacherLabeller:
    """Plan 7.5: label questions with a teacher checkpoint at temperature T and blend with the original label."""

    def __init__(self, path: str, device: torch.device, temperature: float, weight: float, effort: str):
        path = resolve_model_file(path, prefer_ema=True)
        self.model, _ = load_checkpoint(path, map_location="cpu")
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.packer = Rhen2Packer(self.model.cfg)
        self.device, self.temperature, self.weight = device, float(temperature), float(weight)
        self.plan = LoopPlan.effort(self.model.cfg, effort)
        self.skipped = 0

    @staticmethod
    def _hard(q: QuestionSpec, k: int) -> Optional[List[float]]:
        if q.soft is not None:
            return [float(x) for x in q.soft]
        if q.label is None:
            return None
        y = [0.0] * (k + 1)
        if isinstance(q.label, str) and q.label == NULL_LABEL:
            y[k] = 1.0
        elif q.type == "noul":
            y[0 if bool(q.label) else 1] = 1.0
        else:
            y[Rhen2Packer._option_index(q, q.label)] = 1.0
        return y

    @torch.no_grad()
    def blend(self, examples: Sequence[Rhen2Example]) -> List[Rhen2Example]:
        stripped = [Rhen2Example(e.state, [dataclasses.replace(q, label=None, soft=None) for q in e.questions]) for e in examples]
        try:
            batch = self.packer.pack(stripped).to(self.device)
        except ValueError:                       # teacher limits are tighter than the student's: keep original labels
            self.skipped += len(examples)
            return list(examples)
        out = self.model(batch, self.plan)
        probs = out.probs.float()
        m_max = batch.max_questions
        result: List[Rhen2Example] = []
        for b, ex in enumerate(examples):
            qs: List[QuestionSpec] = []
            for m, q in enumerate(ex.questions):
                if q.type not in SOFT_TYPES:
                    qs.append(q)
                    continue
                k = n_options(q)
                row = torch.cat([probs[b * m_max + m, :k], probs[b * m_max + m, -1:]])
                row = row.clamp_min(1e-9)
                teacher = torch.softmax(row.log() / self.temperature, dim=-1).tolist()
                hard = self._hard(q, k)
                w = self.weight
                mixed = teacher if hard is None else [w * t + (1.0 - w) * h for t, h in zip(teacher, hard)]
                qs.append(dataclasses.replace(q, soft=mixed))
            result.append(Rhen2Example(ex.state, qs))
        return result


# --------------------------------------------------------------------------- checkpoint helpers
def resolve_model_file(path: str, prefer_ema: bool = True) -> str:
    """A model file from a file, a step directory or a run directory (latest checkpoint)."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        step_dir = latest_checkpoint(path) if os.path.isdir(os.path.join(path, "checkpoints")) else path
        if step_dir is None:
            raise FileNotFoundError(f"no checkpoint under {path}")
        names = ("model_ema.pt", "model.pt") if prefer_ema else ("model.pt", "model_ema.pt")
        for name in names:
            cand = os.path.join(step_dir, name)
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(f"cannot find a model checkpoint at {path}")


def capture_rng() -> Dict[str, Any]:
    version, internal, gauss = random.getstate()
    state: Dict[str, Any] = {"torch": torch.get_rng_state(), "py_version": int(version),
                             "py_internal": [int(x) for x in internal], "py_gauss": gauss}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].to(torch.uint8).cpu())
    random.setstate((int(state["py_version"]), tuple(state["py_internal"]), state["py_gauss"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


# --------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: TrainConfig, ctx: Optional["distmod.DistributedContext"] = None):
        self.cfg = cfg
        if cfg.stage == "adapter":
            raise NotImplementedError(
                "stage 'adapter' (plan 5.1 option b: pretrained open LLM as state encoder through an adapter) is not "
                "implemented. It needs a pretrained backbone (adapter_backbone) and the v1 rhen/adapter.py as the "
                "starting point; this config validates but cannot train yet.")
        self.ctx = ctx or distmod.init_distributed(cfg.device)
        self.device = self.ctx.device
        self.model_cfg = cfg.model_config()
        self.run_dir = cfg.run_dir
        self.is_main = self.ctx.is_main
        self.precision = devmod.resolve_precision(cfg.precision, self.device)
        devmod.seed_everything(cfg.seed)
        self.model = Rhen2Model(self.model_cfg)
        if cfg.init_from:
            init, _ = load_checkpoint(resolve_model_file(cfg.init_from), map_location="cpu")
            if init.cfg.to_dict() != self.model_cfg.to_dict():
                raise ValueError("init_from checkpoint was trained with a different model config")
            self.model.load_state_dict(init.state_dict())
            del init
        self.model.to(self.device)
        self.head = MLMHead.for_model(self.model).to(self.device) if cfg.stage == "pretrain" else None
        self.packer = Rhen2Packer(self.model_cfg)
        self.ema = EMA(self.model, cfg.ema_decay)
        self.loss_module = LossModule(self.model, self.head)
        self.strategy = choose_strategy(cfg.dist_strategy, self.ctx.world_size, self.model_cfg)
        self.wrapped = wrap_for_training(self.loss_module, self.ctx, self.strategy)
        self.params = [p for p in self.loss_module.parameters() if p.requires_grad]
        groups = build_param_groups(self.model, cfg.lr, cfg.weight_decay, cfg.base_width)
        if self.head is not None:
            mats = [p for p in self.head.parameters() if p.ndim >= 2]
            vecs = [p for p in self.head.parameters() if p.ndim < 2]
            groups.append({"name": "mlm_decay", "params": mats, "lr": cfg.lr, "lr_mult": 1.0, "weight_decay": cfg.weight_decay})
            groups.append({"name": "mlm_no_decay", "params": vecs, "lr": cfg.lr, "lr_mult": 1.0, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        self.loop_gen = torch.Generator()
        self.loop_gen.manual_seed(cfg.seed + 12345 + self.ctx.rank)
        self.step = 0
        self.run_id = ""
        self.manifest: Dict[str, Any] = {}
        self.nonfinite_skips = 0
        self.writer = None
        self.labeller: Optional[TeacherLabeller] = None
        self.data: Optional[DecisionData] = None
        self.mixture: Optional[PretrainMixture] = None
        self._mixture_limit = -1
        self.eval_examples: List[Rhen2Example] = []
        self.eval_families: List[str] = []
        self.data_info: Dict[str, Any] = {}
        self.passages: List[str] = []

    # -- data
    def build_data(self) -> None:
        cfg = self.cfg
        blocklist, bl_info = make_blocklist(cfg)
        screen = _Screen(blocklist, cfg.blocklist_action, cfg.blocklist_ngram_threshold)
        passages: List[str] = []
        corpus_info: Dict[str, Any] = {}
        if cfg.pretrain_corpus_dir:
            corpus = TextCorpus(cfg.pretrain_corpus_dir)
            passages = [p for i, p in enumerate(corpus.passages())
                        if screen.ok(Rhen2Example(p, []), f"{cfg.pretrain_corpus_dir}[{i}]")]
            corpus_info = {"dir": cfg.pretrain_corpus_dir, "files": {f: file_sha256(f) for f in corpus.files},
                           "passages": len(passages)}
        self.passages = passages
        if cfg.stage == "pretrain":
            self.data_info = {"stage": "pretrain", "corpus": corpus_info, "weights": list(cfg.pretrain_weights)}
        else:
            self.data = DecisionData(cfg, self.packer, self.model_cfg, screen, passages)
            self.data_info = {"stage": cfg.stage, "sources": self.data.describe(), "corpus": corpus_info,
                              "effective_weights": self.data.effective_weights(),
                              "weights_dropped_for_empty_sources": self.data.dropped_weight}
        self.data_info["blocklist"] = bl_info
        self.data_info["blocklist_dropped"] = screen.dropped
        self._build_eval()
        if cfg.stage == "distill":
            self.labeller = TeacherLabeller(cfg.teacher_checkpoint, self.device, cfg.distill_temperature,
                                            cfg.distill_teacher_weight, cfg.teacher_effort)

    def _build_eval(self) -> None:
        cfg = self.cfg
        if cfg.stage == "pretrain":
            return
        none_screen = _Screen(Blocklist({}), "raise", None)
        examples: List[Rhen2Example] = []
        fams: List[str] = []
        for p in cfg.eval_jsonl:
            src = JsonlSource(p, none_screen)
            examples += [src.get(i) for i in range(len(src))]
            fams += ["jsonl"] * len(src)
        spec = cfg.eval_synthetic
        if spec is None and not cfg.eval_jsonl:
            spec = {"split": "val", "n": 32, "seed": cfg.seed}
        if spec is not None:
            exs = generate_split(spec.get("split", "val"), int(spec.get("n", 32)), int(spec.get("seed", cfg.seed)),
                                 spec.get("families") or None)
            examples += exs
            fams += ["synthetic"] * len(exs)
        keep = []
        for ex in examples:
            try:
                self.packer.pack([ex])
                keep.append(ex)
            except ValueError:
                pass
        self.eval_dropped = len(examples) - len(keep)
        self.eval_examples = keep

    def _pretrain_mixture(self, limit: int) -> PretrainMixture:
        if self.mixture is None or limit != self._mixture_limit:
            cfg = self.cfg
            self.mixture = PretrainMixture(
                self.packer, self.passages, tuple(cfg.pretrain_weights), seed=cfg.seed * 100003 + self.ctx.rank,
                structured_frac=cfg.pretrain_structured_frac,
                max_passage_tokens=max(8, min(cfg.max_passage_tokens, limit - 48)))
            self._mixture_limit = limit
        return self.mixture

    def _fits(self, ex: Rhen2Example, limit: int) -> bool:
        try:
            return int(self.packer.pack([ex]).state_mask.sum()) <= limit
        except ValueError:
            return False

    def _mix_to_work(self, mix, limit: int, mlm_rng: random.Random) -> List[WorkItem]:
        work: List[WorkItem] = []
        if mix.mlm:
            batch = collate_mlm(mix.mlm, self.packer, 0.3, mlm_rng, max_tokens=limit - 1)
            work.append(WorkItem("mlm", len(mix.mlm), mlm=batch))
        exs = [e for e in mix.examples if self._fits(e, limit)]
        for chunk in make_batches(exs, self.cfg.batch_questions):
            work.append(WorkItem("decision", sum(len(e.questions) for e in chunk), examples=chunk))
        return work

    def make_work(self, step: int) -> List[WorkItem]:
        """All chunks of optimizer step ``step`` (0-based): a pure function of (seed, step, rank)."""
        cfg = self.cfg
        limit = cfg.limit_at(step)
        work: List[WorkItem] = []
        for micro in range(cfg.grad_accum):
            key = step * cfg.grad_accum + micro
            if cfg.stage == "pretrain":
                mix = self._pretrain_mixture(limit).sample_batch(cfg.batch_questions, key)
                work += self._mix_to_work(mix, limit, random.Random(f"mlm:{cfg.seed}:{self.ctx.rank}:{key}"))
            else:
                rng = random.Random(f"decision:{cfg.seed}:{self.ctx.rank}:{key}")
                exs = self.data.draw(rng, cfg.batch_questions, limit, key + self.ctx.rank * 10_000_019)
                for chunk in make_batches(exs, cfg.batch_questions):
                    work.append(WorkItem("decision", sum(len(e.questions) for e in chunk), examples=chunk))
        if not work:
            raise RuntimeError(f"step {step} produced no work")
        return work

    # -- optimisation
    def _full_params(self):
        if distmod.is_fsdp(self.wrapped):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            return FSDP.summon_full_params(self.wrapped, writeback=False)
        return contextlib.nullcontext()

    def _next_plan(self) -> LoopPlan:
        if self.cfg.fixed_loops is not None:
            n = self.cfg.fixed_loops
            return LoopPlan(n_loops=n, k_bptt=self.model_cfg.k_bptt, min_loops=n, train=True)
        if self.cfg.loop_sampling:
            return LoopPlan.sample(self.model_cfg, self.loop_gen)
        r = self.model_cfg.r_mean
        return LoopPlan(n_loops=r, k_bptt=self.model_cfg.k_bptt, min_loops=r, train=True)

    def accumulate(self, work: Sequence[WorkItem], plan: Optional[LoopPlan] = None) -> Dict[str, float]:
        """Zero gradients, then backpropagate ``sum_i units_i / total_units * loss_i``; gradients are left in place.
        Returns unit-weighted means of the logged loss parts. ``plan`` overrides loop sampling (tests)."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = float(sum(w.units for w in work))
        sums: Dict[str, float] = {}
        loops = []
        for i, w in enumerate(work):
            if w.kind == "decision":
                examples = self.labeller.blend(w.examples) if self.labeller is not None else w.examples
                payload: Any = self.packer.pack(examples).to(self.device)
                item_plan = plan if plan is not None else self._next_plan()
            else:
                payload = {k: v.to(self.device) for k, v in w.mlm.items()}
                item_plan = None
            sync = distmod.no_sync(self.wrapped, i < len(work) - 1)
            with sync, devmod.autocast(self.device, self.precision):
                loss, parts, n_loops = self.wrapped(w.kind, payload, item_plan)
            (loss * (w.units / total)).backward()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v * w.units / total
            if w.kind == "decision":
                loops.append(n_loops)
        if loops:
            sums["loops"] = sum(loops) / len(loops)
        sums["questions"] = total
        return sums

    def train_step(self, step: int) -> Dict[str, float]:
        cfg = self.cfg
        lr = lr_at(step, cfg.steps, cfg.lr, cfg.warmup_frac, cfg.min_lr_ratio)
        set_lr(self.optimizer, lr)
        stats = self.accumulate(self.make_work(step))
        if distmod.is_fsdp(self.wrapped):
            gnorm = float(self.wrapped.clip_grad_norm_(cfg.grad_clip))
        else:
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip))
        if math.isfinite(gnorm) and math.isfinite(stats.get("loss", 0.0)):
            self.optimizer.step()
            with self._full_params():
                self.ema.update(self.model)
        else:
            self.nonfinite_skips += 1
            log.warning("step %d: non-finite loss or gradient norm, update skipped", step + 1)
        stats.update({"lr": lr, "grad_norm": gnorm, "nonfinite_skips": float(self.nonfinite_skips)})
        if self.data is not None:
            stats["rejected_oversize"] = float(self.data.rejected)
        return stats

    # -- evaluation
    @torch.no_grad()
    def _eval_pass(self, model: Rhen2Model) -> Dict[str, float]:
        """Metrics for the weights currently in ``model`` (EMA weights when called inside ``ema.swap_into``)."""
        cfg = self.cfg
        was = model.training
        model.eval()
        plan = LoopPlan(n_loops=self.model_cfg.r_mean, min_loops=self.model_cfg.r_mean)
        loss_sum = n_total = correct = n_disc = 0.0
        out_metrics: Dict[str, float] = {}
        try:
            groups: List[List[Rhen2Example]] = []
            mlm_losses, mlm_n = 0.0, 0
            if cfg.stage == "pretrain":
                limit = self.model_cfg.max_state_tokens
                exs: List[Rhen2Example] = []
                for i in range(cfg.eval_batches):
                    mix = self._pretrain_mixture(cfg.limit_at(0)).sample_batch(cfg.batch_questions, 10**9 + i)
                    if mix.mlm:
                        b = collate_mlm(mix.mlm, self.packer, 0.3, random.Random(f"eval-mlm:{cfg.seed}:{i}"), max_tokens=limit - 1)
                        b = {k: v.to(self.device) for k, v in b.items()}
                        with devmod.autocast(self.device, self.precision):
                            l = mlm_loss(model, self.head, b["state_ids"], b["targets"], b["state_mask"], b["state_num"])
                        mlm_losses += float(l) * len(mix.mlm)
                        mlm_n += len(mix.mlm)
                    exs += [e for e in mix.examples if self._fits(e, limit)]
                groups = make_batches(exs, cfg.batch_questions)
            else:
                groups = make_batches(self.eval_examples, cfg.batch_questions)
            for group in groups:
                batch = self.packer.pack(group).to(self.device)
                with devmod.autocast(self.device, self.precision):
                    out = model(batch, plan)
                nq = float(batch.q_valid.sum())
                loss_sum += float(out.loss) * nq
                n_total += nq
                k = out.probs.shape[1] - 1
                idx = batch.label_idx.reshape(-1)
                has = (idx >= 0) | (idx == NULL_INDEX)
                gold = torch.where(idx == NULL_INDEX, torch.full_like(idx, k), idx)
                hit = (out.probs.argmax(-1) == gold) & has
                correct += float(hit.sum())
                n_disc += float(has.sum())
            if mlm_n:
                out_metrics["mlm_loss"] = mlm_losses / mlm_n
            if n_total:
                out_metrics["decision_loss"] = loss_sum / n_total
            if n_disc:
                out_metrics["accuracy"] = correct / n_disc
            total = (mlm_losses + loss_sum) / max(1.0, mlm_n + n_total)
            out_metrics["loss"] = total
            out_metrics["n_questions"] = float(mlm_n + n_total)
        finally:
            model.train(was)
        return out_metrics

    def evaluate(self, step: int) -> Dict[str, Any]:
        rec: Dict[str, Any] = {"kind": "eval", "step": step}
        with self._full_params():
            if self.cfg.eval_raw:
                for k, v in self._eval_pass(self.model).items():
                    rec[f"raw_{k}"] = v
            with self.ema.swap_into(self.model):
                for k, v in self._eval_pass(self.model).items():
                    rec[f"ema_{k}"] = v
        return rec

    # -- persistence
    def _ckpt_extra(self, step: int) -> Dict[str, Any]:
        return {"stage": self.cfg.stage, "step": step, "training_run_id": self.run_id, "preset": self.cfg.preset,
                "smoke_only": self.cfg.preset == "r2-tiny", "train_config_sha256": self.cfg.sha256()}

    def save(self, step: int) -> Optional[str]:
        with self._full_params():
            if self.is_main:
                final = checkpoint_dir(self.run_dir, step)
                tmp = final + ".tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp)
                extra = self._ckpt_extra(step)
                save_checkpoint(os.path.join(tmp, "model.pt"), self.model, self.model_cfg, extra)
                with self.ema.swap_into(self.model):
                    save_checkpoint(os.path.join(tmp, "model_ema.pt"), self.model, self.model_cfg, dict(extra, weights="ema"))
                state = {
                    "format": TRAINER_STATE_FORMAT, "step": step, "optimizer": self.optimizer.state_dict(),
                    "ema": self.ema.state_dict(), "head": self.head.state_dict() if self.head is not None else None,
                    "rng": capture_rng(), "loop_gen": self.loop_gen.get_state(),
                    "nonfinite_skips": self.nonfinite_skips, "train_config_sha256": self.cfg.sha256(),
                    "training_run_id": self.run_id,
                }
                torch.save(state, os.path.join(tmp, "trainer_state.pt"))
                if os.path.isdir(final):
                    shutil.rmtree(final)
                os.replace(tmp, final)
                pointer = os.path.join(self.run_dir, "checkpoints", "latest")
                with open(pointer + ".tmp", "w", encoding="utf-8") as fh:
                    fh.write(os.path.basename(final))
                os.replace(pointer + ".tmp", pointer)
                root = os.path.join(self.run_dir, "checkpoints")
                steps = sorted(d for d in os.listdir(root) if d.startswith("step_") and not d.endswith(".tmp"))
                for old in steps[:-self.cfg.keep_last]:
                    shutil.rmtree(os.path.join(root, old), ignore_errors=True)
        distmod.barrier(self.ctx)
        return checkpoint_dir(self.run_dir, step)

    def load(self, path: str) -> None:
        state = torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu", weights_only=True)
        if state.get("format") != TRAINER_STATE_FORMAT:
            raise ValueError(f"unsupported trainer state format {state.get('format')!r}")
        init, _ = load_checkpoint(os.path.join(path, "model.pt"), map_location="cpu")
        self.model.load_state_dict(init.state_dict())
        del init
        self.optimizer.load_state_dict(state["optimizer"])
        self.ema.load_state_dict(state["ema"])
        self.ema.shadow = {k: v.to(self.device) for k, v in self.ema.shadow.items()}
        if self.head is not None and state["head"] is not None:
            self.head.load_state_dict(state["head"])
        self.loop_gen.set_state(state["loop_gen"].to(torch.uint8).cpu())
        restore_rng(state["rng"])
        self.nonfinite_skips = int(state["nonfinite_skips"])
        self.step = int(state["step"])
        self.run_id = state.get("training_run_id", self.run_id)

    # -- bookkeeping
    @property
    def metrics_path(self) -> str:
        return os.path.join(self.run_dir, "metrics.jsonl")

    def _log(self, rec: Dict[str, Any]) -> None:
        if not self.is_main:
            return
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        if self.writer is not None:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and k not in ("step", "wall"):
                    self.writer.add_scalar(f"{rec['kind']}/{k}", v, rec["step"])

    def _truncate_metrics(self, step: int) -> None:
        if not self.is_main or not os.path.exists(self.metrics_path):
            return
        with open(self.metrics_path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        keep = [ln for ln in lines if json.loads(ln).get("step", 0) <= step]
        with open(self.metrics_path, "w", encoding="utf-8") as fh:
            fh.writelines(keep)

    def _write_manifest(self) -> None:
        cfg = self.cfg
        formula = count_parameters(self.model_cfg)
        actual = int(sum(p.numel() for p in self.model.parameters()))
        extra = {
            "training_run_id": self.run_id, "stage": cfg.stage, "preset": cfg.preset,
            "smoke_only": cfg.preset == "r2-tiny", "notice": SMOKE_NOTICE if cfg.preset == "r2-tiny" else "",
            "parameters": {"count_parameters_total": formula["total"], "actual": actual,
                           "match": formula["total"] == actual, "by_component": formula,
                           "mlm_head": self.head.count() if self.head is not None else 0},
            "data": self.data_info, "strategy": self.strategy, "precision": self.precision.to_dict(),
            "train_config_sha256_critical": canonical_hash(
                {k: v for k, v in cfg.to_dict().items() if k not in _NOT_RESUME_CRITICAL}),
        }
        self.manifest = build_manifest(cfg.to_dict(), self.model_cfg.to_dict(), self.device, self.ctx.world_size, extra)
        write_json(os.path.join(self.run_dir, "manifest.json"), self.manifest)

    def _prepare_run(self) -> Optional[str]:
        """Create or resume the run directory; returns the checkpoint path resumed from (or None)."""
        cfg = self.cfg
        os.makedirs(self.run_dir, exist_ok=True)
        latest = latest_checkpoint(self.run_dir) if cfg.resume else None
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        if latest and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as fh:
                old = json.load(fh)
            now = canonical_hash({k: v for k, v in cfg.to_dict().items() if k not in _NOT_RESUME_CRITICAL})
            if old.get("train_config_sha256_critical") != now:
                raise ValueError(f"cannot resume {latest}: the config differs from the one the run started with "
                                 "(only cadence, eval, device and logging keys may change)")
            self.run_id = old.get("training_run_id", "")
        if latest:
            self.load(latest)
            self._truncate_metrics(self.step)
            if not self.run_id:
                self.run_id = distmod.broadcast_object(uuid.uuid4().hex[:16], self.ctx)
            return latest
        self.run_id = distmod.broadcast_object(uuid.uuid4().hex[:16], self.ctx)
        if self.is_main:
            if os.path.exists(self.metrics_path):
                os.remove(self.metrics_path)
            self._write_manifest()
        return None

    # -- main loop
    def train(self, stop_at: Optional[int] = None) -> Dict[str, Any]:
        cfg = self.cfg
        if cfg.preset == "r2-tiny":
            log.warning(SMOKE_NOTICE)
        self.build_data()
        resumed = self._prepare_run()
        if cfg.tensorboard and self.is_main:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(os.path.join(self.run_dir, "tb"))
            except ImportError:
                log.info("tensorboard is not installed; metrics go to metrics.jsonl only")
        if self.is_main and resumed is None:
            self._log({"kind": "notice", "step": 0, "text": SMOKE_NOTICE if cfg.preset == "r2-tiny" else "run started",
                       "stage": cfg.stage})
        target = cfg.steps if stop_at is None else min(stop_at, cfg.steps)
        log.info("stage=%s steps %d -> %d (total %d) run_id=%s resumed_from=%s", cfg.stage, self.step, target, cfg.steps,
                 self.run_id, resumed)
        t0 = time.time()
        last: Dict[str, Any] = {}
        while self.step < target:
            stats = self.train_step(self.step)
            self.step += 1
            s = self.step
            if s % cfg.log_every == 0 or s == target:
                rec = {"kind": "train", "step": s, "wall": round(time.time() - t0, 3)}
                rec.update({k: float(v) for k, v in stats.items()})
                self._log(rec)
                last = rec
            if s % cfg.eval_every == 0 or s == cfg.steps:
                if self.is_main:
                    ev = self.evaluate(s)
                    self._log(ev)
                    log.info("eval step %d: %s", s, {k: round(v, 4) for k, v in ev.items() if isinstance(v, float)})
                distmod.barrier(self.ctx)
            if s % cfg.save_every == 0 or s == target:
                self.save(s)
        if self.writer is not None:
            self.writer.close()
        summary = {"run_dir": self.run_dir, "step": self.step, "training_run_id": self.run_id, "resumed_from": resumed,
                   "last_train": last, "smoke_only": cfg.preset == "r2-tiny",
                   "notice": SMOKE_NOTICE if cfg.preset == "r2-tiny" else ""}
        return summary


# --------------------------------------------------------------------------- dry run
def plan_run(cfg: TrainConfig, check_data: bool = True) -> Dict[str, Any]:
    """Validate the config and data and describe the run without training or writing anything."""
    mc = cfg.model_config()
    formula = count_parameters(mc)
    real: Optional[int] = None
    if formula["total"] <= 120_000_000:
        real = int(sum(p.numel() for p in Rhen2Model(mc).parameters()))
    plan: Dict[str, Any] = {
        "stage": cfg.stage, "preset": cfg.preset, "dry_run": True, "run_dir": cfg.run_dir,
        "parameters": {"count_parameters_total": formula["total"], "actual": real,
                       "actual_note": None if real is not None else "not instantiated in a dry run above 120M parameters"},
        "steps": cfg.steps, "batch_questions": cfg.batch_questions, "grad_accum": cfg.grad_accum,
        "effective_questions_per_step_per_rank": cfg.batch_questions * cfg.grad_accum,
        "lr_schedule": {"peak": cfg.lr, "warmup_steps": max(1, round(cfg.steps * cfg.warmup_frac)),
                        "final": cfg.lr * cfg.min_lr_ratio, "muP_base_width": cfg.base_width,
                        "hidden_matrix_lr_multiplier": cfg.base_width / mc.d_model},
        "precision": cfg.precision, "ema_decay": cfg.ema_decay, "gradient_checkpointing": mc.gradient_checkpointing,
        "loops": {"r_min": mc.r_min, "r_mean": mc.r_mean, "r_max": mc.r_max,
                  "mode": f"fixed {cfg.fixed_loops}" if cfg.fixed_loops else ("sampled" if cfg.loop_sampling else "r_mean")},
        "state_len_curriculum": cfg.state_len_curriculum,
        "checkpoints": {"every": cfg.save_every, "keep_last": cfg.keep_last}, "eval_every": cfg.eval_every,
        "strategy_if_multi_process": choose_strategy(cfg.dist_strategy, 2, mc),
    }
    if cfg.preset == "r2-tiny":
        plan["notice"] = SMOKE_NOTICE
    if cfg.stage == "adapter":
        plan["note"] = "adapter stage validates but training raises NotImplementedError (needs a pretrained backbone)"
        return plan
    if cfg.init_from:
        plan["init_from"] = {"path": cfg.init_from, "exists": os.path.exists(cfg.init_from)}
        if check_data:
            resolve_model_file(cfg.init_from)
    if cfg.stage == "distill" and check_data:
        resolve_model_file(cfg.teacher_checkpoint)
        plan["teacher_checkpoint"] = cfg.teacher_checkpoint
    if check_data:
        packer = Rhen2Packer(mc)
        blocklist, bl_info = make_blocklist(cfg)
        screen = _Screen(blocklist, cfg.blocklist_action, cfg.blocklist_ngram_threshold)
        data_plan: Dict[str, Any] = {"blocklist": bl_info}
        passages: List[str] = []
        if cfg.pretrain_corpus_dir:
            corpus = TextCorpus(cfg.pretrain_corpus_dir)
            passages = [p for i, p in enumerate(corpus.passages()) if screen.ok(Rhen2Example(p, []), f"corpus[{i}]")]
            data_plan["corpus"] = {"dir": cfg.pretrain_corpus_dir, "files": len(corpus.files), "passages": len(passages)}
        if cfg.stage == "pretrain":
            data_plan["weights"] = list(cfg.pretrain_weights)
            mix = PretrainMixture(packer, passages, tuple(cfg.pretrain_weights), seed=cfg.seed,
                                  structured_frac=cfg.pretrain_structured_frac,
                                  max_passage_tokens=max(8, min(cfg.max_passage_tokens, mc.max_state_tokens - 48)))
            b = mix.sample_batch(min(cfg.batch_questions, 16), 0)
            for ex in b.examples:
                packer.pack([ex])
            data_plan["sample_batch"] = {"mlm": len(b.mlm), "decision_examples": len(b.examples)}
        else:
            dd = DecisionData(cfg, packer, mc, screen, passages)
            data_plan.update({"sources": dd.describe(), "effective_weights": dd.effective_weights(),
                              "weights_dropped_for_empty_sources": dd.dropped_weight})
            rng = random.Random(f"dryrun:{cfg.seed}")
            drawn = dd.draw(rng, min(cfg.batch_questions, 16), cfg.limit_at(0), 0)
            data_plan["sample_draw"] = {"examples": len(drawn), "rejected_oversize": dd.rejected}
            data_plan["blocklist_dropped"] = screen.dropped
        plan["data"] = data_plan
    return plan


def run_training(cfg: TrainConfig, dry_run: bool = False, stop_at: Optional[int] = None) -> Dict[str, Any]:
    """Entry point used by the CLI and the tests."""
    if dry_run:
        return plan_run(cfg)
    trainer = Trainer(cfg)
    try:
        return trainer.train(stop_at=stop_at)
    finally:
        distmod.cleanup()


__all__ = ["TrainConfig", "Trainer", "WorkItem", "apply_overrides", "choose_strategy", "wrap_for_training", "plan_run",
           "run_training", "resolve_model_file", "SMOKE_NOTICE", "BlockedExampleError"]
