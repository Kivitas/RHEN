"""Train EOS on language-model data, Rhen decision data, or both.

Single process (CPU, one GPU/NPU/XPU, Apple MPS)::

    python -m eos_research.train --config configs/tiny_decisions.json

Multi-device or multi-node (any launcher that sets RANK/WORLD_SIZE/LOCAL_RANK)::

    torchrun --nproc_per_node 8 -m eos_research.train --config cfg.json --set strategy=fsdp

Any field of ``TrainConfig`` can be overridden with ``--set key=value`` (values are parsed as
JSON when possible; ``model_overrides.hidden_size=512`` reaches into the nested dict).

The run directory receives ``manifest.json`` (git commit, config hashes, environment),
``train_config.json``, ``metrics.jsonl`` (one line per logged step and per evaluation),
``checkpoints/`` and ``final_metrics.json``. A run restarted with the same ``run_dir``
resumes from its latest checkpoint and continues with exactly the batches and random
numbers an uninterrupted run would have used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from eos_research import checkpointing as ckpt
from eos_research import device as devmod
from eos_research import distributed as distmod
from eos_research.data import (
    DecisionExample,
    TokenWindows,
    lm_inputs,
    load_decision_jsonl,
    read_meta,
    sample_decision_batch,
)
from eos_research.manifest import build_manifest, write_json
from eos_research.presets import preset_config

SCHEDULES = ("cosine", "wsd", "constant")
OPTIMIZERS = ("adamw", "muon")


@dataclass
class TrainConfig:
    # Run
    run_dir: str = "runs/default"
    seed: int = 0
    deterministic: bool = False
    resume: str = "auto"  # "auto" (latest checkpoint in run_dir, if any) | "never" | a checkpoint dir
    # Hardware
    device: str = "auto"  # auto | cpu | cuda | npu | xpu | mps | cuda:1 ...
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    strategy: str = "ddp"  # used when WORLD_SIZE > 1: ddp | fsdp | none
    compile: bool = False
    # Model
    preset: str = "tiny"
    model_overrides: Dict[str, Any] = field(default_factory=dict)
    init_from: Optional[str] = None  # checkpoint dir whose weights initialise the model
    gradient_checkpointing: bool = False
    tokenizer_path: Optional[str] = None
    # Language-model data (output of eos_research.prepare)
    lm_data: Optional[str] = None
    seq_len: int = 256
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    lm_loss_coef: float = 1.0
    # Decision data (JSON lines, see eos_research.data)
    decision_train: Optional[str] = None
    decision_val: Optional[str] = None
    decision_batch_size: int = 4  # examples per micro-step
    decision_every: int = 1  # decision micro-steps on every N-th optimizer step
    decision_loss_coef: float = 1.0
    order_consistency_coef: float = 0.0  # weight of the option-order consistency objective
    decision_num_loops: Optional[int] = None
    # Optimisation
    optimizer: str = "adamw"  # adamw | muon (Muon for 2-D weights, AdamW for the rest)
    lr: float = 3e-4
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    schedule: str = "cosine"
    warmup_steps: int = 100
    max_steps: int = 1000
    min_lr_ratio: float = 0.1
    decay_fraction: float = 0.2  # WSD: final fraction of training spent decaying
    # Logging, evaluation, checkpoints
    log_every: int = 10
    eval_every: int = 200
    eval_lm_batches: int = 20
    eval_decision_examples: int = 200
    save_every: int = 500
    keep_last: int = 3
    tensorboard: bool = False
    wandb_project: Optional[str] = None

    def validate(self) -> None:
        if not self.lm_data and not self.decision_train:
            raise ValueError("set lm_data, decision_train or both")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.strategy not in distmod.STRATEGIES:
            raise ValueError(f"strategy must be one of {distmod.STRATEGIES}")
        if self.optimizer == "muon" and self.strategy == "fsdp":
            raise ValueError("Muon orthogonalises whole matrices and cannot run on FSDP-sharded parameters")
        for name in ("grad_accum_steps", "micro_batch_size", "decision_batch_size", "decision_every", "max_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig fields: {sorted(unknown)}")
        return cls(**data)


def lr_factor(step: int, cfg: TrainConfig) -> float:
    """Learning-rate multiplier after ``step`` optimizer steps (linear warmup, then the schedule)."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return 1.0
    if cfg.schedule == "wsd":
        decay_steps = max(1, int(round(cfg.decay_fraction * cfg.max_steps)))
        decay_start = cfg.max_steps - decay_steps
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / decay_steps)
        return 1.0 - (1.0 - cfg.min_lr_ratio) * progress
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class TrainingModule(nn.Module):
    """One ``forward`` for every objective, so DDP/FSDP synchronise all of their gradients.

    ``forward("lm", (input_ids, labels))`` returns the language-model loss;
    ``forward("decision", (examples, generator))`` returns the mean Rhen multi-task loss over
    the examples, plus the option-order consistency objective when its weight is positive.
    """

    def __init__(self, model: nn.Module, tokenizer=None, order_consistency_coef: float = 0.0,
                 decision_num_loops: Optional[int] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.order_consistency_coef = float(order_consistency_coef)
        self.decision_num_loops = decision_num_loops
        self._packer = None

    @property
    def packer(self):
        if self._packer is None:
            from rhen_packer import get_rhen_packer

            self._packer = get_rhen_packer(self.tokenizer)
        return self._packer

    def forward(self, kind: str, payload):
        if kind == "lm":
            input_ids, labels = payload
            out = self.model(input_ids=input_ids, labels=labels)
            return out.loss, {"lm_loss": float(out.loss.detach())}
        if kind == "decision":
            return self._decision_loss(*payload)
        raise ValueError(f"unknown objective {kind!r}")

    def _decision_loss(self, examples: Sequence[DecisionExample], generator: torch.Generator):
        from rhen_loss import compute_option_order_consistency_loss, compute_rhen_multi_task_loss

        core = distmod.unwrap(self.model)
        device = next(core.parameters()).device
        total = torch.zeros((), device=device)
        stats = {"decision_loss": 0.0, "order_consistency": 0.0}
        for ex in examples:
            items = [q.packer_item() for q in ex.questions]
            packed = self.packer.pack(state=ex.state, questions_and_options=items)
            results = core.forward_rhen(packed, num_loops=self.decision_num_loops)
            loss, _ = compute_rhen_multi_task_loss(
                core, packed, [q.label for q in ex.questions], device, precomputed_results=results
            )
            stats["decision_loss"] += float(loss.detach()) / len(examples)
            if self.order_consistency_coef > 0.0:
                consistency, _ = compute_option_order_consistency_loss(
                    core, ex.state, items, original_results=results, generator=generator,
                    tokenizer=self.tokenizer, num_loops=self.decision_num_loops,
                )
                loss = loss + self.order_consistency_coef * consistency
                stats["order_consistency"] += float(consistency.detach()) / len(examples)
            total = total + loss / len(examples)
        return total, stats


def build_optimizers(model: nn.Module, cfg: TrainConfig) -> List[torch.optim.Optimizer]:
    if cfg.optimizer == "muon":
        from optimizers import build_muon_adamw

        muon, adamw, _ = build_muon_adamw(
            model, adamw_lr=cfg.lr, muon_lr=cfg.muon_lr, adamw_betas=(cfg.beta1, cfg.beta2),
            adamw_weight_decay=cfg.weight_decay,
        )
        return [muon, adamw]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return [torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))]


class MetricsLogger:
    """JSON lines on rank 0, plus TensorBoard and Weights & Biases when requested and installed."""

    def __init__(self, cfg: TrainConfig, ctx: distmod.DistributedContext):
        self.enabled = ctx.is_main
        self.path = os.path.join(cfg.run_dir, "metrics.jsonl")
        self.tb = None
        self.wandb = None
        if not self.enabled:
            return
        if cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(os.path.join(cfg.run_dir, "tensorboard"))
            except ImportError:
                print("tensorboard is not installed; logging to metrics.jsonl only")
        if cfg.wandb_project:
            try:
                import wandb

                self.wandb = wandb.init(project=cfg.wandb_project, dir=cfg.run_dir, config=asdict(cfg), resume="allow")
            except ImportError:
                print("wandb is not installed; logging to metrics.jsonl only")

    def log(self, step: int, values: Dict[str, Any], kind: str = "train") -> None:
        if not self.enabled:
            return
        record = {"kind": kind, "step": step, **values}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=float) + "\n")
        scalars = {k: v for k, v in values.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(f"{kind}/{k}", v, step)
        if self.wandb is not None:
            self.wandb.log({f"{kind}/{k}": v for k, v in scalars.items()}, step=step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        cfg.validate()
        self.cfg = cfg
        self.ctx = distmod.init_distributed(cfg.device)
        self.device = self.ctx.device
        devmod.seed_everything(cfg.seed, self.ctx.rank, cfg.deterministic)
        self.policy = devmod.resolve_precision(cfg.precision, self.device)
        if self.ctx.is_main:
            os.makedirs(cfg.run_dir, exist_ok=True)

        self.tokenizer = self._load_tokenizer()
        self.lm_train = TokenWindows(cfg.lm_data, "train", cfg.seq_len) if cfg.lm_data else None
        self.lm_val = self._optional_windows("val")
        self.decision_train = load_decision_jsonl(cfg.decision_train) if cfg.decision_train else []
        self.decision_val = load_decision_jsonl(cfg.decision_val) if cfg.decision_val else []

        self.model = self._build_model()
        self.module = TrainingModule(self.model, self.tokenizer, cfg.order_consistency_coef, cfg.decision_num_loops)
        if cfg.compile:
            self.module.model = torch.compile(self.model)
        self.wrapped = distmod.wrap_model(self.module, self.ctx, cfg.strategy)
        self.optimizers = build_optimizers(self.model, cfg)
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, lambda s, c=cfg: lr_factor(s, c)) for opt in self.optimizers
        ]
        self.scaler = devmod.make_grad_scaler(self.device, self.policy)
        self.step = 0
        self.logger = MetricsLogger(cfg, self.ctx)
        self._resume()

    # -- setup ------------------------------------------------------------------------------
    def _load_tokenizer(self):
        if not (self.cfg.decision_train or self.cfg.tokenizer_path):
            return None
        from tokenizer import DEFAULT_TOKENIZER_DIR, get_tokenizer

        return get_tokenizer(self.cfg.tokenizer_path or DEFAULT_TOKENIZER_DIR)

    def _optional_windows(self, split: str) -> Optional[TokenWindows]:
        if not self.cfg.lm_data:
            return None
        try:
            return TokenWindows(self.cfg.lm_data, split, self.cfg.seq_len)
        except (FileNotFoundError, ValueError):
            return None

    def _build_model(self):
        from config import EOSConfig
        from model import EOSModel

        kwargs = preset_config(self.cfg.preset, **self.cfg.model_overrides)
        if "vocab_size" not in kwargs:
            needed = []
            if self.cfg.lm_data:
                needed.append(int(read_meta(self.cfg.lm_data)["vocab_size"]))
            if self.tokenizer is not None:
                needed.append(int(self.tokenizer.vocab_size))
            if needed:
                kwargs["vocab_size"] = max(needed)
        if self.cfg.gradient_checkpointing:
            kwargs["gradient_checkpointing"] = True
        model = EOSModel(EOSConfig(**kwargs))
        if self.cfg.init_from:
            ckpt.load_model_weights(model, self.cfg.init_from)
        model.to(self.device)
        # One congestion-bias update per optimizer step, averaged over data-parallel ranks.
        model.set_moe_balance_update_mode("step")
        model.auto_apply_moe_balance = False
        return model

    def _resume(self) -> None:
        cfg = self.cfg
        path = None
        if cfg.resume == "auto":
            path = ckpt.latest_checkpoint(cfg.run_dir)
        elif cfg.resume not in ("never", "", None):
            path = cfg.resume
        if path:
            state = ckpt.load_checkpoint(
                path, self.wrapped, self.model, self.optimizers, self.schedulers, self.scaler, self.ctx
            )
            self.step = int(state["step"])
            if self.ctx.is_main:
                print(f"resumed from {path} at step {self.step}")
        elif self.ctx.is_main:
            manifest = build_manifest(
                asdict(cfg), self.model.config.to_dict(), self.device, self.ctx.world_size,
                extra={"parameters": self.model.count_parameters(), "precision": self.policy.to_dict()},
            )
            write_json(os.path.join(cfg.run_dir, "manifest.json"), manifest)
            write_json(os.path.join(cfg.run_dir, "train_config.json"), asdict(cfg))

    # -- one optimizer step ------------------------------------------------------------------
    def _micro_tasks(self, step: int) -> List[Tuple[str, int]]:
        tasks: List[Tuple[str, int]] = []
        if self.lm_train is not None:
            tasks += [("lm", m) for m in range(self.cfg.grad_accum_steps)]
        if self.decision_train and step % self.cfg.decision_every == 0:
            tasks += [("decision", m) for m in range(self.cfg.grad_accum_steps)]
        return tasks

    def train_step(self) -> Dict[str, float]:
        cfg, ctx, step = self.cfg, self.ctx, self.step
        self.wrapped.train()
        tasks = self._micro_tasks(step)
        totals: Dict[str, float] = {}
        tokens = 0
        for i, (kind, micro) in enumerate(tasks):
            last = i == len(tasks) - 1
            with distmod.no_sync(self.wrapped, active=ctx.enabled and not last):
                with devmod.autocast(self.device, self.policy):
                    if kind == "lm":
                        window = self.lm_train.sample(cfg.micro_batch_size, cfg.seed, ctx.rank, step, micro)
                        inputs, labels = (t.to(self.device) for t in lm_inputs(window))
                        tokens += inputs.numel()
                        loss, stats = self.wrapped("lm", (inputs, labels))
                        coef = cfg.lm_loss_coef
                    else:
                        batch = sample_decision_batch(self.decision_train, cfg.decision_batch_size, cfg.seed, ctx.rank, step, micro)
                        gen = torch.Generator().manual_seed((cfg.seed * 1_000_003 + step * 1_009 + micro * 31 + ctx.rank) % (2 ** 63))
                        loss, stats = self.wrapped("decision", (batch, gen))
                        coef = cfg.decision_loss_coef
                scaled = loss * (coef / cfg.grad_accum_steps)
                if self.scaler is not None:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v / cfg.grad_accum_steps

        if self.scaler is not None:
            for opt in self.optimizers:
                self.scaler.unscale_(opt)
        if distmod.is_fsdp(self.wrapped):
            grad_norm = self.wrapped.clip_grad_norm_(cfg.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        for opt in self.optimizers:
            if self.scaler is not None:
                self.scaler.step(opt)
            else:
                opt.step()
        if self.scaler is not None:
            self.scaler.update()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        for sched in self.schedulers:
            sched.step()
        reduce_fn = (lambda t: distmod.all_reduce_mean(t, ctx)) if ctx.enabled else None
        self.model.apply_moe_balance_updates(reduce_fn=reduce_fn)
        self.step += 1

        totals["grad_norm"] = float(grad_norm)
        totals["lr"] = self.schedulers[0].get_last_lr()[0]
        totals["lm_tokens"] = tokens * ctx.world_size
        return totals

    # -- evaluation --------------------------------------------------------------------------
    def evaluate(self) -> Dict[str, Any]:
        from eos_research.evaluate import decision_records, evaluate_lm
        from eos_research.metrics import summarize

        results: Dict[str, Any] = {}
        if self.lm_val is not None:
            results.update(evaluate_lm(
                self.model, self.lm_val, self.cfg.micro_batch_size, self.cfg.eval_lm_batches,
                self.device, self.policy, self.ctx,
            ))
        if self.decision_val and self.ctx.is_main:
            examples = self.decision_val[: self.cfg.eval_decision_examples]
            with devmod.autocast(self.device, self.policy):
                records = decision_records(self.model, examples, self.tokenizer, self.cfg.decision_num_loops, order_check=True)
            results["decisions"] = summarize(records)
        distmod.barrier(self.ctx)
        return results

    def save(self) -> str:
        return ckpt.save_checkpoint(
            self.cfg.run_dir, self.step, self.wrapped, self.model, self.optimizers, self.schedulers,
            self.scaler, {"train_config": asdict(self.cfg)}, self.ctx, self.cfg.keep_last,
        )

    # -- main loop ---------------------------------------------------------------------------
    def fit(self) -> Dict[str, Any]:
        cfg = self.cfg
        window: Dict[str, float] = {}
        window_steps, window_start = 0, time.perf_counter()
        last_eval: Dict[str, Any] = {}
        run_start = time.perf_counter()
        while self.step < cfg.max_steps:
            stats = self.train_step()
            for k, v in stats.items():
                window[k] = window.get(k, 0.0) + v
            window_steps += 1
            if self.step % cfg.log_every == 0 or self.step == cfg.max_steps:
                elapsed = time.perf_counter() - window_start
                record = {k: v / window_steps for k, v in window.items() if k != "lm_tokens"}
                record["lr"] = stats["lr"]
                record["step_seconds"] = elapsed / window_steps
                if window.get("lm_tokens"):
                    record["tokens_per_second"] = window["lm_tokens"] / max(elapsed, 1e-9)
                record.update(devmod.memory_stats(self.device))
                self.logger.log(self.step, record)
                if self.ctx.is_main:
                    shown = ", ".join(f"{k}={v:.4g}" for k, v in record.items() if isinstance(v, float))
                    print(f"step {self.step}: {shown}", flush=True)
                window, window_steps, window_start = {}, 0, time.perf_counter()
            if cfg.eval_every > 0 and (self.step % cfg.eval_every == 0 or self.step == cfg.max_steps):
                last_eval = self.evaluate()
                flat: Dict[str, Any] = {}
                _flatten("", last_eval, flat)
                self.logger.log(self.step, flat, kind="eval")
            if cfg.save_every > 0 and (self.step % cfg.save_every == 0 or self.step == cfg.max_steps):
                self.save()

        final = {
            "steps": self.step,
            "train_seconds_this_process": time.perf_counter() - run_start,
            "eval": last_eval,
            "parameters": self.model.count_parameters(),
        }
        if self.ctx.is_main:
            write_json(os.path.join(cfg.run_dir, "final_metrics.json"), final)
        self.logger.close()
        return final


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(data: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``key=value`` strings; dotted keys reach into nested dicts."""
    data = json.loads(json.dumps(data))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _parse_value(value)
    return data


def config_from_args(argv=None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train EOS (language modelling and/or Rhen decisions).")
    parser.add_argument("--config", help="JSON file with TrainConfig fields")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a field")
    args = parser.parse_args(argv)
    data: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    return TrainConfig.from_dict(apply_overrides(data, args.set))


def main(argv=None) -> None:
    cfg = config_from_args(argv)
    try:
        Trainer(cfg).fit()
    finally:
        distmod.cleanup()


if __name__ == "__main__":
    main()
