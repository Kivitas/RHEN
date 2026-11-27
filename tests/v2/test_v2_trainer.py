"""Trainer tests (plan T-401, T-403). Everything uses r2-tiny and a handful of steps: these prove plumbing only."""

import glob
import json
import os
import random

import pytest
import torch

from rhen.v2.config import count_parameters, preset
from rhen.v2.data.blocklist import Blocklist, BlockedExampleError
from rhen.v2.data.schema import write_jsonl
from rhen.v2.data.synth import generate_split
from rhen.v2.data.teacher import FakeTeacher, run_labelling
from rhen.v2.io import load_checkpoint
from rhen.v2.train import (SMOKE_NOTICE, TrainConfig, Trainer, WorkItem, apply_overrides, choose_strategy, plan_run,
                           run_training, wrap_for_training)
from rhen.research import distributed as distmod

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAMILIES = ["num_extreme", "pair_numeric", "pair_threshold", "negation_choice", "majority_vote", "score_pass_rate"]


def tiny(tmp_path, **kw):
    base = dict(run_dir=str(tmp_path / "run"), stage="decision", preset="r2-tiny", seed=3,
                synthetic=[{"n": 24, "seed": 1, "families": FAMILIES}], span_choice_frac=0.1,
                steps=6, batch_questions=8, lr=3e-3, ema_decay=0.9, device="cpu", log_every=1, eval_every=3,
                save_every=3, keep_last=2, eval_synthetic={"split": "val", "n": 6, "seed": 0, "families": FAMILIES})
    base.update(kw)
    return TrainConfig(**base)


def read_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.jsonl")) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def final_state(run_dir, name="model.pt"):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*")))
    model, extra = load_checkpoint(os.path.join(ckpts[-1], name))
    return model.state_dict(), extra


# ---------------------------------------------------------------------------- config
def test_config_round_trip_and_validation(tmp_path):
    cfg = tiny(tmp_path)
    again = TrainConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and again.sha256() == cfg.sha256()
    path = tmp_path / "c.json"
    cfg.save(str(path))
    assert TrainConfig.load(str(path)) == cfg
    with pytest.raises(ValueError, match="unknown TrainConfig keys"):
        TrainConfig.from_dict(dict(cfg.to_dict(), bogus=1))
    for bad in (dict(stage="nope"), dict(preset="r2-huge"), dict(steps=0), dict(blocklist_action="ignore"),
                dict(pretrain_weights=[1, 1]), dict(decision_weights={"other": 1.0}), dict(precision="fp8"),
                dict(stage="distill"), dict(stage="adapter"), dict(model_overrides={"nope": 1}),
                dict(synthetic=[{"n": 3, "families": ["no_such_family"]}]), dict(ema_decay=1.0),
                dict(state_len_curriculum=[[0.5, 100], [0.2, 200]])):
        with pytest.raises(ValueError):
            tiny(tmp_path, **bad)
    merged = apply_overrides({"a": 1, "model_overrides": {}, "synthetic": [{"n": 1}]},
                             ["a=2", "model_overrides.d_model=128", "synthetic.0.n=5", "name=abc"])
    assert merged == {"a": 2, "model_overrides": {"d_model": 128}, "synthetic": [{"n": 5}], "name": "abc"}


def test_curriculum_limit(tmp_path):
    cfg = tiny(tmp_path, steps=10, state_len_curriculum=[[0.0, 64], [0.6, 128]],
               model_overrides={"max_state_tokens": 256})
    assert [cfg.limit_at(s) for s in (0, 5, 6, 9)] == [64, 64, 128, 128]


def test_every_shipped_config_loads():
    paths = sorted(glob.glob(os.path.join(REPO, "configs", "v2", "*.json")))
    names = {os.path.basename(p) for p in paths}
    assert {"r2_tiny_cpu.json", "r2_50m_pretrain.json", "r2_50m_decision.json", "r2_300m_pretrain.json",
            "r2_300m_decision.json", "r2_1b_pretrain.json", "r2_1b_decision.json", "r2_8b_adapter.json"} <= names
    for p in paths:
        cfg = TrainConfig.load(p)
        assert cfg.to_dict() == TrainConfig.from_dict(cfg.to_dict()).to_dict()
        text = open(p).read()
        assert "accuracy" not in text and "ece" not in text.lower()      # no invented results
    c300 = TrainConfig.load(os.path.join(REPO, "configs/v2/r2_300m_pretrain.json"))
    assert c300.state_len_curriculum == [[0.0, 2048], [0.6, 8192]] and c300.model_config().global_every == 3
    assert TrainConfig.load(os.path.join(REPO, "configs/v2/r2_1b_decision.json")).model_config().gradient_checkpointing
    with pytest.raises(NotImplementedError):
        Trainer(TrainConfig.load(os.path.join(REPO, "configs/v2/r2_8b_adapter.json")))


def test_strategy_selection_and_wrapper(tmp_path):
    small, big = preset("r2-50m"), preset("r2-1b")
    assert choose_strategy("auto", 1, big) == "none"
    assert choose_strategy("auto", 8, small) == "ddp" and choose_strategy("auto", 8, big) == "fsdp"
    assert choose_strategy("ddp", 4, big) == "ddp" and choose_strategy("none", 4, big) == "none"
    with pytest.raises(ValueError):
        choose_strategy("zero3", 2, small)
    module = torch.nn.Linear(2, 2)
    ctx = distmod.DistributedContext()
    assert wrap_for_training(module, ctx, "none") is module and wrap_for_training(module, ctx, "ddp") is module


# ---------------------------------------------------------------------------- smoke runs
def test_pretrain_smoke_writes_checkpoints_metrics_manifest(tmp_path):
    cfg = tiny(tmp_path, stage="pretrain", steps=10, eval_every=5, save_every=5, batch_questions=12)
    summary = run_training(cfg)
    assert summary["step"] == 10 and summary["smoke_only"] and "plumbing only" in summary["notice"]
    run = cfg.run_dir
    rows = read_metrics(run)
    assert rows[0]["kind"] == "notice" and "plumbing only" in rows[0]["text"]
    train = [r for r in rows if r["kind"] == "train"]
    assert [r["step"] for r in train] == list(range(1, 11))
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in train)
    assert any("mlm" in r for r in train)
    evals = [r for r in rows if r["kind"] == "eval"]
    assert [r["step"] for r in evals] == [5, 10] and "ema_mlm_loss" in evals[0] and "raw_loss" in evals[0]
    ckpts = sorted(os.listdir(os.path.join(run, "checkpoints")))
    assert "latest" in ckpts and len([c for c in ckpts if c.startswith("step_")]) == 2      # keep_last=2
    step_dir = os.path.join(run, "checkpoints", "step_00000010")
    assert sorted(os.listdir(step_dir)) == ["model.pt", "model_ema.pt", "trainer_state.pt"]
    state = torch.load(os.path.join(step_dir, "trainer_state.pt"), weights_only=True)      # safe loader
    assert state["head"] is not None and state["step"] == 10                                # MLM head checkpointed
    manifest = json.load(open(os.path.join(run, "manifest.json")))
    p = manifest["parameters"]
    assert p["match"] and p["actual"] == p["count_parameters_total"] == count_parameters(preset("r2-tiny"))["total"]
    assert p["mlm_head"] > 0
    for key in ("git", "environment", "training_run_id", "train_config", "data", "notice"):
        assert manifest[key]
    assert "dirty" in manifest["git"] or manifest["git"].get("available") is False
    assert manifest["training_run_id"] == summary["training_run_id"]


def test_decision_smoke_with_soft_labels_and_data_hashes(tmp_path):
    exs = generate_split("train", 10, 5, FAMILIES, 0.0)
    teacher_path = str(tmp_path / "teacher.jsonl")
    run_labelling(exs, FakeTeacher(seed=1), teacher_path)
    conv_path = str(tmp_path / "conv.jsonl")
    write_jsonl(generate_split("train", 8, 6, FAMILIES, 0.0), conv_path)
    cfg = tiny(tmp_path, steps=10, eval_every=5, save_every=5, teacher_jsonl=[teacher_path], converted_jsonl=[conv_path])
    summary = run_training(cfg)
    assert summary["step"] == 10
    manifest = json.load(open(os.path.join(cfg.run_dir, "manifest.json")))
    srcs = manifest["data"]["sources"]
    assert srcs["teacher"][0]["sha256"] and srcs["converted"][0]["examples"] == 8
    w = manifest["data"]["effective_weights"]
    assert w["scp"] == pytest.approx(0.1) and w["teacher"] == pytest.approx(0.9 * 0.45 / 1.0)
    rows = read_metrics(cfg.run_dir)
    assert [r for r in rows if r["kind"] == "eval"]
    # soft labels reach the batch
    t = Trainer(cfg)
    t.build_data()
    rng = random.Random(0)
    drawn = t.data.draw(rng, 64, 256, 0)
    assert any(q.soft is not None for e in drawn for q in e.questions)


# ---------------------------------------------------------------------------- resume
def _metric_view(rows):
    return [{k: v for k, v in r.items() if k not in ("wall",)} for r in rows if r["kind"] != "notice"]


@pytest.mark.parametrize("stage", ["decision", "pretrain"])
def test_resume_is_identical_to_uninterrupted(tmp_path, stage):
    a = tiny(tmp_path / "a", stage=stage, steps=6, eval_every=3, save_every=3)
    b = tiny(tmp_path / "b", stage=stage, steps=6, eval_every=3, save_every=3)
    run_training(a)
    first = run_training(b, stop_at=3)
    assert first["step"] == 3
    second = run_training(b)                      # picks up the step-3 checkpoint
    assert second["step"] == 6 and second["resumed_from"].endswith("step_00000003")
    assert second["training_run_id"] == first["training_run_id"]
    sa, _ = final_state(a.run_dir)
    sb, _ = final_state(b.run_dir)
    for k in sa:
        assert torch.allclose(sa[k], sb[k], atol=1e-6, rtol=0), k
    ea, _ = final_state(a.run_dir, "model_ema.pt")
    eb, _ = final_state(b.run_dir, "model_ema.pt")
    for k in ea:
        assert torch.allclose(ea[k], eb[k], atol=1e-6, rtol=0), k
    ma, mb = _metric_view(read_metrics(a.run_dir)), _metric_view(read_metrics(b.run_dir))
    assert len(ma) == len(mb)
    for ra, rb in zip(ma, mb):
        assert ra.keys() == rb.keys()
        for k in ra:
            if isinstance(ra[k], float):
                assert abs(ra[k] - rb[k]) <= 1e-6, (k, ra, rb)
            else:
                assert ra[k] == rb[k]


def test_resume_refuses_changed_config(tmp_path):
    cfg = tiny(tmp_path, steps=6)
    run_training(cfg, stop_at=3)
    with pytest.raises(ValueError, match="cannot resume"):
        run_training(tiny(tmp_path, steps=6, lr=1e-3))


# ---------------------------------------------------------------------------- EMA at eval
def test_ema_weights_used_at_eval_and_restored(tmp_path):
    cfg = tiny(tmp_path, steps=2, eval_every=100)
    t = Trainer(cfg)
    t.build_data()
    t.step = 0
    for s in range(2):
        t.train_step(s)
    seen = []
    orig = t._eval_pass

    def spy(model):
        seen.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return orig(model)

    t._eval_pass = spy
    before = {k: v.detach().clone() for k, v in t.model.state_dict().items()}
    rec = t.evaluate(2)
    assert len(seen) == 2 and "raw_loss" in rec and "ema_loss" in rec
    for k, v in seen[0].items():
        assert torch.equal(v, before[k])                         # raw pass sees the training weights
    moved = False
    for k, v in seen[1].items():
        assert torch.equal(v, t.ema.shadow[k])                   # EMA pass sees the shadow weights
        moved |= not torch.equal(v, before[k])
    assert moved
    for k, v in t.model.state_dict().items():
        assert torch.equal(v, before[k])                         # restored


# ---------------------------------------------------------------------------- blocklist
def test_blocklisted_example_is_refused(tmp_path):
    exs = generate_split("train", 6, 9, FAMILIES, 0.0)
    path = str(tmp_path / "train.jsonl")
    write_jsonl(exs, path)
    bl = Blocklist({})
    bl.add_examples("fake-benchmark-test", [exs[2]], with_ngrams=False)
    bl_path = str(tmp_path / "bl.json")
    bl.save(bl_path)
    cfg = tiny(tmp_path, synthetic=[], converted_jsonl=[path], blocklist_path=bl_path)
    with pytest.raises(BlockedExampleError, match="fake-benchmark-test"):
        run_training(cfg)
    with pytest.raises(BlockedExampleError):
        run_training(tiny(tmp_path, synthetic=[], teacher_jsonl=[path], blocklist_path=bl_path), dry_run=True)
    # synthetic sources are screened too
    bl2 = Blocklist({})
    bl2.add_examples("synth-hit", generate_split("train", 24, 1, FAMILIES), with_ngrams=False)
    bl2.save(bl_path)
    with pytest.raises(BlockedExampleError, match="synth-hit"):
        run_training(tiny(tmp_path, blocklist_path=bl_path), dry_run=True)
    # opt-in drop mode removes the example and records it
    bl.save(bl_path)
    cfg = tiny(tmp_path, synthetic=[], converted_jsonl=[path], blocklist_path=bl_path, blocklist_action="drop")
    plan = plan_run(cfg)
    assert plan["data"]["blocklist_dropped"] == 1 and plan["data"]["sources"]["converted"][0]["examples"] == 5


# ---------------------------------------------------------------------------- gradient accumulation
def test_grad_accumulation_equivalence(tmp_path):
    cfg = tiny(tmp_path)
    t = Trainer(cfg)
    exs = generate_split("train", 4, 2, ["num_extreme"], 0.0)           # one labelled choice question each
    assert all(len(e.questions) == 1 and e.questions[0].type == "choice" for e in exs)
    plan = type(t._next_plan())(n_loops=2, k_bptt=2, min_loops=2, train=True)

    def grads(chunks):
        work = [WorkItem("decision", sum(len(e.questions) for e in c), examples=c) for c in chunks]
        t.accumulate(work, plan)
        return {n: p.grad.detach().clone() for n, p in t.model.named_parameters() if p.grad is not None}

    full = grads([exs])
    split = grads([exs[:2], exs[2:]])
    assert full.keys() == split.keys() and len(full) > 10
    worst = max((full[k] - split[k]).abs().max().item() for k in full)
    assert worst < 1e-5, worst


# ---------------------------------------------------------------------------- distillation
def test_distillation_smoke(tmp_path):
    teacher_cfg = tiny(tmp_path / "t", steps=4, eval_every=4, save_every=4)
    run_training(teacher_cfg)
    cfg = tiny(tmp_path / "s", stage="distill", teacher_checkpoint=teacher_cfg.run_dir, steps=4, eval_every=2, save_every=2,
               distill_temperature=2.0, distill_teacher_weight=0.5)
    t = Trainer(cfg)
    t.build_data()
    exs = [e for e in generate_split("train", 12, 3, ["num_extreme", "pair_numeric"], 0.0) if t._fits(e, 256)][:4]
    assert exs
    out = t.labeller.blend(exs)
    assert t.labeller.skipped == 0
    for src, new in zip(exs, out):
        for q0, q1 in zip(src.questions, new.questions):
            assert q0.soft is None and q1.soft is not None
            assert sum(q1.soft) == pytest.approx(1.0, abs=1e-5) and len(q1.soft) == (2 if q0.type == "noul" else len(q0.options)) + 1
            # the blend keeps weight on the original (one-hot) label
            gold = q0.label if q0.type != "noul" else (0 if q0.label else 1)
            assert q1.soft[gold] >= 0.5 * 0.5 - 1e-6
    summary = run_training(cfg)
    assert summary["step"] == 4
    manifest = json.load(open(os.path.join(cfg.run_dir, "manifest.json")))
    assert manifest["stage"] == "distill"


# ---------------------------------------------------------------------------- dry run
def test_dry_run_does_not_train(tmp_path):
    cfg = tiny(tmp_path)
    plan = run_training(cfg, dry_run=True)
    assert plan["dry_run"] and plan["stage"] == "decision" and plan["notice"] == SMOKE_NOTICE
    assert plan["parameters"]["actual"] == plan["parameters"]["count_parameters_total"]
    assert plan["data"]["effective_weights"]["scp"] == pytest.approx(0.1)
    assert not os.path.exists(cfg.run_dir)
    pre = run_training(tiny(tmp_path, stage="pretrain"), dry_run=True)
    assert pre["data"]["sample_batch"]["mlm"] + pre["data"]["sample_batch"]["decision_examples"] > 0
    with pytest.raises(FileNotFoundError):
        run_training(tiny(tmp_path, synthetic=[], converted_jsonl=[str(tmp_path / "missing.jsonl")]), dry_run=True)


def test_cli_entry_points(tmp_path, capsys):
    from rhen.v2.cli import main_eval, main_train

    cfg_path = tmp_path / "c.json"
    tiny(tmp_path).save(str(cfg_path))
    assert main_train(["--config", str(cfg_path), "--dry-run", "--set", "steps=4"]) == 0
    assert json.loads(capsys.readouterr().out)["steps"] == 4
    assert not os.path.exists(str(tmp_path / "run"))
    assert main_train(["--config", str(cfg_path), "--set", "bogus=1"]) == 2
    assert main_train(["--config", str(cfg_path), "--set", "steps=2", "--set", "eval_every=2", "--set", "save_every=2"]) == 0
    capsys.readouterr()
    out = tmp_path / "eval.json"
    code = main_eval(["--checkpoint", str(tmp_path / "run"), "-n", "4", "--out", str(out)])
    assert code == 0 and "metrics" in json.loads(out.read_text())
