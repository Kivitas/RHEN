"""
Regression tests for the research tooling in ``eos_research``.

Covers device and precision selection, single-process distributed fallbacks, deterministic
data sampling, the decision benchmark's ground truth, hand-checked metrics, calibration,
learning-rate schedules, the option-order consistency objective, trainer checkpoints with
bit-exact resume, evaluation reports and sweep aggregation. Everything runs on the CPU with
the ``tiny`` preset.
"""

import json
import math
import os
import shutil
import tempfile
import unittest

import torch

from eos_research import calibration, data, device, distributed, metrics, tasks
from eos_research.train import TrainConfig, Trainer, apply_overrides, lr_factor


class DeviceTests(unittest.TestCase):
    def test_cpu_and_auto_resolve(self):
        self.assertEqual(device.resolve_device("cpu"), torch.device("cpu"))
        auto = device.resolve_device("auto")
        self.assertIn(auto.type, device.available_device_types())

    def test_unknown_and_unavailable_devices_raise(self):
        with self.assertRaises(ValueError):
            device.resolve_device("tpu")
        for kind in ("cuda", "npu", "xpu", "mps"):
            if not device.is_available(kind):
                with self.assertRaises(RuntimeError):
                    device.resolve_device(kind)

    def test_precision_policies(self):
        cpu = torch.device("cpu")
        self.assertEqual(device.resolve_precision("auto", cpu).name, "fp32")
        bf16 = device.resolve_precision("bf16", cpu)
        self.assertEqual(bf16.autocast_dtype, torch.bfloat16)
        self.assertFalse(bf16.use_grad_scaler)
        with self.assertRaises(RuntimeError):
            device.resolve_precision("fp16", cpu)
        with self.assertRaises(ValueError):
            device.resolve_precision("fp8", cpu)
        self.assertIsNone(device.make_grad_scaler(cpu, device.resolve_precision("fp32", cpu)))

    def test_collective_backend_follows_the_device(self):
        self.assertEqual(device.collective_backend(torch.device("cpu")), "gloo")
        self.assertEqual(device.collective_backend(torch.device("cuda", 0)), "nccl")
        self.assertEqual(device._COLLECTIVE_BACKENDS["npu"], "hccl")
        self.assertEqual(device._COLLECTIVE_BACKENDS["xpu"], "xccl")

    def test_seed_everything_is_repeatable_and_rank_specific(self):
        device.seed_everything(5, rank=0)
        a = torch.rand(3)
        device.seed_everything(5, rank=0)
        self.assertTrue(torch.equal(a, torch.rand(3)))
        device.seed_everything(5, rank=1)
        self.assertFalse(torch.equal(a, torch.rand(3)))

    def test_environment_description(self):
        info = device.describe_environment(torch.device("cpu"))
        self.assertIn("torch", info)
        self.assertIn("cpu", info["available_devices"])


class SingleProcessDistributedTests(unittest.TestCase):
    def test_single_process_is_a_no_op(self):
        ctx = distributed.DistributedContext(device=torch.device("cpu"))
        self.assertFalse(ctx.enabled)
        self.assertTrue(ctx.is_main)
        t = torch.tensor([1.0, 3.0])
        self.assertTrue(torch.equal(distributed.all_reduce_mean(t, ctx), t))
        module = torch.nn.Linear(2, 2)
        self.assertIs(distributed.wrap_model(module, ctx, "ddp"), module)
        with self.assertRaises(ValueError):
            distributed.wrap_model(module, ctx, "zero3")

    def test_unwrap_follows_module_attributes(self):
        inner = torch.nn.Linear(2, 2)

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.module = m

        self.assertIs(distributed.unwrap(Wrapper(Wrapper(inner))), inner)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _corpus(self):
        docs = [" ".join(str(i % 50) for i in range(j, j + 400)) for j in range(20)]
        encode = lambda text: [int(t) for t in text.split()]
        return data.prepare_token_corpus(docs, encode, self.tmp, vocab_size=60, eos_token_id=59, val_fraction=0.2)

    def test_prepare_writes_meta_and_splits(self):
        meta = self._corpus()
        self.assertEqual(meta["dtype"], "uint16")
        self.assertEqual(meta["train_documents"] + meta["val_documents"], 20)
        self.assertEqual(meta["train_tokens"] + meta["val_tokens"], 20 * 401)
        self.assertEqual(data.read_meta(self.tmp)["content_sha256"], meta["content_sha256"])

    def test_out_of_range_tokens_are_rejected(self):
        with self.assertRaises(ValueError):
            data.prepare_token_corpus(["5 99"], lambda t: [int(x) for x in t.split()], self.tmp, vocab_size=10)

    def test_window_sampling_is_a_function_of_seed_rank_and_step(self):
        self._corpus()
        windows = data.TokenWindows(self.tmp, "train", seq_len=16)
        a = windows.sample(4, seed=1, rank=0, step=3)
        self.assertEqual(tuple(a.shape), (4, 17))
        self.assertTrue(torch.equal(a, windows.sample(4, seed=1, rank=0, step=3)))
        self.assertFalse(torch.equal(a, windows.sample(4, seed=1, rank=0, step=4)))
        self.assertFalse(torch.equal(a, windows.sample(4, seed=1, rank=1, step=3)))
        inputs, labels = data.lm_inputs(a)
        self.assertTrue(torch.equal(inputs, labels))
        self.assertEqual(inputs.shape[1], 16)

    def test_sequential_windows_cover_the_split_in_order(self):
        self._corpus()
        windows = data.TokenWindows(self.tmp, "train", seq_len=16)
        batches = list(windows.sequential(batch_size=3, max_batches=2))
        self.assertEqual(len(batches), 2)
        self.assertTrue(torch.equal(batches[0][0], torch.from_numpy(windows._window(0))))

    def test_decision_question_validation(self):
        with self.assertRaises(ValueError):
            data.DecisionQuestion("q", ["a", "b"], "choice", 2)
        with self.assertRaises(ValueError):
            data.DecisionQuestion("q", ["Yes", "No"], "noul", 1)
        with self.assertRaises(ValueError):
            data.DecisionQuestion("q", ["a", "b"], "ranking", 0)
        q = data.DecisionQuestion("q", ["l0", "l1", "l2"], "score", 2)
        self.assertEqual(q.label, 2.0)
        self.assertEqual(q.class_index(), 2)
        self.assertEqual(data.DecisionQuestion("q", ["Yes", "No"], "noul", False).class_index(), 1)

    def test_jsonl_round_trip_and_single_question_lines(self):
        examples = tasks.generate_split("val", 7, seed=3)
        path = os.path.join(self.tmp, "d.jsonl")
        data.write_decision_jsonl(examples, path)
        loaded = data.load_decision_jsonl(path)
        self.assertEqual([e.to_dict() for e in loaded], [e.to_dict() for e in examples])
        single = os.path.join(self.tmp, "s.jsonl")
        with open(single, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"state": {"x": 1}, "question": "q", "options": ["a", "b"], "primitive": "choice", "label": 1}) + "\n")
        self.assertEqual(data.load_decision_jsonl(single)[0].questions[0].label, 1)

    def test_decision_batches_are_deterministic(self):
        examples = tasks.generate_split("train", 30, seed=0)
        a = data.sample_decision_batch(examples, 5, seed=0, rank=0, step=2)
        b = data.sample_decision_batch(examples, 5, seed=0, rank=0, step=2)
        self.assertEqual([e.to_dict() for e in a], [e.to_dict() for e in b])


class BenchmarkTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        a = [e.to_dict() for e in tasks.generate_split("test", 25, seed=4)]
        b = [e.to_dict() for e in tasks.generate_split("test", 25, seed=4)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, [e.to_dict() for e in tasks.generate_split("test", 25, seed=5)])

    def test_labels_follow_the_published_rules(self):
        for ex in tasks.generate_split("train", 200, seed=1) + tasks.generate_split("test_ood", 200, seed=1):
            q, s = ex.questions[0], ex.state
            if ex.task == "lookup":
                self.assertEqual(q.options[q.label], s["city"])
            elif ex.task == "routing":
                expected = "Manual review" if s["flagged"] else ("Senior approver" if s["amount"] >= 10_000 else "Auto-approve")
                self.assertEqual(q.options[q.label], expected)
            elif ex.task == "approval":
                self.assertEqual(q.label, s["refund_amount"] <= s["policy_limit"] and not s["fraud_flag"])
            elif ex.task == "risk":
                self.assertEqual(q.label, float(min(4, s["exposure"] // 2_500)))
            elif ex.task == "compare":
                self.assertEqual(q.options[q.label], max(("sensor_a", "sensor_b"), key=lambda k: s[k]))
            elif ex.task == "checks":
                over = {k for k, v in s["readings"].items() if v > s["limits"][k]}
                self.assertEqual({q.options[i] for i in q.label}, over)

    def test_ood_split_is_shifted(self):
        id_regions = {e.state.get("region") for e in tasks.generate_split("train", 300, seed=0)}
        ood_regions = {e.state.get("region") for e in tasks.generate_split("test_ood", 300, seed=0)}
        self.assertFalse((ood_regions - {None}) & (id_regions - {None}))
        id_q = {e.questions[0].question for e in tasks.generate_split("train", 300, seed=0)}
        ood_q = {e.questions[0].question for e in tasks.generate_split("test_ood", 300, seed=0)}
        self.assertFalse(id_q & ood_q)

    def test_option_order_is_shuffled(self):
        firsts = {tuple(e.questions[0].options) for e in tasks.generate_split("train", 200, seed=0, families=["routing"])}
        self.assertGreater(len(firsts), 1)


class MetricTests(unittest.TestCase):
    def test_ece_by_hand(self):
        # Bin of 0.9: 2 predictions, 1 correct -> |0.5 - 0.9|; bin of 0.6: 2 predictions, 2 correct -> |1 - 0.6|.
        ece = metrics.top_label_ece([0.9, 0.9, 0.6, 0.6], [True, False, True, True], n_bins=10)
        self.assertAlmostEqual(ece, 0.5 * 0.4 + 0.5 * 0.4)

    def test_auroc_extremes_and_ties(self):
        self.assertEqual(metrics.auroc([3, 4], [1, 2]), 1.0)
        self.assertEqual(metrics.auroc([1, 2], [3, 4]), 0.0)
        self.assertEqual(metrics.auroc([1, 1], [1, 1]), 0.5)

    def test_risk_coverage_oracle_ranking_has_zero_excess(self):
        out = metrics.risk_coverage([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
        self.assertAlmostEqual(out["excess_aurc"], 0.0)
        self.assertEqual(out["selective_acc@50"], 1.0)
        worse = metrics.risk_coverage([0.1, 0.2, 0.8, 0.9], [True, True, False, False])
        self.assertGreater(worse["aurc"], out["aurc"])

    def test_single_label_and_multichoice_summaries(self):
        records = [
            {"primitive": "choice", "task": "t", "probs": [0.7, 0.3], "label_index": 0, "free_energy": -1.0},
            {"primitive": "choice", "task": "t", "probs": [0.6, 0.4], "label_index": 1, "free_energy": -0.5},
            {"primitive": "multichoice", "task": "m", "probs": [0.9, 0.2, 0.6], "label_set": [0, 2]},
        ]
        out = metrics.summarize(records, qhat=0.5)
        single = out["single_label"]
        self.assertEqual(single["n"], 2)
        self.assertAlmostEqual(single["accuracy"], 0.5)
        self.assertAlmostEqual(single["nll"], 0.5 * (-math.log(0.7) - math.log(0.4)))
        self.assertAlmostEqual(single["conformal_coverage"], 0.5)  # the set is {0} for both; only the first is covered
        self.assertEqual(out["multichoice"]["exact_match"], 1.0)
        self.assertIn("t", out["by_task"])

    def test_conformal_set(self):
        self.assertEqual(metrics.conformal_set([0.7, 0.2, 0.1], qhat=0.85), [0, 1])


class CalibrationTests(unittest.TestCase):
    def _synthetic(self, sharpness, n=400, k=4, seed=0):
        gen = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, k, (n,), generator=gen)
        logits = torch.randn(n, k, generator=gen)
        logits[torch.arange(n), labels] += 1.0
        probs = torch.softmax(logits * sharpness, dim=-1)
        return probs.tolist(), labels.tolist()

    def _nll(self, probs, labels):
        return sum(-math.log(p[y]) for p, y in zip(probs, labels)) / len(labels)

    def test_overconfident_models_get_a_temperature_above_one(self):
        probs, labels = self._synthetic(sharpness=4.0)
        t = calibration.fit_temperature(probs, labels)
        self.assertGreater(t, 1.5)
        scaled = [calibration.apply_temperature(p, t) for p in probs]
        self.assertLess(self._nll(scaled, labels), self._nll(probs, labels))

    def test_underconfident_models_get_a_temperature_below_one(self):
        probs, labels = self._synthetic(sharpness=0.3)
        self.assertLess(calibration.fit_temperature(probs, labels), 0.8)

    def test_uninformative_scores_stay_within_bounds(self):
        probs = [[0.25] * 4 for _ in range(50)]
        t = calibration.fit_temperature(probs, [0] * 50)
        self.assertTrue(calibration.MIN_TEMPERATURE <= t <= calibration.MAX_TEMPERATURE)

    def test_conformal_coverage_on_exchangeable_data(self):
        probs, labels = self._synthetic(sharpness=1.0, n=2000, seed=1)
        cal_p, cal_y, test_p, test_y = probs[:1000], labels[:1000], probs[1000:], labels[1000:]
        qhat = calibration.fit_conformal_threshold(cal_p, cal_y, alpha=0.1)
        covered = sum(y in metrics.conformal_set(p, qhat) for p, y in zip(test_p, test_y)) / len(test_y)
        self.assertGreater(covered, 0.87)

    def test_artifact_round_trip(self):
        records = [{"primitive": "choice", "probs": p, "label_index": y}
                   for p, y in zip(*self._synthetic(sharpness=2.0, n=100))]
        fitted = calibration.fit_decision_calibration(records, alpha=0.1, fitted_on="val")
        path = os.path.join(tempfile.mkdtemp(), "cal.json")
        fitted.save(path)
        loaded = calibration.DecisionCalibration.load(path)
        self.assertEqual(loaded, fitted)
        self.assertEqual(loaded.prediction_set([0.97, 0.01, 0.01, 0.01]), fitted.prediction_set([0.97, 0.01, 0.01, 0.01]))


class ScheduleTests(unittest.TestCase):
    def test_warmup_then_cosine_to_the_floor(self):
        cfg = TrainConfig(decision_train="x", warmup_steps=10, max_steps=110, min_lr_ratio=0.1)
        self.assertAlmostEqual(lr_factor(0, cfg), 0.1)
        self.assertAlmostEqual(lr_factor(9, cfg), 1.0)
        self.assertAlmostEqual(lr_factor(10, cfg), 1.0)
        self.assertAlmostEqual(lr_factor(110, cfg), 0.1)

    def test_wsd_holds_then_decays(self):
        cfg = TrainConfig(decision_train="x", schedule="wsd", warmup_steps=0, max_steps=100, decay_fraction=0.2, min_lr_ratio=0.0)
        self.assertEqual(lr_factor(50, cfg), 1.0)
        self.assertEqual(lr_factor(79, cfg), 1.0)
        self.assertAlmostEqual(lr_factor(90, cfg), 0.5)
        self.assertAlmostEqual(lr_factor(100, cfg), 0.0)

    def test_overrides_and_validation(self):
        merged = apply_overrides({"model_overrides": {"hidden_size": 64}}, ["model_overrides.max_recurrent_loops=4", "lr=0.001", "run_dir=runs/x"])
        self.assertEqual(merged["model_overrides"], {"hidden_size": 64, "max_recurrent_loops": 4})
        self.assertEqual(merged["lr"], 0.001)
        self.assertEqual(merged["run_dir"], "runs/x")
        with self.assertRaises(ValueError):
            TrainConfig().validate()
        with self.assertRaises(ValueError):
            TrainConfig(decision_train="x", optimizer="muon", strategy="fsdp").validate()
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"learning_rate": 1.0})

    def test_shipped_configs_are_valid(self):
        from eos_research.presets import PRESETS

        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(here) if os.path.basename(here) == "tests" else here
        root = os.path.join(repo_root, "configs")
        names = sorted(f for f in os.listdir(root) if f.endswith(".json"))
        self.assertTrue(names)
        for name in names:
            with self.subTest(config=name), open(os.path.join(root, name), encoding="utf-8") as fh:
                cfg = TrainConfig.from_dict(json.load(fh))
                cfg.validate()
                self.assertIn(cfg.preset, PRESETS)


class OrderConsistencyTests(unittest.TestCase):
    def test_metric_is_zero_for_order_free_decisions_and_positive_otherwise(self):
        from rhen_loss import option_order_consistency

        p = torch.tensor([0.6, 0.3, 0.1])
        perm = [2, 0, 1]  # position j shows original option perm[j]
        self.assertAlmostEqual(float(option_order_consistency(p, p[perm], perm)), 0.0, places=6)
        position_biased = torch.tensor([0.6, 0.3, 0.1])  # prefers position 0 whatever sits there
        self.assertGreater(float(option_order_consistency(p, position_biased, perm)), 0.1)
        mc = option_order_consistency(torch.tensor([0.9, 0.2]), torch.tensor([0.2, 0.9]), [1, 0], "multichoice")
        self.assertAlmostEqual(float(mc), 0.0, places=6)

    def test_permutations_are_never_the_identity(self):
        from rhen_loss import sample_option_permutation

        gen = torch.Generator().manual_seed(0)
        for _ in range(50):
            self.assertEqual(sample_option_permutation(2, gen), [1, 0])
        self.assertEqual(sample_option_permutation(1), [0])

    def test_loss_reaches_the_trunk(self):
        from config import EOSConfig
        from eos_research.presets import preset_config
        from model import EOSModel
        from rhen_loss import compute_option_order_consistency_loss
        from tokenizer import get_tokenizer

        tok = get_tokenizer()
        torch.manual_seed(0)
        model = EOSModel(EOSConfig(**preset_config("tiny", vocab_size=tok.vocab_size)))
        items = [("Which route?", ["north", "south", "east"], "choice", None)]
        loss, info = compute_option_order_consistency_loss(
            model, {"load": 3}, items, generator=torch.Generator().manual_seed(1), tokenizer=tok
        )
        self.assertEqual(info["order_consistency_n"], 1)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertGreater(float(model.embed_tokens.weight.grad.abs().sum()), 0.0)


class TrainerAndEvaluationTests(unittest.TestCase):
    """A real (tiny) training run: artifacts, bit-exact resume and the evaluation report."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.bench = os.path.join(cls.tmp, "bench")
        tasks.write_benchmark(cls.bench, n_train=40, n_val=12, n_test=12, n_ood=12, seed=0)
        docs = [" ".join(str((i * 7 + j) % 900) for i in range(300)) for j in range(6)]
        cls.corpus = os.path.join(cls.tmp, "corpus")
        data.prepare_token_corpus(docs, lambda t: [int(x) for x in t.split()], cls.corpus, vocab_size=1000, val_fraction=0.3)
        cls.base = dict(
            preset="tiny", lm_data=cls.corpus, seq_len=32, micro_batch_size=2, grad_accum_steps=2,
            decision_train=os.path.join(cls.bench, "train.jsonl"), decision_val=os.path.join(cls.bench, "val.jsonl"),
            decision_batch_size=2, order_consistency_coef=0.1, max_steps=4, warmup_steps=1, log_every=2,
            eval_every=4, eval_lm_batches=2, eval_decision_examples=4, save_every=2, device="cpu", precision="fp32",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_training_artifacts_and_bit_exact_resume(self):
        run = os.path.join(self.tmp, "run")
        final = Trainer(TrainConfig(run_dir=run, **self.base)).fit()
        self.assertEqual(final["steps"], 4)
        for name in ("manifest.json", "train_config.json", "metrics.jsonl", "final_metrics.json"):
            self.assertTrue(os.path.exists(os.path.join(run, name)), name)
        with open(os.path.join(run, "manifest.json"), encoding="utf-8") as fh:
            self.assertIn("train_config_sha256", json.load(fh))
        self.assertIn("decisions", final["eval"])
        self.assertIn("lm_perplexity", final["eval"])
        with open(os.path.join(run, "metrics.jsonl"), encoding="utf-8") as fh:
            self.assertTrue(any(json.loads(line)["kind"] == "eval" for line in fh))

        resumed = os.path.join(self.tmp, "resumed")
        os.makedirs(os.path.join(resumed, "checkpoints"))
        shutil.copytree(os.path.join(run, "checkpoints", "step_00000002"), os.path.join(resumed, "checkpoints", "step_00000002"))
        with open(os.path.join(resumed, "checkpoints", "latest"), "w", encoding="utf-8") as fh:
            fh.write("step_00000002")
        trainer = Trainer(TrainConfig(run_dir=resumed, **self.base))
        self.assertEqual(trainer.step, 2)
        trainer.fit()
        a = torch.load(os.path.join(run, "checkpoints", "step_00000004", "model.pt"), weights_only=True)
        b = torch.load(os.path.join(resumed, "checkpoints", "step_00000004", "model.pt"), weights_only=True)
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

        from eos_research.checkpointing import load_model_from_checkpoint
        from eos_research.evaluate import decision_report

        model = load_model_from_checkpoint(os.path.join(run, "checkpoints", "step_00000004"))
        splits = {name: data.load_decision_jsonl(os.path.join(self.bench, f"{name}.jsonl"))[:6] for name in ("val", "test", "test_ood")}
        report, cal = decision_report(model, splits)
        self.assertIsNotNone(cal)
        self.assertIn("calibrated", report["splits"]["test"])
        self.assertIn("permutation_consistency", report["splits"]["test"]["raw"])
        self.assertIn("auroc_free_energy", report["ood_detection"])


class SweepTests(unittest.TestCase):
    def test_grid_parsing_and_aggregation_over_seeds(self):
        from eos_research.sweep import combo_name, parse_grid, run_sweep

        grid = parse_grid(["model_overrides.max_recurrent_loops=2,4", "order_consistency_coef=0,0.1"])
        self.assertEqual(grid[0], ("model_overrides.max_recurrent_loops", [2, 4]))
        self.assertEqual(combo_name([("model_overrides.max_recurrent_loops", 2)]), "max_recurrent_loops=2")
        out = tempfile.mkdtemp()
        try:
            # Finished runs are reused, so pre-written results exercise aggregation without training.
            for name, offset in (("max_recurrent_loops=2", 0.0), ("max_recurrent_loops=4", 1.0)):
                for seed, value in ((0, 1.0), (1, 3.0)):
                    run_dir = os.path.join(out, name, f"seed{seed}")
                    os.makedirs(run_dir)
                    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as fh:
                        json.dump({"eval": {"lm_loss": value + offset}}, fh)
            summary = run_sweep("unused.json", out, parse_grid(["model_overrides.max_recurrent_loops=2,4"]), [0, 1], [])
            stats = summary["max_recurrent_loops=2"]["metrics"]["lm_loss"]
            self.assertEqual(stats["mean"], 2.0)
            self.assertAlmostEqual(stats["std"], math.sqrt(2.0))
            self.assertEqual(summary["max_recurrent_loops=4"]["metrics"]["lm_loss"]["mean"], 3.0)
            self.assertTrue(os.path.exists(os.path.join(out, "summary.csv")))
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
