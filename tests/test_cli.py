"""Unit tests for RHEN command-line interface tools (cli.decide, cli.train, cli.eval, cli.tasks)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest

from cli import decide, eval as rhen_eval, tasks, train


class TestCLITasks(unittest.TestCase):
    def test_list_families(self):
        code = tasks.run(["--list-families"])
        self.assertEqual(code, 0)

    def test_generate_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code = tasks.run([
                "--out-dir", tmpdir,
                "--train", "6",
                "--val", "6",
                "--test", "6",
                "--ood", "6",
                "--seed", "42",
                "--quiet",
            ])
            self.assertEqual(code, 0)
            for split in ("train", "val", "test", "test_ood"):
                path = os.path.join(tmpdir, f"{split}.jsonl")
                self.assertTrue(os.path.exists(path), f"Missing {split}.jsonl")
                with open(path, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh if line.strip()]
                self.assertEqual(len(lines), 6)

            meta_path = os.path.join(tmpdir, "benchmark.json")
            self.assertTrue(os.path.exists(meta_path))
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["seed"], 42)


class TestCLIDecide(unittest.TestCase):
    def test_cli_decide_basic(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
            tf.write('{"state": {"amount": 2500, "flagged": false}}')
            tf_path = tf.name

        try:
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as out_f:
                out_path = out_f.name

            code = decide.run([
                "--state", f"@{tf_path}",
                "--choice", "Routing queue", "Auto-approve", "Manual review",
                "--noul", "Is transaction safe?",
                "--preset", "tiny",
                "--format", "json",
                "--out", out_path,
            ])
            self.assertEqual(code, 0)

            with open(out_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            self.assertEqual(data["num_questions"], 2)
            self.assertEqual(len(data["results"]), 2)
            self.assertEqual(data["results"][0]["primitive"], "choice")
            self.assertEqual(data["results"][1]["primitive"], "noul")
            self.assertIn("abstention", data["results"][0])
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_cli_decide_stdin(self):
        input_payload = json.dumps({
            "state": {"temperature": 105, "pressure": 50},
            "questions": [
                {"type": "choice", "question": "Cooling level", "options": ["Low", "High"]},
            ],
        })

        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO(input_payload)
        sys.stdout = io.StringIO()

        try:
            code = decide.run([
                "--stdin",
                "--preset", "tiny",
                "--format", "json",
            ])
            self.assertEqual(code, 0)
            output = sys.stdout.getvalue().strip()
            data = json.loads(output)
            self.assertEqual(len(data["results"]), 1)
            self.assertEqual(data["results"][0]["choice_label"] in ["Low", "High"], True)
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout


class TestCLIEval(unittest.TestCase):
    def test_cli_eval_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate minimal benchmark dataset
            tasks.run([
                "--out-dir", tmpdir,
                "--train", "4",
                "--val", "4",
                "--test", "4",
                "--ood", "4",
                "--quiet",
            ])

            report_path = os.path.join(tmpdir, "report.json")
            code = rhen_eval.run([
                "--decision-dir", tmpdir,
                "--preset", "tiny",
                "--out", report_path,
                "--quiet",
            ])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(report_path))

            with open(report_path, "r", encoding="utf-8") as fh:
                report = json.load(fh)

            self.assertIn("decisions", report)
            splits = report["decisions"]["splits"]
            self.assertIn("val", splits)
            self.assertIn("test", splits)
            self.assertIn("test_ood", splits)
            self.assertIn("ood_detection", report["decisions"])


class TestCLITrain(unittest.TestCase):
    def test_cli_train_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.jsonl")
            with open(train_path, "w", encoding="utf-8") as fh:
                fh.write('{"state": {"x": 1}, "questions": [{"question": "q", "options": ["a", "b"], "primitive": "choice", "label": 0}]}\n')

            code = train.run([
                "--dry-run",
                "--train-data", train_path,
                "--preset", "tiny",
                "--max-steps", "10",
                "--run-dir", os.path.join(tmpdir, "run"),
            ])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
