"""End-to-end runner checks on a tiny config (technical only)."""

import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from acpr import run
from acpr.collect import CollectConfig, collect_split
from acpr.env import N_ACTIONS, MazeConfig
from acpr.models import LatentPredictor, RecurrentEncoder
from acpr.train import TrainConfig, build_windows, train_predictor

TINY = {
    "name": "unit-tiny",
    "run_kind": "smoke",
    "pilot_approved": False,
    "seeds": [0],
    "variants": ["symmetric"],
    "maze": {"corridor_min": 2, "corridor_max": 2, "n_distractors": 2, "timeout_slack": 4},
    "horizon": 2,
    "train_budget": 120,
    "dev_budget": 60,
    "branch_prob": 0.3,
    "behavior_forward_prob": 0.8,
    "model": {"hidden": 4},
    "stage1": {"epochs": 1, "lr": 0.01, "batch_episodes": 4},
    "stage2": {"epochs": 1, "lr": 0.01, "batch_episodes": 4},
    "eval": {"n_calibration": 8, "n_test": 8, "plan_horizon": 2, "gamma": 0.9},
    "arms": ["acp_branch", "msp_branch", "acp_nobranch", "random_frozen"],
    "resource_caps": {"max_wall_s": 300, "max_peak_rss_mb": 4096},
}


def _run(cfg, out):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        code = run.main(["--config", f.name, "--out", str(out)])
    return code, buf_out.getvalue(), buf_err.getvalue()


class TestRunner(unittest.TestCase):
    def test_tiny_run_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            code, _, _ = _run(TINY, out)
            self.assertEqual(code, 0)
            for name in ("config.json", "raw_log.jsonl", "metrics.json", "manifest.json"):
                self.assertTrue((out / name).exists(), name)
            metrics = json.loads((out / "metrics.json").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual([m["status"] for m in metrics], ["OK"] * 4)
            self.assertEqual(manifest["science_status"], "SCIENCE_NOT_EVALUATED")
            for key in ("source_sha256", "config_sha256", "git", "environment", "peak_rss_mb", "time", "total_mac"):
                self.assertIn(key, manifest)
            by_arm = {m["arm"]: m for m in metrics}
            # Equal actual spend, not just equal cap.
            spent = {m["train_transitions_spent"] for m in metrics}
            self.assertEqual(spent, {TINY["train_budget"]})
            self.assertEqual(by_arm["acp_nobranch"]["train_restores_used"], 0)
            self.assertGreater(by_arm["acp_branch"]["train_restores_used"], 0)
            # Same capacity and same initial weights across ACP/MSP.
            self.assertEqual(by_arm["acp_branch"]["stage1_predictor_params"], by_arm["msp_branch"]["stage1_predictor_params"])
            self.assertEqual(by_arm["acp_branch"]["encoder_params"], by_arm["msp_branch"]["encoder_params"])
            for m in metrics:
                self.assertEqual(m["planning"]["ledger"]["restores"], 0)
            self.assertNotIn("stage1", by_arm["random_frozen"])

    def test_non_smoke_config_refused_without_approval(self):
        cfg = dict(TINY, run_kind="pilot", pilot_approved=False)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            code, _, err = _run(cfg, out)
            self.assertEqual(code, 2)
            self.assertIn("REFUSED", err)
            self.assertFalse(out.exists())

    def test_resource_cap_marks_not_run(self):
        cfg = dict(TINY, resource_caps={"max_wall_s": -1, "max_peak_rss_mb": 4096})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            code, _, _ = _run(cfg, out)
            metrics = json.loads((out / "metrics.json").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual({m["status"] for m in metrics}, {"NOT_RUN"})
            self.assertEqual(manifest["software_status"], "TECHNICAL_RUN_INCOMPLETE")
            self.assertEqual(code, 0)  # NOT_RUN is not a failure, but it is not a pass either


class TestOptimizationSanity(unittest.TestCase):
    def test_stage1_loss_decreases_on_small_data(self):
        maze = MazeConfig(corridor_min=2, corridor_max=2, n_distractors=2)
        res = collect_split(maze, CollectConfig(300, 2, 0.3), 0, "train")
        windows = build_windows(res.episodes, 2)
        enc = RecurrentEncoder(maze.n_obs, N_ACTIONS, 8, random.Random(0))
        pred = LatentPredictor("p", 8, maze.n_obs, N_ACTIONS, True, random.Random(1))
        hist = train_predictor(enc, pred, res.episodes, windows, TrainConfig(6, 0.02, 4, 0), train_encoder=True)
        losses = [e["train_loss_per_target"] for e in hist["epochs"]]
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
