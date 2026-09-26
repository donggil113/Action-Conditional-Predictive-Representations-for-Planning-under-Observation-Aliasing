"""Diagnostic stage: controls, checkpoints, leakage guards, verdict logic, caps."""

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from acpr.collect import CollectConfig, collect_split
from acpr.controls import ClueCurrentObsEncoder, CurrentObsEncoder, fit_standardizer
from acpr.data import History
from acpr.diagnose import ENCODERS, judge, make_encoder
from acpr.env import N_ACTIONS, OBS_CLUE_A, OBS_CLUE_B, OBS_JUNCTION, MazeConfig
from acpr.evaluate import EvalConfig, planning_eval
from acpr.models import LatentPredictor, RecurrentEncoder, copy_predictor, params_hash
from acpr.train import TrainConfig, build_windows, train_predictor

REPO = Path(__file__).resolve().parent.parent
MAZE = MazeConfig(corridor_min=2, corridor_max=3, n_distractors=2)


class TestCheckpointHook(unittest.TestCase):
    def test_checkpoint_equals_independent_shorter_run(self):
        res = collect_split(MAZE, CollectConfig(200, 2, 0.3), 0, "train")
        ws = build_windows(res.episodes, 2)
        enc = RecurrentEncoder(MAZE.n_obs, N_ACTIONS, 6, random.Random(0))
        cfg = TrainConfig(4, 0.01, 4, 7, obs_weight=0.0)
        long_head = LatentPredictor("h", 6, MAZE.n_obs, N_ACTIONS, True, random.Random(1))
        ckpt = {}
        train_predictor(enc, long_head, res.episodes, ws, cfg, False,
                        on_epoch_end=lambda n, p: ckpt.__setitem__(n, params_hash(copy_predictor(p).params())))
        short_head = LatentPredictor("h", 6, MAZE.n_obs, N_ACTIONS, True, random.Random(1))
        train_predictor(enc, short_head, res.episodes, ws, TrainConfig(2, 0.01, 4, 7, obs_weight=0.0), False)
        self.assertEqual(ckpt[2], params_hash(short_head.params()))
        self.assertEqual(ckpt[4], params_hash(long_head.params()))
        self.assertEqual(sorted(ckpt), [1, 2, 3, 4])

    def test_copy_is_independent(self):
        p = LatentPredictor("h", 4, MAZE.n_obs, N_ACTIONS, True, random.Random(0))
        q = copy_predictor(p)
        self.assertEqual(params_hash(p.params()), params_hash(q.params()))
        q.Wd.w[0] += 1.0
        self.assertNotEqual(params_hash(p.params()), params_hash(q.params()))


class TestControls(unittest.TestCase):
    def _junction_history(self, clue_obs, distractors):
        obs = (clue_obs, *distractors, OBS_JUNCTION)
        return History(obs, (0,) * (len(obs) - 1), (0.0,) * (len(obs) - 1))

    def test_memoryless_is_clue_blind_and_informative_is_not(self):
        mem, inf = CurrentObsEncoder(MAZE.n_obs, 16), ClueCurrentObsEncoder(MAZE.n_obs, 16)
        ha = self._junction_history(OBS_CLUE_A, (7, 8, 7))
        hb = self._junction_history(OBS_CLUE_B, (8, 8))
        z = lambda e, h: e.encode_values(h.observations, h.actions, h.rewards)[-1]
        self.assertEqual(z(mem, ha), z(mem, hb))
        self.assertNotEqual(z(inf, ha), z(inf, hb))
        # Informative control differs only in the clue slots.
        self.assertEqual(z(inf, ha)[2:], z(inf, hb)[2:])

    def test_features_do_not_depend_on_order_index_or_previous_episodes(self):
        res = collect_split(MAZE, CollectConfig(300, 2, 0.3), 0, "train")
        eps = list(res.episodes)
        for name in ENCODERS:
            enc = fit_standardizer(make_encoder(name, 5, MAZE.n_obs, 16), eps)
            fwd = [enc.encode_values(e.observations, e.actions, e.rewards) for e in eps]
            rev = [enc.encode_values(e.observations, e.actions, e.rewards) for e in reversed(eps)][::-1]
            self.assertEqual(fwd, rev, name)
            refit = fit_standardizer(make_encoder(name, 5, MAZE.n_obs, 16), list(reversed(eps)))
            for a, b in zip(enc.mean + enc.std, refit.mean + refit.std):
                self.assertAlmostEqual(a, b, places=12)

    def test_standardizer_refuses_non_train_data(self):
        dev = collect_split(MAZE, CollectConfig(100, 2, 0.3), 0, "dev")
        with self.assertRaises(ValueError):
            fit_standardizer(CurrentObsEncoder(MAZE.n_obs, 16), dev.episodes)

    def test_memoryless_forced_commit_is_exactly_chance_with_any_head(self):
        enc = fit_standardizer(CurrentObsEncoder(MAZE.n_obs, 16),
                               collect_split(MAZE, CollectConfig(200, 2, 0.3), 0, "train").episodes)
        for head_seed in range(3):
            head = LatentPredictor("h", 16, MAZE.n_obs, N_ACTIONS, True, random.Random(head_seed))
            ev = planning_eval(enc, head, MAZE, 0, EvalConfig(0, 40, 2, 0.9))
            self.assertEqual(ev["forced_commit_accuracy"], 0.5)
            self.assertEqual(ev["forced_commit_expected_accuracy"], 0.5)


class TieHead:
    def predict_rewards(self, z, seq):
        return [0.0] * len(seq)


class TestTieDiagnostics(unittest.TestCase):
    def test_exact_ties_score_half_credit(self):
        enc = CurrentObsEncoder(MAZE.n_obs, 16)
        ev = planning_eval(enc, TieHead(), MAZE, 0, EvalConfig(0, 30, 2, 0.9))
        self.assertEqual(ev["forced_commit_ties"], 30)
        self.assertEqual(ev["forced_commit_expected_accuracy"], 0.5)


def _cell(seed, enc, acc, success=0.5, status="OK"):
    c = {"seed": seed, "encoder": enc, "status": status}
    if status == "OK":
        c["planning"] = {"forced_commit_accuracy": acc, "success_rate": success, "timeout_rate": 0.0, "mean_return": 0.0}
    return c


class TestJudge(unittest.TestCase):
    CFG = json.loads((REPO / "configs" / "diag_eval_discriminability.json").read_text())

    def _cells(self, inf, rnd, mem, seeds=(1000, 1001, 1002)):
        out = []
        for s, a, b, m in zip(seeds, inf, rnd, mem):
            out += [_cell(s, "informative_control", a, 1.0), _cell(s, "random_recurrent", b), _cell(s, "memoryless_control", m)]
        return out

    def test_verdicts(self):
        j = lambda *a: judge(self._cells(*a), self.CFG)["verdict"]
        self.assertEqual(j([1.0] * 3, [0.95, 1.0, 0.92], [0.5] * 3), "CURRENT_BENCHMARK_NONDISCRIMINATIVE")
        self.assertEqual(j([1.0] * 3, [0.6, 0.7, 0.5], [0.5] * 3), "HEADROOM_PRESENT")
        self.assertEqual(j([1.0] * 3, [0.6, 0.95, 0.5], [0.5] * 3), "HEADROOM_MIXED")
        self.assertEqual(j([1.0, 0.85, 1.0], [0.5] * 3, [0.5] * 3), "EVALUATOR_UNRESOLVED")
        self.assertEqual(j([1.0] * 3, [0.5] * 3, [0.5, 0.65, 0.5]), "EVALUATOR_UNRESOLVED")

    def test_missing_or_failed_cells_never_yield_a_verdict(self):
        cells = self._cells([1.0] * 3, [0.95] * 3, [0.5] * 3)
        cells[4] = _cell(1001, "random_recurrent", None, status="CAP_EXCEEDED")
        out = judge(cells, self.CFG)
        self.assertEqual(out["verdict"], "INCOMPLETE")
        self.assertEqual(out["complete_seeds"], [1000, 1002])
        self.assertEqual(judge(cells[:6], self.CFG)["verdict"], "INCOMPLETE")


TINY = {
    "name": "unit-tiny-diag", "run_kind": "diagnostic", "seeds": [7], "variant": "symmetric",
    "maze": {"corridor_min": 2, "corridor_max": 2, "n_distractors": 2, "timeout_slack": 4},
    "horizon": 2, "train_budget": 120, "dev_budget": 60, "branch_prob": 0.3, "behavior_forward_prob": 0.8,
    "hidden": 16, "head": {"lr": 0.01, "batch_episodes": 4, "epoch_candidates": [1, 2]},
    "eval": {"n_calibration": 8, "n_test": 8, "plan_horizon": 2, "gamma": 0.9, "stratify_clue": True},
    "decision": {"metric": "forced_commit_accuracy", "positive_control_min": 0.9,
                 "negative_control_max_abs_dev_from_half": 0.1, "r1_mie": 0.1, "required_complete_seeds": 1},
    "resource_caps": {"cpu_seconds_hard": 120, "cpu_kill_grace_s": 5, "reserve_s": 0,
                      "initial_cell_cost_estimate_s": 1, "address_space_gib": 3},
}


def _run(cfg, out):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
    return subprocess.run([sys.executable, "-m", "acpr.diagnose", "--config", f.name, "--out", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=300)


class TestDiagnoseRunner(unittest.TestCase):
    def test_tiny_run_records_everything(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r"
            p = _run(TINY, out)
            self.assertEqual(p.returncode, 0, p.stderr)
            cells = json.loads((out / "cells.json").read_text())
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual([c["status"] for c in cells], ["OK"] * 3)
            self.assertEqual(man["limits"]["RLIMIT_CPU"], [120, 125])
            self.assertEqual(man["limits"]["RLIMIT_AS"], [3 * 1024**3] * 2)
            for c in cells:
                self.assertEqual(c["standardizer_source_split"], "train")
                self.assertIn(c["selected_epochs"], (1, 2))
                self.assertEqual(c["planning"]["ledger"]["restores"], 0)
                self.assertGreater(c["mac"], 0)
            tr = man["env_transitions"]
            self.assertEqual(tr["train_main"] + tr["train_branch"], 120)
            self.assertEqual(tr["dev_main"] + tr["dev_branch"], 60)
            self.assertTrue((out / "verdict.json").exists())

    def test_projected_cost_marks_not_run(self):
        cfg = json.loads(json.dumps(TINY))
        cfg["resource_caps"]["initial_cell_cost_estimate_s"] = 10**6
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r"
            self.assertEqual(_run(cfg, out).returncode, 0)
            cells = json.loads((out / "cells.json").read_text())
            self.assertEqual({c["status"] for c in cells}, {"NOT_RUN"})
            self.assertEqual(json.loads((out / "verdict.json").read_text())["verdict"], "INCOMPLETE")

    def test_cpu_cap_is_enforced(self):
        cfg = json.loads(json.dumps(TINY))
        cfg["resource_caps"].update(cpu_seconds_hard=1, initial_cell_cost_estimate_s=0)
        cfg["head"]["epoch_candidates"] = [1, 5000]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r"
            p = _run(cfg, out)
            self.assertEqual(p.returncode, 0, p.stderr)
            cells = json.loads((out / "cells.json").read_text())
            self.assertEqual(cells[0]["status"], "CAP_EXCEEDED")
            self.assertEqual({c["status"] for c in cells[1:]}, {"NOT_RUN"})
            self.assertEqual(json.loads((out / "verdict.json").read_text())["verdict"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
