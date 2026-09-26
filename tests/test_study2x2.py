"""2x2 study: shared datasets, restore-free readout pool, verdict rules, caps."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from acpr.collect import CollectConfig, collect_split
from acpr.controls import CurrentObsEncoder, fit_standardizer
from acpr.env import MazeConfig
from acpr.study2x2 import ARMS, judge

REPO = Path(__file__).resolve().parent.parent

TINY = {
    "name": "unit-tiny-2x2", "run_kind": "study", "variant": "symmetric", "seeds": [7],
    "arms": ["informative_control", "random_recurrent", "acp_branch", "acp_nobranch", "msp_branch", "msp_nobranch"],
    "maze": {"corridor_min": 2, "corridor_max": 2, "n_distractors": 2, "timeout_slack": 4},
    "horizon": 2, "behavior_forward_prob": 0.8, "hidden": 16,
    "pretraining": {"budget": 120, "branch_prob": 0.3, "epochs": 1, "lr": 0.01, "batch_episodes": 4},
    "readout_pool": {"train_budget": 100, "dev_budget": 40},
    "head": {"lr": 0.01, "batch_episodes": 4, "epoch_candidates": [1, 2]},
    "eval": {"n_calibration": 8, "n_test": 8, "plan_horizon": 2, "gamma": 0.9, "stratify_clue": True},
    "decision": {"metric": "forced_commit_accuracy", "mie": 0.1, "positive_control_min": 0.9,
                 "saturation_min": 0.95, "required_complete_seeds": 1},
    "resource_caps": {"cpu_seconds_hard": 300, "cpu_kill_grace_s": 5, "reserve_s": 0, "address_space_gib": 3,
                      "initial_cell_cost_estimate_s": {"learned": 1, "fixed": 1}},
}


def _run(cfg, out):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
    return subprocess.run([sys.executable, "-m", "acpr.study2x2", "--config", f.name, "--out", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=600)


class TestStudyRunner(unittest.TestCase):
    def test_tiny_run_shares_data_and_keeps_readout_restore_free(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r"
            p = _run(TINY, out)
            self.assertEqual(p.returncode, 0, p.stderr)
            cells = {c["arm"]: c for c in json.loads((out / "cells.json").read_text())}
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual({c["status"] for c in cells.values()}, {"OK"})
            # Exact dataset sharing within a collection method.
            self.assertEqual(cells["acp_branch"]["pretrain_dataset_sha256"], cells["msp_branch"]["pretrain_dataset_sha256"])
            self.assertEqual(cells["acp_nobranch"]["pretrain_dataset_sha256"], cells["msp_nobranch"]["pretrain_dataset_sha256"])
            self.assertNotEqual(cells["acp_branch"]["pretrain_dataset_sha256"], cells["acp_nobranch"]["pretrain_dataset_sha256"])
            # Same initial encoder -> random arm's encoder hash equals nothing trained; learned ones differ from it.
            self.assertNotEqual(cells["acp_branch"]["encoder_sha256"], cells["random_recurrent"]["encoder_sha256"])
            rec = man["data"]["7"]
            self.assertEqual(rec["pretrain_branch"]["ledger"]["transitions_total"], 120)
            self.assertEqual(rec["pretrain_nobranch"]["ledger"]["transitions_total"], 120)
            self.assertGreater(rec["pretrain_branch"]["ledger"]["restores"], 0)
            for key in ("pretrain_nobranch", "head_train", "head_dev"):
                self.assertEqual(rec[key]["ledger"]["restores"], 0, key)
            self.assertEqual(rec["head_train"]["ledger"]["transitions_total"], 100)
            self.assertEqual(rec["head_dev"]["ledger"]["transitions_total"], 40)
            self.assertEqual(set(rec["split_checks"]["latent_seed_overlap_between_splits"].values()), {0})
            self.assertEqual(rec["prefix_replays"], 0)
            for c in cells.values():
                self.assertEqual(c["standardizer_source_split"], "head_train")
                self.assertEqual(c["planning"]["ledger"]["restores"], 0)
                self.assertIn(c["selected_epochs"], (1, 2))
            tr = man["env_transitions"]
            self.assertEqual(tr["test_restores"], 0)
            self.assertGreater(tr["split_check_test_prefix_eval"], 0)
            self.assertTrue((out / "weights" / "7_acp_branch_encoder.json").exists())
            self.assertTrue((out / "weights" / "7_informative_control_head.json").exists())
            self.assertIn("contrasts", json.loads((out / "verdict.json").read_text()))

    def test_cpu_cap_marks_remaining_cells(self):
        cfg = json.loads(json.dumps(TINY))
        cfg["resource_caps"].update(cpu_seconds_hard=1, initial_cell_cost_estimate_s={"learned": 0, "fixed": 0})
        cfg["head"]["epoch_candidates"] = [1, 5000]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r"
            p = _run(cfg, out)
            self.assertEqual(p.returncode, 0, p.stderr)
            statuses = [c["status"] for c in json.loads((out / "cells.json").read_text())]
            self.assertIn("CAP_EXCEEDED", statuses)
            self.assertNotIn("FAILED", statuses)
            self.assertEqual(json.loads((out / "verdict.json").read_text())["verdict"], "INCOMPLETE")


def _cells(vals):
    out = []
    for s, row in vals.items():
        for arm, v in row.items():
            out.append({"seed": s, "arm": arm, "status": "OK", "planning": {"forced_commit_accuracy": v}})
    return out


class TestStudyJudge(unittest.TestCase):
    CFG = {"seeds": [1, 2, 3], "decision": {"metric": "forced_commit_accuracy", "mie": 0.1, "positive_control_min": 0.9,
                                            "saturation_min": 0.95, "required_complete_seeds": 3}}

    def _row(self, acp_b, acp_n, msp_b, msp_n, rnd=0.5, info=1.0):
        return {"acp_branch": acp_b, "acp_nobranch": acp_n, "msp_branch": msp_b, "msp_nobranch": msp_n,
                "random_recurrent": rnd, "informative_control": info}

    def _judge(self, rows):
        return judge(_cells({s: r for s, r in zip((1, 2, 3), rows)}), self.CFG)

    def test_labels(self):
        J = lambda *r: self._judge(r)["verdict"]
        self.assertEqual(J(*[self._row(0.9, 0.6, 0.6, 0.6)] * 3), "PRELIMINARY_COLLECTION_BENEFIT")
        self.assertEqual(J(*[self._row(0.8, 0.8, 0.5, 0.5)] * 3), "ACTION_CONDITIONING_ONLY")
        self.assertEqual(J(*[self._row(0.55, 0.9, 0.5, 0.5)] * 3), "LEARNING_BENEFIT_NOT_SUPPORTED")
        self.assertEqual(J(*[self._row(1.0, 1.0, 0.97, 0.96)] * 3), "SATURATED_COMPARISON")
        self.assertEqual(J(*[self._row(0.7, 0.7, 0.7, 0.7)] * 3), "NO_CONSISTENT_MIE_CONTRAST")
        self.assertEqual(J(self._row(0.9, 0.6, 0.6, 0.6, info=0.8), *[self._row(0.9, 0.6, 0.6, 0.6)] * 2),
                         "EVALUATOR_UNRESOLVED")

    def test_inconsistent_seed_blocks_mie_claim(self):
        out = self._judge([self._row(0.9, 0.6, 0.6, 0.6), self._row(0.9, 0.6, 0.6, 0.6), self._row(0.65, 0.6, 0.6, 0.6)])
        self.assertIs(out["contrasts"]["acp_branch_minus_acp_nobranch"]["all_seeds_ge_mie"], False)
        self.assertNotEqual(out["verdict"], "PRELIMINARY_COLLECTION_BENEFIT")

    def test_mie_only_on_preregistered_contrasts_and_interaction(self):
        out = self._judge([self._row(0.9, 0.6, 0.7, 0.6)] * 3)
        c = out["contrasts"]
        self.assertEqual(c["msp_branch_minus_msp_nobranch"]["all_seeds_ge_mie"], "n/a")
        self.assertEqual(c["interaction"]["all_seeds_ge_mie"], "n/a")
        self.assertAlmostEqual(c["interaction"]["mean"], (0.9 - 0.7) - (0.6 - 0.6))

    def test_missing_cell_is_incomplete(self):
        cells = _cells({s: self._row(0.9, 0.6, 0.6, 0.6) for s in (1, 2, 3)})
        cells = [c for c in cells if not (c["seed"] == 3 and c["arm"] == "msp_nobranch")]
        out = judge(cells, self.CFG)
        self.assertEqual(out["verdict"], "INCOMPLETE")
        self.assertEqual(out["complete_seeds"], [1, 2])
        self.assertEqual(len(ARMS), 6)


class TestReadoutPoolStandardizer(unittest.TestCase):
    def test_standardizer_bound_to_named_split(self):
        maze = MazeConfig(corridor_min=2, corridor_max=2, n_distractors=2)
        pool = collect_split(maze, CollectConfig(80, 2, 0.0), 0, "head_train")
        enc = fit_standardizer(CurrentObsEncoder(maze.n_obs, 16), pool.episodes, split="head_train")
        self.assertEqual(enc.source_split, "head_train")
        with self.assertRaises(ValueError):
            fit_standardizer(CurrentObsEncoder(maze.n_obs, 16), pool.episodes)  # default "train" refuses


if __name__ == "__main__":
    unittest.main()
