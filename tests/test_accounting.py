"""Transition accounting: every branch step is charged; caps are hard."""

import random
import unittest

from acpr.accounting import BudgetExceeded, MeteredSimulator, PrivilegeError, TransitionLedger
from acpr.collect import CollectConfig, collect_split
from acpr.env import ACT_FORWARD, N_ACTIONS, AliasedTMaze, MazeConfig
from acpr.evaluate import EvalConfig, planning_eval
from acpr.models import LatentPredictor, RecurrentEncoder
from acpr.splits import episode_seed

MAZE = MazeConfig(corridor_min=2, corridor_max=3, n_distractors=2)


class TestLedger(unittest.TestCase):
    def test_hard_cap_blocks_the_step(self):
        ledger = TransitionLedger(cap=3, label="t")
        sim = MeteredSimulator(AliasedTMaze(MAZE), ledger)
        sim.reset(0)
        for _ in range(3):
            sim.step(ACT_FORWARD, kind="main")
        before = sim._sim.get_state()
        with self.assertRaises(BudgetExceeded):
            sim.step(ACT_FORWARD, kind="branch")
        self.assertEqual(sim._sim.get_state(), before)  # not executed
        self.assertEqual(ledger.steps, 3)

    def test_unknown_kind_rejected(self):
        ledger = TransitionLedger(cap=None, label="t")
        with self.assertRaises(ValueError):
            ledger.charge("free")

    def test_restore_forbidden_ledger(self):
        sim = MeteredSimulator(AliasedTMaze(MAZE), TransitionLedger(None, "eval", allow_restore=False))
        sim.reset(0)
        with self.assertRaises(PrivilegeError):
            sim.snapshot()
        with self.assertRaises(PrivilegeError):
            sim.restore(sim._sim.get_state())


class TestCollectorAccounting(unittest.TestCase):
    def test_totals_match_records_and_cap(self):
        for budget in (1, 7, 50, 333):
            for horizon in (1, 2, 4):
                for bp in (0.0, 0.3, 1.0):
                    res = collect_split(MAZE, CollectConfig(budget, horizon, bp), run_seed=5, split="train")
                    led = res.ledger
                    recorded_main = sum(len(e.actions) for e in res.episodes)
                    recorded_branch = sum(len(b.actions) for e in res.episodes for b in e.branches)
                    self.assertEqual(led["transitions_by_kind"]["main"], recorded_main)
                    self.assertEqual(led["transitions_by_kind"]["branch"], recorded_branch)
                    self.assertEqual(led["transitions_total"], recorded_main + recorded_branch)
                    self.assertEqual(led["transitions_total"], sum(e.n_transitions() for e in res.episodes))
                    self.assertLessEqual(led["transitions_total"], budget)
                    # The collector spends the budget exactly (equal actual spend, not just equal cap).
                    self.assertEqual(led["transitions_total"], budget)
                    n_sets = res.stats["branch_sets"]
                    self.assertEqual(led["snapshots"], n_sets)
                    self.assertEqual(led["restores"], n_sets * (N_ACTIONS + 1))
                    self.assertEqual(led["resets"], len(res.episodes))
                    for e in res.episodes:
                        for b in e.branches:
                            self.assertLessEqual(len(b.actions), horizon)
                            self.assertGreaterEqual(len(b.actions), 1)

    def test_branch_and_nobranch_spend_equal_budget(self):
        b = collect_split(MAZE, CollectConfig(400, 3, 0.5), 1, "train")
        n = collect_split(MAZE, CollectConfig(400, 3, 0.0), 1, "train")
        self.assertEqual(b.ledger["transitions_total"], n.ledger["transitions_total"])
        self.assertEqual(n.ledger["restores"], 0)
        self.assertGreater(b.ledger["restores"], 0)
        self.assertGreater(len(n.episodes), len(b.episodes))

    def test_branching_does_not_change_main_trajectory(self):
        kw = dict(horizon=3, max_episodes=6)
        b = collect_split(MAZE, CollectConfig(10_000, branch_prob=0.7, **kw), 2, "train")
        n = collect_split(MAZE, CollectConfig(10_000, branch_prob=0.0, **kw), 2, "train")
        for eb, en in zip(b.episodes, n.episodes):
            self.assertEqual(
                (eb.observations, eb.actions, eb.rewards, eb.terminated, eb.truncated),
                (en.observations, en.actions, en.rewards, en.terminated, en.truncated),
            )

    def test_branches_start_from_the_recorded_history_state(self):
        res = collect_split(MAZE, CollectConfig(2_000, 3, 0.5), 3, "train")
        sim = AliasedTMaze(MAZE)
        checked = 0
        for i, ep in enumerate(res.episodes):
            for br in ep.branches:
                sim.reset(episode_seed(3, "train", i))
                for a in ep.actions[: br.t]:
                    sim.step(a)
                replay = [sim.step(a)[:2] for a in br.actions]
                self.assertEqual([o for o, _ in replay], list(br.observations))
                self.assertEqual([r for _, r in replay], list(br.rewards))
                checked += 1
        self.assertGreater(checked, 50)


class TestEvaluationAccounting(unittest.TestCase):
    def test_planner_uses_no_hidden_rollouts(self):
        rng = random.Random(0)
        enc = RecurrentEncoder(MAZE.n_obs, N_ACTIONS, 6, rng)
        head = LatentPredictor("head", 6, MAZE.n_obs, N_ACTIONS, True, rng)
        cfg = EvalConfig(n_calibration=0, n_test=15, plan_horizon=2)
        res = planning_eval(enc, head, MAZE, run_seed=0, cfg=cfg)
        led = res["ledger"]
        self.assertFalse(led["restore_allowed"])
        self.assertEqual(led["snapshots"], 0)
        self.assertEqual(led["restores"], 0)
        self.assertEqual(led["resets"], cfg.n_test)
        self.assertEqual(led["transitions_total"], led["transitions_by_kind"]["eval"])
        self.assertLessEqual(led["transitions_total"], led["cap"])


if __name__ == "__main__":
    unittest.main()
