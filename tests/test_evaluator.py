"""Evaluator mechanics: stratification, probe sensitivity, planner choice."""

import random
import unittest

from acpr.data import History
from acpr.env import ACT_FORWARD, ACT_LEFT, ACT_RIGHT, MazeConfig
from acpr.evaluate import EvalConfig, clue_probe, collect_probe_histories, plan_action

MAZE = MazeConfig(corridor_min=2, corridor_max=3, n_distractors=2)
CFG = EvalConfig(n_calibration=0, n_test=0, plan_horizon=2)


class FakeEncoder:
    """Maps a history to a fixed feature via a user function."""

    def __init__(self, fn):
        self.fn = fn

    def encode_values(self, observations, actions, rewards):
        return [self.fn(observations)]


class FakeHead:
    def __init__(self, table):
        self.table = table  # first action -> reward list

    def predict_rewards(self, z, seq):
        return [self.table[seq[0]]] + [0.0] * (len(seq) - 1)


class TestStratification(unittest.TestCase):
    def test_eval_clues_are_balanced_and_alternate(self):
        _, clues, ledger = collect_probe_histories(MAZE, 0, "test", 40)
        self.assertEqual(clues, [i % 2 for i in range(40)])
        self.assertEqual(ledger["restores"], 0)
        _, clues_free, _ = collect_probe_histories(MAZE, 0, "test", 40, stratify_clue=False)
        self.assertEqual(len(clues_free), 40)


class TestProbe(unittest.TestCase):
    def _histories(self, n, rng):
        # One history per item; the feature function reads a private tag in o_0.
        return [History.start(i) for i in range(n)], [rng.randrange(2) for _ in range(n)]

    def test_probe_is_scale_invariant(self):
        rng = random.Random(0)
        labels = {}

        def feat(obs):
            i = obs[0]
            return [1e-4 * labels[i] + 1e-6 * ((i * 7919) % 13), 0.3]

        n = 120
        hs = [History.start(i) for i in range(n)]
        for i in range(n):
            labels[i] = rng.randrange(2)
        ys = [labels[i] for i in range(n)]
        res = clue_probe(FakeEncoder(feat), hs[:60], ys[:60], hs[60:], ys[60:], CFG)
        self.assertEqual(res["status"], "OK")
        self.assertGreaterEqual(res["test_accuracy"], 0.95)

    def test_uninformative_features_do_not_decode(self):
        hs = [History.start(i) for i in range(80)]
        ys = [i % 2 for i in range(80)]
        res = clue_probe(FakeEncoder(lambda o: [0.5, -0.5]), hs[:40], ys[:40], hs[40:], ys[40:], CFG)
        self.assertEqual(res["test_accuracy"], 0.5)

    def test_probe_not_run_without_both_classes(self):
        hs = [History.start(i) for i in range(4)]
        res = clue_probe(FakeEncoder(lambda o: [0.0]), hs, [1, 1, 1, 1], hs, [0, 1, 0, 1], CFG)
        self.assertEqual(res["status"], "NOT_RUN")


class TestPlanner(unittest.TestCase):
    def test_argmax_and_forced_commit(self):
        enc = FakeEncoder(lambda o: [0.0])
        head = FakeHead({ACT_FORWARD: 0.5, ACT_LEFT: 0.2, ACT_RIGHT: -1.0})
        h = History.start(2)
        rng = random.Random(0)
        self.assertEqual(plan_action(enc, head, h, 2, 0.9, rng), ACT_FORWARD)
        self.assertEqual(plan_action(enc, head, h, 2, 0.9, rng, first_actions=(ACT_LEFT, ACT_RIGHT)), ACT_LEFT)

    def test_ties_are_broken_randomly(self):
        enc = FakeEncoder(lambda o: [0.0])
        head = FakeHead({ACT_FORWARD: -1.0, ACT_LEFT: 0.0, ACT_RIGHT: 0.0})
        rng = random.Random(0)
        picks = {plan_action(enc, head, History.start(2), 1, 0.9, rng) for _ in range(50)}
        self.assertEqual(picks, {ACT_LEFT, ACT_RIGHT})


class ClueAndCurrentEncoder:
    """History-only features: identity of o_0 and whether o_t is the junction."""

    def __init__(self, hidden):
        self.hidden = hidden

    def encode_values(self, observations, actions, rewards):
        from acpr.env import OBS_CLUE_A, OBS_JUNCTION

        clue = [1.0, 0.0] if observations[0] == OBS_CLUE_A else [0.0, 1.0]
        return [clue + [float(o == OBS_JUNCTION)] + [0.0] * (self.hidden - 3) for o in observations]


class TestEvaluationPathwayPositiveControl(unittest.TestCase):
    """If a representation carries the clue, stage 2 + planner must be able to use it.

    Numeric fixture check of the evaluation pathway, not a proof. The 0.9 bar
    is a validity requirement for a perfectly informative representation; an
    exploratory run of the same setup reached 1.0 before this test was
    written, and the bar must not be lowered to make a regression pass.
    """

    def test_informative_history_features_are_exploitable(self):
        from acpr.collect import CollectConfig, collect_split
        from acpr.env import N_ACTIONS
        from acpr.evaluate import planning_eval
        from acpr.models import LatentPredictor
        from acpr.train import TrainConfig, build_windows, train_predictor

        for symmetric in (True, False):
            maze = MazeConfig(corridor_min=2, corridor_max=3, n_distractors=2, symmetric=symmetric)
            res = collect_split(maze, CollectConfig(800, 2, 0.25), 0, "train")
            enc = ClueAndCurrentEncoder(8)
            head = LatentPredictor("head", 8, maze.n_obs, N_ACTIONS, True, random.Random(3))
            train_predictor(enc, head, res.episodes, build_windows(res.episodes, 2),
                            TrainConfig(30, 0.01, 8, 0, obs_weight=0.0), train_encoder=False)
            ev = planning_eval(enc, head, maze, 0, EvalConfig(0, 40, 2, 0.9))
            self.assertGreaterEqual(ev["forced_commit_accuracy"], 0.9, symmetric)
            self.assertGreaterEqual(ev["success_rate"], 0.9, symmetric)

    def test_uninformative_features_give_chance_forced_accuracy(self):
        from acpr.collect import CollectConfig, collect_split
        from acpr.env import N_ACTIONS
        from acpr.evaluate import planning_eval
        from acpr.models import LatentPredictor
        from acpr.train import TrainConfig, build_windows, train_predictor

        maze = MazeConfig(corridor_min=2, corridor_max=3, n_distractors=2, symmetric=True)
        res = collect_split(maze, CollectConfig(800, 2, 0.25), 0, "train")
        enc = FakeEncoder(lambda o: [0.0] * 8)
        enc.encode_values = lambda obs, acts, rews: [[0.0] * 8 for _ in obs]
        head = LatentPredictor("head", 8, maze.n_obs, N_ACTIONS, True, random.Random(3))
        train_predictor(enc, head, res.episodes, build_windows(res.episodes, 2),
                        TrainConfig(10, 0.01, 8, 0, obs_weight=0.0), train_encoder=False)
        ev = planning_eval(enc, head, maze, 0, EvalConfig(0, 40, 2, 0.9))
        # A clue-blind model cannot beat chance on a clue-balanced test set.
        self.assertEqual(ev["forced_commit_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
