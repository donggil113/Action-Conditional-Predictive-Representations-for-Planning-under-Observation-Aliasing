"""Horizon semantics: alignment, termination padding, truncation masking."""

import random
import unittest

from acpr import autodiff as ad
from acpr.collect import CollectConfig, collect_split
from acpr.data import Branch, Episode
from acpr.env import N_ACTIONS, OBS_TERMINAL_PAD, MazeConfig
from acpr.models import LatentPredictor, RecurrentEncoder
from acpr.train import TrainConfig, _window_terms, episode_windows

MAZE = MazeConfig(corridor_min=2, corridor_max=3, n_distractors=2)


def make_episode(terminated, truncated=False, budget_truncated=False, branches=()):
    return Episode(
        episode_id="x", split="train",
        observations=(0, 7, 8, 2, 3) if terminated else (0, 7, 8, 2, 2),
        actions=(0, 0, 0, 1) if terminated else (0, 0, 0, 0),
        rewards=(0.0, 0.0, 0.0, 1.0) if terminated else (0.0, 0.0, 0.0, 0.0),
        terminated=terminated, truncated=truncated, budget_truncated=budget_truncated,
        branches=tuple(branches),
    )


class TestWindows(unittest.TestCase):
    def test_main_windows_align_with_episode(self):
        res = collect_split(MAZE, CollectConfig(600, 3, 0.3), 0, "train")
        for ep in res.episodes:
            ws = [w for w in episode_windows(ep, 3) if w.source == "main"]
            self.assertEqual([w.t for w in ws], list(range(len(ep.actions))))
            for w in ws:
                n_real = min(3, len(ep.actions) - w.t)
                self.assertEqual(w.actions[:n_real], ep.actions[w.t : w.t + n_real])
                self.assertEqual(w.observations[:n_real], ep.observations[w.t + 1 : w.t + 1 + n_real])
                self.assertEqual(w.rewards[:n_real], ep.rewards[w.t : w.t + n_real])
                self.assertEqual(len(w.actions), 3)
                self.assertEqual(len(w.mask), 3)

    def test_termination_pads_are_supervised(self):
        ws = episode_windows(make_episode(terminated=True), 3)
        last = ws[3]  # t=3: one real step (the terminal arm), then two pads
        self.assertEqual(last.observations, (3, OBS_TERMINAL_PAD, OBS_TERMINAL_PAD))
        self.assertEqual(last.rewards, (1.0, 0.0, 0.0))
        self.assertEqual(last.mask, (1, 1, 1))

    def test_truncation_tail_is_masked(self):
        for kw in ({"truncated": True}, {"budget_truncated": True}):
            ws = episode_windows(make_episode(terminated=False, **kw), 3)
            self.assertEqual(ws[3].mask, (1, 0, 0))
            self.assertEqual(ws[2].mask, (1, 1, 0))
            self.assertEqual(ws[1].mask, (1, 1, 1))

    def test_branch_windows(self):
        branches = [
            Branch(3, (1,), (3,), (1.0,), terminated=True, truncated=False),
            Branch(3, (0, 0), (2, 2), (0.0, 0.0), terminated=False, truncated=True),
            Branch(1, (0, 1, 2), (8, 2, 4), (0.0, 0.0, -1.0), terminated=True, truncated=False),
        ]
        ws = [w for w in episode_windows(make_episode(True, branches=branches), 3) if w.source == "branch"]
        self.assertEqual([w.mask for w in ws], [(1, 1, 1), (1, 1, 0), (1, 1, 1)])
        self.assertEqual(ws[0].observations, (3, OBS_TERMINAL_PAD, OBS_TERMINAL_PAD))
        with self.assertRaises(ValueError):
            episode_windows(make_episode(True, branches=[Branch(0, (0,) * 4, (7,) * 4, (0.0,) * 4, False, False)]), 3)

    def test_horizon_one_is_exactly_the_transitions(self):
        res = collect_split(MAZE, CollectConfig(500, 1, 0.5), 2, "train")
        n_targets = 0
        for ep in res.episodes:
            for w in episode_windows(ep, 1):
                self.assertEqual(len(w.actions), 1)
                self.assertEqual(w.mask, (1,))
                self.assertNotEqual(w.observations[0], OBS_TERMINAL_PAD)
                n_targets += 1
        self.assertEqual(n_targets, res.ledger["transitions_total"])

    def test_branch_steps_supervised_exactly_once(self):
        res = collect_split(MAZE, CollectConfig(700, 3, 0.5), 3, "train")
        real = 0
        for ep in res.episodes:
            for w in episode_windows(ep, 3):
                if w.source == "branch":
                    real += sum(1 for o, m in zip(w.observations, w.mask) if m and o != OBS_TERMINAL_PAD)
        self.assertEqual(real, res.ledger["transitions_by_kind"]["branch"])

    def test_pad_actions_are_deterministic(self):
        ep = make_episode(True)
        self.assertEqual(episode_windows(ep, 4), episode_windows(ep, 4))


class TestMaskedLoss(unittest.TestCase):
    def test_masked_targets_do_not_affect_loss_or_gradient(self):
        rng = random.Random(0)
        enc = RecurrentEncoder(MAZE.n_obs, N_ACTIONS, 5, rng)
        pred = LatentPredictor("p", 5, MAZE.n_obs, N_ACTIONS, True, rng)
        ep = make_episode(terminated=False, truncated=True)
        w = episode_windows(ep, 3)[3]
        self.assertEqual(w.mask, (1, 0, 0))
        w2 = w.__class__(w.t, w.actions, (w.observations[0], 5, 6), (w.rewards[0], 9.0, -9.0), w.mask, w.source)
        cfg = TrainConfig(1, 0.01, 1, 0)
        results = []
        for win in (w, w2):
            for p in enc.params() + pred.params():
                p.zero_grad()
            zs = enc.forward(ep.observations, ep.actions, ep.rewards)
            loss = ad.sum_scalars(_window_terms(pred, zs[win.t], win, cfg, 1.0))
            ad.backward(loss)
            results.append((loss.value[0], [list(p.g) for p in enc.params() + pred.params()]))
        self.assertEqual(results[0], results[1])

    def test_rollout_length_equals_horizon(self):
        pred = LatentPredictor("p", 4, MAZE.n_obs, N_ACTIONS, True, random.Random(0))
        for k in (1, 2, 5):
            out = pred.rollout(ad.const([0.1] * 4), [0] * k)
            self.assertEqual(len(out), k)
            self.assertEqual(len(out[0][0].value), MAZE.n_obs)


if __name__ == "__main__":
    unittest.main()
