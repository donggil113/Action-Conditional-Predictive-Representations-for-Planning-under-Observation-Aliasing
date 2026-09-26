"""Environment validity: aliasing exists and has the intended structure."""

import itertools
import random
import unittest
from collections import Counter

from acpr.env import (
    ACT_FORWARD,
    ACT_LEFT,
    ACT_RIGHT,
    N_ACTIONS,
    N_FIXED_OBS,
    OBS_CLUE_A,
    OBS_CLUE_B,
    OBS_JUNCTION,
    OBS_TERMINAL_PAD,
    AliasedTMaze,
    MazeConfig,
)


def walk_to_junction(sim):
    obs_seq = []
    obs = None
    while obs != OBS_JUNCTION:
        obs, _, term, trunc = sim.step(ACT_FORWARD)
        obs_seq.append(obs)
        assert not (term or trunc)
    return obs_seq


def future_outcomes(sim, state, horizon):
    """Outcome multiset over all action sequences of length `horizon` (uniform)."""
    outcomes = Counter()
    for seq in itertools.product(range(N_ACTIONS), repeat=horizon):
        sim.set_state(state)
        trace = []
        for a in seq:
            obs, r, term, trunc = sim.step(a)
            trace.append((obs, r))
            if term or trunc:
                break
        outcomes[tuple(trace)] += 1
    return outcomes


class TestAliasedTMaze(unittest.TestCase):
    def setUp(self):
        self.cfg = MazeConfig(corridor_min=2, corridor_max=4, n_distractors=3, symmetric=True)

    def test_same_current_observation_different_action_outcome(self):
        a, b = AliasedTMaze(self.cfg), AliasedTMaze(self.cfg)
        self.assertEqual(a.reset(7, clue=0), OBS_CLUE_A)
        self.assertEqual(b.reset(7, clue=1), OBS_CLUE_B)
        # Same seed: identical corridor observations and identical junction obs.
        self.assertEqual(walk_to_junction(a), walk_to_junction(b))
        oa, ra, _, _ = a.step(ACT_LEFT)
        ob, rb, _, _ = b.step(ACT_LEFT)
        self.assertNotEqual((oa, ra), (ob, rb))

    def test_clue_only_visible_at_clue_cell(self):
        for seed in range(50):
            sims = [AliasedTMaze(self.cfg) for _ in range(2)]
            traces = []
            for clue, sim in enumerate(sims):
                sim.reset(seed, clue=clue)
                traces.append(walk_to_junction(sim))
            self.assertEqual(traces[0], traces[1])
            for o in traces[0]:
                self.assertNotIn(o, (OBS_CLUE_A, OBS_CLUE_B))

    def test_symmetric_action_marginal_future_is_clue_independent(self):
        sim = AliasedTMaze(self.cfg)
        for seed in range(10):
            for steps_before_junction in (0, 1):
                states = []
                for clue in (0, 1):
                    sim.reset(seed, clue=clue)
                    for _ in range(sim.junction_pos - steps_before_junction):
                        sim.step(ACT_FORWARD)
                    states.append(sim.get_state())
                for horizon in (1, 2, 3):
                    m0 = future_outcomes(sim, states[0], horizon)
                    m1 = future_outcomes(sim, states[1], horizon)
                    self.assertEqual(m0, m1, (seed, steps_before_junction, horizon))
                    if horizon <= steps_before_junction:
                        continue  # no arm reachable within the horizon
                    # ... while the action-conditional futures differ.
                    self.assertNotEqual(
                        self._conditional(sim, states[0], horizon),
                        self._conditional(sim, states[1], horizon),
                    )

    @staticmethod
    def _conditional(sim, state, horizon):
        out = {}
        for seq in itertools.product(range(N_ACTIONS), repeat=horizon):
            sim.set_state(state)
            trace = []
            for a in seq:
                obs, r, term, trunc = sim.step(a)
                trace.append((obs, r))
                if term or trunc:
                    break
            out[seq] = tuple(trace)
        return out

    def test_asymmetric_action_marginal_future_depends_on_clue(self):
        cfg = MazeConfig(corridor_min=2, corridor_max=4, n_distractors=3, symmetric=False)
        sim = AliasedTMaze(cfg)
        states = []
        for clue in (0, 1):
            sim.reset(3, clue=clue)
            walk_to_junction(sim)
            states.append(sim.get_state())
        self.assertNotEqual(future_outcomes(sim, states[0], 1), future_outcomes(sim, states[1], 1))

    def test_memoryless_policies_succeed_at_most_half(self):
        # A memoryless policy's junction choice is a function of OBS_JUNCTION only,
        # hence clue-independent; with a uniform clue it succeeds for at most one clue.
        for symmetric in (True, False):
            cfg = MazeConfig(symmetric=symmetric)
            for junction_action in (ACT_LEFT, ACT_RIGHT):
                wins = 0
                for clue in (0, 1):
                    sim = AliasedTMaze(cfg)
                    sim.reset(11, clue=clue)
                    walk_to_junction(sim)
                    sim.step(junction_action)
                    wins += junction_action == sim.optimal_arm(clue)
                self.assertEqual(wins, 1)

    def test_history_based_policy_solves_task(self):
        for symmetric in (True, False):
            cfg = MazeConfig(symmetric=symmetric)
            for seed in range(20):
                sim = AliasedTMaze(cfg)
                first = sim.reset(seed)
                walk_to_junction(sim)
                arm = ACT_LEFT if first == OBS_CLUE_A else ACT_RIGHT
                _, _, term, _ = sim.step(arm)
                self.assertTrue(term)
                self.assertEqual(arm, sim.ground_truth()["optimal_arm"])

    def test_timeout_truncates(self):
        sim = AliasedTMaze(self.cfg)
        sim.reset(0)
        walk_to_junction(sim)
        steps = sim.get_state().t
        truncated = False
        while not truncated:
            _, _, term, truncated = sim.step(ACT_FORWARD)
            self.assertFalse(term)
            steps += 1
        self.assertEqual(steps, self.cfg.max_steps(sim.get_state().corridor_len))
        with self.assertRaises(RuntimeError):
            sim.step(ACT_FORWARD)

    def test_state_roundtrip_reproduces_future(self):
        sim = AliasedTMaze(self.cfg)
        rng = random.Random(0)
        for seed in range(20):
            sim.reset(seed)
            sim.step(ACT_FORWARD)
            state = sim.get_state()
            seq = [rng.randrange(N_ACTIONS) for _ in range(4)]
            traces = []
            for _ in range(2):
                sim.set_state(state)
                trace = []
                for a in seq:
                    out = sim.step(a)
                    trace.append(out)
                    if out[2] or out[3]:
                        break
                traces.append(trace)
            self.assertEqual(traces[0], traces[1])

    def test_terminal_pad_never_emitted(self):
        sim = AliasedTMaze(self.cfg)
        rng = random.Random(1)
        for seed in range(200):
            obs = sim.reset(seed)
            self.assertLess(obs, self.cfg.n_obs)
            done = False
            while not done:
                obs, _, term, trunc = sim.step(rng.randrange(N_ACTIONS))
                self.assertNotEqual(obs, OBS_TERMINAL_PAD)
                self.assertLess(obs, self.cfg.n_obs)
                done = term or trunc

    def test_distractors_cover_vocabulary(self):
        sim = AliasedTMaze(self.cfg)
        seen = set()
        for seed in range(100):
            sim.reset(seed)
            seen.update(o for o in walk_to_junction(sim) if o >= N_FIXED_OBS)
        self.assertEqual(seen, set(range(N_FIXED_OBS, self.cfg.n_obs)))


if __name__ == "__main__":
    unittest.main()
