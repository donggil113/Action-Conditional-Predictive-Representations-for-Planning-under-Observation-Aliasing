"""Information access: actors and learners see only permitted history."""

import dataclasses
import inspect
import random
import unittest

from acpr import autodiff as ad
from acpr.collect import CollectConfig, collect_split, make_behavior_policy
from acpr.data import (
    ACTOR_VISIBLE_BRANCH_FIELDS,
    ACTOR_VISIBLE_EPISODE_FIELDS,
    Episode,
    EvaluatorLabel,
    History,
    episode_to_json,
)
from acpr.env import N_ACTIONS, MazeConfig
from acpr.evaluate import plan_action, sequence_scores
from acpr.models import LatentPredictor, RecurrentEncoder, effective_param_count, param_count, params_hash
from acpr.train import TrainConfig, build_windows, episode_windows, train_predictor

MAZE = MazeConfig(corridor_min=2, corridor_max=3, n_distractors=2)
FORBIDDEN = {"clue", "corridor_len", "pos", "state", "rng_state", "optimal_arm", "ground_truth", "labels", "label", "sim", "simulator"}


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _keys(v)


class TestActorVisibleRecords(unittest.TestCase):
    def test_history_has_only_observable_fields(self):
        self.assertEqual({f.name for f in dataclasses.fields(History)}, {"observations", "actions", "rewards"})

    def test_episode_and_branch_fields_whitelisted(self):
        self.assertEqual(
            set(ACTOR_VISIBLE_EPISODE_FIELDS),
            {"episode_id", "split", "observations", "actions", "rewards", "terminated", "truncated", "budget_truncated", "branches"},
        )
        self.assertEqual(
            set(ACTOR_VISIBLE_BRANCH_FIELDS),
            {"t", "actions", "observations", "rewards", "terminated", "truncated"},
        )
        self.assertFalse(FORBIDDEN & set(ACTOR_VISIBLE_EPISODE_FIELDS))

    def test_serialized_dataset_contains_no_ground_truth(self):
        res = collect_split(MAZE, CollectConfig(300, 2, 0.5), 0, "train")
        for ep in res.episodes:
            self.assertFalse(FORBIDDEN & set(_keys(episode_to_json(ep))))
        # Ground truth exists, but only in the separate evaluator mapping.
        self.assertEqual(set(res.labels), {e.episode_id for e in res.episodes})
        self.assertTrue(all(isinstance(v, EvaluatorLabel) for v in res.labels.values()))

    def test_behavior_policy_receives_only_history(self):
        seen = []
        inner = make_behavior_policy(0.8)

        def spy(history, rng):
            self.assertIs(type(history), History)
            self.assertIs(type(rng), random.Random)
            seen.append(history)
            return inner(history, rng)

        res = collect_split(MAZE, CollectConfig(200, 2, 0.5), 0, "train", behavior=spy)
        self.assertEqual(len(seen), sum(len(e.actions) for e in res.episodes))

    def test_learning_and_planning_signatures_take_no_privileged_inputs(self):
        for fn in (train_predictor, build_windows, episode_windows, plan_action, sequence_scores,
                   RecurrentEncoder.forward, LatentPredictor.rollout):
            names = set(inspect.signature(fn).parameters)
            self.assertFalse(FORBIDDEN & names, fn.__name__)


class TestEncoderCausality(unittest.TestCase):
    def test_z_t_ignores_the_future(self):
        enc = RecurrentEncoder(MAZE.n_obs, N_ACTIONS, 6, random.Random(0))
        obs, acts, rews = [0, 7, 8, 2, 3], [0, 0, 0, 1], [0.0, 0.0, 0.0, 1.0]
        base = enc.encode_values(obs, acts, rews)
        alt = enc.encode_values(obs[:3] + [8, 4], acts[:2] + [2, 2], rews[:2] + [0.0, -1.0])
        for t in range(3):
            self.assertEqual(base[t], alt[t])
        self.assertNotEqual(base[3], alt[3])


class TestArmParity(unittest.TestCase):
    def setUp(self):
        self.acp = LatentPredictor("p", 6, MAZE.n_obs, N_ACTIONS, True, random.Random(1))
        self.msp = LatentPredictor("p", 6, MAZE.n_obs, N_ACTIONS, False, random.Random(1))

    def test_same_parameter_count_and_init(self):
        self.assertEqual(param_count(self.acp.params()), param_count(self.msp.params()))
        self.assertEqual(params_hash(self.acp.params()), params_hash(self.msp.params()))
        self.assertEqual(effective_param_count(self.acp) - effective_param_count(self.msp), 6 * N_ACTIONS)

    def test_msp_ignores_actions_and_acp_does_not(self):
        z = [0.3, -0.2, 0.1, 0.5, -0.4, 0.0]
        seqs = [(0, 0), (1, 2), (2, 1)]
        msp = {self.msp.predict_rewards(z, s)[1] for s in seqs}
        acp = {self.acp.predict_rewards(z, s)[1] for s in seqs}
        self.assertEqual(len(msp), 1)
        self.assertEqual(len(acp), 3)

    def test_frozen_stage2_leaves_encoder_untouched(self):
        res = collect_split(MAZE, CollectConfig(150, 2, 0.5), 0, "train")
        windows = build_windows(res.episodes, 2)
        enc = RecurrentEncoder(MAZE.n_obs, N_ACTIONS, 6, random.Random(0))
        before = params_hash(enc.params())
        head = LatentPredictor("head", 6, MAZE.n_obs, N_ACTIONS, True, random.Random(2))
        head_before = params_hash(head.params())
        train_predictor(enc, head, res.episodes, windows, TrainConfig(1, 0.01, 4, 0, obs_weight=0.0), train_encoder=False)
        self.assertEqual(params_hash(enc.params()), before)
        self.assertNotEqual(params_hash(head.params()), head_before)
        self.assertTrue(all(g == 0.0 for p in enc.params() for g in p.g))


if __name__ == "__main__":
    unittest.main()
