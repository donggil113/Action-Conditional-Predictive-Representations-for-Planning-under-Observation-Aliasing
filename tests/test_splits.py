"""Episode splits: disjoint seed streams, determinism, branch ownership."""

import unittest

from acpr.collect import CollectConfig, collect_split
from acpr.data import dataset_hash
from acpr.env import MazeConfig
from acpr.splits import SPLITS, content_overlap, episode_seed

MAZE = MazeConfig(corridor_min=2, corridor_max=4, n_distractors=3)


class TestSplits(unittest.TestCase):
    def test_seed_streams_are_disjoint(self):
        seen = {}
        for run_seed in range(3):
            for split in SPLITS:
                for i in range(2000):
                    s = episode_seed(run_seed, split, i)
                    self.assertNotIn(s, seen, (run_seed, split, i, seen.get(s)))
                    seen[s] = (run_seed, split, i)

    def test_unknown_split_rejected(self):
        with self.assertRaises(ValueError):
            episode_seed(0, "validation", 0)

    def test_collection_is_deterministic_and_split_specific(self):
        cfg = CollectConfig(500, 3, 0.4)
        a = collect_split(MAZE, cfg, 0, "train")
        b = collect_split(MAZE, cfg, 0, "train")
        c = collect_split(MAZE, cfg, 0, "dev")
        self.assertEqual(dataset_hash(a.episodes), dataset_hash(b.episodes))
        self.assertNotEqual(dataset_hash(a.episodes), dataset_hash(c.episodes))
        ids_a = {e.episode_id for e in a.episodes}
        ids_c = {e.episode_id for e in c.episodes}
        self.assertFalse(ids_a & ids_c)
        self.assertTrue(all(e.split == "train" for e in a.episodes))
        self.assertTrue(all(e.split == "dev" for e in c.episodes))

    def test_branches_belong_to_their_episode(self):
        res = collect_split(MAZE, CollectConfig(800, 3, 0.5), 1, "train")
        for ep in res.episodes:
            for br in ep.branches:
                # A branch set is taken before main action t, so t indexes a real step.
                self.assertLess(br.t, len(ep.actions))
                self.assertGreaterEqual(br.t, 0)

    def test_content_overlap(self):
        self.assertEqual(content_overlap([(1, 2), (3,)], [(1, 2), (4,)]), 0.5)
        self.assertEqual(content_overlap([], [(1,)]), 0.0)
        self.assertEqual(content_overlap([(1,)], []), 0.0)


if __name__ == "__main__":
    unittest.main()
