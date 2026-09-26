"""Deterministic, split-disjoint seed derivation.

Episode seeds are derived from (run seed, split name, episode index) with
SHA-256, so train/development/calibration/test episodes come from disjoint
seed streams. Seed disjointness does not imply content disjointness in a tiny
environment; `content_overlap` measures the latter.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

# head_train / head_dev: restore-free readout pool shared by all arms of the
# 2x2 study (added later; existing splits' seeds are unchanged because the
# split name is part of the hash input).
SPLITS = ("train", "dev", "calibration", "test", "head_train", "head_dev")


def derive_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def episode_seed(run_seed: int, split: str, index: int) -> int:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return derive_seed("episode", run_seed, split, index)


def episode_id(run_seed: int, split: str, index: int) -> str:
    return f"s{run_seed}-{split}-{index:06d}"


def content_overlap(
    train_keys: Iterable[Sequence], eval_keys: Iterable[Sequence]
) -> float:
    """Fraction of eval items whose key also occurs in train."""
    train = {tuple(k) for k in train_keys}
    eval_list = [tuple(k) for k in eval_keys]
    if not eval_list:
        return 0.0
    return sum(k in train for k in eval_list) / len(eval_list)
