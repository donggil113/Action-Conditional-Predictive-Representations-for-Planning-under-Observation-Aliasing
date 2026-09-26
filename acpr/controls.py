"""Fixed (untrained) history encoders for evaluator diagnostics.

All encoders read only the actor-visible history (observations, actions,
rewards) of a single episode; none receives simulator state, episode index,
seed or evaluator labels.

- `ClueCurrentObsEncoder` (informative control): identity of the clue
  observation at o_0 plus a one-hot of the current observation o_t. The clue
  is legitimately in the history; this is hand-designed, not learned.
- `CurrentObsEncoder` (memoryless negative control): one-hot of o_t only.
- `Standardized`: per-dimension z-scoring with statistics fitted on the
  TRAIN split only, applied identically to every encoder.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from .data import Episode
from .env import OBS_CLUE_A, OBS_CLUE_B


def _pad(v: List[float], hidden: int) -> List[float]:
    if len(v) > hidden:
        raise ValueError(f"feature size {len(v)} exceeds hidden size {hidden}")
    return v + [0.0] * (hidden - len(v))


class CurrentObsEncoder:
    def __init__(self, n_obs: int, hidden: int):
        self.n_obs = n_obs
        self.hidden = hidden

    def encode_values(self, observations, actions, rewards) -> List[List[float]]:
        out = []
        for o in observations:
            v = [0.0] * self.n_obs
            v[o] = 1.0
            out.append(_pad(v, self.hidden))
        return out


class ClueCurrentObsEncoder:
    def __init__(self, n_obs: int, hidden: int):
        self.n_obs = n_obs
        self.hidden = hidden

    def encode_values(self, observations, actions, rewards) -> List[List[float]]:
        first = observations[0]
        clue = [float(first == OBS_CLUE_A), float(first == OBS_CLUE_B)]
        out = []
        for o in observations:
            v = [0.0] * self.n_obs
            v[o] = 1.0
            out.append(_pad(clue + v, self.hidden))
        return out


class Standardized:
    def __init__(self, base, mean: Sequence[float], std: Sequence[float], source_split: str):
        self.base = base
        self.hidden = base.hidden
        self.mean = list(mean)
        self.std = list(std)
        self.source_split = source_split

    def encode_values(self, observations, actions, rewards) -> List[List[float]]:
        return [
            [(x - m) / s for x, m, s in zip(z, self.mean, self.std)]
            for z in self.base.encode_values(observations, actions, rewards)
        ]


def fit_standardizer(base, episodes: Sequence[Episode]) -> Standardized:
    """Fit per-dimension mean/std over every history prefix of TRAIN episodes."""
    if not episodes:
        raise ValueError("no episodes to fit the standardizer")
    splits = {ep.split for ep in episodes}
    if splits != {"train"}:
        raise ValueError(f"standardizer must be fitted on the train split only, got {sorted(splits)}")
    rows = []
    for ep in episodes:
        rows.extend(base.encode_values(ep.observations, ep.actions, ep.rewards))
    d = len(rows[0])
    n = len(rows)
    mean = [sum(r[k] for r in rows) / n for k in range(d)]
    std = [math.sqrt(sum((r[k] - mean[k]) ** 2 for r in rows) / n) for k in range(d)]
    std = [s if s > 1e-12 else 1.0 for s in std]
    return Standardized(base, mean, std, "train")
