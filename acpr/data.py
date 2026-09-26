"""Actor-visible records and evaluator-only labels.

`History`, `Branch` and `Episode` contain only what an actor is allowed to
see: observations, actions, rewards and episode-end flags. Ground truth
(clue, corridor length, positions) lives in `EvaluatorLabel`, which the
collector returns in a separate mapping that training code never receives.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class History:
    observations: Tuple[int, ...]  # o_0 .. o_t
    actions: Tuple[int, ...]  # a_0 .. a_{t-1}
    rewards: Tuple[float, ...]  # r_1 .. r_t

    def __post_init__(self) -> None:
        if len(self.observations) != len(self.actions) + 1:
            raise ValueError("need len(observations) == len(actions) + 1")
        if len(self.rewards) != len(self.actions):
            raise ValueError("need len(rewards) == len(actions)")

    @staticmethod
    def start(obs: int) -> "History":
        return History((obs,), (), ())

    def extend(self, action: int, obs: int, reward: float) -> "History":
        return History(
            self.observations + (obs,),
            self.actions + (action,),
            self.rewards + (float(reward),),
        )

    @property
    def t(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class Branch:
    """Rollout from the main-trajectory state after observing o_t.

    actions[0] is the enumerated first action; later actions come from the
    continuation policy. terminated/truncated describe why the rollout ended
    before the horizon (neither is set if it reached the horizon).
    """

    t: int
    actions: Tuple[int, ...]
    observations: Tuple[int, ...]
    rewards: Tuple[float, ...]
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Episode:
    episode_id: str
    split: str
    observations: Tuple[int, ...]
    actions: Tuple[int, ...]
    rewards: Tuple[float, ...]
    terminated: bool
    truncated: bool  # environment timeout
    budget_truncated: bool  # collection stopped because the budget ran out
    branches: Tuple[Branch, ...]

    def history(self, t: int) -> History:
        return History(
            self.observations[: t + 1], self.actions[:t], self.rewards[:t]
        )

    def n_transitions(self) -> int:
        return len(self.actions) + sum(len(b.actions) for b in self.branches)


ACTOR_VISIBLE_EPISODE_FIELDS = frozenset(f.name for f in dataclasses.fields(Episode))
ACTOR_VISIBLE_BRANCH_FIELDS = frozenset(f.name for f in dataclasses.fields(Branch))


@dataclass(frozen=True)
class EvaluatorLabel:
    """Privileged ground truth for one episode. Evaluator use only."""

    episode_id: str
    clue: int
    corridor_len: int
    optimal_arm: int


EvaluatorLabels = Dict[str, EvaluatorLabel]


def episode_to_json(ep: Episode) -> dict:
    return dataclasses.asdict(ep)


def dataset_hash(episodes: Iterable[Episode]) -> str:
    h = hashlib.sha256()
    for ep in episodes:
        h.update(json.dumps(episode_to_json(ep), sort_keys=True).encode())
    return h.hexdigest()
