"""Aliased T-maze: a minimal POMDP with observation aliasing.

An episode starts at a clue cell whose observation reveals a hidden binary
clue. The agent then walks a corridor whose observations are clue-independent
distractor symbols and reaches a junction whose observation is identical for
both clues. The consequence of LEFT/RIGHT at the junction depends on the clue,
so two histories with different past clues can share the same current
observation while future action outcomes differ.

`AliasedTMaze` is the privileged simulator. Its ground-truth state (clue,
position, RNG state) is only meant for collectors and evaluators. Actors see
observations, actions and rewards through `acpr.data.History`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

# Observation vocabulary. Distractor corridor symbols follow N_FIXED_OBS.
OBS_CLUE_A = 0
OBS_CLUE_B = 1
OBS_JUNCTION = 2
OBS_GOAL = 3
OBS_WRONG = 4
OBS_TRAP = 5
# Synthetic absorbing symbol used only as a post-termination prediction
# target. The simulator never emits it.
OBS_TERMINAL_PAD = 6
N_FIXED_OBS = 7

ACT_FORWARD = 0
ACT_LEFT = 1
ACT_RIGHT = 2
N_ACTIONS = 3
ACTION_NAMES = ("FORWARD", "LEFT", "RIGHT")


@dataclass(frozen=True)
class MazeConfig:
    """Static environment parameters.

    symmetric=True: the correct arm yields GOAL (+1) and the other arm yields
    WRONG (-1). Under any clue-independent action distribution the marginal
    distribution of future observations is then clue-independent.

    symmetric=False (control variant): LEFT yields GOAL (+1) under clue A and
    TRAP (-1) under clue B; RIGHT always yields WRONG (0). Action-marginal
    futures then do depend on the clue.
    """

    corridor_min: int = 2
    corridor_max: int = 4
    n_distractors: int = 3
    symmetric: bool = True
    timeout_slack: int = 6

    def __post_init__(self) -> None:
        if not 1 <= self.corridor_min <= self.corridor_max:
            raise ValueError("need 1 <= corridor_min <= corridor_max")
        if self.n_distractors < 1:
            raise ValueError("need at least one distractor symbol")
        if self.timeout_slack < 1:
            raise ValueError("timeout_slack must be >= 1")

    @property
    def n_obs(self) -> int:
        return N_FIXED_OBS + self.n_distractors

    def max_steps(self, corridor_len: int) -> int:
        # clue cell -> corridor_len cells -> junction -> arm, plus slack.
        return corridor_len + 2 + self.timeout_slack


@dataclass(frozen=True)
class GroundTruthState:
    """Complete simulator state. Privileged: collector/evaluator only."""

    clue: int
    corridor_len: int
    pos: int  # 0 clue cell, 1..L corridor, L+1 junction, L+2 terminal arm
    t: int
    done: bool
    rng_state: tuple


class AliasedTMaze:
    """Privileged simulator with save/restore for branching."""

    def __init__(self, config: MazeConfig):
        self.config = config
        self._rng = random.Random(0)
        self._clue = 0
        self._corridor_len = config.corridor_min
        self._pos = 0
        self._t = 0
        self._done = True
        self._max_steps = config.max_steps(self._corridor_len)

    # ---- privileged state access (collector / evaluator only) ----------
    def get_state(self) -> GroundTruthState:
        return GroundTruthState(
            clue=self._clue,
            corridor_len=self._corridor_len,
            pos=self._pos,
            t=self._t,
            done=self._done,
            rng_state=self._rng.getstate(),
        )

    def set_state(self, state: GroundTruthState) -> None:
        self._clue = state.clue
        self._corridor_len = state.corridor_len
        self._pos = state.pos
        self._t = state.t
        self._done = state.done
        self._rng.setstate(state.rng_state)
        self._max_steps = self.config.max_steps(self._corridor_len)

    def ground_truth(self) -> dict:
        return {
            "clue": self._clue,
            "corridor_len": self._corridor_len,
            "pos": self._pos,
            "t": self._t,
            "optimal_arm": self.optimal_arm(self._clue),
        }

    def optimal_arm(self, clue: int) -> int:
        if self.config.symmetric:
            return ACT_LEFT if clue == 0 else ACT_RIGHT
        # Asymmetric: LEFT gives +1 under clue A, -1 under clue B; RIGHT gives 0.
        return ACT_LEFT if clue == 0 else ACT_RIGHT

    # ---- dynamics ------------------------------------------------------
    @property
    def junction_pos(self) -> int:
        return self._corridor_len + 1

    def reset(self, seed: int, clue: Optional[int] = None) -> int:
        """Start an episode. `clue` may be forced by tests/evaluators."""
        self._rng = random.Random(seed)
        sampled_clue = self._rng.randrange(2)
        self._clue = sampled_clue if clue is None else int(clue)
        self._corridor_len = self._rng.randint(
            self.config.corridor_min, self.config.corridor_max
        )
        self._max_steps = self.config.max_steps(self._corridor_len)
        self._pos = 0
        self._t = 0
        self._done = False
        return self._emit()

    def _emit(self) -> int:
        if self._pos == 0:
            return OBS_CLUE_A if self._clue == 0 else OBS_CLUE_B
        if self._pos <= self._corridor_len:
            return N_FIXED_OBS + self._rng.randrange(self.config.n_distractors)
        if self._pos == self.junction_pos:
            return OBS_JUNCTION
        raise AssertionError("no observation is emitted from a terminal arm")

    def _arm_outcome(self, action: int) -> Tuple[int, float]:
        if self.config.symmetric:
            if action == self.optimal_arm(self._clue):
                return OBS_GOAL, 1.0
            return OBS_WRONG, -1.0
        if action == ACT_LEFT:
            return (OBS_GOAL, 1.0) if self._clue == 0 else (OBS_TRAP, -1.0)
        return OBS_WRONG, 0.0

    def step(self, action: int) -> Tuple[int, float, bool, bool]:
        """Returns (observation, reward, terminated, truncated)."""
        if self._done:
            raise RuntimeError("step() called on a finished episode")
        if action not in (ACT_FORWARD, ACT_LEFT, ACT_RIGHT):
            raise ValueError(f"invalid action {action!r}")
        self._t += 1
        if self._pos == self.junction_pos and action != ACT_FORWARD:
            obs, reward = self._arm_outcome(action)
            self._pos = self.junction_pos + 1
            self._done = True
            return obs, reward, True, False
        if action == ACT_FORWARD and self._pos < self.junction_pos:
            self._pos += 1
        obs = self._emit()
        truncated = self._t >= self._max_steps
        self._done = truncated
        return obs, 0.0, False, truncated
