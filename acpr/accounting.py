"""Environment transition accounting.

Every simulator transition -- main trajectory, branch rollout or evaluation
step -- goes through `MeteredSimulator.step`, which charges the ledger before
the transition executes. The cap is hard: a step that would exceed it raises
`BudgetExceeded` and is not executed. Snapshots and restores are not
transitions, but they are a simulator privilege that a real environment does
not offer, so they are counted separately and can be forbidden outright.
"""

from __future__ import annotations

from typing import Optional

from .env import AliasedTMaze, GroundTruthState

TRANSITION_KINDS = ("main", "branch", "eval")


class BudgetExceeded(RuntimeError):
    pass


class PrivilegeError(RuntimeError):
    pass


class TransitionLedger:
    def __init__(self, cap: Optional[int], label: str, allow_restore: bool = True):
        if cap is not None and cap < 0:
            raise ValueError("cap must be non-negative")
        self.cap = cap
        self.label = label
        self.allow_restore = allow_restore
        self.steps = 0
        self.by_kind = {k: 0 for k in TRANSITION_KINDS}
        self.resets = 0
        self.snapshots = 0
        self.restores = 0

    def remaining(self) -> Optional[int]:
        return None if self.cap is None else self.cap - self.steps

    def charge(self, kind: str) -> None:
        if kind not in self.by_kind:
            raise ValueError(f"unknown transition kind {kind!r}")
        if self.cap is not None and self.steps + 1 > self.cap:
            raise BudgetExceeded(
                f"ledger {self.label!r}: step {self.steps + 1} exceeds cap {self.cap}"
            )
        self.steps += 1
        self.by_kind[kind] += 1

    def summary(self) -> dict:
        return {
            "label": self.label,
            "cap": self.cap,
            "transitions_total": self.steps,
            "transitions_by_kind": dict(self.by_kind),
            "resets": self.resets,
            "snapshots": self.snapshots,
            "restores": self.restores,
            "restore_allowed": self.allow_restore,
        }


class MeteredSimulator:
    """The only simulator handle collectors and evaluators receive."""

    def __init__(self, sim: AliasedTMaze, ledger: TransitionLedger):
        self._sim = sim
        self.ledger = ledger

    @property
    def config(self):
        return self._sim.config

    def reset(self, seed: int, clue: Optional[int] = None) -> int:
        self.ledger.resets += 1
        return self._sim.reset(seed, clue=clue)

    def step(self, action: int, kind: str):
        self.ledger.charge(kind)
        return self._sim.step(action)

    def snapshot(self) -> GroundTruthState:
        if not self.ledger.allow_restore:
            raise PrivilegeError(f"ledger {self.ledger.label!r} forbids snapshots")
        self.ledger.snapshots += 1
        return self._sim.get_state()

    def restore(self, state: GroundTruthState) -> None:
        if not self.ledger.allow_restore:
            raise PrivilegeError(f"ledger {self.ledger.label!r} forbids restores")
        self.ledger.restores += 1
        self._sim.set_state(state)

    def evaluator_ground_truth(self) -> dict:
        """Ground truth for evaluator labels. Never passed to an actor."""
        return self._sim.ground_truth()
