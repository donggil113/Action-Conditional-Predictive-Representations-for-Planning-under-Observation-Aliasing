"""Evaluator: frozen-representation planning and a ground-truth clue probe.

Protocol (identical for every arm):
1. A scripted, observation-only prefix presses FORWARD until the junction
   observation appears (same actions for every arm).
2. From then on an exhaustive model-predictive planner uses only the frozen
   encoder and the stage-2 reward head: it scores every action sequence of
   length `plan_horizon` by discounted predicted reward and executes the
   first action of the best sequence (ties broken by an RNG shared across
   arms). No simulator snapshot/restore is allowed: the evaluation ledger
   forbids it, so the planner cannot perform hidden rollouts.
3. Success = the episode terminated in the optimal arm. Timeout = failure.
   Forced-commit accuracy = whether the same planner, restricted to
   first actions {LEFT, RIGHT}, prefers the optimal arm at the first
   junction arrival (computed from the model only; no extra env steps).

Evaluation episodes are clue-stratified by the evaluator (clue = index % 2)
so that every constant policy scores exactly 0.5; the actor never sees this.

Ground-truth labels (clue, optimal arm) are read by the evaluator after the
fact and never passed to the planner. The clue probe is an evaluator-only
diagnostic fitted on the calibration split.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .accounting import MeteredSimulator, TransitionLedger
from .data import History
from .env import (
    ACT_FORWARD,
    ACT_LEFT,
    ACT_RIGHT,
    N_ACTIONS,
    OBS_JUNCTION,
    AliasedTMaze,
    MazeConfig,
)
from .models import LatentPredictor, RecurrentEncoder
from .splits import derive_seed, episode_seed


@dataclass(frozen=True)
class EvalConfig:
    n_calibration: int
    n_test: int
    plan_horizon: int
    gamma: float = 0.9
    probe_steps: int = 300
    probe_lr: float = 0.5
    probe_l2: float = 1e-2
    tie_tol: float = 1e-9
    stratify_clue: bool = True


def _eval_clue(stratify: bool, index: int):
    return index % 2 if stratify else None


def _eval_ledger(maze_cfg: MazeConfig, n_episodes: int, label: str) -> TransitionLedger:
    cap = n_episodes * maze_cfg.max_steps(maze_cfg.corridor_max)
    return TransitionLedger(cap=cap, label=label, allow_restore=False)


def _run_prefix(sim: MeteredSimulator, history: History) -> Tuple[History, bool]:
    """Scripted FORWARD until the junction observation. Returns (history, done)."""
    while history.observations[-1] != OBS_JUNCTION:
        obs, r, term, trunc = sim.step(ACT_FORWARD, kind="eval")
        history = history.extend(ACT_FORWARD, obs, r)
        if term or trunc:
            return history, True
    return history, False


def sequence_scores(
    encoder: RecurrentEncoder,
    head: LatentPredictor,
    history: History,
    horizon: int,
    gamma: float,
    first_actions: Sequence[int] = tuple(range(N_ACTIONS)),
) -> List[Tuple[Tuple[int, ...], float]]:
    z = encoder.encode_values(history.observations, history.actions, history.rewards)[-1]
    out = []
    for first in first_actions:
        for rest in itertools.product(range(N_ACTIONS), repeat=horizon - 1):
            seq = (first,) + rest
            rewards = head.predict_rewards(z, seq)
            out.append((seq, sum(gamma**j * r for j, r in enumerate(rewards))))
    return out


def plan_action(
    encoder, head, history: History, horizon: int, gamma: float, rng: random.Random,
    first_actions: Sequence[int] = tuple(range(N_ACTIONS)), tie_tol: float = 1e-9,
) -> int:
    scored = sequence_scores(encoder, head, history, horizon, gamma, first_actions)
    best = max(s for _, s in scored)
    candidates = sorted({seq[0] for seq, s in scored if s >= best - tie_tol})
    return candidates[0] if len(candidates) == 1 else rng.choice(candidates)


def collect_probe_histories(
    maze_cfg: MazeConfig, run_seed: int, split: str, n_episodes: int, stratify_clue: bool = True
) -> Tuple[List[History], List[int], dict]:
    """Histories at the first junction arrival (scripted prefix) + evaluator clue labels."""
    ledger = _eval_ledger(maze_cfg, n_episodes, f"{split}:probe")
    sim = MeteredSimulator(AliasedTMaze(maze_cfg), ledger)
    histories, clues = [], []
    for i in range(n_episodes):
        history = History.start(sim.reset(episode_seed(run_seed, split, i), clue=_eval_clue(stratify_clue, i)))
        clue = sim.evaluator_ground_truth()["clue"]
        history, done = _run_prefix(sim, history)
        if not done:
            histories.append(history)
            clues.append(clue)
    return histories, clues, ledger.summary()


def planning_eval(
    encoder: RecurrentEncoder,
    head: LatentPredictor,
    maze_cfg: MazeConfig,
    run_seed: int,
    cfg: EvalConfig,
    split: str = "test",
) -> dict:
    ledger = _eval_ledger(maze_cfg, cfg.n_test, f"{split}:planning")
    sim = MeteredSimulator(AliasedTMaze(maze_cfg), ledger)
    successes, returns, forced_correct = [], [], []
    forced_expected, forced_ties = [], 0
    junction_histories, junction_clues = [], []
    first_decisions = {name: 0 for name in ("FORWARD", "LEFT", "RIGHT")}
    timeouts = 0
    for i in range(cfg.n_test):
        # Tie-breaking RNGs depend only on (run seed, episode): shared across arms.
        rng = random.Random(derive_seed("plan-ties", run_seed, split, i))
        forced_rng = random.Random(derive_seed("forced-ties", run_seed, split, i))
        history = History.start(sim.reset(episode_seed(run_seed, split, i), clue=_eval_clue(cfg.stratify_clue, i)))
        gt = sim.evaluator_ground_truth()  # evaluator-side only
        history, done = _run_prefix(sim, history)
        terminated = False
        if not done:
            junction_histories.append(history)
            junction_clues.append(gt["clue"])
            forced = plan_action(
                encoder, head, history, cfg.plan_horizon, cfg.gamma, forced_rng,
                first_actions=(ACT_LEFT, ACT_RIGHT), tie_tol=cfg.tie_tol,
            )
            forced_correct.append(float(forced == gt["optimal_arm"]))
            # Diagnostic only: expected credit under uniform tie-breaking, so an
            # exact LEFT/RIGHT tie scores 0.5 independent of the tie RNG.
            scored = sequence_scores(encoder, head, history, cfg.plan_horizon, cfg.gamma, (ACT_LEFT, ACT_RIGHT))
            best = max(sc for _, sc in scored)
            cands = {seq[0] for seq, sc in scored if sc >= best - cfg.tie_tol}
            forced_ties += int(len(cands) > 1)
            forced_expected.append((gt["optimal_arm"] in cands) / len(cands))
            first = True
            while not done:
                a = plan_action(encoder, head, history, cfg.plan_horizon, cfg.gamma, rng, tie_tol=cfg.tie_tol)
                if first:
                    first_decisions[("FORWARD", "LEFT", "RIGHT")[a]] += 1
                    first = False
                obs, r, terminated, truncated = sim.step(a, kind="eval")
                history = history.extend(a, obs, r)
                done = terminated or truncated
        success = terminated and history.actions[-1] == gt["optimal_arm"]
        timeouts += int(not terminated)
        successes.append(float(success))
        returns.append(sum(history.rewards))
    n = max(len(successes), 1)
    return {
        "success_rate": sum(successes) / n,
        "mean_return": sum(returns) / n,
        "timeout_rate": timeouts / n,
        "forced_commit_accuracy": (sum(forced_correct) / len(forced_correct)) if forced_correct else None,
        "forced_commit_expected_accuracy": (sum(forced_expected) / len(forced_expected)) if forced_expected else None,
        "forced_commit_ties": forced_ties,
        "first_planner_decisions": first_decisions,
        "n_episodes": len(successes),
        "ledger": ledger.summary(),
        "_junction_histories": junction_histories,
        "_junction_clues": junction_clues,
    }


# ---- evaluator-only clue probe (logistic regression, pure Python) --------
def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def standardizer(features: Sequence[Sequence[float]]):
    """Per-dimension z-scoring fitted on calibration features only."""
    d = len(features[0])
    n = len(features)
    mean = [sum(x[k] for x in features) / n for k in range(d)]
    std = [math.sqrt(sum((x[k] - mean[k]) ** 2 for x in features) / n) for k in range(d)]
    std = [s if s > 1e-12 else 1.0 for s in std]
    return lambda xs: [[(x[k] - mean[k]) / std[k] for k in range(d)] for x in xs]


def fit_probe(features: Sequence[Sequence[float]], labels: Sequence[int], cfg: EvalConfig):
    d = len(features[0])
    w = [0.0] * d
    b = 0.0
    n = len(features)
    for _ in range(cfg.probe_steps):
        gw = [cfg.probe_l2 * wi for wi in w]
        gb = 0.0
        for x, y in zip(features, labels):
            p = _sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)
            err = (p - y) / n
            gw = [g + err * xi for g, xi in zip(gw, x)]
            gb += err
        w = [wi - cfg.probe_lr * g for wi, g in zip(w, gw)]
        b -= cfg.probe_lr * gb
    return w, b


def probe_accuracy(w, b, features, labels) -> Optional[float]:
    if not features:
        return None
    correct = 0
    for x, y in zip(features, labels):
        p = _sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)
        correct += int((p >= 0.5) == bool(y))
    return correct / len(features)


def clue_probe(
    encoder: RecurrentEncoder,
    calib_histories: Sequence[History],
    calib_clues: Sequence[int],
    test_histories: Sequence[History],
    test_clues: Sequence[int],
    cfg: EvalConfig,
) -> dict:
    def feats(hs):
        return [encoder.encode_values(h.observations, h.actions, h.rewards)[-1] for h in hs]

    if not calib_histories or len(set(calib_clues)) < 2:
        return {"status": "NOT_RUN", "reason": "calibration set lacks both clue classes"}
    calib = feats(calib_histories)
    scale = standardizer(calib)
    w, b = fit_probe(scale(calib), calib_clues, cfg)
    return {
        "status": "OK",
        "calibration_accuracy": probe_accuracy(w, b, scale(calib), calib_clues),
        "test_accuracy": probe_accuracy(w, b, scale(feats(test_histories)), test_clues),
        "n_calibration": len(calib_histories),
        "n_test": len(test_histories),
    }
