"""Prediction windows, masked multi-step losses and training loops.

A window is a (history index t, k future actions, k targets, mask) tuple
taken either from the main trajectory or from a branch at t.

Horizon semantics:
- terminated before k steps: remaining targets are the absorbing
  OBS_TERMINAL_PAD symbol with reward 0 and ARE supervised (the episode end
  is actor-visible);
- truncated before k steps (env timeout, budget stop, or the end of a
  truncated main trajectory): remaining targets are masked out.
Padding actions after the last real action are drawn from an RNG seeded by
the episode id, so every arm sees exactly the same windows.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import autodiff as ad
from .data import Episode
from .env import N_ACTIONS, OBS_TERMINAL_PAD
from .models import LatentPredictor, RecurrentEncoder
from .splits import derive_seed


@dataclass(frozen=True)
class Window:
    t: int
    actions: Tuple[int, ...]
    observations: Tuple[int, ...]
    rewards: Tuple[float, ...]
    mask: Tuple[int, ...]
    source: str  # "main" or "branch"

    @property
    def n_valid(self) -> int:
        return sum(self.mask)


def _complete(
    t: int,
    actions: Sequence[int],
    observations: Sequence[int],
    rewards: Sequence[float],
    terminated: bool,
    horizon: int,
    pad_rng: random.Random,
    source: str,
) -> Window:
    n = len(actions)
    if not 1 <= n <= horizon or len(observations) != n or len(rewards) != n:
        raise ValueError("window pieces must have equal length in [1, horizon]")
    pad = horizon - n
    pad_actions = tuple(pad_rng.randrange(N_ACTIONS) for _ in range(pad))
    mask_tail = (1,) * pad if terminated else (0,) * pad
    return Window(
        t=t,
        actions=tuple(actions) + pad_actions,
        observations=tuple(observations) + (OBS_TERMINAL_PAD,) * pad,
        rewards=tuple(float(r) for r in rewards) + (0.0,) * pad,
        mask=(1,) * n + mask_tail,
        source=source,
    )


def episode_windows(ep: Episode, horizon: int, include_branches: bool = True) -> List[Window]:
    pad_rng = random.Random(derive_seed("pad", ep.episode_id, horizon))
    T = len(ep.actions)
    out = []
    for t in range(T):
        end = min(t + horizon, T)
        out.append(
            _complete(
                t,
                ep.actions[t:end],
                ep.observations[t + 1 : end + 1],
                ep.rewards[t:end],
                ep.terminated and end == T,
                horizon,
                pad_rng,
                "main",
            )
        )
    if include_branches:
        for b in ep.branches:
            if len(b.actions) > horizon:
                raise ValueError("branch longer than horizon")
            out.append(
                _complete(
                    b.t, b.actions, b.observations, b.rewards, b.terminated,
                    horizon, pad_rng, "branch",
                )
            )
    return out


def build_windows(episodes: Sequence[Episode], horizon: int) -> List[List[Window]]:
    return [episode_windows(ep, horizon) for ep in episodes]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float
    batch_episodes: int
    seed: int
    obs_weight: float = 1.0
    reward_weight: float = 1.0
    clip_norm: Optional[float] = 5.0


def _window_terms(
    predictor: LatentPredictor, z: ad.Node, w: Window, cfg: TrainConfig, norm: float
) -> List[ad.Node]:
    terms = []
    for j, (logits, reward) in enumerate(predictor.rollout(z, w.actions)):
        if not w.mask[j]:
            continue
        if cfg.obs_weight:
            terms.append(ad.softmax_xent(logits, w.observations[j], cfg.obs_weight / norm))
        if cfg.reward_weight:
            terms.append(ad.sq_err(reward, 0, w.rewards[j], cfg.reward_weight / norm))
    return terms


def _z_nodes(encoder: RecurrentEncoder, ep: Episode, frozen: bool) -> List[ad.Node]:
    if frozen:
        return [ad.const(v) for v in encoder.encode_values(ep.observations, ep.actions, ep.rewards)]
    return encoder.forward(ep.observations, ep.actions, ep.rewards)


def train_predictor(
    encoder: RecurrentEncoder,
    predictor: LatentPredictor,
    episodes: Sequence[Episode],
    windows: Sequence[Sequence[Window]],
    cfg: TrainConfig,
    train_encoder: bool,
    log: Optional[Callable[[dict], None]] = None,
    tag: str = "",
) -> dict:
    """Stage 1 (train_encoder=True) or frozen-encoder stage 2 (False)."""
    params = predictor.params() + (encoder.params() if train_encoder else [])
    opt = ad.Adam(params, cfg.lr, clip_norm=cfg.clip_norm)
    rng = random.Random(cfg.seed)
    order = list(range(len(episodes)))
    frozen_cache: Dict[int, List[ad.Node]] = {}
    history = []
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_terms = 0
        for start in range(0, len(order), cfg.batch_episodes):
            batch = order[start : start + cfg.batch_episodes]
            n_valid = sum(w.n_valid for i in batch for w in windows[i])
            if n_valid == 0:
                continue
            opt.zero_grad()
            for i in batch:
                if train_encoder:
                    zs = _z_nodes(encoder, episodes[i], frozen=False)
                else:
                    if i not in frozen_cache:
                        frozen_cache[i] = _z_nodes(encoder, episodes[i], frozen=True)
                    zs = frozen_cache[i]
                terms = []
                for w in windows[i]:
                    terms.extend(_window_terms(predictor, zs[w.t], w, cfg, n_valid))
                if terms:
                    loss = ad.sum_scalars(terms)
                    ad.backward(loss)
                    epoch_loss += loss.value[0] * n_valid
            epoch_terms += n_valid
            opt.step()
        rec = {
            "event": "epoch",
            "tag": tag,
            "epoch": epoch,
            "train_loss_per_target": epoch_loss / max(epoch_terms, 1),
            "elapsed_s": time.perf_counter() - t0,
        }
        history.append(rec)
        if log:
            log(rec)
    return {"epochs": history, "wall_s": time.perf_counter() - t0}


def evaluate_predictor(
    encoder: RecurrentEncoder,
    predictor: LatentPredictor,
    episodes: Sequence[Episode],
    windows: Sequence[Sequence[Window]],
) -> dict:
    """Mean observation NLL and reward squared error per supervised target."""
    nll = 0.0
    sq = 0.0
    n = 0
    for ep, ws in zip(episodes, windows):
        zs = _z_nodes(encoder, ep, frozen=True)
        for w in ws:
            for j, (logits, reward) in enumerate(predictor.rollout(zs[w.t], w.actions)):
                if not w.mask[j]:
                    continue
                logp = ad.log_softmax_values(logits.value)
                nll -= logp[w.observations[j]]
                sq += (reward.value[0] - w.rewards[j]) ** 2
                n += 1
    return {"obs_nll": nll / max(n, 1), "reward_mse": sq / max(n, 1), "n_targets": n}
