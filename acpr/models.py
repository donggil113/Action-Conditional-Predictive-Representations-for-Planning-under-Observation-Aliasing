"""Recurrent history encoder and latent multi-step predictors.

Encoder (Elman RNN, tanh):
    x_t = [onehot(o_t); onehot(a_{t-1}) or 0 at t=0; r_t or 0 at t=0]
    z_t = tanh(W_e [x_t; z_{t-1}] + b_e),  z_{-1} = 0

Predictor (latent rollout, same form as a small latent dynamics model):
    u_0 = z_t
    u_{j+1} = tanh(W_d [u_j; c_j] + b_d)
    obs logits_{j+1} = W_o u_{j+1} + b_o,  reward_{j+1} = W_r u_{j+1} + b_r

Action-conditional (ACP): c_j = onehot(a_{t+j}).
Action-marginal multi-step baseline (MSP): c_j = 0. Parameter count is
identical; the action columns of W_d simply receive no signal in MSP, which
`effective_param_count` reports.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import List, Sequence, Tuple

from . import autodiff as ad


def onehot(index: int, n: int) -> List[float]:
    v = [0.0] * n
    v[index] = 1.0
    return v


def _scale(fan_in: int) -> float:
    return 1.0 / math.sqrt(fan_in)


class RecurrentEncoder:
    def __init__(self, n_obs: int, n_act: int, hidden: int, rng: random.Random):
        self.n_obs = n_obs
        self.n_act = n_act
        self.hidden = hidden
        self.in_dim = n_obs + n_act + 1
        fan_in = self.in_dim + hidden
        self.W = ad.Param("enc.W", hidden, fan_in, rng, _scale(fan_in))
        self.b = ad.Param("enc.b", hidden, 1, rng, 0.0)

    def params(self) -> List[ad.Param]:
        return [self.W, self.b]

    def input_vector(self, obs: int, prev_action, reward: float) -> List[float]:
        a = [0.0] * self.n_act if prev_action is None else onehot(prev_action, self.n_act)
        return onehot(obs, self.n_obs) + a + [float(reward)]

    def forward(
        self,
        observations: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
    ) -> List[ad.Node]:
        """Returns z_0..z_T. z_t depends only on o_0..o_t, a_0..a_{t-1}, r_1..r_t."""
        if len(observations) != len(actions) + 1 or len(rewards) != len(actions):
            raise ValueError("inconsistent history lengths")
        z = ad.const([0.0] * self.hidden)
        out = []
        for t, obs in enumerate(observations):
            prev_a = actions[t - 1] if t > 0 else None
            r = rewards[t - 1] if t > 0 else 0.0
            x = ad.const(self.input_vector(obs, prev_a, r))
            z = ad.tanh(ad.linear(self.W, ad.concat([x, z]), self.b))
            out.append(z)
        return out

    def encode_values(self, observations, actions, rewards) -> List[List[float]]:
        return [z.value for z in self.forward(observations, actions, rewards)]


class LatentPredictor:
    def __init__(
        self,
        name: str,
        hidden: int,
        n_obs: int,
        n_act: int,
        action_conditional: bool,
        rng: random.Random,
    ):
        self.name = name
        self.hidden = hidden
        self.n_obs = n_obs
        self.n_act = n_act
        self.action_conditional = action_conditional
        fan_in = hidden + n_act
        self.Wd = ad.Param(f"{name}.Wd", hidden, fan_in, rng, _scale(fan_in))
        self.bd = ad.Param(f"{name}.bd", hidden, 1, rng, 0.0)
        self.Wo = ad.Param(f"{name}.Wo", n_obs, hidden, rng, _scale(hidden))
        self.bo = ad.Param(f"{name}.bo", n_obs, 1, rng, 0.0)
        self.Wr = ad.Param(f"{name}.Wr", 1, hidden, rng, _scale(hidden))
        self.br = ad.Param(f"{name}.br", 1, 1, rng, 0.0)

    def params(self) -> List[ad.Param]:
        return [self.Wd, self.bd, self.Wo, self.bo, self.Wr, self.br]

    def action_input(self, action: int) -> List[float]:
        if self.action_conditional:
            return onehot(action, self.n_act)
        return [0.0] * self.n_act

    def rollout(self, z: ad.Node, actions: Sequence[int]) -> List[Tuple[ad.Node, ad.Node]]:
        """Returns [(obs_logits_{t+j+1}, reward_{t+j+1}) for j in range(len(actions))]."""
        u = z
        out = []
        for a in actions:
            c = ad.const(self.action_input(a))
            u = ad.tanh(ad.linear(self.Wd, ad.concat([u, c]), self.bd))
            out.append((ad.linear(self.Wo, u, self.bo), ad.linear(self.Wr, u, self.br)))
        return out

    def predict_rewards(self, z_values: Sequence[float], actions: Sequence[int]) -> List[float]:
        return [r.value[0] for _, r in self.rollout(ad.const(z_values), actions)]


def copy_predictor(src: LatentPredictor) -> LatentPredictor:
    """Independent copy of a predictor's weights (for epoch checkpoints)."""
    dst = LatentPredictor(src.name, src.hidden, src.n_obs, src.n_act, src.action_conditional, random.Random(0))
    for d, s in zip(dst.params(), src.params()):
        d.w = list(s.w)
    return dst


def param_count(params: Sequence[ad.Param]) -> int:
    return sum(p.size for p in params)


def effective_param_count(predictor: LatentPredictor) -> int:
    """Parameters that can receive gradient. MSP's action columns of Wd cannot."""
    total = param_count(predictor.params())
    if predictor.action_conditional:
        return total
    return total - predictor.Wd.rows * predictor.n_act


def params_hash(params: Sequence[ad.Param]) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.name.encode())
        h.update(struct.pack(f"<{p.size}d", *p.w))
    return h.hexdigest()
