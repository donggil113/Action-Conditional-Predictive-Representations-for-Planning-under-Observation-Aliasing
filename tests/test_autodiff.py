"""Finite-difference checks of the pure-Python autodiff and full model loss.

These are implementation checks on specific numeric fixtures, not proofs.
"""

import random
import unittest

from acpr import autodiff as ad
from acpr.env import N_ACTIONS
from acpr.models import LatentPredictor, RecurrentEncoder
from acpr.train import TrainConfig, Window, _window_terms


def numeric_grad(loss_fn, param, idx, eps=1e-6):
    old = param.w[idx]
    param.w[idx] = old + eps
    up = loss_fn()
    param.w[idx] = old - eps
    down = loss_fn()
    param.w[idx] = old
    return (up - down) / (2 * eps)


class TestOps(unittest.TestCase):
    def test_small_graph_gradients(self):
        rng = random.Random(0)
        W1 = ad.Param("W1", 4, 5, rng, 0.8)
        b1 = ad.Param("b1", 4, 1, rng, 0.3)
        W2 = ad.Param("W2", 3, 6, rng, 0.8)
        W3 = ad.Param("W3", 1, 3, rng, 0.8)
        x = [0.5, -1.0, 0.25, 0.0, 2.0]
        extra = [0.7, -0.3]

        def build():
            h = ad.tanh(ad.linear(W1, ad.const(x), b1))
            h2 = ad.add(ad.linear(W2, ad.concat([h, ad.const(extra)])), ad.const([0.1, 0.2, 0.3]))
            r = ad.linear(W3, ad.tanh(h2))
            return ad.sum_scalars([ad.softmax_xent(h2, 2, 0.7), ad.sq_err(r, 0, 0.4, 1.3)])

        for p in (W1, b1, W2, W3):
            p.zero_grad()
        ad.backward(build())
        for p in (W1, b1, W2, W3):
            for idx in range(p.size):
                num = numeric_grad(lambda: build().value[0], p, idx)
                self.assertAlmostEqual(p.g[idx], num, places=6, msg=f"{p.name}[{idx}]")

    def test_shared_node_gradient_accumulates(self):
        rng = random.Random(1)
        W = ad.Param("W", 2, 2, rng, 1.0)

        def build():
            h = ad.tanh(ad.linear(W, ad.const([1.0, -0.5])))
            return ad.sum_scalars([ad.sq_err(h, 0, 0.3), ad.sq_err(h, 1, -0.2), ad.sq_err(ad.add(h, h), 0, 0.1)])

        W.zero_grad()
        ad.backward(build())
        for idx in range(W.size):
            self.assertAlmostEqual(W.g[idx], numeric_grad(lambda: build().value[0], W, idx), places=6)

    def test_adam_reduces_a_quadratic(self):
        rng = random.Random(2)
        W = ad.Param("W", 1, 3, rng, 1.0)
        opt = ad.Adam([W], lr=0.05)
        losses = []
        for _ in range(200):
            opt.zero_grad()
            loss = ad.sq_err(ad.linear(W, ad.const([1.0, 2.0, -1.0])), 0, 3.0)
            ad.backward(loss)
            opt.step()
            losses.append(loss.value[0])
        self.assertLess(losses[-1], 1e-3 * losses[0])


class TestModelGradient(unittest.TestCase):
    def _check(self, action_conditional):
        rng = random.Random(3)
        n_obs = 9
        enc = RecurrentEncoder(n_obs, N_ACTIONS, 4, rng)
        pred = LatentPredictor("p", 4, n_obs, N_ACTIONS, action_conditional, rng)
        obs, acts, rews = (0, 7, 8, 2), (0, 1, 0), (0.0, 0.0, 0.0)
        windows = [
            Window(1, (0, 2), (8, 2), (0.0, 0.0), (1, 1), "main"),
            Window(2, (1, 0), (3, 6), (1.0, 0.0), (1, 1), "branch"),
            Window(0, (2, 1), (7, 5), (0.0, 0.0), (1, 0), "branch"),
        ]
        cfg = TrainConfig(1, 0.01, 1, 0, obs_weight=1.0, reward_weight=0.5)

        def build():
            zs = enc.forward(obs, acts, rews)
            terms = []
            for w in windows:
                terms.extend(_window_terms(pred, zs[w.t], w, cfg, 5.0))
            return ad.sum_scalars(terms)

        params = enc.params() + pred.params()
        for p in params:
            p.zero_grad()
        ad.backward(build())
        check_rng = random.Random(4)
        for p in params:
            for idx in check_rng.sample(range(p.size), min(p.size, 12)):
                num = numeric_grad(lambda: build().value[0], p, idx)
                self.assertAlmostEqual(p.g[idx], num, places=6, msg=f"{p.name}[{idx}]")
        if not action_conditional:
            # Action columns of Wd get exactly zero gradient in the marginal baseline.
            cols = pred.Wd.cols
            for i in range(pred.Wd.rows):
                for j in range(pred.hidden, cols):
                    self.assertEqual(pred.Wd.g[i * cols + j], 0.0)

    def test_acp_model_gradient(self):
        self._check(True)

    def test_msp_model_gradient(self):
        self._check(False)


if __name__ == "__main__":
    unittest.main()
