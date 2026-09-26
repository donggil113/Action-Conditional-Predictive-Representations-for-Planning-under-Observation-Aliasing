"""Minimal vector-level reverse-mode autodiff in pure Python.

Only the operations needed by the recurrent encoder and latent predictor are
implemented. Parameters are dense matrices stored row-major in flat lists;
their gradients accumulate into `Param.g` during `backward`. Nodes built from
constants carry `needs_grad=False`, which is how the frozen-encoder protocol
stops gradients.

`COUNTERS["mac"]` counts multiply-accumulates in `linear` (forward and
backward) so that runs can report actual arithmetic cost, not just caps.
"""

from __future__ import annotations

import math
import random
from operator import mul
from typing import Iterable, List, Optional, Sequence

COUNTERS = {"mac": 0}


class Param:
    __slots__ = ("name", "rows", "cols", "w", "g")

    def __init__(self, name: str, rows: int, cols: int, rng: random.Random, scale: float):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.w = [rng.uniform(-scale, scale) for _ in range(rows * cols)]
        self.g = [0.0] * (rows * cols)

    @property
    def size(self) -> int:
        return self.rows * self.cols

    def zero_grad(self) -> None:
        self.g = [0.0] * (self.rows * self.cols)


class Node:
    __slots__ = ("value", "grad", "parents", "backward_fn", "needs_grad")

    def __init__(self, value: List[float], parents=(), backward_fn=None, needs_grad=False):
        self.value = value
        self.grad: Optional[List[float]] = None
        self.parents = parents
        self.backward_fn = backward_fn
        self.needs_grad = needs_grad


def _acc(node: Node, g: Sequence[float]) -> None:
    if not node.needs_grad:
        return
    if node.grad is None:
        node.grad = list(g)
    else:
        node.grad = [a + b for a, b in zip(node.grad, g)]


def const(values: Iterable[float]) -> Node:
    return Node([float(v) for v in values])


def linear(W: Param, x: Node, b: Optional[Param] = None) -> Node:
    xv = x.value
    cols = W.cols
    if len(xv) != cols:
        raise ValueError(f"{W.name}: expected input of size {cols}, got {len(xv)}")
    w = W.w
    out = [sum(map(mul, w[i * cols : (i + 1) * cols], xv)) for i in range(W.rows)]
    if b is not None:
        out = [o + bi for o, bi in zip(out, b.w)]
    COUNTERS["mac"] += W.rows * cols

    def backward_fn(gy: List[float]) -> None:
        wg = W.g
        gx = [0.0] * cols if x.needs_grad else None
        for i, g in enumerate(gy):
            if g == 0.0:
                continue
            base = i * cols
            wg[base : base + cols] = [a + g * xj for a, xj in zip(wg[base : base + cols], xv)]
            if gx is not None:
                gx = [a + g * wij for a, wij in zip(gx, w[base : base + cols])]
        COUNTERS["mac"] += W.rows * cols * (2 if gx is not None else 1)
        if b is not None:
            b.g = [a + g for a, g in zip(b.g, gy)]
        if gx is not None:
            _acc(x, gx)

    return Node(out, (x,), backward_fn, True)


def add(a: Node, b: Node) -> Node:
    if len(a.value) != len(b.value):
        raise ValueError("add: size mismatch")
    out = [x + y for x, y in zip(a.value, b.value)]

    def backward_fn(g):
        _acc(a, g)
        _acc(b, g)

    return Node(out, (a, b), backward_fn, a.needs_grad or b.needs_grad)


def tanh(a: Node) -> Node:
    out = [math.tanh(v) for v in a.value]

    def backward_fn(g):
        _acc(a, [gi * (1.0 - o * o) for gi, o in zip(g, out)])

    return Node(out, (a,), backward_fn, a.needs_grad)


def concat(nodes: Sequence[Node]) -> Node:
    out: List[float] = []
    sizes = []
    for n in nodes:
        out.extend(n.value)
        sizes.append(len(n.value))

    def backward_fn(g):
        start = 0
        for n, s in zip(nodes, sizes):
            _acc(n, g[start : start + s])
            start += s

    return Node(out, tuple(nodes), backward_fn, any(n.needs_grad for n in nodes))


def log_softmax_values(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    lse = m + math.log(sum(math.exp(v - m) for v in logits))
    return [v - lse for v in logits]


def softmax_xent(logits: Node, target: int, weight: float = 1.0) -> Node:
    """weight * -log softmax(logits)[target] as a scalar node."""
    logp = log_softmax_values(logits.value)
    probs = [math.exp(v) for v in logp]
    out = [-weight * logp[target]]

    def backward_fn(g):
        gg = g[0] * weight
        grad = [gg * p for p in probs]
        grad[target] -= gg
        _acc(logits, grad)

    return Node(out, (logits,), backward_fn, logits.needs_grad)


def sq_err(pred: Node, index: int, target: float, weight: float = 1.0) -> Node:
    """weight * (pred[index] - target)^2 as a scalar node."""
    diff = pred.value[index] - target
    out = [weight * diff * diff]

    def backward_fn(g):
        grad = [0.0] * len(pred.value)
        grad[index] = g[0] * weight * 2.0 * diff
        _acc(pred, grad)

    return Node(out, (pred,), backward_fn, pred.needs_grad)


def sum_scalars(nodes: Sequence[Node]) -> Node:
    out = [sum(n.value[0] for n in nodes)]

    def backward_fn(g):
        for n in nodes:
            _acc(n, g)

    return Node(out, tuple(nodes), backward_fn, any(n.needs_grad for n in nodes))


def backward(loss: Node) -> None:
    if len(loss.value) != 1:
        raise ValueError("backward() expects a scalar node")
    topo: List[Node] = []
    seen = set()
    stack = [(loss, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            topo.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for p in node.parents:
            if id(p) not in seen:
                stack.append((p, False))
    loss.grad = [1.0]
    for node in reversed(topo):
        if node.backward_fn is not None and node.grad is not None and node.needs_grad:
            node.backward_fn(node.grad)


class Adam:
    def __init__(self, params: Sequence[Param], lr: float, betas=(0.9, 0.999), eps=1e-8, clip_norm: Optional[float] = 5.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.clip_norm = clip_norm
        self.t = 0
        self.m = [[0.0] * p.size for p in self.params]
        self.v = [[0.0] * p.size for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def grad_norm(self) -> float:
        return math.sqrt(sum(g * g for p in self.params for g in p.g))

    def step(self) -> float:
        norm = self.grad_norm()
        scale = 1.0
        if self.clip_norm is not None and norm > self.clip_norm:
            scale = self.clip_norm / (norm + 1e-12)
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            w = p.w
            for i, g in enumerate(p.g):
                g *= scale
                m[i] = self.b1 * m[i] + (1.0 - self.b1) * g
                v[i] = self.b2 * v[i] + (1.0 - self.b2) * g * g
                w[i] -= self.lr * (m[i] / c1) / (math.sqrt(v[i] / c2) + self.eps)
        return norm
