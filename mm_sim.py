"""Vectorized simulation of quoting strategies, for research rather than play.

`mm_engine` resolves one order at a time for the interactive game. This module
runs the same model over tens of thousands of rounds at once with numpy, and
defines the bots the human is compared against:

    Static(h)            symmetric quote at fair +/- h, never updated
    Chaser(h, step)      moves its mid `step` per lot toward each fill's side
    BayesMid(h)          exact posterior over V; quotes posterior mean +/- h
    BayesMyopic()        exact posterior over V; picks each side's price to
                         maximize the expected P&L of the next fill. No
                         tuned parameters.

The posterior is computed on a grid. Every order outcome -- lift, hit or no
trade, and the fill size -- has a known likelihood as a function of V under the
model, so Bayes' rule is exact up to the grid spacing.

`test_mm.py` checks that `simulate` here reproduces `mm_engine.play_round`
exactly, round by round, so the game and the research share one model.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr  # standard normal CDF

import mm_engine as mm


# --------------------------------------------------------------------- batches

@dataclass
class Batch:
    """Many rounds of order flow as arrays. Shapes: values (n, A+1), rest (n, A)."""
    values: np.ndarray
    informed: np.ndarray
    z: np.ndarray
    size_a: np.ndarray
    size_b: np.ndarray

    @property
    def n(self):
        return self.values.shape[0]


def draw_batch(cfg, n, seed):
    rng = np.random.default_rng(seed)
    A, M = cfg.arrivals, max(cfg.max_size, 1)
    v0 = np.rint(rng.normal(cfg.fair, cfg.sigma_v, n))
    steps = rng.normal(0, cfg.sigma_step, (n, A)) if cfg.sigma_step > 0 else np.zeros((n, A))
    values = np.concatenate([v0[:, None], v0[:, None] + np.cumsum(steps, axis=1)], axis=1)
    return Batch(values=values,
                 informed=rng.random((n, A)) < cfg.p_informed,
                 z=rng.standard_normal((n, A)),
                 size_a=rng.integers(1, M + 1, (n, A)),
                 size_b=rng.integers(1, M + 1, (n, A)))


def from_scenarios(scenarios):
    """Convert mm_engine.Scenario objects (e.g. the rounds a human played)."""
    return Batch(
        values=np.array([s.values for s in scenarios], dtype=float),
        informed=np.array([[o.informed for o in s.orders] for s in scenarios]),
        z=np.array([[o.z for o in s.orders] for s in scenarios]),
        size_a=np.array([[o.size_a for o in s.orders] for s in scenarios]),
        size_b=np.array([[o.size_b for o in s.orders] for s in scenarios]),
    )


def size_pmfs(cfg):
    """P(size = s | informed) and P(size = s | noise), indexed by s (index 0 unused)."""
    M = max(cfg.max_size, 1)
    s = np.arange(M + 1, dtype=float)
    if M == 1:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    p_max = np.where(s >= 1, (2 * s - 1) / M**2, 0.0)
    p_min = np.where(s >= 1, (2 * (M - s) + 1) / M**2, 0.0)
    return p_max, p_min


# ------------------------------------------------------------------ strategies

class Static:
    name = "static"

    def __init__(self, h):
        self.h = h

    def reset(self, n, cfg):
        self.mid = np.full(n, cfg.fair)

    def quote(self, k, inv):
        return self.mid - self.h, self.mid + self.h

    def observe(self, bid, ask, sell, buy, size):
        pass


class Chaser(Static):
    """The intuitive heuristic: lifted -> mark up, hit -> mark down, by a fixed
    amount per lot. Uses no model of who is informed."""
    name = "chaser"

    def __init__(self, h, step):
        self.h, self.step = h, step

    def observe(self, bid, ask, sell, buy, size):
        self.mid = self.mid + self.step * size * (sell.astype(float) - buy)


class _Bayes:
    """Exact grid posterior over V, updated on every order outcome."""

    SPACING = 0.5

    def reset(self, n, cfg):
        self.cfg = cfg
        spread_v = np.sqrt(cfg.sigma_v**2 + cfg.arrivals * cfg.sigma_step**2)
        J = int(np.ceil(6 * spread_v / self.SPACING))
        self.grid = cfg.fair + self.SPACING * np.arange(-J, J + 1)
        # V0 = round(N(fair, sigma_v)): mass on integer grid points only.
        on_int = np.isclose(self.grid, np.rint(self.grid))
        prior = np.where(on_int,
                         ndtr((self.grid + 0.5 - cfg.fair) / cfg.sigma_v)
                         - ndtr((self.grid - 0.5 - cfg.fair) / cfg.sigma_v), 0.0)
        self.post = np.tile(prior / prior.sum(), (n, 1))
        self.p_max, self.p_min = size_pmfs(cfg)
        self.kernel = None
        if cfg.sigma_step > 0:  # transition V_k -> V_{k+1}, rows normalised
            d = self.grid[None, :] - self.grid[:, None]
            K = np.exp(-0.5 * (d / cfg.sigma_step) ** 2)
            self.kernel = K / K.sum(axis=1, keepdims=True)

    def mean_sd(self):
        m = self.post @ self.grid
        var = self.post @ self.grid**2 - m**2
        return m, np.sqrt(np.maximum(var, 0.0))

    def _lift_hit(self, bid, ask):
        """P(an informed / noise order lifts or hits), per grid point."""
        c = self.cfg
        inf_lift = ndtr((self.grid[None, :] - ask[:, None]) / c.sigma_signal)
        inf_hit = ndtr((bid[:, None] - self.grid[None, :]) / c.sigma_signal)
        noise_lift = ndtr((c.fair - ask) / c.sigma_noise)[:, None]
        noise_hit = ndtr((bid - c.fair) / c.sigma_noise)[:, None]
        return inf_lift, inf_hit, noise_lift, noise_hit

    def observe(self, bid, ask, sell, buy, size):
        p = self.cfg.p_informed
        inf_lift, inf_hit, noise_lift, noise_hit = self._lift_hit(bid, ask)
        pm, pn = self.p_max[size][:, None], self.p_min[size][:, None]
        L = np.where(sell[:, None], p * inf_lift * pm + (1 - p) * noise_lift * pn,
            np.where(buy[:, None], p * inf_hit * pm + (1 - p) * noise_hit * pn,
                     p * (1 - inf_lift - inf_hit) + (1 - p) * (1 - noise_lift - noise_hit)))
        post = self.post * L
        total = post.sum(axis=1, keepdims=True)
        # A likelihood that underflows to zero everywhere carries no usable
        # information; keep the previous posterior rather than divide by zero.
        self.post = np.where(total > 0, post / np.where(total > 0, total, 1.0), self.post)
        if self.kernel is not None:
            self.post = self.post @ self.kernel


class BayesMid(_Bayes):
    name = "bayes-mid"

    def __init__(self, h):
        self.h = h

    def quote(self, k, inv):
        m, _ = self.mean_sd()
        return m - self.h, m + self.h


class BayesMyopic(_Bayes):
    """For each side, the price that maximizes expected P&L from the next order:
        ask* = argmax_a  sum_V post(V) * P(lift at a | V) * E[size | lift, V] * (a - V)
    and symmetrically for the bid. Myopic: it ignores that a tighter quote also
    buys information about V for later orders."""
    name = "bayes-myopic"

    OFFSETS = np.arange(0.0, 20.01, 0.25)

    def quote(self, k, inv):
        c, p = self.cfg, self.cfg.p_informed
        s = np.arange(len(self.p_max))
        e_inf, e_noise = (s * self.p_max).sum(), (s * self.p_min).sum()
        m, _ = self.mean_sd()
        best = {}
        for side in ("ask", "bid"):
            best_val = np.full(m.shape, -np.inf)
            best_px = m.copy()
            for d in self.OFFSETS:
                px = m + d if side == "ask" else m - d
                if side == "ask":
                    inf = ndtr((self.grid[None, :] - px[:, None]) / c.sigma_signal)
                    noise = ndtr((c.fair - px) / c.sigma_noise)[:, None]
                    edge = px[:, None] - self.grid[None, :]
                else:
                    inf = ndtr((px[:, None] - self.grid[None, :]) / c.sigma_signal)
                    noise = ndtr((px - c.fair) / c.sigma_noise)[:, None]
                    edge = self.grid[None, :] - px[:, None]
                val = (self.post * (p * inf * e_inf + (1 - p) * noise * e_noise) * edge).sum(axis=1)
                better = val > best_val
                best_val = np.where(better, val, best_val)
                best_px = np.where(better, px, best_px)
            best[side] = best_px
        return best["bid"], best["ask"]


# ------------------------------------------------------------------ simulation

def simulate(strategy, batch, cfg):
    """Run `strategy` over every round in `batch`. Returns per-round arrays."""
    n, A = batch.informed.shape
    strategy.reset(n, cfg)
    cash = np.zeros(n)
    inv = np.zeros(n)
    spread = np.zeros(n)
    adverse = np.zeros(n)
    volume = np.zeros(n)
    fills = np.zeros(n)
    M = max(cfg.max_size, 1)
    for k in range(A):
        V = batch.values[:, k]
        bid, ask = strategy.quote(k, inv)
        bid, ask = np.minimum(bid, ask), np.maximum(bid, ask)
        inf = batch.informed[:, k]
        perceived = np.where(inf, V + cfg.sigma_signal * batch.z[:, k],
                             cfg.fair + cfg.sigma_noise * batch.z[:, k])
        if M <= 1:
            size = np.ones(n, dtype=int)
        else:
            size = np.where(inf, np.maximum(batch.size_a[:, k], batch.size_b[:, k]),
                            np.minimum(batch.size_a[:, k], batch.size_b[:, k]))
        sell = perceived > ask
        buy = ~sell & (perceived < bid)
        mid = (bid + ask) / 2
        cash += np.where(sell, ask * size, np.where(buy, -bid * size, 0.0))
        inv += np.where(buy, size, 0) - np.where(sell, size, 0)
        spread += np.where(sell, ask - mid, np.where(buy, mid - bid, 0.0)) * size
        adverse += np.where(sell, mid - V, np.where(buy, V - mid, 0.0)) * size
        traded = sell | buy
        volume += np.where(traded, size, 0)
        fills += traded
        strategy.observe(bid, ask, sell, buy, size)
    pnl = cash + inv * batch.values[:, -1]
    return dict(pnl=pnl, spread=spread, adverse=adverse,
                drift=pnl - spread - adverse, volume=volume, fills=fills)


def tune(make, grid, cfg, batch):
    """Grid-search a strategy's parameters on `batch` (the TRAINING set).
    Returns (best_params, mean P&L on the training set)."""
    best, best_pnl = None, -np.inf
    for params in grid:
        pnl = simulate(make(*params), batch, cfg)["pnl"].mean()
        if pnl > best_pnl:
            best, best_pnl = params, pnl
    return best, best_pnl


HALF_SPREADS = [(h,) for h in np.arange(0.5, 40.01, 0.5)]
CHASER_GRID = [(h, st) for h in np.arange(1.0, 15.01, 1.0)
               for st in np.arange(0.0, 3.01, 0.25)]
BAYES_HALF_SPREADS = [(h,) for h in np.arange(0.0, 12.01, 0.5)]


def tuned_strategies(cfg, train):
    """The four bots, with parameters fitted on `train` only."""
    (h_s,), _ = tune(Static, HALF_SPREADS, cfg, train)
    (h_c, st_c), _ = tune(Chaser, CHASER_GRID, cfg, train)
    (h_b,), _ = tune(BayesMid, BAYES_HALF_SPREADS, cfg, train)
    return [Static(h_s), Chaser(h_c, st_c), BayesMid(h_b), BayesMyopic()]
