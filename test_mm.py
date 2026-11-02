"""Checks on the engine, the vectorized simulator and the Bayesian bots.

    python -m pytest -q test_mm.py
"""

import random
from dataclasses import replace

import numpy as np
import pytest

import mm_engine as mm
import mm_sim as ms

SETTINGS = [replace(mm.PRESETS[k][1], sigma_step=d)
            for k in "123" for d in (0.0, mm.DRIFT_STEP)]


def scenarios(cfg, n, seed):
    rng = random.Random(seed)
    return [mm.draw_round(cfg, rng) for _ in range(n)]


def chaser_quoter(cfg, h, step):
    """The Chaser bot written as a scalar quoter for mm_engine.play_round."""
    state = {"mid": cfg.fair}

    def quote(k, inv, history):
        if history:
            _, _, action, size = history[-1]
            state["mid"] += step * size * ((action == "sell") - (action == "buy"))
        return state["mid"] - h, state["mid"] + h
    return quote


@pytest.mark.parametrize("cfg", SETTINGS)
def test_vectorized_simulator_matches_engine_round_by_round(cfg):
    """The game (mm_engine) and the research code (mm_sim) are one model."""
    scen = scenarios(cfg, 300, seed=7)
    scalar = [mm.play_round(s, chaser_quoter(cfg, 4.0, 0.75), cfg) for s in scen]
    vector = ms.simulate(ms.Chaser(4.0, 0.75), ms.from_scenarios(scen), cfg)
    for field in ("pnl", "spread", "adverse", "drift", "volume", "fills"):
        assert np.allclose([r[field] for r in scalar], vector[field]), field


@pytest.mark.parametrize("cfg", SETTINGS)
def test_pnl_decomposes_exactly(cfg):
    """pnl = spread + adverse + drift; spread is never negative; drift is
    identically zero when the value is fixed."""
    r = ms.simulate(ms.Chaser(5.0, 1.0), ms.draw_batch(cfg, 5000, seed=3), cfg)
    assert np.allclose(r["pnl"], r["spread"] + r["adverse"] + r["drift"])
    assert (r["spread"] >= -1e-12).all()
    if cfg.sigma_step == 0:
        assert np.allclose(r["drift"], 0.0)


def test_size_distributions():
    """Informed sizes are the max of two uniforms, noise the min; both pmfs sum
    to 1, match a simulation, and informed orders are larger on average."""
    cfg = mm.PRESETS["2"][1]
    p_max, p_min = ms.size_pmfs(cfg)
    assert np.isclose(p_max.sum(), 1) and np.isclose(p_min.sum(), 1)
    b = ms.draw_batch(cfg, 200_000, seed=5)
    emp_max = np.bincount(np.maximum(b.size_a, b.size_b).ravel(), minlength=5) / b.size_a.size
    emp_min = np.bincount(np.minimum(b.size_a, b.size_b).ravel(), minlength=5) / b.size_a.size
    assert np.allclose(emp_max, p_max, atol=0.005)
    assert np.allclose(emp_min, p_min, atol=0.005)
    s = np.arange(len(p_max))
    assert (s * p_max).sum() > (s * p_min).sum()


@pytest.mark.parametrize("cfg", SETTINGS)
def test_posterior_is_calibrated(cfg):
    """If Bayes' rule is implemented correctly, the true value V is a draw from
    the bot's own posterior, so the posterior CDF evaluated at V (the PIT) is
    uniform on [0, 1]. A wrong likelihood -- say, mis-weighting informed flow --
    makes the posterior over- or under-confident and bends the PIT histogram.

    The PIT is used rather than (V - mean) / sd because posteriors here can be
    lopsided after one-sided flow: that z has |z| ~ 10 in rare rounds, which
    makes its sample variance too noisy to test against 1."""
    n = 20_000
    batch = ms.draw_batch(cfg, n, seed=11)
    bot = ms.BayesMid(6.0)
    ms.simulate(bot, batch, cfg)
    # After the last order (and the drift step that follows it) the posterior
    # is over the closing value values[:, -1]. With drift on, V is continuous,
    # so snap it to the nearest grid point; the grid posterior is a
    # discretization of the true one.
    target = batch.values[:, -1]
    rows = np.arange(n)
    idx = np.clip(np.rint((target - bot.grid[0]) / bot.SPACING).astype(int),
                  0, len(bot.grid) - 1)
    cdf_below = np.cumsum(bot.post, axis=1)[rows, idx] - bot.post[rows, idx]
    # Randomized PIT: exactly uniform for a correct discrete posterior.
    u = np.random.default_rng(0).random(n)
    pit = cdf_below + u * bot.post[rows, idx]
    counts = np.histogram(pit, bins=10, range=(0, 1))[0]
    chi2 = ((counts - n / 10) ** 2 / (n / 10)).sum()
    assert chi2 < 27.9, (chi2, counts)  # chi-square(9) at p = 0.001


def test_myopic_quotes_are_symmetric_at_the_prior():
    """Before any order the posterior is symmetric about fair, so the myopic
    bot's first quote must be too."""
    cfg = mm.PRESETS["2"][1]
    bot = ms.BayesMyopic()
    bot.reset(4, cfg)
    bid, ask = bot.quote(0, np.zeros(4))
    assert np.allclose(cfg.fair - bid, ask - cfg.fair)
    assert (ask > bid).all()


def test_static_tuning_does_not_reuse_the_test_set():
    """Picking the best of many widths on one sample overstates it; on a fresh
    sample the tuned width should do no better on average than it did in
    training (selection bias goes one way)."""
    cfg = mm.PRESETS["2"][1]
    train, test = ms.draw_batch(cfg, 3000, seed=21), ms.draw_batch(cfg, 20000, seed=22)
    (h,), train_pnl = ms.tune(ms.Static, ms.HALF_SPREADS, cfg, train)
    test_pnl = ms.simulate(ms.Static(h), test, cfg)["pnl"].mean()
    se = 30 / np.sqrt(3000)  # one round's sd is about 30 on this preset
    assert test_pnl < train_pnl + 3 * se
