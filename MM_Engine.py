"""Pure market-making simulation logic. No I/O -- import this from a front-end,
a notebook, or a backtest. Standard library only.

Model (sequential-trade / Glosten-Milgrom flavor): a hidden value V is drawn
each round. Orders arrive ONE AT A TIME. Each order is INFORMED (its reservation
price tracks V through noise, so it trades against you when your quote is wrong
in its favor) or NOISE (its reservation price is anchored on the prior, so it's
uncorrelated with V and pays you the spread). Between arrivals a front-end may
re-quote.

Each fill's P&L is split into spread captured vs. adverse selection paid,
measured against the mid IN FORCE AT THE MOMENT OF THE FILL. That's the whole
lesson: as you mark your mid toward V, the adverse-selection term shrinks, so
reading order flow and updating literally reduces the tax informed traders
levy on you.
"""

from dataclasses import dataclass
import random


@dataclass
class Config:
    fair: float = 100.0          # prior mean = your starting rational mid
    sigma_v: float = 8.0         # stdev of true value around fair (your uncertainty)
    sigma_signal: float = 3.0    # informed trader's signal noise (smaller = sharper)
    p_informed: float = 0.45     # fraction of arrivals that are informed
    sigma_noise: float = 8.0     # dispersion of noise traders' reservation prices
    arrivals: int = 6            # orders per round
    rounds: int = 10             # rounds per session
    sigma_step: float = 0.0      # per-order drift of V (0 = fixed value all round)
    max_size: int = 4            # largest order size; informed orders skew larger


PRESETS = {
    "1": ("easy   (dull informed traders, plenty of noise flow)",
          Config(sigma_v=6,  sigma_signal=4.0, p_informed=0.30, sigma_noise=9.0, arrivals=6)),
    "2": ("medium (balanced)",
          Config(sigma_v=8,  sigma_signal=3.0, p_informed=0.45, sigma_noise=8.0, arrivals=6)),
    "3": ("hard   (sharp, frequent informed traders)",
          Config(sigma_v=10, sigma_signal=2.0, p_informed=0.55, sigma_noise=8.0, arrivals=8)),
}


def draw_value(cfg, rng):
    """The hidden true value for a round. You never see it while quoting."""
    return round(rng.gauss(cfg.fair, cfg.sigma_v))


def step_value(V, cfg, rng):
    """Random-walk the true value one step. With sigma_step == 0 this is a no-op
    and V stays fixed for the round (pure adverse-selection mode). With drift on,
    a position held across steps carries timing risk even if priced correctly."""
    if cfg.sigma_step <= 0:
        return V
    return V + rng.gauss(0, cfg.sigma_step)


def draw_size(informed, cfg, rng):
    """Order size. Informed orders skew LARGER (a big print more likely carries
    information); noise skews smaller. Distributions OVERLAP, so a large order is
    only *probably* informed, never certainly -- size is a signal, not a tell."""
    if cfg.max_size <= 1:
        return 1
    a, b = rng.randint(1, cfg.max_size), rng.randint(1, cfg.max_size)
    return max(a, b) if informed else min(a, b)


def one_arrival(bid, ask, V, cfg, rng):
    """A single order hits your current quote. Returns (action, price, size):
        ('sell', ask, s)  counterparty lifted your offer  -> you sold s, inv down
        ('buy',  bid, s)  counterparty hit your bid        -> you bought s, inv up
        ('none', None, 0) no trade
    Informed reservation tracks V; noise reservation is anchored on the prior.
    Either only crosses if your quote beats their price. Size doesn't change
    WHETHER they trade -- only HOW MUCH -- but informed flow arrives bigger, so a
    large fill both loads your inventory faster and warns the fill was toxic."""
    informed = rng.random() < cfg.p_informed
    if informed:
        perceived = rng.gauss(V, cfg.sigma_signal)
    else:
        perceived = rng.gauss(cfg.fair, cfg.sigma_noise)
    if perceived > ask:
        return ("sell", ask, draw_size(informed, cfg, rng))
    if perceived < bid:
        return ("buy", bid, draw_size(informed, cfg, rng))
    return ("none", None, 0)


def fill_pnl(action, price, size, mid, V):
    """Split one fill's mark-to-V P&L into (spread, adverse) at the fill's mid,
    scaled by size.
        sell size s at ask a: pnl = (a - V)*s = (a - mid)*s + (mid - V)*s
        buy  size s at bid b: pnl = (V - b)*s = (mid - b)*s + (V - mid)*s
    spread is always >= 0 (your half-spread); adverse is negative exactly when
    someone traded the correct side of your mispricing -- and a big size
    multiplies both."""
    if action == "sell":
        return ((price - mid) * size, (mid - V) * size)
    if action == "buy":
        return ((mid - price) * size, (V - mid) * size)
    return (0.0, 0.0)


def simulate_static(h, cfg, rng):
    """Ground-truth P&L of a symmetric quote that is NEVER re-quoted, over one
    round -- tracks inventory and liquidates at the closing value, so it's
    correct with drift on or off. This is the naive benchmark the interactive
    player tries to beat by reading flow and skewing on inventory."""
    V = draw_value(cfg, rng)
    bid, ask = cfg.fair - h, cfg.fair + h
    cash, inv = 0.0, 0
    for _ in range(cfg.arrivals):
        action, price, size = one_arrival(bid, ask, V, cfg, rng)
        if action == "sell":
            cash += price * size
            inv -= size
        elif action == "buy":
            cash -= price * size
            inv += size
        V = step_value(V, cfg, rng)
    return cash + inv * V  # mark remaining inventory to the closing value


def best_static_half_spread(cfg, rng, trials=3000):
    """Grid-search the P&L-maximizing symmetric STATIC half-spread.
    Returns (half_spread, avg_pnl_per_round). This is the 'if you never
    updated' bar; good flow-reading should exceed its per-round P&L."""
    best_h, best_pnl = 0.5, -1e18
    for k in range(1, 41):
        h = 0.5 * k
        avg = sum(simulate_static(h, cfg, rng) for _ in range(trials)) / trials
        if avg > best_pnl:
            best_pnl, best_h = avg, h
    return best_h, best_pnl