"""Pure market-making simulation logic. No I/O -- import this from a front-end,
a notebook, or a backtest. Standard library only.

Model (sequential-trade / Glosten-Milgrom flavor): a hidden value V is drawn
each round. Orders arrive ONE AT A TIME. Each order is INFORMED (its reservation
price tracks V through noise, so it trades against you when your quote is wrong
in its favor) or NOISE (its reservation price is anchored on the prior, so it's
uncorrelated with V and pays you the spread). Between arrivals a front-end may
re-quote.

All of a round's randomness is drawn up front by `draw_round` into a
`Scenario`: the value path and, for every arrival, whether it is informed, its
reservation-price shock and its two size draws. `resolve` then decides what an
order does against a given quote. Separating the two means any number of
quoting strategies -- a human, a bot, a benchmark -- can be replayed against
the *same* order flow, so they can be compared on identical rounds (common
random numbers) instead of on independent, noisy samples.

Each fill's P&L is split into spread captured vs. adverse selection paid,
measured against the mid IN FORCE AT THE MOMENT OF THE FILL. As you mark your
mid toward V, the adverse-selection term shrinks, so reading order flow and
updating literally reduces the tax informed traders levy on you.
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

DRIFT_STEP = 2.0  # per-order stdev of V in "drifting" mode


@dataclass(frozen=True)
class Order:
    """One arrival's randomness, independent of any quote."""
    informed: bool
    z: float        # standard-normal shock to the reservation price
    size_a: int     # two uniform draws on 1..max_size; informed orders take
    size_b: int     # the max, noise the min (see `order_size`)


@dataclass(frozen=True)
class Scenario:
    """Everything random about one round. values[k] is V when order k arrives;
    values[-1] is the closing value inventory is marked to."""
    values: tuple
    orders: tuple


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


def draw_round(cfg, rng):
    """Draw one round's value path and order flow."""
    V = draw_value(cfg, rng)
    values, orders = [V], []
    for _ in range(cfg.arrivals):
        orders.append(Order(
            informed=rng.random() < cfg.p_informed,
            z=rng.gauss(0, 1),
            size_a=rng.randint(1, max(cfg.max_size, 1)),
            size_b=rng.randint(1, max(cfg.max_size, 1)),
        ))
        V = step_value(V, cfg, rng)
        values.append(V)
    return Scenario(values=tuple(values), orders=tuple(orders))


def order_size(order, cfg):
    """Informed orders skew LARGER (a big print more likely carries
    information); noise skews smaller. The distributions OVERLAP, so a large
    order is only *probably* informed -- size is a signal, not a tell.
        P(size = s | informed) = (2s - 1) / M^2        (max of two uniforms)
        P(size = s | noise)    = (2(M - s) + 1) / M^2  (min of two uniforms)"""
    if cfg.max_size <= 1:
        return 1
    if order.informed:
        return max(order.size_a, order.size_b)
    return min(order.size_a, order.size_b)


def reservation_price(order, V, cfg):
    """Informed reservation tracks V; noise reservation is anchored on the prior."""
    if order.informed:
        return V + cfg.sigma_signal * order.z
    return cfg.fair + cfg.sigma_noise * order.z


def resolve(order, V, bid, ask, cfg):
    """What one order does against your current quote. Returns (action, price, size):
        ('sell', ask, s)  counterparty lifted your offer  -> you sold s, inv down
        ('buy',  bid, s)  counterparty hit your bid        -> you bought s, inv up
        ('none', None, 0) no trade
    An order only crosses if your quote beats its reservation price. Size doesn't
    change WHETHER it trades -- only HOW MUCH."""
    perceived = reservation_price(order, V, cfg)
    if perceived > ask:
        return ("sell", ask, order_size(order, cfg))
    if perceived < bid:
        return ("buy", bid, order_size(order, cfg))
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


def play_round(scenario, quoter, cfg):
    """Replay one scenario against `quoter`, a callable
        quoter(k, inv, history) -> (bid, ask)
    where history is the list of (bid, ask, action, size) for orders so far.
    Returns a dict with pnl and its spread / adverse / drift split.
    pnl = spread + adverse + drift exactly; drift is 0 when V is fixed."""
    cash, inv = 0.0, 0
    spread = adverse = 0.0
    fills = volume = 0
    history = []
    for k, order in enumerate(scenario.orders):
        V = scenario.values[k]
        bid, ask = quoter(k, inv, history)
        action, price, size = resolve(order, V, bid, ask, cfg)
        s, a = fill_pnl(action, price, size, (bid + ask) / 2, V)
        spread += s
        adverse += a
        if action == "sell":
            cash += price * size
            inv -= size
        elif action == "buy":
            cash -= price * size
            inv += size
        if action != "none":
            fills += 1
            volume += size
        history.append((bid, ask, action, size))
    pnl = cash + inv * scenario.values[-1]
    return dict(pnl=pnl, spread=spread, adverse=adverse,
                drift=pnl - spread - adverse, fills=fills, volume=volume)
