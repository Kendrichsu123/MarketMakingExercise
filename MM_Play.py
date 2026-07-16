"""Interactive market-making trainer with mid-round RE-QUOTING and inventory.

Run:  python mm_play.py        (standard library only; matplotlib optional)

You post a two-way market. Orders arrive one at a time. Between arrivals you
HOLD (press Enter) or RE-QUOTE (type a new bid and ask). Two skills this trains
that a static quote can't:

  1. Order flow is information. If you keep getting lifted, the value is above
     your price -- raise your market before the next order picks you off. The
     P&L split shows adverse selection shrinking as you mark toward the truth.
  2. Inventory is risk. Every fill moves your position; at round end it marks
     to the true value. Skew your quote (drop both sides when long, lift both
     when short) to flatten before a wrong-way position burns you.

The engine lives in mm_engine.py -- import it into a notebook to run thousands
of sessions, or wrap a different UI (Streamlit, web) around it later.
"""

import random
from dataclasses import replace
import MM_Engine as mm


def parse_quote(raw):
    for ch in (",", "b", "a", "x", "/", "@"):
        raw = raw.replace(ch, " ")
    nums = [float(p) for p in raw.split()]
    if len(nums) != 2:
        raise ValueError
    return min(nums), max(nums)


def ask_quote(prompt, allow_hold):
    """Returns ('quote', (bid, ask)) | ('hold', None) | ('skip', None) | ('quit', None)."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("q", "quit", "exit"):
            return ("quit", None)
        if allow_hold and raw == "":
            return ("hold", None)
        if not allow_hold and raw in ("skip", "s"):
            return ("skip", None)
        try:
            return ("quote", parse_quote(raw))
        except ValueError:
            tail = "  (or Enter to hold)" if allow_hold else "  (or 'skip')"
            print(f"   couldn't read that -- try:  96 104{tail}")


def choose_config():
    print("Pick difficulty:")
    for key, (label, _) in mm.PRESETS.items():
        print(f"  {key}) {label}")
    while True:
        pick = input("> ").strip()
        if pick in mm.PRESETS:
            return mm.PRESETS[pick][1]
        print("Enter 1, 2, or 3.")


def choose_drift(cfg):
    print("\nFair value behavior:")
    print("  1) fixed     V is drawn once and held all round (pure adverse selection)")
    print("  2) drifting  V random-walks between orders (adds inventory / timing risk)")
    while True:
        pick = input("> ").strip()
        if pick == "1":
            return replace(cfg, sigma_step=0.0)
        if pick == "2":
            return replace(cfg, sigma_step=2.0)
        print("Enter 1 or 2.")


def play_round(cfg, r, rng):
    """Play one round. Returns (status, record) where status is
    'done' | 'skip' | 'quit'. record is None unless status == 'done'."""
    V = mm.draw_value(cfg, rng)
    N = cfg.arrivals
    cash = 0.0
    inv = 0
    spread_tot = adverse_tot = 0.0
    fills = sells = buys = volume = 0
    widths = []
    skews = []
    max_abs_inv = 0

    print(f"\nRound {r}/{cfg.rounds}   fair starts at {cfg.fair:.0f}, "
          f"{N} orders will arrive.")

    # Initial quote (required -- nothing to hold yet).
    status, q = ask_quote(f"  order 1/{N}  post your market > ", allow_hold=False)
    if status in ("quit", "skip"):
        if status == "skip":
            print(f"  passed. (V was {V})")
        return (status, None)
    bid, ask = q
    widths.append(ask - bid)
    skews.append(abs((bid + ask) / 2 - cfg.fair))

    for k in range(1, N + 1):
        if k > 1:
            mid = (bid + ask) / 2
            print(f"  inv {inv:+d}   cash {cash:+.1f}   "
                  f"quote {bid:.1f}/{ask:.1f}  (mid {mid:.1f}, w {ask - bid:.1f})")
            status, q = ask_quote(f"  order {k}/{N}  hold or re-quote > ",
                                  allow_hold=True)
            if status == "quit":
                return ("quit", None)
            if status == "quote":
                bid, ask = q
                widths.append(ask - bid)
                skews.append(abs((bid + ask) / 2 - cfg.fair))

        mid = (bid + ask) / 2
        action, price, size = mm.one_arrival(bid, ask, V, cfg, rng)
        s, a = mm.fill_pnl(action, price, size, mid, V)
        spread_tot += s
        adverse_tot += a
        if action == "sell":
            cash += price * size
            inv -= size
            sells += 1
            fills += 1
            volume += size
            print(f"    -> offer lifted @ {price:.1f} x{size}   inv {inv:+d}")
        elif action == "buy":
            cash -= price * size
            inv += size
            buys += 1
            fills += 1
            volume += size
            print(f"    -> bid hit @ {price:.1f} x{size}   inv {inv:+d}")
        else:
            print(f"    -> no trade")
        max_abs_inv = max(max_abs_inv, abs(inv))
        V = mm.step_value(V, cfg, rng)  # value moves before the next order

    # Mark remaining inventory to the closing value.
    carried = inv
    cash += inv * V
    round_pnl = cash
    # Total = spread + adverse(at trade time) + drift(inventory carried through
    # value moves). Drift is the residual and is ~0 when V is fixed.
    drift = round_pnl - spread_tot - adverse_tot

    vclose = f"{V:.1f}" if cfg.sigma_step > 0 else f"{V:.0f}"
    print(f"  V closed at {vclose}.   carried {carried:+d} into the close.")
    if cfg.sigma_step > 0:
        print(f"  round P&L {round_pnl:+.1f}   =  spread {spread_tot:+.1f}  +  "
              f"adverse {adverse_tot:+.1f}  +  drift {drift:+.1f}")
    else:
        print(f"  round P&L {round_pnl:+.1f}   =  spread {spread_tot:+.1f}  +  "
              f"adverse selection {adverse_tot:+.1f}")

    return ("done", dict(
        pnl=round_pnl, spread=spread_tot, adverse=adverse_tot, drift=drift,
        fills=fills, volume=volume, avg_width=sum(widths) / len(widths),
        avg_skew=sum(skews) / len(skews),
        max_abs_inv=max_abs_inv, carried=abs(carried)))


def summarize(hist, cfg, rng):
    played = [h for h in hist if h is not None]
    if not played:
        print("\nNo rounds played.")
        return

    n = len(played)
    pnl = sum(h["pnl"] for h in played)
    spread = sum(h["spread"] for h in played)
    adverse = sum(h["adverse"] for h in played)
    drift = sum(h["drift"] for h in played)
    fills = sum(h["fills"] for h in played)
    volume = sum(h.get("volume", h["fills"]) for h in played)
    avg_width = sum(h["avg_width"] for h in played) / n
    avg_skew = sum(h["avg_skew"] for h in played) / n
    avg_maxinv = sum(h["max_abs_inv"] for h in played) / n
    avg_carried = sum(h["carried"] for h in played) / n

    best_h, static_pnl = mm.best_static_half_spread(cfg, rng)

    print("\n" + "=" * 60)
    print(f"SESSION SUMMARY  ({n} rounds)")
    print("=" * 60)
    print(f"  Total P&L .................... {pnl:+.1f}   "
          f"({pnl / n:+.2f} / round)")
    print(f"    spread captured ........... {spread:+.1f}")
    print(f"    adverse selection paid .... {adverse:+.1f}")
    if cfg.sigma_step > 0:
        print(f"    inventory drift / carry ... {drift:+.1f}")
    print(f"  Fills ....................... {fills} orders / {volume} lots")
    if volume:
        print(f"  Per lot:  spread {spread / volume:+.2f}   "
              f"adverse {adverse / volume:+.2f}   net {pnl / volume:+.2f}")
    print(f"  Avg quoted width ............ {avg_width:.1f}")
    print(f"  Avg mid skew off fair ....... {avg_skew:.2f}")
    print(f"  Avg peak inventory / round .. {avg_maxinv:.1f}")
    print(f"  Avg carried to close ........ {avg_carried:.1f}")

    print(f"\n  Benchmark: best NON-updating symmetric width for these settings")
    print(f"  is ~{2 * best_h:.0f} (half-spread {best_h:.1f}), worth "
          f"~{static_pnl:+.2f} / round.")
    print(f"  You made {pnl / n:+.2f} / round.")

    print("\n  Read:")
    edge = pnl / n - static_pnl
    if edge > 0.15 * abs(static_pnl) + 0.5:
        print("  - You beat the no-update benchmark. You're pulling information")
        print("    out of order flow -- marking toward V as it reveals itself.")
    elif edge < -0.15 * abs(static_pnl) - 0.5:
        print("  - You're below the naive static benchmark. Either you're not")
        print("    re-quoting on flow, or you're over-reacting and paying for it")
        print("    in wrong-way inventory. Update, but don't chase every fill.")
    else:
        print("  - You're roughly at the no-update benchmark. Re-quoting isn't")
        print("    helping yet -- watch which side keeps trading and mark that way.")

    if fills and spread > 0 and adverse < -0.6 * spread:
        print("  - Adverse selection is eating most of your spread. You're getting")
        print("    picked off -- widen, or mark your mid faster when flow is one-sided.")
    if avg_maxinv >= 1.2 * cfg.max_size:
        print("  - Inventory ran large. A single big fill can load you fast now.")
        print("    When you're long, drop BOTH quotes to invite sellers and deter")
        print("    buyers (mirror when short). Skewing to flatten is the point of")
        print("    re-quoting -- and it matters more once orders come in size.")
    if avg_skew > cfg.sigma_v:
        print("  - Big average skew. Skewing to manage inventory is good; skewing")
        print("    beyond what flow justifies is just a directional bet.")
    if fills and spread > 0 and adverse < -0.55 * spread:
        print("  - A big fill is more likely informed. After you get hit in size,")
        print("    treat it as a stronger signal you're mispriced -- mark harder and")
        print("    consider widening, not just flattening the position.")
    if cfg.sigma_step > 0 and drift < -0.4 * abs(spread) - 1:
        print("  - Drift is costing you: value moved against inventory you were")
        print("    holding. When V can walk, carry less -- flatten faster and don't")
        print("    let a position sit through several orders hoping to round-trip.")

    # Optional cumulative-P&L plot.
    try:
        import matplotlib.pyplot as plt
        cum, run = [], 0.0
        for h in played:
            run += h["pnl"]
            cum.append(run)
        plt.figure()
        plt.plot(range(1, n + 1), cum, marker="o")
        plt.axhline(0, color="grey", lw=0.8)
        plt.title("Cumulative P&L")
        plt.xlabel("round")
        plt.ylabel("P&L")
        plt.tight_layout()
        plt.savefig("mm_pnl.png", dpi=120)
        print("\n  Saved P&L curve to mm_pnl.png")
    except ImportError:
        pass


def main():
    rng = random.Random()
    cfg = choose_config()
    cfg = choose_drift(cfg)
    drift_note = (f"V drifts ~{cfg.sigma_step:.1f} per order, so inventory carries timing risk."
                  if cfg.sigma_step > 0 else
                  "V is fixed for the round; your only value risk is mis-estimating it.")
    print(f"""
True value V starts around {cfg.fair:.0f} (stdev {cfg.sigma_v:.0f}) and is hidden while you quote.
{int(cfg.p_informed * 100)}% of orders are informed (signal noise {cfg.sigma_signal:.1f}); the rest are noise.
Orders arrive in sizes up to {cfg.max_size} lots; informed orders skew larger, so a big
fill loads your inventory fast AND warns the fill was probably toxic -- react to it.
{drift_note}
Post a market (e.g. 96 104). After each order: Enter to hold, or type a new
market to re-quote. 'skip' to pass a round, 'q' to stop and see results.
""")
    hist = []
    for r in range(1, cfg.rounds + 1):
        status, rec = play_round(cfg, r, rng)
        if status == "quit":
            break
        hist.append(rec if status == "done" else None)
    summarize(hist, cfg, rng)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")