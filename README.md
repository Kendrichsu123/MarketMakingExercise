# Market-Making Trainer

A terminal trainer for quoting a two-way market against **informed order flow**.
You post a bid/ask on a hidden value, orders arrive one at a time (in varying
sizes), and you re-quote as the flow reveals information. Every fill's P&L is
decomposed into **spread captured** vs. **adverse selection paid**, so you can
see exactly where your edge comes from and where it leaks.

**What this demonstrates:** a working model of adverse selection in the spirit
of Glosten–Milgrom — informed vs. noise flow, a spread that exists purely
because some counterparties know more than the market maker, inventory risk, and
a clean decomposition of realized P&L into the two forces that drive it. Built
to drill the skills trading-interview market-making games test: pricing under
adverse selection, reading order flow, and managing inventory.

## The model

A sequential-trade setup in the spirit of Glosten–Milgrom (1985):

- Each round draws a hidden true value **V** from a known prior (mean 100).
- Orders arrive one at a time, in **varying sizes**. Each is **informed** — its
  reservation price tracks V through noise, so it only trades when your quote is
  wrong in its favor — or **noise** — anchored on the prior, uncorrelated with V.
  Informed orders skew larger, so a big fill both loads inventory faster and
  warns the fill was probably toxic.
- You **re-quote between every order**. Order flow is information: if your offer
  keeps getting lifted, V is above your price, so mark up.
- Every fill moves your **inventory**, which marks to V at the close. Skew your
  quote to flatten a position before it burns you.

### P&L decomposition

Each fill's mark-to-value P&L splits by using your mid `m` as the pivot:

```
sell size s at ask a:  P&L = (a − V)·s = (a − m)·s + (m − V)·s
buy  size s at bid b:  P&L = (V − b)·s = (m − b)·s + (V − m)·s
                                         |_ spread _| |_ adverse _|
```

Spread is always non-negative (your half-spread — what you charge everyone).
Adverse selection is negative exactly when someone traded the correct side of
your mispricing. Against noise flow the adverse term averages to zero; against
informed flow it is systematically negative. Reading flow and marking your mid
toward V shrinks the adverse term without touching the spread.

## Running it

Standard library only. `matplotlib` is optional (saves a cumulative-P&L plot).

```bash
python3 MM_Play.py
```

Pick a difficulty, then choose fixed or drifting value:

- **fixed** — V is drawn once and held all round. Isolates pure adverse selection.
- **drifting** — V random-walks between orders, adding inventory/timing risk.
  P&L then splits three ways: spread, adverse, and **drift** (value moving
  against inventory you held).

Post a market (e.g. `96 104`). After each order, press Enter to hold or type a
new market to re-quote. `skip` passes a round, `q` ends and shows the summary.

The end-of-session summary reports your spread/adverse split, net edge per lot,
average width and inventory, and — as a benchmark — the P&L-maximizing
*non-updating* width for those settings (found by Monte Carlo). Beating that
benchmark means you're extracting information from order flow rather than sitting
on a static quote.

<!-- Optional: drop a screenshot of a session summary here:
![session summary](summary.png) -->

## Structure

- `MM_Engine.py` — pure simulation logic, no I/O. Import it into a notebook to
  run thousands of sessions, sweep parameters, or wrap a different UI around it.
- `MM_Play.py` — the interactive terminal front-end.

## Notes / possible extensions

- The informed/noise mix and prior-anchored noise are deliberately simple; real
  toxicity is more structured (news, correlated flow, size signaling).
- Drift is a trend-free random walk — only risk to manage, no directional edge.
- Natural next steps: quoting your own size/depth (not just reacting to it), a
  proper inventory-penalty objective, competing market makers, and a Streamlit
  or web front-end on top of the same engine.
