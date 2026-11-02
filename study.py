"""How much is reading order flow worth? Runs every bot on every setting.

For each difficulty preset, with the value fixed and drifting:
  1. tune each bot's parameters on a TRAINING batch of rounds,
  2. evaluate all bots on a separate TEST batch -- the same rounds for every
     bot (common random numbers), so differences between bots are paired,
  3. report P&L per round with 95% confidence intervals, the spread / adverse
     split per lot, and the paired gains between bots.

Tuning and evaluating on separate batches matters: picking the best of ~40
widths on the same noisy sample you then report would overstate the winner.

    python study.py > results/study_output.md    # ~4 min; also writes figures/
"""

import sys
from dataclasses import replace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mm_engine as mm
import mm_sim as ms

N_TRAIN, N_TEST = 5_000, 20_000
SEED_TRAIN, SEED_TEST = 1, 2

LABELS = {"static": "Static", "chaser": "Chaser",
          "bayes-mid": "Bayes-mid", "bayes-myopic": "Bayes-myopic"}
COLORS = {"static": "#2a78d6", "chaser": "#eb6834",
          "bayes-mid": "#1baf7a", "bayes-myopic": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def ci(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def params(s):
    if s.name == "static":
        return f"h={s.h:g}"
    if s.name == "chaser":
        return f"h={s.h:g}, step={s.step:g}"
    if s.name == "bayes-mid":
        return f"h={s.h:g}"
    return "none"


def run_setting(key, drift):
    cfg = replace(mm.PRESETS[key][1], sigma_step=drift)
    train = ms.draw_batch(cfg, N_TRAIN, SEED_TRAIN)
    test = ms.draw_batch(cfg, N_TEST, SEED_TEST)
    strats = ms.tuned_strategies(cfg, train)
    return cfg, {s.name: (s, ms.simulate(s, test, cfg)) for s in strats}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    settings = [(k, d) for k in "123" for d in (0.0, mm.DRIFT_STEP)]
    results = {}
    for key, drift in settings:
        name = f"{mm.PRESETS[key][0].split()[0]}, {'drifting' if drift else 'fixed'}"
        print(f"running {name} ...", file=sys.stderr, flush=True)
        results[(key, drift)] = (name,) + run_setting(key, drift)

    print("\n## P&L per round on the test set (95% CI), and the split per lot\n")
    print("| setting | bot | tuned params | P&L / round | spread / lot | adverse / lot |")
    print("|---|---|---|---:|---:|---:|")
    for key, drift in settings:
        name, cfg, res = results[(key, drift)]
        for bot, (s, r) in res.items():
            vol = r["volume"].sum()
            if vol < 0.01 * len(r["pnl"]):  # fewer than 1 lot per 100 rounds
                split = "barely trades | — "
            else:
                split = f"{r['spread'].sum() / vol:+.2f} | {r['adverse'].sum() / vol:+.2f} "
            print(f"| {name} | {LABELS[bot]} | {params(s)} | "
                  f"{r['pnl'].mean():+.2f} ± {ci(r['pnl']):.2f} | {split}|")

    print("\n## Paired gains on identical rounds (95% CI)\n")
    print("| setting | chaser − static | bayes-mid − chaser | bayes-myopic − bayes-mid |")
    print("|---|---:|---:|---:|")
    for key, drift in settings:
        name, cfg, res = results[(key, drift)]
        p = {b: r["pnl"] for b, (_, r) in res.items()}
        cells = []
        for a, b in [("chaser", "static"), ("bayes-mid", "chaser"), ("bayes-myopic", "bayes-mid")]:
            d = p[a] - p[b]
            cells.append(f"{d.mean():+.2f} ± {ci(d):.2f}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n## How many rounds it takes to grade a player\n")
    print("The Chaser stands in for a player who reads flow reasonably well; the")
    print("question is how precisely a session measures its edge over Bayes-mid.\n")
    print("| setting | true gap (bayes-mid − chaser) | sd of one round's P&L | "
          "sd of the paired gap | ±95% on the gap, 10 rounds | rounds to resolve the gap |")
    print("|---|---:|---:|---:|---:|---:|")
    for key, drift in settings:
        name, cfg, res = results[(key, drift)]
        a, b = res["chaser"][1]["pnl"], res["bayes-mid"][1]["pnl"]
        d = b - a
        sd, sd_d = a.std(ddof=1), d.std(ddof=1)
        need = (1.96 * sd_d / abs(d.mean())) ** 2  # rounds for the CI to exclude 0
        print(f"| {name} | {d.mean():+.2f} | {sd:.1f} | {sd_d:.1f} | "
              f"{1.96 * sd_d / np.sqrt(10):.1f} | {need:,.0f} |")

    # ---------------------------------------------------------------- figures
    # 1. P&L per round, every bot, every setting.
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax)
    bots = list(LABELS)
    rows = [results[s][0] for s in settings]
    for j, bot in enumerate(bots):
        for i, s in enumerate(settings):
            r = results[s][2][bot][1]["pnl"]
            y = len(settings) - 1 - i + (1.5 - j) * 0.17
            ax.errorbar(r.mean(), y, xerr=ci(r), fmt="o", ms=6, color=COLORS[bot],
                        mec=SURFACE, mew=1.5, elinewidth=1.5, capsize=0,
                        label=LABELS[bot] if i == 0 else None)
    ax.set_yticks(range(len(settings)))
    ax.set_yticklabels(rows[::-1], color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, s in enumerate(settings):
        if s[0] == "3":  # static's best width on hard is so wide it never trades
            ax.text(0.4, len(settings) - 1 - i + 0.26, "static's best: quote so wide nobody trades",
                    fontsize=8, color=INK2, va="center")
    ax.set_xlabel("expected P&L per round (95% CI, test set)", color=INK2)
    ax.set_title("Updating on order flow is worth more the sharper the informed traders",
                 color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig("figures/pnl_by_bot.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. Where the edge comes from: spread earned vs adverse selection paid, per lot.
    name, cfg, res = results[("2", 0.0)]
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    style(ax)
    for i, bot in enumerate(bots):
        r = res[bot][1]
        vol = r["volume"].sum()
        sp, ad = r["spread"].sum() / vol, r["adverse"].sum() / vol
        y = len(bots) - 1 - i
        ax.barh(y, sp, height=0.55, color=COLORS[bot], alpha=0.9)
        ax.barh(y, ad, height=0.55, color=COLORS[bot], alpha=0.45)
        ax.text(sp + 0.2, y, f"spread {sp:+.2f}", va="center", fontsize=8.5, color=INK2)
        ax.text(ad - 0.2, y, f"adverse {ad:+.2f}", va="center", ha="right",
                fontsize=8.5, color=INK2)
        ax.text(19.5, y, f"net {sp + ad:+.2f}", va="center", ha="right",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(range(len(bots)))
    ax.set_yticklabels([LABELS[b] for b in bots][::-1], color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-13, 19.5)
    ax.set_xlabel("P&L per lot traded (medium, fixed value)", color=INK2)
    ax.set_title("Updating mostly shrinks the adverse-selection bill",
                 color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/spread_vs_adverse.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("\nwrote figures/pnl_by_bot.png, figures/spread_vs_adverse.png")


if __name__ == "__main__":
    main()
