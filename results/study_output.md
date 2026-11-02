## P&L per round on the test set (95% CI), and the split per lot

| setting | bot | tuned params | P&L / round | spread / lot | adverse / lot |
|---|---|---|---:|---:|---:|
| easy, fixed | Static | h=8 | +23.46 ± 0.37 | +8.00 | -2.69 |
| easy, fixed | Chaser | h=7, step=0.75 | +24.90 ± 0.31 | +7.00 | -2.22 |
| easy, fixed | Bayes-mid | h=8 | +25.03 ± 0.31 | +8.00 | -2.30 |
| easy, fixed | Bayes-myopic | none | +25.20 ± 0.32 | +7.62 | -2.22 |
| easy, drifting | Static | h=9 | +21.72 ± 0.44 | +9.00 | -3.44 |
| easy, drifting | Chaser | h=8, step=1 | +23.83 ± 0.36 | +8.00 | -2.84 |
| easy, drifting | Bayes-mid | h=8.5 | +23.88 ± 0.35 | +8.50 | -2.89 |
| easy, drifting | Bayes-myopic | none | +24.20 ± 0.37 | +8.07 | -2.79 |
| medium, fixed | Static | h=10 | +8.05 ± 0.42 | +10.00 | -7.59 |
| medium, fixed | Chaser | h=9, step=2 | +14.56 ± 0.27 | +9.00 | -5.01 |
| medium, fixed | Bayes-mid | h=8.5 | +14.88 ± 0.27 | +8.50 | -4.77 |
| medium, fixed | Bayes-myopic | none | +15.55 ± 0.28 | +8.86 | -4.71 |
| medium, drifting | Static | h=11 | +6.53 ± 0.48 | +11.00 | -8.83 |
| medium, drifting | Chaser | h=9, step=2 | +13.73 ± 0.33 | +9.00 | -5.53 |
| medium, drifting | Bayes-mid | h=9 | +13.98 ± 0.30 | +9.00 | -5.47 |
| medium, drifting | Bayes-myopic | none | +14.42 ± 0.31 | +9.66 | -5.68 |
| hard, fixed | Static | h=39.5 | +0.00 ± 0.00 | barely trades | — |
| hard, fixed | Chaser | h=10, step=2.25 | +14.51 ± 0.33 | +10.00 | -6.60 |
| hard, fixed | Bayes-mid | h=10.5 | +15.93 ± 0.32 | +10.50 | -6.36 |
| hard, fixed | Bayes-myopic | none | +15.12 ± 0.33 | +11.92 | -7.24 |
| hard, drifting | Static | h=38.5 | +0.00 ± 0.01 | barely trades | — |
| hard, drifting | Chaser | h=11, step=2.75 | +13.37 ± 0.35 | +11.00 | -7.70 |
| hard, drifting | Bayes-mid | h=10.5 | +15.10 ± 0.34 | +10.50 | -7.03 |
| hard, drifting | Bayes-myopic | none | +13.02 ± 0.36 | +13.55 | -9.11 |

## Paired gains on identical rounds (95% CI)

| setting | chaser − static | bayes-mid − chaser | bayes-myopic − bayes-mid |
|---|---:|---:|---:|
| easy, fixed | +1.44 ± 0.20 | +0.13 ± 0.16 | +0.17 ± 0.12 |
| easy, drifting | +2.10 ± 0.25 | +0.05 ± 0.15 | +0.32 ± 0.14 |
| medium, fixed | +6.51 ± 0.31 | +0.33 ± 0.15 | +0.67 ± 0.15 |
| medium, drifting | +7.19 ± 0.34 | +0.25 ± 0.15 | +0.44 ± 0.16 |
| hard, fixed | +14.51 ± 0.33 | +1.42 ± 0.24 | -0.81 ± 0.29 |
| hard, drifting | +13.36 ± 0.35 | +1.74 ± 0.25 | -2.08 ± 0.32 |

## How many rounds it takes to grade a player

The Chaser stands in for a player who reads flow reasonably well; the
question is how precisely a session measures its edge over Bayes-mid.

| setting | true gap (bayes-mid − chaser) | sd of one round's P&L | sd of the paired gap | ±95% on the gap, 10 rounds | rounds to resolve the gap |
|---|---:|---:|---:|---:|---:|
| easy, fixed | +0.13 | 22.6 | 11.7 | 7.2 | 30,059 |
| easy, drifting | +0.05 | 26.2 | 11.0 | 6.8 | 175,914 |
| medium, fixed | +0.33 | 19.8 | 10.8 | 6.7 | 4,213 |
| medium, drifting | +0.25 | 23.5 | 10.5 | 6.5 | 6,638 |
| hard, fixed | +1.42 | 23.5 | 17.2 | 10.7 | 568 |
| hard, drifting | +1.74 | 25.5 | 18.4 | 11.4 | 429 |

wrote figures/pnl_by_bot.png, figures/spread_vs_adverse.png
