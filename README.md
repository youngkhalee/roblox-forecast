# roblox-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/youngkhalee/roblox-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/youngkhalee/roblox-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01)

A forecaster for Roblox's Up-and-Coming chart. It reads the chart's
history (`history.ndjson` + `task.json`) and writes one forecast —
for each of places 1–10, a calibrated probability for every plausible
game, plus a top-10 guess — scored against every hourly chart of the
72 hours after the history ends.

## How it works

The forecast is a *time-averaged occupancy distribution* over the whole
72-hour window, not a snapshot: the persistence assumption ("the chart
will not move") decays within hours on this chart. Per game, from the
history alone:

- statistics are weighted by exponential decay in **hours** (archive
  reads are hours apart, recorder reads hourly), half-life 16 h;
- expected starting rank blends the last read's rank with the
  recency-weighted mean rank, penalised for intermittent presence;
- drift combines the rank slope with the log player-count slope (the
  chart is sorted by user growth);
- over the window the expected rank mean-reverts toward the mid-chart
  while the rank spread grows like sqrt(horizon); place probabilities
  are normal rank-bin masses averaged over six horizon slices;
- games whose 28th day since creation falls inside the window (from
  games/v1 details, when the history carries them) contribute nothing
  after that day — the chart only lists games published in the last
  28 days;
- every place leaves 20% of its mass unnamed, which the scorer spreads
  over chart games the forecast does not name — that is what covers
  games that enter the chart during the window.

Tuning was by rolling-origin backtest on the kit history with the
competition's own log-score rule: on origins with at least eight
future reads this model averages about −40.2 against the persistence
baseline's −50.5.

Pure Python 3 standard library, no network, deterministic: two runs on
the same history write byte-identical output.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json
```

Without the environment variables it reads `/input` and writes
`/output`, as in the grading sandbox.

---

<sub>Entered on [make.fun](https://make.fun/bounty/roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
