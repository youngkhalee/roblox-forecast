#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for make.fun
(roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson + /input/task.json and writes
/output/forecast.json: for each of places 1..10 a calibrated probability
for every plausible game, plus a top-10 guess in order.

Approach
--------
The scored object is ONE distribution judged against every hourly chart
of the 72 hours after the history ends, so the forecast must be the
*time-averaged* occupancy distribution over that window, not a snapshot.
The persistence baseline (bet the chart never moves) scores badly on the
practice window because the chart churns: consecutive archive reads
~6.4 h apart already differ by ~1.3 top-10 places on average, and only
about half of the closing top 10 is still there 72 h later.

Model, per game, from the history alone (stdlib only, deterministic):

* Work in hours, not read index: archive reads are ~4-11 h apart while
  the house recorder reads hourly, so every statistic is weighted by
  exp decay in time (half-life HALF_LIFE_H) toward the history's end.
* mu0, the game's expected rank at the window's start: a blend of its
  rank in the last read and its recency-weighted mean rank when charted,
  pulled toward the mid-chart when the game is only intermittently
  charted (low weighted presence).
* drift: a clipped recency-weighted rank slope (ranks/hour), plus the
  game's log player-count slope folded in (the chart is sorted by user
  growth, so compounding player counts lead rank climbs). Climbers
  keep climbing for a while; fallers keep falling.
* Mean reversion: over the window a game's expected rank relaxes toward
  REVERT_TO with time constant REVERT_TAU_H (yesterday's #1 is usually
  not tomorrow's).
* Spread: at horizon h the rank is Normal(mu(h), sigma(h)) with
  sigma(h) = SIGMA0 + SIGMA_SQRT * sqrt(h); the place probabilities are
  the normal mass of each unit rank bin, averaged over six 12-hour
  horizon slices spanning the window.
* Ageing out: the chart only lists games published in the last 28 days.
  When the history carries games/v1 details, a game whose 28th day
  falls inside the window contributes nothing to slices after that day.
  A like-ratio from the chart's votes (recorder reads only) nudges mu0
  slightly: heavily disliked games fade faster.
* Newcomers: games that enter the chart during the window cannot be
  named, so every place deliberately leaves RESERVE of its mass unnamed;
  the scorer spreads that share over exactly those unnamed games.
  Per-place rows are normalised to sum to 1 - RESERVE and rows below
  the implied unnamed share are dropped (naming them would score worse
  than leaving them unnamed).

Two runs on the same history write byte-identical output: no randomness,
no wall clock, sorted serialisation.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# --- Tunables (chosen by rolling-origin backtests on the kit history) ---
# Backtest protocol: origins at archive reads 13..30, forecast scored with
# the competition's log rule over the reads of the following 72 h. On the
# 13 origins with >= 8 future reads the persistence baseline averages
# -50.51; this configuration averages about -40.2.
HALF_LIFE_H = 16.0        # recency weighting half-life, hours
REVERT_TO = 16.0          # rank the expected rank relaxes toward
REVERT_TAU_H = 55.0       # mean-reversion time constant, hours
SIGMA0 = 2.5              # rank spread at horizon 0
SIGMA_SQRT = 1.0          # extra spread per sqrt(hour)
DRIFT_CLIP = 0.09         # max |rank drift|, ranks per hour
LAST_BLEND = 0.7          # weight of last-read rank inside mu0
RESERVE = 0.20            # per-place mass left unnamed (newcomers)
MIN_ROW_P = 0.004         # drop rows below ~the unnamed share
CHART_SPAN = 100.0        # assumed chart length for absence handling
MAX_PLACE_ROWS = 100      # format cap per place
PLAY_W = 2.0              # rank drift per unit log-playing growth / hour
PLAY_CLIP = 0.05          # max |log playing slope| used, per hour

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


def parse_time(text):
    """ISO 8601 (with Z) -> aware datetime in UTC."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: r["at"])
    return task, reads


def game_stats(reads, history_until):
    """Per-game time-weighted statistics from every read.

    Returns {universe_id: dict} with the weighted mean rank when charted,
    weighted presence, last rank, rank slope, like ratio and the hour at
    which the game ages off the chart (None when unknown).
    """
    total_w = 0.0
    acc = {}
    for read in reads:
        age_h = (history_until - parse_time(read["at"])).total_seconds() / 3600.0
        w = 0.5 ** (max(age_h, 0.0) / HALF_LIFE_H)
        total_w += w
        t_h = -age_h  # hours relative to history end (<= 0)
        details = read.get("details") or {}
        for row in read.get("up_and_coming") or []:
            uid = row["universe_id"]
            g = acc.setdefault(uid, {
                "w": 0.0, "w_rank": 0.0, "w_t": 0.0, "w_tt": 0.0,
                "w_trank": 0.0, "last_rank": None, "last_age": None,
                "exit_h": None, "like": None, "like_w": 0.0,
                "pw": 0.0, "p_wt": 0.0, "p_wtt": 0.0, "p_wl": 0.0, "p_wtl": 0.0,
            })
            rank = float(row["rank"])
            g["w"] += w
            g["w_rank"] += w * rank
            g["w_t"] += w * t_h
            g["w_tt"] += w * t_h * t_h
            g["w_trank"] += w * t_h * rank
            playing = row.get("playing")
            if playing is not None and playing > 0:
                lp = math.log(float(playing))
                g["pw"] += w
                g["p_wt"] += w * t_h
                g["p_wtt"] += w * t_h * t_h
                g["p_wl"] += w * lp
                g["p_wtl"] += w * t_h * lp
            if g["last_age"] is None or age_h < g["last_age"]:
                g["last_age"] = age_h
                g["last_rank"] = rank
            up = row.get("up_votes")
            down = row.get("down_votes")
            if up is not None and down is not None and up + down >= 50:
                g["like"] = up / float(up + down)
                g["like_w"] += w
            det = details.get(str(uid)) if isinstance(details, dict) else None
            if det and det.get("created"):
                created = parse_time(det["created"])
                exit_at = created.timestamp() + 28.0 * 86400.0
                g["exit_h"] = (exit_at - history_until.timestamp()) / 3600.0
    stats = {}
    for uid, g in acc.items():
        if g["w"] <= 0:
            continue
        mean_rank = g["w_rank"] / g["w"]
        # Weighted least-squares slope of rank against time (ranks/hour).
        denom = g["w"] * g["w_tt"] - g["w_t"] * g["w_t"]
        if denom > 1e-9 and g["w"] > 0.15 * total_w:
            slope = (g["w"] * g["w_trank"] - g["w_t"] * g["w_rank"]) / denom
        else:
            slope = 0.0
        slope = max(-DRIFT_CLIP, min(DRIFT_CLIP, slope))
        # The chart is sorted by user growth, so a game whose player count
        # is compounding tends to climb even before its rank slope shows
        # it: fold the log-playing slope into the drift (growth lowers the
        # rank number). Gated on enough weighted playing observations.
        if g["pw"] > 0.15 * total_w:
            pden = g["pw"] * g["p_wtt"] - g["p_wt"] * g["p_wt"]
            if pden > 1e-9:
                gslope = (g["pw"] * g["p_wtl"] - g["p_wt"] * g["p_wl"]) / pden
                gslope = max(-PLAY_CLIP, min(PLAY_CLIP, gslope))
                slope = max(-DRIFT_CLIP, min(DRIFT_CLIP, slope - PLAY_W * gslope))
        stats[uid] = {
            "mean_rank": mean_rank,
            "presence": g["w"] / total_w if total_w else 0.0,
            "last_rank": g["last_rank"],
            "slope": slope,
            "exit_h": g["exit_h"],
            "like": g["like"],
        }
    return stats


def expected_rank_start(st):
    """mu0: expected rank at the window's first hour."""
    if st["last_rank"] is not None:
        mu = LAST_BLEND * st["last_rank"] + (1.0 - LAST_BLEND) * st["mean_rank"]
    else:
        mu = st["mean_rank"]
    # Intermittent games are probably worse than their charted mean:
    # on the reads they missed they sat below the chart entirely.
    mu += (1.0 - st["presence"]) * 0.35 * max(0.0, CHART_SPAN - mu) * 0.25
    if st["like"] is not None:
        # 0.95 liked ~ -1 rank; 0.70 liked ~ +1.5 ranks (fade faster).
        mu -= max(-1.5, min(1.0, (st["like"] - 0.88) * 8.0))
    return max(1.0, mu)


def place_probs(mu0, drift, exit_h, horizon_h, places):
    """Time-averaged P(at each place 1..places) over the window."""
    out = [0.0] * places
    slices = 6
    used = 0.0
    for s in range(slices):
        h = horizon_h * (s + 0.5) / slices
        if exit_h is not None and h > exit_h:
            continue  # aged off the chart before this slice
        used += 1.0
        mu = REVERT_TO + (mu0 - REVERT_TO) * math.exp(-h / REVERT_TAU_H)
        mu += drift * h * math.exp(-h / (REVERT_TAU_H * 0.7))
        sigma = SIGMA0 + SIGMA_SQRT * math.sqrt(h)
        prev = phi((0.5 - mu) / sigma)
        for k in range(1, places + 1):
            edge = phi((k + 0.5 - mu) / sigma)
            out[k - 1] += edge - prev
            prev = edge
    if used == 0:
        return out
    return [v / slices for v in out]  # dead slices give their mass to nobody


def main():
    task, reads = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    history_until = parse_time(task["history_until"])
    window = task.get("window") or {}
    if window.get("from") and window.get("until"):
        horizon_h = (parse_time(window["until"]) - parse_time(window["from"])).total_seconds() / 3600.0
    else:
        horizon_h = 72.0
    horizon_h = max(1.0, horizon_h)

    stats = game_stats(reads, history_until)

    raw = {}  # uid -> [p per place]
    for uid, st in stats.items():
        mu0 = expected_rank_start(st)
        raw[uid] = place_probs(mu0, st["slope"], st["exit_h"], horizon_h, places)

    # Per-place normalisation to 1 - RESERVE (the rest stays unnamed).
    rows = []
    for k in range(places):
        col = [(uid, probs[k]) for uid, probs in raw.items() if probs[k] > 0.0]
        total = sum(p for _, p in col)
        if total <= 0:
            continue
        scale = (1.0 - RESERVE) / total
        col = [(uid, p * scale) for uid, p in col if p * scale >= MIN_ROW_P]
        col.sort(key=lambda t: (-t[1], t[0]))
        for uid, p in col[:MAX_PLACE_ROWS]:
            rows.append({"universe_id": int(uid), "place": k + 1, "p": round(p, 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # Top 10 guess: agreement-style score, expected places weighted most.
    def guess_score(uid):
        probs = raw[uid]
        return sum(p * (places + 1 - k) for k, p in enumerate(probs, start=1))

    ranked = sorted(raw.keys(), key=lambda u: (-guess_score(u), u))
    top10 = [int(u) for u in ranked[:places]]

    forecast = {"probabilities": rows, "top10": top10}
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
