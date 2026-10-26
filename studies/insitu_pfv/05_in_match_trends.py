"""
Step 5 - how do a player's PFv metrics change during a match?
Reproduces Table 5 and Figure 4 of the in-situ PFv paper.

Method
------
For every player in every match:
  1. Split the trace into 1st and 2nd half at the largest time gap
     (half-time). Halves shorter than 10 min are dropped.
  2. Fit one profile to the whole game (both halves). This is the reference
     each window is expressed against (100 % = the player's full-game value).
  3. Slide a window (7 or 12 min) through each half in 10-s steps and fit a
     profile in every window that contains enough fast efforts. Each half
     is stretched or squeezed onto a nominal 30-min axis so halves of slightly
     different length line up.
  4. Change across a half = last window - first window (in % of full game).
     Change across the game = last window of H2 - first window of H1.

Table 5 reports the mean change across player-matches with a t-based 95 % CI.

Note: on a straight A-S line, PB = A0·S0/2 and Pmax = A0·S0/4, so PB is
exactly twice Pmax. Expressed as % of the full-game value they are therefore
identical, and so are their changes.

Outputs: outputs/in_match_windows.csv, outputs/in_match_changes.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import common
from gpskit.io import split_at_largest_gap
from gpskit.profile import profile_from_speed

METRICS = ["A0", "S0", "SFv", "Pmax", "PB", "dFv"]


def fit(speed):
    """Profile for one window; same rules as the original in-game analysis."""
    p = profile_from_speed(speed, common.FS, plausible_range=None)
    return p if (p is not None and p.S0 > 0) else None


def halves_of(session, min_half_s=600.0):
    """[(half_number, t, speed), ...] for halves of at least 10 min."""
    if len(session["t"]) < 200:
        return []
    parts = split_at_largest_gap(session["t"], session["speed"], min_gap_s=120.0)
    out = []
    for i, (t, v) in enumerate(parts, start=1):
        if len(t) >= 100 and t[-1] - t[0] >= min_half_s:
            out.append((i, t, v))
    return out


def sliding_windows(t, speed, window_min, step_s):
    """Fit a profile in each window position across one half."""
    duration = t[-1] - t[0]
    scale = duration / (common.HALF_LENGTH_MIN * 60)  # real seconds per nominal second
    width, step = window_min * 60 * scale, step_s * scale
    rows, start = [], 0.0
    while start + width <= duration + 0.1:
        inside = (t >= start - 0.01) & (t <= start + width + 0.01)
        if inside.sum() >= 50:
            p = fit(speed[inside])
            if p is not None:
                mid = (start + width / 2) / duration * common.HALF_LENGTH_MIN
                rows.append({"mid_min": mid, **{m: getattr(p, m) for m in METRICS}})
        start += step
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[7, 12], help="window widths (min)")
    ap.add_argument("--step", type=float, default=10, help="window step (s)")
    args = ap.parse_args()

    windows = []
    for player, sessions in common.iter_players():
        for s in (x for x in sessions if x["type"] == "match"):
            halves = halves_of(s)
            if not halves:
                continue
            ref = fit(np.concatenate([v for _, _, v in halves]))
            if ref is None:
                continue
            for w in args.windows:
                for half, t, v in halves:
                    for r in sliding_windows(t, v, w, args.step):
                        pct = {f"{m}_pct": 100 * r[m] / getattr(ref, m) for m in METRICS}
                        windows.append({"player": player, "date": s["date"], "window_min": w,
                                        "half": half, **r, **pct})
    windows = pd.DataFrame(windows)
    windows.to_csv(common.output_path("in_match_windows.csv"), index=False, float_format="%.4f")

    # Change from first to last window, per player-match.
    changes = []
    for (player, date, w), g in windows.groupby(["player", "date", "window_min"]):
        halves = {h: gh for h, gh in g.groupby("half") if len(gh) >= 5}
        if not halves:
            continue
        row = {"player": player, "date": date, "window_min": w}
        for m in METRICS:
            col = f"{m}_pct"
            for h, gh in halves.items():
                row[f"{m}_H{h}"] = gh[col].iloc[-1] - gh[col].iloc[0]
            if 1 in halves and 2 in halves:
                row[f"{m}_game"] = halves[2][col].iloc[-1] - halves[1][col].iloc[0]
        changes.append(row)
    changes = pd.DataFrame(changes)
    changes.to_csv(common.output_path("in_match_changes.csv"), index=False, float_format="%.4f")

    for w, g in changes.groupby("window_min"):
        print(f"\n{w}-min window (n = {len(g)} player-matches): change, % of full-game value")
        print(f"{'':6s} {'1st half':>24s} {'2nd half':>24s} {'full game':>24s}")
        for m in METRICS:
            cells = []
            for period in ("H1", "H2", "game"):
                v = g[f"{m}_{period}"].dropna()
                half = stats.t.ppf(0.975, len(v) - 1) * v.std() / np.sqrt(len(v))
                cells.append(f"{v.mean():+5.1f} [{v.mean() - half:+5.1f}, {v.mean() + half:+5.1f}]")
            print(f"{m:6s} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
