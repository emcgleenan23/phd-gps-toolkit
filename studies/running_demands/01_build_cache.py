"""
Step 1 - read each match export once and cache the cleaned 10 Hz traces.

For every *Match*.csv in the raw folder whose date is a league fixture:
  * keep the "1st Half" and "2nd Half" drills (no warm-up or half-time),
  * keep one row per player per timestamp (true 10 Hz),
  * crop each half to its first 30 minutes (regulation time),
  * split each half at GNSS drop-outs and filter each piece separately
    (1 Hz, 2nd-order, zero-phase Butterworth),
  * compute acceleration within each piece.

Usage:  python 01_build_cache.py [--raw /path/to/exports] [--anonymise]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common
from gpskit.efforts import continuous_runs
from gpskit.filtering import acceleration, butterworth
from gpskit.io import DRILL, MATCH_DRILLS, PLAYER, SPEED, TIME, read_export, session_info


def is_league(date: str, fixtures: pd.DataFrame) -> bool:
    row = fixtures[fixtures["Date"] == date]
    if row.empty:
        print(f"  [skip] {date}: not in fixtures.csv")
        return False
    comp = str(row["Competition"].iloc[0]).lower()
    return any(k in comp for k in common.COMPETITION_KEYWORDS)


def clean_half(g: pd.DataFrame) -> pd.DataFrame:
    """Crop to 30 min, label continuous runs, filter and differentiate within each run."""
    g = g.sort_values(TIME)
    t = (g[TIME] - g[TIME].iloc[0]).dt.total_seconds().to_numpy()
    g = g[t <= common.HALF_MINUTES * 60].copy()
    g["t"] = t[t <= common.HALF_MINUTES * 60]
    raw = pd.to_numeric(g[SPEED], errors="coerce").fillna(0).to_numpy()
    speed, accel, run = np.empty_like(raw), np.empty_like(raw), np.empty(len(raw), int)
    for k, (a, b) in enumerate(continuous_runs(g["t"].to_numpy(), common.MAX_GAP_S)):
        speed[a:b] = np.clip(butterworth(raw[a:b], common.FS, common.FILTER.cutoff_hz,
                                         common.FILTER.order), 0, None)
        accel[a:b] = acceleration(speed[a:b], common.FS) if b - a > 1 else 0.0
        run[a:b] = k
    g["speed_raw"], g["speed"], g["accel"], g["run"] = raw, speed, accel, run
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=common.RAW_DIR)
    ap.add_argument("--anonymise", action="store_true",
                    help="replace player names with P01, P02, ... (key saved to cache/name_key.csv)")
    args = ap.parse_args()

    fixtures = pd.read_csv(common.FIXTURES_CSV, dtype={"Date": str})
    files = sorted(f for f in args.raw.rglob("*.csv") if session_info(f)["type"] == "match")
    common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = {}
    for f in files:
        date = session_info(f)["date"]
        if not is_league(date, fixtures):
            continue
        df = read_export(f, columns=[PLAYER, DRILL, TIME, SPEED, "Lat", "Lon"], drills=MATCH_DRILLS)
        halves = []
        for (player, half), g in df.groupby([PLAYER, DRILL]):
            if len(g) < 10:
                continue
            h = clean_half(g)
            if args.anonymise:
                player = key.setdefault(player, f"P{len(key) + 1:02d}")
            halves.append(pd.DataFrame({
                "date": date, "season": int(date[:4]), "player": player,
                "half": 1 if half.startswith("1") else 2, "run": h["run"], "t": h["t"],
                "speed_raw": h["speed_raw"], "speed": h["speed"], "accel": h["accel"],
                "lat": h["Lat"].to_numpy(), "lon": h["Lon"].to_numpy()}))
        pd.concat(halves, ignore_index=True).to_pickle(common.CACHE_DIR / f"{date}.pkl.gz")
        print(f"  {f.name}: {df[PLAYER].nunique()} players")
    if args.anonymise:
        pd.Series(key, name="pseudonym").rename_axis("player").to_csv(common.CACHE_DIR / "name_key.csv")


if __name__ == "__main__":
    main()
