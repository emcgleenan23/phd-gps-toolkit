"""
Step 1 - read raw STATSports exports once and cache each player's filtered speed.

For every export file in the raw-data folder:
  * Match files     (*Match*.csv)    -> keep the "1st Half" and "2nd Half" drills,
                                        so warm-up and the half-time break are excluded.
  * Training files  (*Training*.csv) -> keep the "Entire Session" drill.
  * Collapse the 100 Hz rows to one 10 Hz sample per timestamp.
  * Filter speed (1 Hz, 2nd-order, zero-phase Butterworth).

The two halves of a match are filtered as one trace, as in the original analysis.
The half-time gap is short relative to a 30-min half, so this only affects the
few samples either side of the join.

Usage
-----
    python 01_build_cache.py                          # demo data
    python 01_build_cache.py --raw /path/to/exports   # your own data
    python 01_build_cache.py --raw ... --anonymise    # replace names with P01, P02, ...
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

import common
from gpskit.filtering import butterworth
from gpskit.io import MATCH_DRILLS, PLAYER, TRAINING_DRILL, player_speed, read_export, session_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=common.RAW_DATA_DIR, help="folder of raw exports")
    ap.add_argument("--cache", type=Path, default=common.CACHE_DIR, help="where to write the cache")
    ap.add_argument("--anonymise", action="store_true",
                    help="store players as P01, P02, ... and write the key to name_key.csv")
    args = ap.parse_args()

    files = sorted(f for f in args.raw.rglob("*.csv")
                   if not f.name.startswith("._") and session_info(f)["type"] != "unknown")
    if not files:
        raise SystemExit(f"No *Match*.csv or *Training*.csv files found in {args.raw}")
    print(f"{len(files)} export files in {args.raw}")

    by_player = defaultdict(list)
    for f in files:
        info = session_info(f)
        drills = MATCH_DRILLS if info["type"] == "match" else TRAINING_DRILL
        df = read_export(f, drills=drills)
        for player in df[PLAYER].unique():
            t, speed = player_speed(df, player)
            if len(speed) < common.MIN_SAMPLES:
                continue
            by_player[player].append({
                "date": info["date"], "type": info["type"], "file": info["file"],
                "duration_s": float(t[-1] - t[0]), "t": t - t[0],
                "speed": butterworth(speed, common.FS, common.FILTER.cutoff_hz, common.FILTER.order),
            })
        print(f"  {f.name}: {df[PLAYER].nunique()} players")

    names = sorted(by_player)
    key = {n: (f"P{i:02d}" if args.anonymise else n) for i, n in enumerate(names, 1)}
    args.cache.mkdir(parents=True, exist_ok=True)
    for name in names:
        common.save_player(args.cache / f"{key[name].replace(' ', '_')}.npz", by_player[name])
    if args.anonymise:
        # Keep this file private: it links pseudonyms back to real players.
        pd.Series(key, name="pseudonym").rename_axis("player").to_csv(args.cache / "name_key.csv")
    print(f"Cached {len(names)} players -> {args.cache}")


if __name__ == "__main__":
    main()
