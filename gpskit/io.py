"""
Reading STATSports raw-data exports.

What the export looks like
--------------------------
A STATSports "raw data extended" CSV holds every player in a session, one row
per inertial-sensor sample (100 Hz). The GNSS fields (Lat, Lon, Speed) only
update at 10 Hz, so each GNSS value is repeated on ~10 consecutive rows that
share the same `Time` stamp:

    Player Display Name, Date, Time,            Lat, Lon, Speed (m/s), ..., Drill Title, Accl X, ...
    P01,                 2024-06-07, 19:30:00.100, ...,  ..., 4.21,       ..., 1st Half,    ...
    P01,                 2024-06-07, 19:30:00.100, ...,  ..., 4.21,       ..., 1st Half,    ...   <- same GNSS sample
    ...

The `Drill Title` column labels segments created in the STATSports software,
e.g. "Entire Session", "1st Half", "2nd Half", "Warm Up", "Rep 1", ...
Note that "Entire Session" overlaps every other drill, so a row can appear
once per drill it belongs to.

To work with the GNSS signal we therefore:
  1. keep the columns we need,
  2. keep the drill(s) we want,
  3. keep one row per player per timestamp (-> true 10 Hz series).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PLAYER = "Player Display Name"
DRILL = "Drill Title"
TIME = "Time"
SPEED = "Speed (m/s)"

# Columns needed for speed-based analyses. Reading only these keeps memory low:
# a full match export can be several GB.
SPEED_COLUMNS = [PLAYER, DRILL, TIME, SPEED]

MATCH_DRILLS = ["1st Half", "2nd Half"]
TRAINING_DRILL = "Entire Session"


def read_export(path, columns=SPEED_COLUMNS, drills=None, players=None,
                chunksize: int | None = 2_000_000) -> pd.DataFrame:
    """
    Read a STATSports export and return one row per player per GNSS sample.

    Parameters
    ----------
    path : str or Path
        CSV export.
    columns : list of str
        Columns to read. Must include player, time and drill columns.
    drills : str, list of str or None
        Keep only these drill titles. None keeps everything.
    players : list of str or None
        Keep only these players. None keeps everyone.
    chunksize : int or None
        Rows per chunk when reading. Large exports are read in chunks and
        filtered as they go so they never sit in memory all at once.

    Returns
    -------
    pd.DataFrame
        Columns as requested, plus `t_s`: seconds since the player's first
        sample in the returned data. Sorted by player then time.
    """
    if isinstance(drills, str):
        drills = [drills]

    def _keep(chunk: pd.DataFrame) -> pd.DataFrame:
        if drills is not None:
            chunk = chunk[chunk[DRILL].isin(drills)]
        if players is not None:
            chunk = chunk[chunk[PLAYER].isin(players)]
        # Collapse the ~10 repeated 100 Hz rows to one 10 Hz sample.
        return chunk.drop_duplicates(subset=[PLAYER, DRILL, TIME])

    reader = pd.read_csv(path, usecols=columns, low_memory=False,
                         chunksize=chunksize)
    parts = [_keep(c) for c in reader] if chunksize else [_keep(reader)]
    df = pd.concat(parts, ignore_index=True)

    # A timestamp can sit on a chunk boundary, and "Entire Session" overlaps
    # other drills; keep the first occurrence per player and time.
    df = df.drop_duplicates(subset=[PLAYER, TIME], keep="first")

    df[TIME] = pd.to_datetime(df[TIME], format="mixed")
    df = df.sort_values([PLAYER, TIME]).reset_index(drop=True)
    df["t_s"] = (df[TIME] - df.groupby(PLAYER)[TIME].transform("min")).dt.total_seconds()
    return df


def player_speed(df: pd.DataFrame, player: str):
    """
    Return (t_s, speed) arrays for one player from `read_export` output,
    dropping samples where speed is missing.
    """
    p = df[df[PLAYER] == player]
    speed = pd.to_numeric(p[SPEED], errors="coerce").to_numpy(float)
    t = p["t_s"].to_numpy(float)
    ok = ~np.isnan(speed)
    return t[ok], speed[ok]


def session_info(path) -> dict:
    """
    Parse the date and session type from a file name like
    `2024-06-07-ClubName-Match.csv` or `2024-06-11-ClubName-Training.csv`.
    """
    stem = Path(path).stem
    m = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    kind = "match" if "match" in stem.lower() else (
        "training" if "training" in stem.lower() else "unknown")
    return {"date": m.group(1) if m else None, "type": kind, "file": Path(path).name}


def split_at_largest_gap(t_s, values, min_gap_s: float = 120.0):
    """
    Split a time series into two parts at its largest time gap, e.g. the
    half-time break when 1st and 2nd half have been read together.

    Returns a list of (t_s, values) tuples, each re-zeroed to start at 0.
    If no gap is at least `min_gap_s` long, returns the whole series.
    """
    t = np.asarray(t_s, float)
    v = np.asarray(values, float)
    if len(t) < 2:
        return [(t, v)]
    gaps = np.diff(t)
    i = int(np.argmax(gaps))
    if gaps[i] < min_gap_s:
        return [(t - t[0], v)]
    cut = i + 1
    return [(t[:cut] - t[0], v[:cut]), (t[cut:] - t[cut], v[cut:])]
