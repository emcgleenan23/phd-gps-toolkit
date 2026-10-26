"""
Settings and the session cache shared by the in-situ PFv scripts.

Why a cache?
------------
A season of raw STATSports exports is tens of GB (one match file is ~3 GB).
Every analysis in this study only needs each player's *filtered speed trace*
for each session, which is a few MB per player. `01_build_cache.py` reads
the raw files once and saves those traces; every other script reads the cache
and runs in seconds to minutes.

Cache format: one compressed NumPy file per player (`<player>.npz`) with one
entry per session:

    dates           'YYYY-MM-DD'
    session_types   'match' or 'training'
    filenames       source export
    durations_s     elapsed time from first to last sample (s)
    times           seconds since the session's first sample
    speed_filtered  Butterworth-filtered speed (m/s)

(Caches written by the original analysis code call `speed_filtered`
`raw_speeds` and `times` `raw_times`; `load_player` reads both spellings.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Make `gpskit` importable when running a script directly from this folder.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gpskit.filtering import DEFAULT_FS_HZ, STUDY_FILTERS  # noqa: E402

STUDY_DIR = Path(__file__).resolve().parent

# Where things live. Override with environment variables or CLI flags.
RAW_DATA_DIR = Path(os.environ.get("PFV_RAW_DIR", REPO_ROOT / "demo_data" / "insitu_pfv"))
CACHE_DIR = Path(os.environ.get("PFV_CACHE_DIR", STUDY_DIR / "cache"))
OUTPUT_DIR = Path(os.environ.get("PFV_OUTPUT_DIR", STUDY_DIR / "outputs"))

FS = DEFAULT_FS_HZ                       # 10 Hz
FILTER = STUDY_FILTERS["insitu_pfv"]     # 1 Hz cut-off, 2nd order, zero-phase
DT = 1.0 / FS

MIN_SESSIONS = 5           # sessions of a type needed for a pooled profile
MIN_SAMPLES = 100          # ~10 s at 10 Hz; shorter sessions are skipped
HALF_LENGTH_MIN = 30.0     # Gaelic football (club): 2 x 30 min


def load_player(path) -> list[dict]:
    """Read one player's cache file into a list of session dicts, sorted by date."""
    d = np.load(path, allow_pickle=True)
    speeds = d["speed_filtered"] if "speed_filtered" in d.files else d["raw_speeds"]
    times = d["times"] if "times" in d.files else d["raw_times"]
    sessions = []
    for i in range(len(d["dates"])):
        kind = str(d["session_types"][i])
        sessions.append({
            "date": str(d["dates"][i]),
            # old caches used 'match_h1'/'match_h2' for halves
            "type": "match" if kind.startswith("match") else kind,
            "file": str(d["filenames"][i]),
            "duration_s": float(d["durations_s"][i]),
            "t": np.asarray(times[i], float),
            "speed": np.asarray(speeds[i], float),
        })
    sessions.sort(key=lambda s: s["date"])
    return sessions


def save_player(path, sessions: list[dict]) -> None:
    """Write a player's sessions in the cache format described above."""
    obj = lambda xs: np.array(xs + [None], dtype=object)[:-1]  # keep ragged arrays
    np.savez_compressed(
        path,
        dates=np.array([s["date"] for s in sessions]),
        session_types=np.array([s["type"] for s in sessions]),
        filenames=np.array([s["file"] for s in sessions]),
        durations_s=np.array([s["duration_s"] for s in sessions]),
        times=obj([s["t"].astype(np.float32) for s in sessions]),
        speed_filtered=obj([s["speed"].astype(np.float32) for s in sessions]),
    )


def iter_players(cache_dir=None):
    """Yield (player_name, sessions) for every cache file, alphabetically."""
    for f in sorted(Path(cache_dir or CACHE_DIR).glob("*.npz")):
        yield f.stem.replace("_", " "), load_player(f)


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name
