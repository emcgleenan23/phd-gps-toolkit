"""
Settings shared by the running-demands scripts.

Every threshold below is the one described in the manuscript's methods. Change
them here, not inside the scripts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gpskit.filtering import DEFAULT_FS_HZ, STUDY_FILTERS  # noqa: E402

STUDY_DIR = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("DEMANDS_RAW_DIR", REPO_ROOT / "demo_data" / "running_demands"))
CACHE_DIR = Path(os.environ.get("DEMANDS_CACHE_DIR", STUDY_DIR / "cache"))
OUTPUT_DIR = Path(os.environ.get("DEMANDS_OUTPUT_DIR", STUDY_DIR / "outputs"))

# Small tables you provide (templates in the demo data folder):
#   fixtures.csv  : Date, Competition, Opposition, Venue (Home/Away), Opponent Points
#   positions.csv : Player, Position
FIXTURES_CSV = Path(os.environ.get("DEMANDS_FIXTURES", RAW_DIR / "fixtures.csv"))
POSITIONS_CSV = Path(os.environ.get("DEMANDS_POSITIONS", RAW_DIR / "positions.csv"))

# ---- Signal processing ------------------------------------------------------
FS = DEFAULT_FS_HZ                         # 10 Hz
FILTER = STUDY_FILTERS["running_demands"]  # 1 Hz, 2nd order, zero-phase
MAX_GAP_S = 0.5                            # split the trace at GNSS drop-outs longer than this

# ---- Match and player inclusion ---------------------------------------------
HALF_MINUTES = 30.0          # each half cropped to regulation time
MIN_MATCH_MINUTES = 40.0     # player-match kept only if >= 40 of 60 min played
MIN_GAMES_PER_SEASON = 3     # player-season kept only with >= 3 qualifying matches
COMPETITION_KEYWORDS = ["league"]   # fixtures kept if Competition contains one of these

# ---- Speed bands (entry threshold; effort ends below 90 % of it) ------------
ABSOLUTE_BANDS = {"HSR": 4.72, "Sprint": 6.11, "Near-maximal": 7.00}      # m/s
NORMALISED_BANDS = {"HSR": 0.75, "Sprint": 0.85, "Near-maximal": 0.90}    # x MSS
SPEED_EXPOSURE = {"Speed 95": 0.95}        # extra count of efforts above 95 % MSS
EXIT_FRACTION = 0.90
MIN_EFFORT_S = 1.0
MAX_EFFORT_S = 20.0          # longer "efforts" are treated as artefacts

# ---- Maximal seasonal speed (MSS) -------------------------------------------
MSS_K = 3                    # highest speed reached on >= 3 separate efforts
MSS_DETECT = 4.72            # efforts used to find peaks (m/s)
MSS_PLAUSIBLE = 10.5         # peaks above this are GNSS artefacts (m/s)
MSS_MIN_HALVES = 4           # fewer halves -> fall back to the single highest peak (flagged)

# ---- Acceleration events ----------------------------------------------------
ACCEL_FLOOR = 1.0            # m/s², event detection floor
ACCEL_MIN_S = 0.5
ACCEL_ABS_EDGES = (1.0, 2.0, 3.0)      # low / moderate / high (Abbott et al., 2018)
ACCEL_REL_EDGES = (0.25, 0.50, 0.75)   # x MSA or MSD (Sonderegger et al., 2016)
SPEED_CONTEXT_MS = 4.72                # onset speed below / at-or-above HSR

# ---- Curvilinear ------------------------------------------------------------
TANGENT_POINTS = 3
ANGLE_BINS = [0, 5, 10, 15, 180]       # degrees

# ---- Worst-case scenario windows -------------------------------------------
WCS_WINDOWS_MIN = [1, 2, 3, 5, 10]


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def load_cache() -> pd.DataFrame:
    """All cached matches as one long table (one row per player per 10 Hz sample)."""
    files = sorted(CACHE_DIR.glob("*.pkl.gz"))
    if not files:
        raise SystemExit(f"No cache in {CACHE_DIR}. Run 01_build_cache.py first.")
    return pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
