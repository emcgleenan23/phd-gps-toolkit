"""
Synthetic STATSports-style exports, so every script can run without real data.

The files mimic the real export layout (same column names, drill titles,
time format and 100 Hz rows that repeat each 10 Hz GNSS sample), but every
player, value and session is invented. To keep the demo small, the match and
training files contain one row per 10 Hz sample; the repeated-sprint file keeps
the real layout of ~10 rows per sample. `gpskit.io.read_export` handles both. Nothing here is derived from real
athletes.

How the movement is simulated
-----------------------------
Each player gets an acceleration-speed line a_max(v) = A0 · (1 - v / S0).
An effort starts from the current speed and accelerates at a fraction of that
maximum (most efforts are sub-maximal, a few are near-maximal), then the
player decelerates and jogs or walks until the next effort. This produces an
A-S cloud whose upper edge sits close to the player's line, which is exactly
the structure in-situ profiling relies on.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

FS = 10.0
ROWS_PER_SAMPLE = 10  # real exports have ~10 inertial rows per GNSS sample


def _players(n, rng):
    """Invented player capacities: A0 (m/s²) and S0 (m/s)."""
    return {f"P{i:02d}": (rng.normal(7.0, 0.5), rng.normal(9.0, 0.6)) for i in range(1, n + 1)}


def _movement(duration_s, A0, S0, rng, effort_rate=1 / 45, near_max=0.15):
    """Speed trace (m/s) at 10 Hz for intermittent team-sport movement."""
    n = int(duration_s * FS)
    v = np.empty(n)
    speed, target, state = 1.5, 1.5, "cruise"
    intensity = 0.0
    for i in range(n):
        if state == "cruise" and rng.random() < effort_rate / FS:
            state = "effort"
            intensity = rng.uniform(0.85, 1.0) if rng.random() < near_max else rng.uniform(0.4, 0.85)
            target = rng.uniform(0.7, 0.97) * S0
        if state == "effort":
            speed += intensity * A0 * (1 - speed / S0) / FS
            if speed >= target:
                state, target = "slow", rng.uniform(0.5, 3.5)
        elif state == "slow":
            speed -= rng.uniform(0.8, 3.5) / FS
            if speed <= target:
                speed, state = target, "cruise"
        else:  # cruise: wander around a jogging/walking speed
            target += rng.normal(0, 0.05)
            target = np.clip(target, 0.3, 4.0)
            speed += 0.5 * (target - speed) / FS
        v[i] = max(speed, 0.0)
    return np.clip(v + rng.normal(0, 0.08, n), 0, None)  # GNSS measurement noise


def _rows(player, date, start, speed, drill, repeat=ROWS_PER_SAMPLE):
    t = [start + timedelta(seconds=k / FS) for k in range(len(speed))]
    df = pd.DataFrame({
        "Player Display Name": player, "Date": date,
        "Time": [x.strftime("%H:%M:%S.%f") for x in t],
        "Speed (m/s)": np.round(speed, 4), "Drill Title": drill,
    })
    return df.loc[df.index.repeat(repeat)]


def match_or_training(date: str, kind: str, players: dict, rng) -> pd.DataFrame:
    """One session export. Matches: 2 x 30-min halves; training: 60-75 min."""
    t0 = datetime.strptime(f"{date} 19:30", "%Y-%m-%d %H:%M")
    parts = []
    for name, (A0, S0) in players.items():
        if kind == "match":
            h1 = _movement(1800, A0, S0, rng, effort_rate=1 / 40, near_max=0.2)
            h2 = _movement(1800, A0, S0, rng, effort_rate=1 / 45, near_max=0.15)
            parts += [_rows(name, date, t0, h1, "1st Half", repeat=1),
                      _rows(name, date, t0 + timedelta(minutes=45), h2, "2nd Half", repeat=1)]
        else:
            v = _movement(rng.uniform(3600, 4500), A0, S0, rng, effort_rate=1 / 70, near_max=0.08)
            parts.append(_rows(name, date, t0, v, "Entire Session", repeat=1))
    return pd.concat(parts, ignore_index=True)


def repeated_sprint_session(date: str, players: dict, rng, distance_m: float = 50.0) -> pd.DataFrame:
    """3 x 5 x 50 m sprints, drills 'Rep 1' ... 'Rep 15', with fatigue across reps."""
    t0 = datetime.strptime(f"{date} 19:00", "%Y-%m-%d %H:%M")
    parts = []
    for name, (A0, S0) in players.items():
        start = t0
        for rep in range(1, 16):
            within_set = (rep - 1) % 5
            set_no = (rep - 1) // 5
            fatigue = 1 - 0.035 * within_set - 0.03 * set_no + rng.normal(0, 0.04)
            a0, s0 = A0 * fatigue * 0.85, S0 * (0.5 + 0.5 * fatigue) * 0.95
            v, x, trace = 0.0, 0.0, [0.0] * 5  # standing start
            while x < distance_m:
                v += a0 * (1 - v / s0) / FS
                x += v / FS
                trace.append(v)
            while v > 0.5:  # run-out after the line
                v -= 3.0 / FS
                trace.append(max(v, 0))
            trace += [0.2] * 10
            speed = np.clip(np.array(trace) + rng.normal(0, 0.05, len(trace)), 0, None)
            parts.append(_rows(name, date, start, speed, f"Rep {rep}"))
            start += timedelta(seconds=30 if within_set < 4 else 180)
    return pd.concat(parts, ignore_index=True)


def make_insitu_season(out_dir, n_players=6, n_matches=6, n_training=6, seed=1):
    """Write a small synthetic season of match and training exports."""
    from pathlib import Path
    rng = np.random.default_rng(seed)
    players = _players(n_players, rng)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    day = datetime(2024, 4, 1)
    for i in range(max(n_matches, n_training)):
        for kind, n in (("Training", n_training), ("Match", n_matches)):
            if i < n:
                date = (day + timedelta(days=7 * i + (5 if kind == "Match" else 2))).strftime("%Y-%m-%d")
                df = match_or_training(date, kind.lower(), players, rng)
                df.to_csv(out / f"{date}-DemoClub-{kind}.csv", index=False)
    return players


def make_rsa_session(out_file, n_players=8, seed=2):
    """Write one synthetic repeated-sprint test export."""
    rng = np.random.default_rng(seed)
    players = _players(n_players, rng)
    df = repeated_sprint_session("2024-05-20", players, rng)
    df.to_csv(out_file, index=False)
    return players


# ---------------------------------------------------------------------------
# Running-demands demo: three seasons of matches with positions
# ---------------------------------------------------------------------------

PITCH_LAT, PITCH_LON = 53.5, -7.5   # an arbitrary point; not a real club's pitch
PITCH_HALF_LEN, PITCH_HALF_WID = 65.0, 40.0  # metres


def _path(speed, rng):
    """Latitude/longitude for a speed trace: a heading that drifts, bending
    gently during fast running and more freely when jogging, kept on the pitch."""
    n = len(speed)
    x, y, heading = rng.uniform(-30, 30), rng.uniform(-20, 20), rng.uniform(0, 2 * np.pi)
    xs, ys = np.empty(n), np.empty(n)
    for i, v in enumerate(speed):
        turn_sd = 0.02 if v > 4.7 else 0.12
        heading += rng.normal(0, turn_sd)
        x += v * np.cos(heading) / FS
        y += v * np.sin(heading) / FS
        if abs(x) > PITCH_HALF_LEN or abs(y) > PITCH_HALF_WID:  # turn back infield
            heading += np.pi
            x, y = np.clip(x, -PITCH_HALF_LEN, PITCH_HALF_LEN), np.clip(y, -PITCH_HALF_WID, PITCH_HALF_WID)
        xs[i], ys[i] = x, y
    lat = PITCH_LAT + np.degrees(ys / 6_371_000.0)
    lon = PITCH_LON + np.degrees(xs / (6_371_000.0 * np.cos(np.radians(PITCH_LAT))))
    return lat + rng.normal(0, 1e-6, n), lon + rng.normal(0, 1e-6, n)  # ~0.1 m GNSS noise


def make_demands_seasons(out_dir, seasons=(2024, 2025, 2026), matches_per_season=3,
                         n_players=8, seed=3):
    """
    Write synthetic league matches for several seasons, plus the two small
    tables the running-demands scripts need:

    * fixtures.csv  - Date, Competition, Opposition, Venue, Opponent Points
    * positions.csv - Player, Position (inside defender / middle eight / inside forward)

    Each match has one substitute appearance (2nd half only), which the
    40-minute inclusion rule removes. Season 2 runs slightly less than season 1,
    so there is something to detect.
    """
    from pathlib import Path
    rng = np.random.default_rng(seed)
    players = _players(n_players, rng)
    lines = ["Inside defender", "Middle eight", "Middle eight", "Inside forward"]
    positions = {p: lines[i % len(lines)] for i, p in enumerate(players)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for s_idx, season in enumerate(seasons):
        work_rate = {0: 1.0, 1: 0.85, 2: 0.9}.get(s_idx, 1.0)
        for m in range(matches_per_season):
            date = f"{season}-0{4 + m}-1{m}"
            t0 = datetime.strptime(f"{date} 19:30", "%Y-%m-%d %H:%M")
            sub = list(players)[(m + s_idx) % n_players]
            parts = []
            for name, (A0, S0) in players.items():
                for half, start in (("1st Half", t0), ("2nd Half", t0 + timedelta(minutes=45))):
                    if name == sub and half == "1st Half":
                        continue
                    v = _movement(1800 + rng.uniform(0, 120), A0, S0, rng,
                                  effort_rate=work_rate / 40, near_max=0.2)
                    lat, lon = _path(v, rng)
                    df = _rows(name, date, start, v, half, repeat=1)
                    df["Lat"], df["Lon"] = np.round(lat, 7), np.round(lon, 7)
                    parts.append(df)
            pd.concat(parts, ignore_index=True).to_csv(out / f"{date}-DemoClub-Match.csv", index=False)
            fixtures.append({"Date": date, "Competition": "League", "Opposition": f"Team {chr(65 + m)}",
                             "Venue": "Home" if m % 2 == 0 else "Away",
                             "Opponent Points": int(rng.integers(4, 16))})
    pd.DataFrame(fixtures).to_csv(out / "fixtures.csv", index=False)
    pd.Series(positions, name="Position").rename_axis("Player").to_csv(out / "positions.csv")
