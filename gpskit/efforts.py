"""
Detecting running efforts and acceleration events in a speed trace.

All functions work on ONE continuous run of 10 Hz samples (no gaps). Use
`continuous_runs` to split a half at GNSS drop-outs first, so an effort can
never bridge a gap in the data.

High-speed efforts (hysteresis)
-------------------------------
An effort starts on the first sample at or above the entry threshold and ends
when speed drops below `exit_fraction` (90 %) of that threshold. The lower exit
stops one run being counted several times when speed hovers around the
threshold (Varley et al., 2017). Efforts shorter than the minimum effort
duration (1 s; Sweeting et al., 2017) are ignored, and efforts longer than
`max_s` (20 s) are treated as artefacts.

Acceleration events
-------------------
A contiguous stretch where acceleration stays at or above +1 m/s² (or at or
below -1 m/s²) for at least 0.5 s. Each event is then put in a zone by its
peak magnitude, e.g. low 1-2, moderate 2-3, high >= 3 m/s² (Abbott et al., 2018).

Seasonal references
-------------------
`kth_highest` returns the highest value reached on at least k (default 3)
separate occasions, i.e. the k-th highest effort peak. Used for maximal
seasonal speed (MSS), acceleration (MSA) and deceleration (MSD). Taking the
3rd-highest rather than the single maximum reduces the influence of one
unusual effort or GNSS spike.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

_trapz = getattr(np, "trapezoid", None) or np.trapz
EARTH_RADIUS_M = 6_371_000.0


def continuous_runs(t_s, max_gap_s: float = 0.5):
    """Yield (start, stop) slices of `t_s` with no gap larger than `max_gap_s`."""
    t = np.asarray(t_s, float)
    if len(t) == 0:
        return
    breaks = np.flatnonzero(np.diff(t) > max_gap_s)
    start = 0
    for b in breaks:
        yield start, b + 1
        start = b + 1
    yield start, len(t)


def effort_windows(speed, entry: float, *, fs: float = 10.0, exit_fraction: float = 0.90,
                   min_s: float = 1.0, max_s: float = 20.0) -> list[tuple[int, int]]:
    """
    (first, last) sample indices (inclusive) of each effort at or above `entry`.
    """
    v = np.asarray(speed, float)
    exit_ = entry * exit_fraction
    out, i, n = [], 0, len(v)
    while i < n:
        if v[i] >= entry:
            j = i
            while j + 1 < n and v[j + 1] >= exit_:
                j += 1
            duration = (j - i + 1) / fs
            if min_s <= duration <= max_s:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def effort_summary(speed, accel, i: int, j: int, fs: float = 10.0) -> dict:
    """Duration, distance, peak speed and peak acceleration of samples i..j."""
    v, a = speed[i:j + 1], accel[i:j + 1]
    return {"duration_s": (j - i + 1) / fs, "distance_m": float(_trapz(v, dx=1 / fs)),
            "peak_speed": float(v.max()), "peak_accel": float(a.max()),
            "onset_speed": float(v[0])}


def accel_events(speed, accel, *, fs: float = 10.0, floor: float = 1.0,
                 min_s: float = 0.5) -> list[dict]:
    """
    Acceleration and deceleration events in one continuous run.

    Returns dicts with direction ('acc' / 'dec'), start/stop indices, duration,
    distance, signed peak acceleration and the speed at onset.
    """
    v, a = np.asarray(speed, float), np.asarray(accel, float)
    min_n = max(1, int(round(min_s * fs)))
    events = []
    for direction, mask in (("acc", a >= floor), ("dec", a <= -floor)):
        # Edges of each run of True values in `mask`.
        padded = np.concatenate([[False], mask, [False]]).astype(int)
        starts = np.flatnonzero(np.diff(padded) == 1)
        stops = np.flatnonzero(np.diff(padded) == -1)  # exclusive
        for s, e in zip(starts, stops):
            if e - s < min_n:
                continue
            seg = a[s:e]
            events.append({
                "direction": direction, "start": int(s), "stop": int(e),
                "duration_s": (e - s) / fs,
                "distance_m": float(v[s:e].sum() / fs),
                "peak_accel": float(seg.max() if direction == "acc" else seg.min()),
                "onset_speed": float(v[s]),
            })
    return events


def zone(value: float, edges: tuple, labels=("low", "moderate", "high")):
    """Label for |value| given ascending lower edges, e.g. (1, 2, 3); None if below edges[0]."""
    x = abs(value)
    label = None
    for edge, name in zip(edges, labels):
        if x >= edge:
            label = name
    return label


def kth_highest(values, k: int = 3):
    """k-th highest value, or nan if there are fewer than k values."""
    v = np.sort(np.asarray(values, float))[::-1]
    return float(v[k - 1]) if len(v) >= k else np.nan


def tangent_chord_angle(lat, lon, n_tangent: int = 3, min_chord_m: float = 0.5) -> float:
    """
    Unsigned angle (degrees) between an effort's initial heading and its overall
    direction (Fitzpatrick et al., 2019).

    * tangent: from the first position to the `n_tangent`-th position
    * chord:   from the first position to the last position

    0° means the player finished the effort heading the way they started
    (straight); larger angles mean a more curved path. Returns nan for paths
    too short to measure. Exploratory: it relies on a few GNSS positions.
    """
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[ok], lon[ok]
    if len(lat) < max(3, n_tangent):
        return np.nan
    # Project to local metres (equirectangular; fine over a pitch).
    x = np.deg2rad(lon - lon.mean()) * np.cos(np.deg2rad(lat.mean())) * EARTH_RADIUS_M
    y = np.deg2rad(lat - lat.mean()) * EARTH_RADIUS_M
    cx, cy = x[-1] - x[0], y[-1] - y[0]
    k = min(n_tangent, len(x)) - 1
    tx, ty = x[k] - x[0], y[k] - y[0]
    if np.hypot(cx, cy) < min_chord_m or np.hypot(tx, ty) < 1e-9:
        return np.nan
    return float(np.degrees(np.arctan2(abs(cx * ty - cy * tx), cx * tx + cy * ty)))


def peak_window_distance(speed, window_s: float, fs: float = 10.0):
    """
    Greatest distance (m) covered in any `window_s`-second window of one
    continuous run ("worst-case scenario"). None if the run is shorter than
    the window.
    """
    v = np.asarray(speed, float)
    w = int(round(window_s * fs))
    if w < 1 or len(v) <= w:
        return None
    cum = cumulative_trapezoid(v, dx=1 / fs, initial=0.0)
    return float((cum[w:] - cum[:-w]).max())
