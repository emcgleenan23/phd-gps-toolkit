"""
In-situ acceleration-speed (A-S) and force-velocity (PFv) profiling.

The idea (Morin et al., 2021)
-----------------------------
During training or a match a player rarely sprints flat out from a standstill,
but across many efforts they do produce close-to-maximal accelerations at a wide
range of speeds. If we plot every (speed, acceleration) sample and keep only the
top of that cloud, the upper edge is close to a straight line:

    a = A0 + SFv · v          (SFv < 0)

* A0  : y-intercept, the theoretical maximal acceleration at zero speed (m/s²).
        Because force per kg = acceleration, this is also the mass-normalised
        maximal force (N/kg).
* S0  : x-intercept, the theoretical maximal speed (m/s), S0 = -A0 / SFv.
* SFv : slope of the line (1/s).

The "intensity frame" (IF) points that define the upper edge are found by:
  1. keeping samples with positive acceleration,
  2. keeping speeds of at least 3 m/s,
  3. cutting the speed axis into 0.2 m/s bins and keeping the 2 highest
     accelerations in each bin.

A straight line is fitted, points more than 1.96 SD of the residuals away from it
are discarded, and the line is fitted again. That second fit is the profile.

Derived power metrics
---------------------
Power per kg = force per kg × speed = a · v. Along the fitted line:

* Pmax : peak of the power-velocity parabola. Analytically A0 · S0 / 4.
* PB   : "ballistic power", the area under the A-S line from 0 to S0.
         Analytically A0 · S0 / 2. It summarises the whole profile in one
         number (McGleenan et al., 2026a).
* dFv  : perpendicular distance from the origin to the A-S line,
         |A0| / sqrt(SFv² + 1). Another single-number summary of the profile.

Pmax and PB are evaluated numerically on a 200-point speed grid, as in the
original analysis, so that results reproduce exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filtering import DEFAULT_FS_HZ, acceleration

MIN_SPEED = 3.0      # m/s; slower samples are not near-maximal efforts
BIN_WIDTH = 0.2      # m/s
TOP_N = 2            # accelerations kept per bin
OUTLIER_SD = 1.96    # residual cut-off for the second fit

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz


@dataclass
class ASProfile:
    """Result of fitting an acceleration-speed profile."""

    A0: float
    S0: float
    SFv: float
    r2: float
    Pmax: float
    PB: float
    dFv: float
    n_points: int          # IF points used in the final fit
    n_outliers: int        # IF points removed by the 1.96 SD rule
    # The points themselves, handy for plotting.
    speed: np.ndarray = field(repr=False, default=None)
    accel: np.ndarray = field(repr=False, default=None)
    kept: np.ndarray = field(repr=False, default=None)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("A0", "S0", "SFv", "r2", "Pmax", "PB", "dFv", "n_points", "n_outliers")}


def select_if_points(speed, accel, min_speed: float = MIN_SPEED,
                     bin_width: float = BIN_WIDTH, top_n: int = TOP_N):
    """
    Pick the intensity-frame points: the `top_n` highest accelerations in each
    `bin_width` speed bin, using only speeds >= `min_speed`.

    Returns (if_speed, if_accel) arrays.
    """
    speed = np.asarray(speed, float)
    accel = np.asarray(accel, float)
    fast = speed >= min_speed
    sp, ac = speed[fast], accel[fast]
    if len(sp) == 0:
        return np.array([]), np.array([])

    edges = np.arange(min_speed, sp.max() + bin_width, bin_width)
    out_sp, out_ac = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = np.flatnonzero((sp >= lo) & (sp < hi))
        if len(in_bin) == 0:
            continue
        top = in_bin[np.argsort(ac[in_bin])[-top_n:]]  # highest accelerations
        out_sp.extend(sp[top])
        out_ac.extend(ac[top])
    return np.array(out_sp), np.array(out_ac)


def line_metrics(slope: float, intercept: float) -> dict:
    """A0, S0, Pmax, PB and dFv from the fitted A-S line a = intercept + slope · v."""
    A0, SFv = intercept, slope
    S0 = -A0 / SFv
    v = np.linspace(0, S0, 200)
    a = A0 + SFv * v
    return {
        "A0": float(A0), "S0": float(S0), "SFv": float(SFv),
        "Pmax": float(np.max(v * a)),
        "PB": float(_trapz(a, x=v)),
        "dFv": float(abs(A0) / np.sqrt(SFv ** 2 + 1)),
    }


def fit_as_profile(speed, accel, *, min_points: int = 5,
                   outlier_sd: float = OUTLIER_SD,
                   plausible_range: tuple | None = (0.0, 15.0)) -> ASProfile | None:
    """
    Fit the A-S line to (already selected) IF points with one round of
    outlier rejection.

    Parameters
    ----------
    speed, accel : array-like
        IF points (see `select_if_points`).
    min_points : int
        Minimum number of points before and after outlier removal.
    outlier_sd : float
        Points with |residual| > outlier_sd × SD(residuals) are dropped.
    plausible_range : (low, high) or None
        Reject fits whose A0 or S0 falls outside this open interval
        (a guard against nonsense fits from sparse data). None disables it.

    Returns
    -------
    ASProfile, or None if the fit is not valid (too few points, or a
    non-negative slope, which has no physical meaning here).
    """
    sp = np.asarray(speed, float)
    ac = np.asarray(accel, float)
    if len(sp) < min_points:
        return None

    # First fit, then drop points far from the line.
    slope, intercept = np.polyfit(sp, ac, 1)
    resid = ac - (slope * sp + intercept)
    kept = np.abs(resid) <= outlier_sd * np.std(resid)
    if kept.sum() < min_points:
        return None

    # Second (final) fit on the kept points.
    slope, intercept = np.polyfit(sp[kept], ac[kept], 1)
    if slope >= 0:
        return None

    m = line_metrics(slope, intercept)
    if plausible_range is not None:
        lo, hi = plausible_range
        if not (lo < m["A0"] < hi and lo < m["S0"] < hi):
            return None

    pred = slope * sp[kept] + intercept
    ss_res = np.sum((ac[kept] - pred) ** 2)
    ss_tot = np.sum((ac[kept] - ac[kept].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return ASProfile(**m, r2=float(r2), n_points=int(kept.sum()),
                     n_outliers=int((~kept).sum()), speed=sp, accel=ac, kept=kept)


def session_if_points(speed_filtered, fs: float = DEFAULT_FS_HZ):
    """
    Filtered speed -> acceleration -> positive-acceleration samples -> IF points.
    This is the per-session step that is repeated for every training or match.
    """
    s = np.asarray(speed_filtered, float)
    a = acceleration(s, fs)
    pos = a > 0
    return select_if_points(s[pos], a[pos])


def profile_from_speed(speed_filtered, fs: float = DEFAULT_FS_HZ,
                       **fit_kwargs) -> ASProfile | None:
    """Fit an A-S profile directly from one continuous filtered speed trace."""
    sp, ac = session_if_points(speed_filtered, fs)
    return fit_as_profile(sp, ac, **fit_kwargs)


def pooled_profile(sessions_if, rebin: bool = True, **fit_kwargs) -> ASProfile | None:
    """
    Fit one profile from several sessions.

    Parameters
    ----------
    sessions_if : list of (if_speed, if_accel) tuples
        IF points from each session (see `session_if_points`).
    rebin : bool
        If True, the pooled points are put through `select_if_points` again so
        only the top 2 per speed bin *across all sessions* remain (the "best
        case" envelope, about 50 points). If False, every session's IF points
        are used (several hundred points).

        The in-situ PFv paper used rebin=True for its Tables 1-2 and
        rebin=False for the in-situ vs isolated comparison (Table 3).
    """
    if not sessions_if:
        return None
    sp = np.concatenate([s for s, _ in sessions_if])
    ac = np.concatenate([a for _, a in sessions_if])
    if rebin:
        sp, ac = select_if_points(sp, ac)
    return fit_as_profile(sp, ac, **fit_kwargs)
