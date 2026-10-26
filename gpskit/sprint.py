"""
Profiling a single, isolated maximal sprint (e.g. one 50 m repetition).

Used by the ballistic-power study (McGleenan et al., 2026a). For one sprint:

  1. Smooth speed (2 Hz, 2nd-order, zero-phase Butterworth).
  2. Keep the effort itself: from the first sample above 0.25 m/s to the last
     sample above 70 % of the sprint's top speed (the run-out after the
     line is discarded).
  3. Fit a 4th-order polynomial of speed against time. This gives a smooth
     speed curve whose derivative is a clean acceleration curve.
  4. Keep the accelerating part (acceleration > 0) and read off the
     acceleration-speed (A-S) curve, which for a sprint from standstill is
     close to a straight line.

Metrics returned (all per kg of body mass)
------------------------------------------
vmax  : top speed on the fitted curve (m/s)
Fmax  : peak acceleration = peak force per kg (N/kg = m/s²)
Pmax  : peak of acceleration x speed (W/kg)
PB    : ballistic power, the area under the A-S curve, i.e. the integral of
        acceleration over speed from the first to the last accelerating
        sample (W/kg). Summarises force across the whole speed range in one
        number and is sensitive to fatigue.
SFv_poly : slope of a straight line through the whole accelerating A-S curve
        (this is the "SFv" reported in the ballistic-power paper)
A0, S0, SFv, dFv : from a straight line fitted to the middle of the A-S curve
        (excluding the first 30 % and last 10 % of the speed range, where the
        curve bends). A0 and S0 are its intercepts; dFv is the perpendicular
        distance from the origin to that line.

Differences from the code used for the published paper
-------------------------------------------------------
This function follows the published method with three corrections. The
original choices are still available for comparison:
* Filter order: 2 here (as described in the thesis); the paper's code used 1
  (`filter_order=1`).
* De-duplicating the raw rows: here one sample per timestamp is kept. The
  original code used `Speed.unique()`, which also drops any genuine repeat of
  a speed value within a repetition and shifts later samples in time
  (`dedupe="unique"` in the study script).
* Derivative spacing: here the exact spacing of the 100-point fitted curve,
  range / 99. The original used range / 100, which scales acceleration (and
  Fmax, Pmax, PB) up by about 1 % (`grid_spacing="n"`).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from .filtering import STUDY_FILTERS, butterworth

_trapz = getattr(np, "trapezoid", None) or np.trapz

START_SPEED = 0.25      # m/s, effort starts once speed exceeds this
END_FRACTION = 0.70     # effort ends at the last sample above 70 % of top speed
POLY_ORDER = 4
TRIM = (0.30, 0.90)     # part of the speed range used for the straight-line fit


def sprint_profile(speed_raw, fs: float = 10.0, *, grid_spacing: str = "n-1",
                   filter_order: int | None = None) -> dict | None:
    """
    Metrics for one sprint repetition.

    Parameters
    ----------
    speed_raw : array-like
        Unfiltered 10 Hz speed for the repetition (one value per GNSS sample).
    grid_spacing : "n-1" (exact, default) or "n" (original code); see module notes.
    filter_order : Butterworth order; None uses the study default (2).

    Returns
    -------
    dict of metrics, or None if the trace is too short to analyse.
    """
    f = STUDY_FILTERS["ballistic_power"]
    order = filter_order or f.order
    v = np.asarray(speed_raw, float)
    if len(v) <= 3 * (order + 1):
        return None
    v_f = butterworth(v, fs, f.cutoff_hz, order)
    t = np.arange(len(v_f)) / fs

    # 2. Isolate the effort.
    moving = v_f > START_SPEED
    t_m, v_m = t[moving], v_f[moving]
    if len(v_m) < POLY_ORDER + 2:
        return None
    end = np.max(np.flatnonzero(v_m > END_FRACTION * v_m.max()))
    t_e, v_e = t_m[:end], v_m[:end]
    if len(v_e) < POLY_ORDER + 2:
        return None

    # 3. Smooth speed-time curve and its derivative.
    coeffs = np.polyfit(t_e, v_e, POLY_ORDER)
    t_fit = np.linspace(t_e.min(), t_e.max(), 100)
    v_fit = np.polyval(coeffs, t_fit)
    n = len(t_fit) if grid_spacing == "n" else len(t_fit) - 1
    a_fit = np.gradient(v_fit, (t_fit.max() - t_fit.min()) / n)

    # 4. Accelerating phase -> A-S curve.
    acc = a_fit > 0
    speed, accel = v_fit[acc], a_fit[acc]
    if len(speed) < 3:
        return None
    power = speed * accel

    # Straight line through the middle of the A-S curve.
    lo = speed.min() + TRIM[0] * np.ptp(speed)
    hi = speed.min() + TRIM[1] * np.ptp(speed)
    mid = (speed > lo) & (speed < hi)
    slope, intercept = np.polyfit(speed[mid], accel[mid], 1)

    return {
        "vmax": float(speed.max()),
        "Fmax": float(accel.max()),
        "Pmax": float(power.max()),
        "PB": float(_trapz(accel, x=speed)),
        "SFv_poly": float(np.polyfit(speed, accel, 1)[0]),
        "A0": float(intercept),
        "S0": float(-intercept / slope),
        "SFv": float(slope),
        "dFv": float(abs(intercept) / np.sqrt(slope ** 2 + 1)),
        "power_skew": float(skew(power)),
        "power_kurtosis": float(kurtosis(power)),
        "speed_at_Pmax": float(speed[np.argmax(power)]),
    }
