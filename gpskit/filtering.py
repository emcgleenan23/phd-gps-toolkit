"""
Smoothing GNSS speed before differentiating it.

Why filter at all?
------------------
Acceleration is the time-derivative of speed. Differentiating amplifies
high-frequency noise, so raw 10 Hz GNSS speed gives a very noisy acceleration
trace. A low-pass filter removes the fast wobble (noise) and keeps the slower
changes that reflect real running. In one example match trace, filtering cut
the standard deviation of acceleration from 1.43 m/s² (raw) to 0.68 (2 Hz cut-off)
and 0.92 m/s² (1 Hz cut-off). See McGleenan (2026) PhD thesis, section 2.5.

Why a *zero-phase* Butterworth filter?
--------------------------------------
* Butterworth: flat pass-band, so frequencies below the cut-off pass through
  essentially unchanged.
* Zero-phase (scipy.signal.filtfilt): the filter is run forwards and then
  backwards, which cancels the time lag a one-way filter introduces. That keeps
  the timing of accelerations and decelerations correct. Running it twice also
  doubles the effective order and puts the attenuation at the cut-off at about
  -6 dB rather than -3 dB.

Which settings did each study use?
----------------------------------
The cut-off was matched to the type of data:

* Isolated 50 m sprints (ballistic power study): 2 Hz, to keep the sharper
  acceleration peaks of a maximal sprint (Ellens et al., 2024).
* Training and match play (in-situ PFv and running-demands studies): 1 Hz,
  a more conservative choice for unstructured activity (Clavel et al., 2023;
  Delves et al., 2022).

All three studies use a 2nd-order filter, as described in the thesis
(section 2.5.2). Note that the original analysis code for the ballistic-power
and running-demands studies used a 1st-order filter, so their published
numbers were produced with order=1. Scripts that need to reproduce those
numbers expose a `--filter-order` option. If you are starting a new analysis,
choose (and report) the cut-off and order deliberately; ideally derive the
cut-off from a residual analysis of your own device's data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

# STATSports Apex units sample position and speed at 10 Hz.
DEFAULT_FS_HZ = 10.0


@dataclass(frozen=True)
class FilterSettings:
    """Low-pass Butterworth settings: cut-off frequency (Hz) and filter order."""

    cutoff_hz: float
    order: int


# Filter settings for each study (all 2nd order, zero-phase).
STUDY_FILTERS = {
    "ballistic_power": FilterSettings(cutoff_hz=2.0, order=2),  # Study 1: isolated sprints
    "insitu_pfv": FilterSettings(cutoff_hz=1.0, order=2),       # Study 2: training + matches
    "running_demands": FilterSettings(cutoff_hz=1.0, order=2),  # Study 3: matches
}


def butterworth(speed, fs: float = DEFAULT_FS_HZ, cutoff_hz: float = 1.0,
                order: int = 2) -> np.ndarray:
    """
    Zero-phase low-pass Butterworth filter (scipy.signal.butter + filtfilt).

    Parameters
    ----------
    speed : array-like
        Speed samples in m/s, evenly spaced in time.
    fs : float
        Sampling frequency in Hz (10 for STATSports Apex).
    cutoff_hz : float
        Cut-off frequency in Hz. Must be below the Nyquist frequency (fs / 2).
    order : int
        Filter order before the forward-backward pass.

    Returns
    -------
    np.ndarray
        The filtered speed. If the trace is too short for filtfilt's edge
        padding, the input is returned unchanged rather than raising an error.

    Notes
    -----
    * The input must be continuous. Filtering straight across a gap in the
      data (e.g. a GNSS dropout or the half-time break) blends the two sides
      together. Filter each continuous segment separately where it matters.
    * Normalised cut-off = cutoff_hz / (fs / 2). Earlier versions of the
      ballistic-power code set the Nyquist frequency equal to fs, which
      silently halved the cut-off. Always divide by fs / 2.
    """
    x = np.asarray(speed, dtype=float)
    nyquist = fs / 2.0
    wn = cutoff_hz / nyquist
    if not 0 < wn < 1:
        raise ValueError(f"cut-off {cutoff_hz} Hz must lie between 0 and the "
                         f"Nyquist frequency ({nyquist} Hz)")

    b, a = signal.butter(order, wn, btype="low")

    # filtfilt pads the signal at both ends; it needs more samples than the pad.
    min_len = 3 * max(len(a), len(b))
    if len(x) <= min_len:
        return x.copy()
    return signal.filtfilt(b, a, x)


def butterworth_runs(t_s, speed, max_gap_s: float = 0.5, **kwargs) -> np.ndarray:
    """
    Filter each continuous stretch of data separately.

    GNSS units occasionally drop samples. Filtering straight across a gap
    blends the speed before and after it, so the trace is split wherever two
    samples are more than `max_gap_s` apart and each piece is filtered on its
    own. `kwargs` are passed to `butterworth` (fs, cutoff_hz, order).
    """
    t = np.asarray(t_s, float)
    v = np.asarray(speed, float)
    out = np.empty_like(v)
    breaks = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(v)]):
        out[a:b] = butterworth(v[a:b], **kwargs)
    return np.clip(out, 0, None)  # speed cannot be negative


def filter_for_study(speed, study: str, fs: float = DEFAULT_FS_HZ) -> np.ndarray:
    """Filter speed with the settings a given study used (see STUDY_FILTERS)."""
    s = STUDY_FILTERS[study]
    return butterworth(speed, fs=fs, cutoff_hz=s.cutoff_hz, order=s.order)


def acceleration(speed_filtered, fs: float = DEFAULT_FS_HZ) -> np.ndarray:
    """
    Acceleration (m/s²) from filtered speed by central differences.

    numpy.gradient uses (v[i+1] - v[i-1]) / (2·dt) for interior points and
    one-sided differences at the two ends.
    """
    return np.gradient(np.asarray(speed_filtered, dtype=float), 1.0 / fs)
