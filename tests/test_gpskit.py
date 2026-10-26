"""
Quick checks that the core functions behave as documented.

Run from the repository root:   python -m pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpskit import efforts, filtering, profile, stats  # noqa: E402


def test_filter_keeps_slow_signal_and_removes_fast_noise():
    t = np.arange(0, 60, 0.1)
    slow = 3 + np.sin(2 * np.pi * 0.1 * t)          # 0.1 Hz: should pass
    fast = 0.5 * np.sin(2 * np.pi * 4.0 * t)        # 4 Hz: should be removed
    out = filtering.butterworth(slow + fast, fs=10, cutoff_hz=1, order=2)
    assert np.abs(out - slow)[50:-50].max() < 0.05


def test_filter_has_no_time_lag():
    t = np.arange(0, 20, 0.1)
    v = np.where(t > 10, 6.0, 2.0)
    out = filtering.butterworth(v, fs=10, cutoff_hz=1, order=2)
    # Zero-phase: the filtered step crosses the midpoint at the step itself.
    assert abs(t[np.argmin(np.abs(out - 4.0))] - 10.05) <= 0.1  # step lies between 10.0 and 10.1 s


def test_filter_runs_does_not_bridge_a_gap():
    t = np.r_[np.arange(0, 10, 0.1), np.arange(20, 30, 0.1)]
    v = np.r_[np.full(100, 2.0), np.full(100, 6.0)]
    out = filtering.butterworth_runs(t, v, cutoff_hz=1, order=2)
    assert np.allclose(out[:100], 2.0) and np.allclose(out[100:], 6.0)


def test_profile_recovers_a_known_line():
    rng = np.random.default_rng(0)
    v = rng.uniform(3, 9, 2000)
    a_max = 7.0 * (1 - v / 9.0)                       # A0 = 7, S0 = 9
    a = a_max * rng.uniform(0.3, 1.0, v.size)         # cloud below the line
    sp, ac = profile.select_if_points(v, a)
    p = profile.fit_as_profile(sp, ac)
    assert p.A0 == pytest.approx(7.0, abs=0.15)
    assert p.S0 == pytest.approx(9.0, abs=0.15)


def test_ballistic_power_is_twice_pmax_on_a_line():
    m = profile.line_metrics(slope=-0.8, intercept=7.2)
    assert m["PB"] == pytest.approx(2 * m["Pmax"], rel=1e-3)
    assert m["PB"] == pytest.approx(7.2 * 9.0 / 2, rel=1e-6)


def test_effort_hysteresis_and_minimum_duration():
    v = np.array([4.0] * 5 + [5.0] * 5 + [4.4] * 5 + [5.0] * 5 + [3.0] * 5 + [5.0] * 5 + [3.0] * 5)
    # 4.4 is above the 90 % exit (4.25), so the first two bursts are ONE effort;
    # the last burst is 0.5 s long and is dropped by the 1-s minimum.
    w = efforts.effort_windows(v, 4.72, fs=10, min_s=1.0)
    assert w == [(5, 19)]


def test_accel_events_minimum_duration():
    a = np.r_[np.zeros(10), np.full(4, 2.5), np.zeros(10), np.full(6, -3.2), np.zeros(10)]
    ev = efforts.accel_events(np.full(a.size, 3.0), a, fs=10, floor=1.0, min_s=0.5)
    assert [e["direction"] for e in ev] == ["dec"]
    assert efforts.zone(ev[0]["peak_accel"], (1, 2, 3)) == "high"


def test_tangent_chord_angle():
    lat0, lon0 = 53.5, -7.5
    straight_lat = lat0 + np.linspace(0, 1e-4, 10)
    assert efforts.tangent_chord_angle(straight_lat, np.full(10, lon0)) == pytest.approx(0, abs=1e-6)
    # Start heading north, finish due east of the start: the chord is ~90° from the tangent.
    lat = np.r_[lat0, lat0 + 1e-5, lat0 + 2e-5, lat0 + 1e-5, lat0]
    lon = np.r_[lon0, lon0, lon0, lon0 + 1e-4, lon0 + 2e-4]
    assert efforts.tangent_chord_angle(lat, lon) == pytest.approx(90, abs=1)


def test_peak_window_distance():
    v = np.r_[np.full(600, 2.0), np.full(600, 5.0)]
    assert efforts.peak_window_distance(v, 60, fs=10) == pytest.approx(300, abs=1)


def test_kth_highest():
    assert efforts.kth_highest([8.1, 9.5, 8.7, 7.9], 3) == 8.1
    assert np.isnan(efforts.kth_highest([8.1, 9.5], 3))


def test_hedges_g_small_sample_correction():
    a, b = np.array([1, 2, 3, 4, 5.0]), np.array([2, 3, 4, 5, 6.0])
    d = 1 / np.std(a, ddof=1)
    assert stats.hedges_g(a, b) == pytest.approx(d * (1 - 3 / (4 * 8 - 1)))
    assert stats.magnitude(0.5) == "small" and stats.magnitude(-1.3) == "large"
