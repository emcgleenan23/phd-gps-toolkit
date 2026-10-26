"""
Step 4 - how much data does a reliable in-situ profile need?
Reproduces Table 4 and Figure 3 of the in-situ PFv paper.

Method
------
For each player, sessions are put in date order and treated as one long,
continuous stream of data (matches only, training only, or both combined).
We step through that stream one minute at a time. At every step we fit a
pooled profile to *all data seen so far*: the IF points of every completed
session plus the IF points of the current session up to this minute. The R²
of each fit shows how quickly the profile becomes linear as data accumulate.

* Matches are placed on a 60-min axis (2 x 30-min halves), whatever their
  actual elapsed time. Training sessions use their real duration.
* Group curve: per-player R² is averaged within 1-min buckets. Buckets with
  too few players are dropped (fewer than 16 match, 25 training or 20
  combined players, as in the paper; capped at the number of players
  available, so the small demo data set still runs).

Two summaries of the group R² curve:
  * threshold - first minute where R² >= 0.90 for 3 consecutive steps
  * knee      - geometric "knee" of the curve (Kneedle algorithm, `kneed`)
The ± in Table 4 is a 95 % CI half-width, 1.96 · SD / sqrt(n), of the same
two quantities computed for each player individually.

Note: the paper describes 5-min (single-session) and 10-min (multi-session)
increments. The published Table 4 values were produced with this continuous
1-min version, which is what this script runs.

Outputs: outputs/minimum_data_load.csv, outputs/minimum_data_load.png
"""

import numpy as np
import pandas as pd
from kneed import KneeLocator

import common
from gpskit.filtering import acceleration
from gpskit.profile import pooled_profile, select_if_points

STEP_MIN = 1
R2_TARGET = 0.90
MIN_PLAYERS = {"match": 16, "training": 25, "combined": 20}


def positive_samples(speed):
    """Filtered speed -> (speed, acceleration) where acceleration > 0."""
    acc = acceleration(speed, common.FS)
    pos = acc > 0
    return speed[pos], acc[pos], pos


def accumulation_curve(sessions):
    """R² (and profile) of the pooled fit at every minute of accumulated data."""
    # IF points of each *complete* session, used once that session is finished.
    full_if = []
    for s in sessions:
        sp, ac, _ = positive_samples(s["speed"])
        full_if.append(select_if_points(sp, ac))

    rows, minutes_done = [], 0.0
    for k, s in enumerate(sessions):
        axis_len = 60.0 if s["type"] == "match" else s["duration_s"] / 60.0
        acc = acceleration(s["speed"], common.FS)
        done = [p for p in full_if[:k] if len(p[0]) >= 2]

        for m in np.arange(STEP_MIN, axis_len + 0.01, STEP_MIN):
            # Minute m on the axis -> the same fraction of the real session.
            cutoff_s = m / axis_len * s["duration_s"]
            upto = s["t"] <= cutoff_s + 0.01
            pos = upto & (acc > 0)
            if pos.sum() < 5:
                continue
            partial = select_if_points(s["speed"][pos], acc[pos])
            if len(partial[0]) < 2:
                continue
            prof = pooled_profile(done + [partial], rebin=True, plausible_range=None)
            if prof is not None and prof.S0 > 0:
                rows.append({"cum_min": minutes_done + m, "r2": prof.r2,
                             "A0": prof.A0, "S0": prof.S0})
        minutes_done += axis_len
    return pd.DataFrame(rows)


def threshold_minute(minutes, r2):
    """First minute at which R² stays >= target for 3 consecutive steps."""
    ok = np.asarray(r2) >= R2_TARGET
    for i in range(len(ok) - 2):
        if ok[i] and ok[i + 1] and ok[i + 2]:
            return float(minutes[i])
    return np.nan


def knee_minute(minutes, r2):
    if len(minutes) < 5:
        return np.nan
    kn = KneeLocator(minutes, r2, curve="concave", direction="increasing",
                     S=1.0, interp_method="interp1d")
    return float(kn.knee) if kn.knee is not None else np.nan


def main():
    curves = []
    for player, sessions in common.iter_players():
        subsets = {"match": [s for s in sessions if s["type"] == "match"],
                   "training": [s for s in sessions if s["type"] == "training"],
                   "combined": sessions}
        for name, sub in subsets.items():
            if len(sessions) < 2 or len(sub) < 2:
                continue
            c = accumulation_curve(sub)
            if len(c):
                curves.append(c.assign(player=player, dataset=name))
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(common.output_path("minimum_data_load.csv"), index=False, float_format="%.4f")

    print(f"{'':10s} {'R² >= 0.90 (min)':>18s} {'knee (min)':>12s}")
    group_curves = {}
    for name in ("match", "training", "combined"):
        d = curves[curves.dataset == name]

        # Per-player values (for the ± in Table 4).
        per_player = [(threshold_minute(g.cum_min.values, g.r2.values),
                       knee_minute(g.cum_min.values, g.r2.values))
                      for _, g in d.groupby("player") if len(g) >= 10]
        per_player = np.array(per_player)

        # Group curve: each player's fit nearest each whole minute, then the mean.
        d = d.assign(bucket=np.maximum(d.cum_min.round().astype(int), STEP_MIN))
        d = d.loc[(d.cum_min - d.bucket).abs().groupby([d.player, d.bucket]).idxmin()]
        d = d[d.groupby("player").cum_min.transform("size") >= 5]
        g = d.groupby("bucket").r2.agg(mean="mean", std=lambda x: x.std(ddof=0), size="size")
        g = g[g["size"] >= min(MIN_PLAYERS[name], d.player.nunique())]
        group_curves[name] = g

        thr, knee = threshold_minute(g.index, g["mean"]), knee_minute(g.index.values, g["mean"].values)
        ci = lambda v: 1.96 * np.nanstd(v) / np.sqrt(np.isfinite(v).sum())
        print(f"{name:10s} {thr:9.0f} ± {ci(per_player[:, 0]):<5.0f} {knee:6.0f} ± {ci(per_player[:, 1]):.0f}")

    plot(group_curves)


def plot(group_curves):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, g) in zip(axes, group_curves.items()):
        half = 1.96 * g["std"] / np.sqrt(g["size"])
        ax.plot(g.index, g["mean"], color="#b03a2e")
        ax.fill_between(g.index, g["mean"] - half, g["mean"] + half, color="#b03a2e", alpha=0.2)
        ax.axhline(R2_TARGET, ls=":", color="grey")
        ax.set(title=name.capitalize(), xlabel="Accumulated data (min)", ylim=(0.5, 1.02))
    axes[0].set_ylabel("R² of pooled A-S fit")
    fig.tight_layout()
    fig.savefig(common.output_path("minimum_data_load.png"), dpi=200)


if __name__ == "__main__":
    main()
