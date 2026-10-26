"""
Step 3 - do in-situ profiles agree with a profile from an isolated sprint?
Reproduces Table 3 of the in-situ PFv paper.

In-situ side
------------
For each player and condition (match / training / combined), every session's
IF points are pooled WITHOUT re-selecting across sessions (several hundred
points), then fitted with the same 1.96 SD outlier step. A player is included
if they have >= 5 sessions of that condition and the pooled fit has R² >= 0.90.

Isolated side
-------------
Repetition 1 of the 3 x 5 x 50 m repeated-sprint test (the un-fatigued sprint),
from the ballistic-power study's per-repetition output.

Metrics compared
----------------
* A0, S0 : intercepts of the fitted A-S line on BOTH sides (the isolated line
           is fitted to 30-90 % of the sprint's speed range).
* Amax, Smax : observed peaks. Isolated: peak acceleration and top speed on
           the sprint's polynomial fit. In-situ: highest acceleration and
           highest speed in the player's filtered data for that condition.
* SFv, Pmax, PB, dFv : as in the paper (isolated SFv, Pmax and PB come from
           the curved sprint A-S profile, dFv from its straight line).

Differences from the published Table 3
--------------------------------------
1. The published isolated A0 and S0 were the observed peaks (now Amax, Smax);
   here they are intercepts, matching the in-situ definition.
2. In the published table the isolated mean ± SD and the bias for the
   Training and Combined rows repeated the Match row. Here every row uses
   its own paired players. d and p values were already paired correctly.

Statistics: paired differences per player; bias = in-situ - isolated;
Cohen's d = mean difference / SD of differences; Wilcoxon signed-rank test.

Usage
-----
    python 03_insitu_vs_isolated.py --rsa ../ballistic_power/outputs/rsa_reps.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import common
from gpskit.filtering import acceleration
from gpskit.profile import fit_as_profile, select_if_points
from gpskit.stats import paired_d

METRICS = ["A0", "S0", "Amax", "Smax", "SFv", "Pmax", "PB", "dFv"]

# Column names in the ballistic-power output (and in the original rsa_test_3.csv).
RSA_COLUMNS = {
    "player": ["player", "Name"], "rep": ["rep", "Rep Number"],
    "Amax": ["Fmax", "A Max"], "Vmax": ["vmax", "S Max"], "Pmax": ["Pmax", "P Max"],
    "PB": ["PB", "Ballistic Power"], "SFv_poly": ["SFv_poly", "S Fv"],
    "dFv": ["dFv", "d Fv"], "A0": ["A0", "A Max Ideal"], "S0": ["S0", "S Max Ideal"],
}


def read_rsa(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    out = {}
    for new, options in RSA_COLUMNS.items():
        col = next(c for c in options if c in raw.columns)
        out[new] = raw[col]
    return pd.DataFrame(out)


def isolated_metrics(rep1) -> dict:
    return {"A0": rep1.A0, "S0": rep1.S0, "Amax": rep1.Amax, "Smax": rep1.Vmax,
            "SFv": rep1.SFv_poly, "Pmax": rep1.Pmax, "PB": rep1.PB, "dFv": rep1.dFv}


def insitu_profile(sessions):
    """Pooled profile (plus observed peaks) with the inclusion rules used for Table 3."""
    pooled, peak_a, peak_v = [], 0.0, 0.0
    for s in sessions:
        if len(s["speed"]) < common.MIN_SAMPLES:
            continue
        acc = acceleration(s["speed"], common.FS)
        sp, ac = select_if_points(s["speed"], acc)
        if len(sp) >= 5:
            pooled.append((sp, ac))
            peak_a, peak_v = max(peak_a, acc.max()), max(peak_v, s["speed"].max())
    if len(pooled) < common.MIN_SESSIONS:
        return None
    sp = np.concatenate([p[0] for p in pooled])
    ac = np.concatenate([p[1] for p in pooled])
    if len(sp) < 20:
        return None
    prof = fit_as_profile(sp, ac, min_points=10, plausible_range=None)
    if prof is None or prof.r2 < 0.90:
        return None
    return {**prof.as_dict(), "Amax": peak_a, "Smax": peak_v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsa", type=Path, required=True, help="per-repetition RSA results CSV")
    args = ap.parse_args()

    rsa = read_rsa(args.rsa)
    rep1 = {r.player: isolated_metrics(r) for r in rsa[rsa.rep == 1].itertuples()}

    conditions = {"Match": ["match"], "Training": ["training"], "Combined": ["match", "training"]}
    rows, paired = [], []
    for player, sessions in common.iter_players():
        if player not in rep1:
            continue
        for cond, kinds in conditions.items():
            prof = insitu_profile([s for s in sessions if s["type"] in kinds])
            if prof is not None:
                paired.append({"player": player, "condition": cond,
                               **{f"{m}_insitu": prof[m] for m in METRICS},
                               **{f"{m}_isolated": rep1[player][m] for m in METRICS}})
    paired = pd.DataFrame(paired)
    paired.to_csv(common.output_path("insitu_vs_isolated_pairs.csv"), index=False)

    print("In-situ vs isolated sprint (repetition 1)\n")
    print(f"{'metric':6s} {'condition':9s} {'n':>3s} {'in-situ':>14s} {'isolated':>14s} {'bias':>7s} {'d':>6s} {'p':>6s}")
    for m in METRICS:
        for cond in conditions:
            g = paired[paired.condition == cond]
            a, b = g[f"{m}_insitu"].values, g[f"{m}_isolated"].values
            p = stats.wilcoxon(a, b).pvalue
            rows.append({"metric": m, "condition": cond, "n": len(g),
                         "insitu_mean": a.mean(), "insitu_sd": a.std(),
                         "isolated_mean": b.mean(), "isolated_sd": b.std(),
                         "bias": (a - b).mean(), "d": paired_d(a, b), "p": p})
            r = rows[-1]
            print(f"{m:6s} {cond:9s} {r['n']:3d} {r['insitu_mean']:7.2f} ± {r['insitu_sd']:4.2f} "
                  f"{r['isolated_mean']:7.2f} ± {r['isolated_sd']:4.2f} {r['bias']:+7.2f} "
                  f"{r['d']:+6.2f} {p:6.3f}")
    pd.DataFrame(rows).to_csv(common.output_path("insitu_vs_isolated.csv"),
                              index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
