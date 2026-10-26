"""
Step 2 - descriptive results of the ballistic-power paper.

* Table 1   : mean ± SD, max and min of each metric over all repetitions.
* Figure 2  : each metric per repetition as % of its repetition-1 mean.
* Fatigue   : per-player % change from the first to the last repetition, with a
              t-based 95 % CI and Cohen's d (mean change / SD of change).

Usage:  python 02_fatigue_summary.py [--reps outputs/rsa_reps.csv]
Outputs: outputs/table1.csv, outputs/percent_of_rep1.csv, outputs/percent_of_rep1.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
# Metric column -> label used in the paper ("SFv" in the paper is the slope
# through the whole accelerating A-S curve, SFv_poly).
METRICS = {"vmax": "vmax", "Fmax": "Fmax", "Pmax": "Pmax", "PB": "PB",
           "SFv_poly": "SFv", "dFv": "dFv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=Path, default=HERE / "outputs" / "rsa_reps.csv")
    args = ap.parse_args()
    d = pd.read_csv(args.reps)
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)

    print(f"{d.player.nunique()} players, {len(d)} repetitions\n")
    table = pd.DataFrame({label: {"mean": d[c].mean(), "sd": d[c].std(), "max": d[c].max(),
                                  "min": d[c].min()} for c, label in METRICS.items()}).T
    table.to_csv(out / "table1.csv", float_format="%.3f")
    print("Table 1\n", table.round(2), "\n")

    # Mean of each repetition as % of the repetition-1 mean (Figure 2).
    rep_means = d.groupby("rep")[list(METRICS)].mean()
    pct = rep_means / rep_means.loc[1] * 100
    pct.rename(columns=METRICS).to_csv(out / "percent_of_rep1.csv", float_format="%.2f")

    first, last = d.rep.min(), d.rep.max()
    print(f"Change from rep {first} to rep {last} (per player, %)")
    for c, label in METRICS.items():
        wide = d.pivot(index="player", columns="rep", values=c)
        change = ((wide[last] - wide[first]) / wide[first] * 100).dropna()
        n, mean, sd = len(change), change.mean(), change.std()
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        print(f"  {label:5s} {mean:+6.1f} %  [{mean - half:+.1f}, {mean + half:+.1f}]  d = {abs(mean) / sd:.2f}")

    import matplotlib.pyplot as plt
    ax = pct.rename(columns=METRICS).plot(marker="o", figsize=(8, 4))
    for boundary in (5.5, 10.5):
        ax.axvline(boundary, color="grey", ls=":")  # 2:30 recovery between sets
    ax.set(xlabel="Repetition", ylabel="% of repetition 1")
    plt.tight_layout()
    plt.savefig(out / "percent_of_rep1.png", dpi=200)


if __name__ == "__main__":
    main()
