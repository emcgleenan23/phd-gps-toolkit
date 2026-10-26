"""
Step 4 - how curved are high-speed efforts and acceleration events?

Uses the tangent-chord angle computed for every effort and event in step 2
(Fitzpatrick et al., 2019): the unsigned angle between the initial heading
(first 3 positions) and the straight line from start to finish. 0° = straight.

The angle is an exploratory descriptor: it rests on a handful of GNSS
positions and has not been validated against a criterion measure.

Output: outputs/curvilinear_summary.csv - by season and band: n, mean ± SD
angle and the % of efforts in each angle bin (0-5, 5-10, 10-15, 15+ degrees).
"""

import numpy as np
import pandas as pd

import common


def summarise(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    labels = [f"{a}-{b}" if b < 180 else f"{a}+" for a, b in zip(common.ANGLE_BINS[:-1], common.ANGLE_BINS[1:])]
    df = df.dropna(subset=["angle_deg"]).assign(
        bin=lambda x: pd.cut(x.angle_deg, common.ANGLE_BINS, labels=labels, right=False))
    out = df.groupby(group_cols).angle_deg.agg(n="size", mean="mean", sd="std")
    pct = pd.crosstab([df[c] for c in group_cols], df["bin"], normalize="index") * 100
    return out.join(pct.add_prefix("% ")).reset_index()


def main():
    eff = pd.read_csv(common.output_path("efforts.csv"))
    ev = pd.read_csv(common.output_path("accel_events.csv"))
    ev = ev[ev.zone_abs == "high"].assign(threshold="Abs",
                                          band=lambda x: np.where(x.direction == "acc", "High accel", "High decel"))
    table = pd.concat([summarise(eff, ["threshold", "band", "season"]),
                       summarise(ev, ["threshold", "band", "season"])], ignore_index=True)
    table.to_csv(common.output_path("curvilinear_summary.csv"), index=False, float_format="%.1f")
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
