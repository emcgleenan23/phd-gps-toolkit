"""
Step 5 - worst-case-scenario (peak) running demands.

For every eligible player-match, the most demanding 1, 2, 3, 5 and 10-minute
windows anywhere in the match, for total distance and for absolute HSR distance
(speed >= 4.72 m/s). Windows never cross half-time or a GNSS drop-out: each
continuous run is searched separately and the match value is the highest.

Match averages hide these passages; the peak values are a better target for
training drills that aim to prepare players for the hardest periods of play.

Outputs: outputs/worst_case_by_match.csv, outputs/worst_case_by_season.csv (m/min)
"""

import numpy as np
import pandas as pd

import common
from gpskit.efforts import peak_window_distance

HSR = common.ABSOLUTE_BANDS["HSR"]


def main():
    data = common.load_cache()
    keep = pd.read_csv(common.output_path("match_metrics.csv"), dtype={"Date": str})[["Season", "Date", "Player"]]
    keep.columns = ["season", "date", "player"]
    data = data.merge(keep, on=["season", "date", "player"])

    rows = []
    for (season, date, player), g in data.groupby(["season", "date", "player"]):
        runs = [r["speed"].to_numpy() for _, r in g.groupby(["half", "run"])]
        for w in common.WCS_WINDOWS_MIN:
            total = [peak_window_distance(v, w * 60, common.FS) for v in runs]
            hsr = [peak_window_distance(np.where(v >= HSR, v, 0.0), w * 60, common.FS) for v in runs]
            total, hsr = [x for x in total if x is not None], [x for x in hsr if x is not None]
            rows.append({"season": season, "date": date, "player": player, "window_min": w,
                         "total_m_per_min": max(total) / w if total else np.nan,
                         "hsr_m_per_min": max(hsr) / w if hsr else np.nan})
    wcs = pd.DataFrame(rows)
    wcs.to_csv(common.output_path("worst_case_by_match.csv"), index=False, float_format="%.2f")

    summary = wcs.groupby(["season", "window_min"])[["total_m_per_min", "hsr_m_per_min"]].agg(["mean", "std"])
    summary.to_csv(common.output_path("worst_case_by_season.csv"), float_format="%.1f")
    print(summary.round(1))


if __name__ == "__main__":
    main()
