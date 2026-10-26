"""
Step 3 - how did match demands change from one season to the next?

A. Squad level (model-based)
   For each metric, a linear mixed-effects model
       metric ~ season + venue + opponent points + (1 | player)
   is fitted twice: with the earlier season of each adjacent pair as the
   reference (e.g. 2024 -> 2025, then 2025 -> 2026). The player random
   intercept accounts for repeated matches by the same player; the season
   estimate is adjusted for venue and opponent strength.

B. Match context
   Venue and opponent strength are the same for every player in a match, so
   their effects are estimated from match means (one row per match):
       team mean ~ season + venue + opponent points (per +1 SD)

C. Positional lines (descriptive)
   Within each position line, % change and Hedges' g (bootstrap 95 % CI)
   between adjacent seasons, from player-match values without adjustment.
   Small groups give wide intervals; treat these as exploratory.

Outputs: outputs/season_contrasts_model.csv, outputs/match_context.csv,
         outputs/season_contrasts_by_position.csv
"""

import pandas as pd

import common
from gpskit.stats import compare_groups, match_context, season_contrasts

METRICS = [
    "Total Distance", "Distance Per Min",
    "HSR Per Minute (Abs)", "Sprint Per Minute (Abs)", "Near-maximal Per Minute (Abs)",
    "HSR Per Minute (Rel)", "Sprint Per Minute (Rel)", "Near-maximal Per Minute (Rel)",
    "Accel High (Abs)", "Decel High (Abs)", "Max Speed",
]


def main():
    d = pd.read_csv(common.output_path("match_metrics.csv"), dtype={"Date": str})
    seasons = sorted(d.Season.unique())
    pairs = list(zip(seasons[:-1], seasons[1:]))

    # A. Mixed models
    rows = []
    for earlier, later in pairs:
        for m in METRICS:
            r = season_contrasts(d, m, season="Season", player="Player", venue="Venue",
                                 opponent="Opponent Points", baseline=earlier)
            rows.append(r[r.contrast == f"{earlier}->{later}"])
    model = pd.concat(rows, ignore_index=True)
    model.to_csv(common.output_path("season_contrasts_model.csv"), index=False, float_format="%.3f")
    print("Adjusted season differences (estimate [95 % CI])")
    for _, r in model.iterrows():
        flag = "" if r.converged else "   (model did not converge cleanly)"
        print(f"  {r.contrast}  {r.metric:32s} {r.estimate:+9.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]{flag}")

    # B. Venue and opponent strength
    ctx = pd.concat([match_context(d, m, match="Date", season="Season", venue="Venue",
                                   opponent="Opponent Points") for m in METRICS], ignore_index=True)
    ctx.to_csv(common.output_path("match_context.csv"), index=False, float_format="%.3f")

    # C. Positional lines
    if "Position" in d:
        rows = []
        for (earlier, later) in pairs:
            for pos, g in d.groupby("Position"):
                for m in METRICS:
                    a, b = g.loc[g.Season == earlier, m], g.loc[g.Season == later, m]
                    if len(a) >= 2 and len(b) >= 2:
                        rows.append({"contrast": f"{earlier}->{later}", "position": pos, "metric": m,
                                     **compare_groups(a, b)})
        pd.DataFrame(rows).to_csv(common.output_path("season_contrasts_by_position.csv"),
                                  index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
