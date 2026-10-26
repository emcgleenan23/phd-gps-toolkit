"""
Step 2 - one pooled A-S / PFv profile per player for matches and for training.
Reproduces Tables 1-2 of the in-situ PFv paper (in-situ columns).

Method (Morin et al., 2021):
  1. For each session: filtered speed -> acceleration -> IF points
     (top 2 positive accelerations per 0.2 m/s bin above 3 m/s).
  2. Pool the IF points of all of a player's matches (or training sessions).
  3. Re-select the top 2 per bin across the pooled points (the "best-case"
     envelope), fit a line, drop points beyond 1.96 SD, refit.
  A player needs at least 5 sessions of a type to get a profile of that type.

Outputs
-------
outputs/pooled_profiles.csv   one row per player x session type
printed group summary         mean ± SD (population SD, as in the paper)
"""

import numpy as np
import pandas as pd

import common
from gpskit.profile import pooled_profile, session_if_points

METRICS = ["r2", "A0", "S0", "SFv", "Pmax", "PB", "dFv", "n_points"]


def main():
    rows = []
    for player, sessions in common.iter_players():
        for kind in ("match", "training"):
            chosen = [s for s in sessions if s["type"] == kind]
            if_points = [session_if_points(s["speed"], common.FS) for s in chosen]
            # Sessions that yield fewer than 2 IF points carry no information.
            if_points = [p for p in if_points if len(p[0]) >= 2]
            if len(if_points) < common.MIN_SESSIONS:
                continue
            prof = pooled_profile(if_points, rebin=True)
            if prof is not None:
                rows.append({"player": player, "type": kind,
                             "n_sessions": len(if_points), **prof.as_dict()})

    df = pd.DataFrame(rows)
    df.to_csv(common.output_path("pooled_profiles.csv"), index=False, float_format="%.4f")

    print("Pooled in-situ profiles (mean ± SD across players)\n")
    for kind, g in df.groupby("type"):
        print(f"{kind:9s} n = {len(g)} players")
        for m in METRICS:
            print(f"   {m:8s} {g[m].mean():7.3f} ± {np.std(g[m]):.3f}")
        print()


if __name__ == "__main__":
    main()
