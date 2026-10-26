"""
Step 1 - sprint metrics for every repetition of the repeated-sprint test.

Protocol: 3 sets x 5 maximal 50 m sprints. In the STATSports export each
sprint is a drill called "Rep 1" ... "Rep 15". For every player and repetition
this script computes vmax, Fmax, Pmax, ballistic power (PB), the A-S slope and
the Fv-offset (dFv) with `gpskit.sprint.sprint_profile`.

Usage
-----
    python 01_sprint_metrics.py                                   # demo data
    python 01_sprint_metrics.py --raw /path/2024-05-20-Club-Training.csv
    python 01_sprint_metrics.py --raw ... --filter-order 1 --dedupe unique --grid-spacing n
                                         # the original paper code's choices

Output: outputs/rsa_reps.csv (one row per player x repetition)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from gpskit.io import PLAYER, DRILL, SPEED, read_export  # noqa: E402
from gpskit.sprint import sprint_profile  # noqa: E402

REPS = [f"Rep {i}" for i in range(1, 16)]
DEFAULT_RAW = HERE.parents[1] / "demo_data" / "ballistic_power" / "2024-05-20-DemoClub-Training.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="STATSports export of the RSA session")
    ap.add_argument("--dedupe", choices=["time", "unique"], default="time",
                    help="'time' keeps one sample per timestamp; 'unique' is the original paper code")
    ap.add_argument("--grid-spacing", choices=["n-1", "n"], default="n-1",
                    help="'n-1' is the exact derivative spacing; 'n' is the original paper code")
    ap.add_argument("--filter-order", type=int, default=None,
                    help="Butterworth order (default 2; the published paper used 1)")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "rsa_reps.csv")
    args = ap.parse_args()

    df = read_export(args.raw, drills=REPS)
    rows = []
    for (player, rep), g in df.groupby([PLAYER, DRILL], sort=False):
        speed = pd.to_numeric(g[SPEED], errors="coerce").dropna()
        if args.dedupe == "unique":
            speed = pd.Series(speed.unique())
        m = sprint_profile(speed.to_numpy(), grid_spacing=args.grid_spacing,
                           filter_order=args.filter_order)
        if m is None:
            print(f"  skipped {player} {rep}: too few samples")
            continue
        rows.append({"player": player, "rep": int(rep.split()[1]), **m})

    out = pd.DataFrame(rows).sort_values(["player", "rep"])
    out["set"] = (out["rep"] - 1) // 5 + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.5f")
    print(f"{out.player.nunique()} players, {len(out)} repetitions -> {args.out}")


if __name__ == "__main__":
    main()
