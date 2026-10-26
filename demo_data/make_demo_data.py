"""
Create the synthetic demo data used by the example commands in each study.

    python demo_data/make_demo_data.py

Writes (about 300 MB in total, not tracked by git):
    demo_data/insitu_pfv/        6 matches + 6 training sessions, 6 invented players
    demo_data/ballistic_power/   one 3 x 5 x 50 m repeated-sprint session, 8 players
    demo_data/running_demands/   3 seasons x 3 league matches with positions, 8 players

All values are simulated; see gpskit/synthetic.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gpskit.synthetic import make_demands_seasons, make_insitu_season, make_rsa_session  # noqa: E402

if __name__ == "__main__":
    print("Writing synthetic in-situ season ...")
    make_insitu_season(HERE / "insitu_pfv")
    print("Writing synthetic repeated-sprint session ...")
    (HERE / "ballistic_power").mkdir(exist_ok=True)
    make_rsa_session(HERE / "ballistic_power" / "2024-05-20-DemoClub-Training.csv")
    print("Writing synthetic running-demands seasons ...")
    make_demands_seasons(HERE / "running_demands")
    print("Done.")
