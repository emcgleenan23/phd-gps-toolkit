# Match running demands across seasons

Code for the three-season study of match running demands in sub-elite Gaelic
football (manuscript under review).

## Inputs

* Match exports (`YYYY-MM-DD-...-Match.csv`) with `1st Half` / `2nd Half`
  drills and `Lat` / `Lon`.
* `fixtures.csv`: `Date, Competition, Opposition, Venue, Opponent Points`
  (Venue is Home/Away; Opponent Points are the opponent's end-of-season league points).
* `positions.csv`: `Player, Position` (e.g. Inside defender / Middle eight / Inside forward).

## Scripts

```bash
python 01_build_cache.py                      # demo data
python 01_build_cache.py --raw /path/exports --anonymise   # your data
python 02_match_metrics.py
python 03_season_contrasts.py
python 04_curvilinear.py
python 05_worst_case.py
```

For your own data, point `DEMANDS_FIXTURES` and `DEMANDS_POSITIONS` at your two
tables (or put `fixtures.csv` and `positions.csv` in the raw folder).

| Script | Does |
|---|---|
| `01_build_cache.py` | Halves cropped to 30 min, filtered within continuous stretches, cached |
| `02_match_metrics.py` | Inclusion rules, MSS / MSA / MSD, all player-match metrics, every effort and acceleration event |
| `03_season_contrasts.py` | Mixed models (adjacent seasons), venue and opponent effects, positional Hedges' g |
| `04_curvilinear.py` | Tangent-chord angles of efforts and high accelerations |
| `05_worst_case.py` | Peak 1-10 min windows for total and HSR distance |

## Definitions

* Player-match included if >= 40 of 60 min played; player-season if >= 3 such matches.
* Speed bands (efforts end below 90 % of entry; minimum 1 s):
  absolute HSR >= 4.72, sprint >= 6.11, near-maximal >= 7.00 m/s;
  normalised >= 75 / 85 / 90 % of maximal seasonal speed (MSS).
* MSS: highest speed reached on >= 3 separate efforts in the season.
* Acceleration events: >= 1 m/s² for >= 0.5 s, zoned by peak
  (1-2, 2-3, >= 3 m/s²) and relative to MSA / MSD (25-50, 50-75, >= 75 %).

See `docs/paper_vs_toolkit.md` for where this differs from the original analysis code.
