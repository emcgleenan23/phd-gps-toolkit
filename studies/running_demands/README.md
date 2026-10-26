# Match running demands across seasons

Code for McGleenan et al., *Match running demands in club Gaelic football
across three seasons spanning rule amendments and promotion: a single-club GNSS
case study* (manuscript under review).

## What the study asks

How did match running demands change from one season to the next when the
rules changed (2024 to 2025, same playing level) and when the club was
promoted (2025 to 2026, same rules)? Demands are described with fixed
(absolute) speed thresholds and with thresholds individualised to each
player's maximal seasonal speed, by squad and by positional line (inside
defenders, middle eight, inside forwards), alongside acceleration profiles,
the curvature of high-speed efforts and the most demanding passages of play.

The design is observational: season contrasts describe what changed, not
what caused it.

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

## Processing

| Step | Setting |
|---|---|
| Sampling | 10 Hz GNSS speed; raw 100 Hz rows collapsed to one sample per timestamp |
| Match segment | `1st Half` and `2nd Half` only (no warm-up or half-time); each half cropped to its first **30 min** |
| Gaps | Trace split wherever samples are > **0.5 s** apart; every step below runs within a continuous piece |
| Filter | Low-pass Butterworth, **2nd order, 1 Hz**, zero-phase (`scipy.signal.butter` + `filtfilt`) |
| Acceleration | Central differences of filtered speed (`numpy.gradient`) |
| Distance | Trapezoidal integration of filtered speed (`numpy.trapz`) |

## Inclusion

| Rule | Value |
|---|---|
| Fixtures | League matches only (`Competition` contains "league"; set `COMPETITION_KEYWORDS`) |
| Player-match | >= **40 of 60** regulation minutes played |
| Player-season | >= **3** qualifying player-matches in that season (seasons treated separately) |

## Speed thresholds

Bands are cumulative lower bounds: a near-maximal effort is also a sprint and an
HSR effort. Every effort starts at the entry speed and ends when speed drops
below **90 %** of it (hysteresis); efforts shorter than **1 s** are ignored and
efforts longer than 20 s are treated as artefacts.

| Band | Absolute | Normalised | Sources |
|---|---|---|---|
| High-speed running (HSR) | >= 4.72 m/s (17.0 km/h) | >= 75 % MSS | Malone et al. (2016, 2017); Freeman et al. (2023) |
| Sprinting | >= 6.11 m/s (22.0 km/h) | >= 85 % MSS | Malone et al. (2016, 2017); Freeman et al. (2023) |
| Near-maximal running | >= 7.00 m/s (25.2 km/h) | >= 90 % MSS | Freeman et al. (2023) |
| Speed exposure (count only) | - | >= 95 % MSS | |

## Seasonal reference values (per player-season)

| Reference | Definition |
|---|---|
| **MSS**, maximal seasonal speed | Highest speed reached on >= 3 separate efforts (3rd-highest peak of efforts >= 4.72 m/s). Peaks > 10.5 m/s are discarded as artefacts. With fewer than 4 halves or 3 efforts, the single highest peak is used and flagged. |
| **Smax** | Single highest effort peak (reported alongside MSS) |
| **MSA**, maximal seasonal acceleration | 3rd-highest peak of acceleration events |
| **MSD**, maximal seasonal deceleration | 3rd-highest peak magnitude of deceleration events |

MSS, MSA and MSD are match-derived references, not physiological maxima.

## Acceleration and deceleration events

An event is a stretch where acceleration stays >= +1 m/s² (or <= -1 m/s²) for
at least **0.5 s**. Each event is put in one zone by its peak magnitude, and its
speed context is the speed at onset.

| Zone | Absolute (Abbott et al., 2018) | Normalised (Sonderegger et al., 2016) |
|---|---|---|
| Low | 1.0-2.0 m/s² | 25-50 % of MSA / MSD |
| Moderate | 2.0-3.0 m/s² | 50-75 % of MSA / MSD |
| High | >= 3.0 m/s² | >= 75 % of MSA / MSD |
| Speed context | below HSR (< 4.72 m/s) or at/above HSR at onset | |

## Curvilinear running

Tangent-chord angle (Fitzpatrick et al., 2019) of every speed-band effort and
every high (>= 3 m/s²) acceleration / deceleration: the unsigned angle between
the initial heading (first 3 positions) and the straight line from start to
finish. 0° = straight. Summarised in bins of 0-5°, 5-10°, 10-15° and 15°+.
Exploratory: based on a few GNSS positions and not validated against a
criterion measure.

## Worst-case scenarios

Greatest total distance and absolute HSR distance (speed >= 4.72 m/s) in any
rolling **1, 2, 3, 5 and 10-min** window, reported as m/min. Windows never
cross half-time or a GNSS drop-out.

## Statistics

| Analysis | Method |
|---|---|
| Squad-level season contrasts | Linear mixed-effects model `metric ~ season + venue + opponent points + (1 \| player)`, REML (statsmodels), fitted for each adjacent pair of seasons; estimates with 95 % CI |
| Venue and opponent strength | Match-level regression of team means (n = matches) on season, venue and opponent points per +1 SD |
| Positional lines | % change and Hedges' g (small-sample corrected) with 95 % bootstrap CI (5,000 resamples) between adjacent seasons; descriptive |
| Magnitude words | Hopkins et al. (2009): trivial < 0.2, small 0.2-0.6, moderate 0.6-1.2, large 1.2-2.0, very large > 2.0 |

## Output metrics

`outputs/match_metrics.csv` has one row per eligible player-match:

| Group | Columns |
|---|---|
| Volume | `Duration (min)`, `Total Distance` (m), `Distance Per Min` (m/min), `Max Speed` (m/s) |
| Speed bands, absolute and normalised | for each of HSR, Sprint, Near-maximal and each of `(Abs)` / `(Rel)`: `Efforts`, `Distance` (m), `Per Minute` (m/min) |
| Speed exposure | `Speed 95 Efforts (Rel)` |
| Acceleration | `Accel Low / Moderate / High (Abs)` and `(Rel)`, counts |
| Deceleration | `Decel Low / Moderate / High (Abs)` and `(Rel)`, counts |
| Match context | `Competition`, `Opposition`, `Venue`, `Opponent Points`, `Position` |

Other outputs:

| File | Contents |
|---|---|
| `seasonal_references.csv` | MSS, Smax, MSA, MSD per player-season, with effort counts and any low-exposure flag |
| `efforts.csv` | every speed-band effort: band, threshold type, entry speed, duration, distance, peak speed, peak acceleration, onset speed, tangent-chord angle |
| `accel_events.csv` | every acceleration / deceleration event: direction, duration, distance, peak, onset speed, absolute and normalised zone, speed context, angle |
| `season_contrasts_model.csv` | mixed-model season estimates, 95 % CI, SE, n, convergence flag |
| `match_context.csv` | venue and opponent-strength effects |
| `season_contrasts_by_position.csv` | positional % change and Hedges' g with 95 % CI |
| `curvilinear_summary.csv` | angle mean ± SD and % in each bin, by band and season |
| `worst_case_by_match.csv`, `worst_case_by_season.csv` | peak window intensities |

All thresholds live in `common.py`; change them there, not in the scripts.

See `docs/paper_vs_toolkit.md` for where this differs from the original analysis code.
