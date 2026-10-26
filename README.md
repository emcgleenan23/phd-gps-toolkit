# phd-gps-toolkit

Python tools for analysing **STATSports GNSS (GPS) data** in field sports,
written during my PhD in the Predictive Sports Analytics group at Queen's
University Belfast, in collaboration with STATSports.

The code is organised so you can run each study's analysis **on your own
data**: point the scripts at a folder of STATSports exports and they produce
the same tables and figures as the papers. No athlete data is included; a
generator creates synthetic files so everything can be tried straight away.

| Study | What it measures | Folder |
|---|---|---|
| Ballistic power (McGleenan et al., 2026a) | Force-velocity metrics from isolated maximal sprints, and how they change with fatigue across a repeated-sprint test | [`studies/ballistic_power`](studies/ballistic_power) |
| In-situ PFv profiling (McGleenan et al., 2026b) | Acceleration-speed / force-velocity profiles from ordinary training and match data, how much data a profile needs, and how profiles change within a match | [`studies/insitu_pfv`](studies/insitu_pfv) |
| Match running demands (McGleenan et al., under review) | How match demands changed across three seasons spanning rule amendments and promotion: distance, high-speed running, sprinting, accelerations and curvilinear running, with absolute and individualised thresholds, mixed-effects season contrasts and worst-case-scenario windows | [`studies/running_demands`](studies/running_demands) |

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/emcgleenan23/phd-gps-toolkit.git
cd phd-gps-toolkit
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python demo_data/make_demo_data.py     # synthetic data, ~2 minutes

cd studies/insitu_pfv
python 01_build_cache.py
python 02_pooled_profiles.py
```

Each study folder has its own README with the full sequence of scripts. The
demo data cover all three studies: one repeated-sprint session, a block of
training and matches, and three short seasons of league matches with positions.

## Using your own data

The scripts expect STATSports **raw data (extended)** CSV exports:

* one file per session, named `YYYY-MM-DD-<anything>-Match.csv` or
  `YYYY-MM-DD-<anything>-Training.csv`;
* columns including `Player Display Name`, `Time`, `Speed (m/s)` and
  `Drill Title` (plus `Lat`, `Lon` for curvilinear analyses);
* drills named `1st Half` / `2nd Half` for matches, `Entire Session` for
  training and `Rep 1` ... `Rep 15` for a repeated-sprint test.

The running-demands study also needs two small tables you write yourself:
a fixture list (date, competition, venue, opponent league points) and a
player-position list. Templates are produced with the demo data.

Point a study at your folder with `--raw /path/to/exports` (or the
environment variable named in the study's `common.py`). Use `--anonymise`
when building a cache to replace names with P01, P02, ...; the key is saved
next to the cache, never alongside results.

`.gitignore` excludes data, caches and outputs, so real athlete data cannot be
committed by accident. Check your own ethics and data-sharing agreements before
publishing any results.

## What's in the box

```
gpskit/                 reusable functions (import these in your own work)
  io.py                 read STATSports exports (one row per player per 10 Hz sample)
  filtering.py          zero-phase Butterworth smoothing, acceleration
  profile.py            in-situ acceleration-speed / force-velocity profiling
  sprint.py             profiling an isolated maximal sprint (ballistic power)
  efforts.py            high-speed efforts, acceleration events, tangent-chord angle, peak windows
  stats.py              effect sizes, bootstrap CIs, Bland-Altman, mixed-effects models
  synthetic.py          invented demo data in the STATSports format
studies/                one folder per paper; numbered scripts run in order
demo_data/              make_demo_data.py (generated files are not tracked)
docs/paper_vs_toolkit.md  where the toolkit differs from the published methods
tests/                  python -m pytest
```

## Processing choices (all studies)

* GNSS speed sampled at 10 Hz.
* Low-pass, **2nd-order, zero-phase Butterworth** filter
  (`scipy.signal.butter` + `filtfilt`): 2 Hz for isolated sprints, 1 Hz for
  training and match data. Traces are filtered within continuous stretches so
  smoothing never bridges a GNSS drop-out.
* Acceleration by central differences of filtered speed (`numpy.gradient`).
* Running efforts (running-demands study): an effort starts at the band's
  entry speed and ends when speed drops below 90 % of it, with a 1-s minimum
  duration; acceleration events are >= 1 m/s² for >= 0.5 s and are zoned by
  their peak.

The comments in each module explain why. Filter settings should ideally be
chosen for your own device (e.g. by residual analysis) and always reported.

## Papers vs toolkit

The first of these analyses was written in 2024. The toolkit is a cleaned-up
rewrite and in a few places recommends something slightly different from what
the original analysis code did (for example, the filter order). Every
difference is listed in [`docs/paper_vs_toolkit.md`](docs/paper_vs_toolkit.md).

## Citing

If you use this code, please cite the relevant paper(s):

* McGleenan, E., Grant, S., McFetridge, L.M., Brown, J.C.J., Shaw, J.W.,
  de Melo Malaquias, T., Montgomery, R. & Jess, D.B. (2026). Ballistic power:
  GPS-derived measurements suitable for the real-time monitoring of
  neuromuscular fatigue during repeated sprints. *International Journal of
  Sports Science & Coaching*. https://doi.org/10.1177/17479541251413077
* McGleenan, E., Grant, S., McFetridge, L.M., Brown, J.C.J., Hawkins, R.,
  de Melo Malaquias, T., Montgomery, R., McCann, R. & Jess, D.B. (2026).
  In-situ PFv profiling in sub-elite Gaelic football: validation, comparison
  with isolated sprints, minimum data load requirements, and exploring
  "real-time" monitoring potential. *International Journal of Sports Science
  & Coaching*. https://doi.org/10.1177/17479541261468202

* McGleenan, E., Harvey, C., Grant, S.D.T., McFetridge, L.M., Brown, J.C.J.,
  de Melo Malaquias, T., Montgomery, R., McGahan, J. & Jess, D.B. Match
  running demands in club Gaelic football across three seasons spanning rule
  amendments and promotion: a single-club GNSS case study. *Manuscript under
  review* (full reference to follow).

`CITATION.cff` lets GitHub produce the citation for you ("Cite this repository").

## Licence

MIT - see [`LICENSE`](LICENSE). Questions and suggestions are welcome via
GitHub issues.
