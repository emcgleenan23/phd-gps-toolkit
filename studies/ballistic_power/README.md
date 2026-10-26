# Ballistic power

Code for McGleenan et al. (2026), *Ballistic power: GPS-derived measurements
suitable for the real-time monitoring of neuromuscular fatigue during repeated
sprints.* https://doi.org/10.1177/17479541251413077

## Protocol

Repeated-sprint test: 3 sets × 5 maximal 50 m sprints, with a longer rest
between sets. In the STATSports export each sprint is a drill named
`Rep 1` ... `Rep 15`.

## What is measured

For each sprint, speed is smoothed (2 Hz, 2nd-order, zero-phase Butterworth),
the effort is isolated (from 0.25 m/s to the last sample above 70 % of top
speed), and a 4th-order polynomial of speed against time is fitted. Its
derivative gives a clean acceleration curve and the acceleration-speed (A-S)
relationship of the sprint.

| Metric | Meaning |
|---|---|
| vmax | top speed (m/s) |
| Fmax | peak acceleration = peak force per kg (N/kg) |
| Pmax | peak power per kg, acceleration × speed (W/kg) |
| PB | **ballistic power**: area under the A-S curve (W/kg) |
| SFv | slope of a line through the A-S curve |
| dFv | Fv-offset: distance from the origin to the A-S line |

PB summarises force across the whole speed range of the sprint and, because
an isolated sprint's A-S curve is curved, it carries information Pmax does not.
In the paper it was the metric most sensitive to fatigue.

## Scripts

```bash
python 01_sprint_metrics.py                                # demo data
python 01_sprint_metrics.py --raw /path/YYYY-MM-DD-Club-Training.csv
python 02_fatigue_summary.py      # Table 1, % of rep 1 per repetition (Fig 2), first-to-last change
python 03_mixed_models.py         # Fmax / Pmax / PB ~ vmax + (1 | player)
```

`outputs/rsa_reps.csv` (one row per player × repetition) is also the
isolated-sprint input to the in-situ PFv study.

## Notes

* The paper's mixed models were fitted in R (nlme). `03_mixed_models.py` fits the
  same models in Python.
* The toolkit corrects three details of the original code (filter order,
  de-duplication of raw rows, derivative spacing). The original choices are
  available as options; see `docs/paper_vs_toolkit.md`.
