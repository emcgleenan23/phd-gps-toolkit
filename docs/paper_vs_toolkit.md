# Methods: papers vs this toolkit

The code for the first of these studies was written in 2024. This toolkit is a
cleaned-up, documented rewrite, and in a few places it recommends something
slightly different from what the original analysis code did. Each difference
is listed here so you can compare toolkit output with the published papers.

Where an option reproduces the original behaviour, it is named.

## All studies

### Butterworth filter order
The thesis (section 2.5.2) describes a 2nd-order, zero-phase (filtfilt)
low-pass Butterworth filter for all studies. The toolkit uses that throughout.
The original analysis code used a **1st-order** filter for the ballistic-power
study (2 Hz) and the running-demands study (1 Hz). The in-situ PFv study used
2nd order (1 Hz), as described.

The toolkit also splits each trace at GNSS drop-outs longer than 0.5 s and
filters each continuous piece separately, so smoothing never bridges a gap.

*Original behaviour:* `--filter-order 1` (ballistic power).

## Ballistic power (McGleenan et al., 2026a)

### De-duplicating raw rows
STATSports exports repeat each 10 Hz GNSS sample on ~10 rows. The original
code reduced a repetition with `Speed.unique()`, which also drops any genuine
repeat of a speed value and shifts later samples in time. The toolkit keeps
one sample per timestamp.
*Original behaviour:* `--dedupe unique`.

### Derivative spacing
Acceleration is the derivative of a 100-point polynomial speed curve. The
original code used a spacing of range / 100; the exact spacing is
range / 99. The original choice scales acceleration, Fmax, Pmax and PB up by
about 1 %.
*Original behaviour:* `--grid-spacing n`.

### Two different "SFv" lines
The paper's SFv is the slope of a straight line through the whole
accelerating A-S curve (`SFv_poly`), while dFv, A0 and S0 come from a line
fitted to 30-90 % of the speed range. The toolkit reports both slopes.

### Mixed-effects models
The paper fitted `Fmax/Pmax/PB ~ vmax + (1 | player)` in R (nlme). The R code
is not part of this repository; `03_mixed_models.py` fits the same models in
Python (statsmodels, REML).

## In-situ PFv (McGleenan et al., 2026b)

### Table 3: in-situ vs isolated sprint
* **A0 and S0 are intercepts on both sides.** In the published table the
  isolated-sprint "A0" and "S0" were the sprint's observed peak acceleration
  and top speed, while the in-situ values were extrapolated intercepts. The
  toolkit compares intercept with intercept, and reports the observed peaks
  separately as **Amax** and **Smax** (in-situ: highest acceleration and speed
  in the player's filtered data; isolated: peaks of the sprint's polynomial
  fit). The same applies to the isolated column of Table 1.
* **Each condition uses its own players.** In the published table, the
  isolated mean ± SD and the bias in the Training and Combined rows repeated
  the Match row. The toolkit pairs every condition with its own players (the
  published d and p values were already paired correctly).

With intercepts compared like for like, in-situ A0 is lower than the
isolated-sprint A0 (match d = -1.07), while in-situ Amax is higher than the
sprint's peak acceleration (match d = +0.93).

### Pmax and PB carry the same information in-situ
Because in-situ profiles are linear, PB = A0·S0/2 = 2·Pmax, so the two
metrics carry the same relative information in-situ (their % changes in
Table 5 are identical). PB's additional sensitivity (ballistic-power study)
arises from the curvature of the isolated-sprint A-S profile.

### Minimum data load (Table 4)
The paper describes 5-min steps within a session and 10-min steps across
sessions. The published values were produced with continuous 1-min steps,
which is what `04_minimum_data_load.py` runs. The reported value is taken
from the group-mean R² curve; the ± is a 95 % CI half-width
(1.96·SD/√n) of the same quantity computed for each player individually.

### Two ways of pooling sessions
Tables 1-2 re-select the top 2 points per speed bin across all pooled
sessions (about 50 points per profile). Table 3 pools every session's points
without re-selecting (several hundred points) and requires R² ≥ 0.90.
`pooled_profile(..., rebin=True/False)` selects between them.

### Reported SDs
"Mean ± SD" in the paper uses the population SD (`numpy.std`, ddof = 0).

## Match running demands (McGleenan et al., under review)

The toolkit applies the method described in the manuscript in one consistent
way. The original analysis combined two code bases (an older club pipeline and
newer study scripts), which differ from the manuscript text in the places below.

### Match table used for the season models
The original season contrasts were fitted to match metrics produced by the
older club pipeline, which:
* summed the speed of every sample above the HSR threshold, without the 90 %
  exit or the 1-s minimum effort duration;
* applied the sprint hysteresis to raw (unfiltered) speed, with no minimum
  duration;
* computed absolute near-maximal distance without hysteresis or a minimum duration;
* counted "high" accelerations and decelerations at ±2.78 m/s², not the
  ≥ 3.0 m/s² band, and added one event for every 0.5 s spent above the
  threshold (so a long acceleration counted several times);
* took maximum speed from raw speed.

The toolkit computes every match metric with the method in the manuscript:
efforts on filtered speed, 90 % exit, 1-s minimum, distance by trapezoidal
integration of filtered speed; acceleration events ≥ 1 m/s² for ≥ 0.5 s,
zoned by their peak (1-2, 2-3, ≥ 3 m/s²); maximum speed from filtered speed.

### Effort, acceleration and curvilinear analyses
* Halves are cropped to 30 min in every analysis (originally only in the
  match table).
* Curvilinear acceleration events use the "high" zone (≥ 3.0 m/s²) rather than
  ±2.78 m/s².

### Statistics
* Hedges' g intervals are 95 % bootstrap intervals, as stated in the
  manuscript (the original code used 90 %).
* Opponent strength in the match-level model is scaled per SD of opponent
  points across player-match rows, as in the original analysis.
