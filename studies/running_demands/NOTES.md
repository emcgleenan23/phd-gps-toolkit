# Running-demands study: original code vs this toolkit

(Kept with the study while the paper is under review; move into
`docs/paper_vs_toolkit.md` when the study is made public.)

The toolkit applies the method described in the manuscript in one consistent
way. The original analysis code mixed two code bases, which differ from the
manuscript text in the places below.

## Filter
* Manuscript: 2nd-order Butterworth, 1 Hz, filtfilt. Original code: 1st order.
* Original code filtered each half (or both halves together) as one trace.
  The toolkit splits at GNSS drop-outs > 0.5 s and filters each piece.

## Match table behind the mixed models (`_model_dataset.csv`)
The season contrasts were fitted to a match table built by the older club
pipeline ("engine"), not the effort detector described in the methods:
* HSR distance: speed of every sample above the threshold, with no 90 %
  exit and no 1-s minimum effort duration.
* Sprint distance: hysteresis applied to *raw* (unfiltered) speed, no
  minimum duration.
* Near-maximal (absolute) distance: no hysteresis or minimum duration.
* "Accel / Decel High (Abs)": threshold ±2.78 m/s² (Johnston et al., 2014),
  not the ≥ 3.0 m/s² band in Table 3, and the counter added one event for
  every 0.5 s spent above the threshold, so a long acceleration counted
  several times.
* "Max Speed": maximum of raw speed.

The toolkit computes every match metric with the method in section 2.3-2.4:
efforts on filtered speed, 90 % exit, 1-s minimum, distance by trapezoidal
integration of filtered speed; acceleration events ≥ 1 m/s² for ≥ 0.5 s,
zoned by peak (1-2, 2-3, ≥ 3 m/s²); maximum speed from filtered speed.

## Effort and acceleration analyses
* Halves were not cropped to 30 min in the effort / acceleration / curvilinear
  analyses (they were in the match table). The toolkit crops everywhere.
* Curvilinear acceleration events used ±2.78 m/s²; the toolkit uses the
  "high" zone (≥ 3.0 m/s²) to match Table 3.

## Statistics
* Hedges' g intervals: the manuscript states 95 % CIs; the original code used
  90 % bootstrap intervals (`CI_LEVEL = 0.90`). The toolkit uses 95 %.
* Opponent strength in the match-level model is scaled per SD of opponent
  points across player-match rows (as in the original analysis).
