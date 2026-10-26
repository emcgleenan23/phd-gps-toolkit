# In-situ PFv profiling

Code for McGleenan et al. (2026), *In-situ PFv profiling in sub-elite Gaelic
football: validation, comparison with isolated sprints, minimum data load
requirements, and exploring "real-time" monitoring potential.*
https://doi.org/10.1177/17479541261468202

## The idea

Players rarely sprint flat out from a standstill in training or matches, but
over many efforts they produce near-maximal accelerations across a wide range
of speeds. Plotting acceleration against speed and keeping the top of that
cloud gives a straight line, the **acceleration-speed (A-S) profile**
(Morin et al., 2021):

```
a = A0 + SFv · v          A0: max acceleration at 0 speed   S0 = -A0 / SFv: max speed
```

Power metrics follow from the line: Pmax = A0·S0/4, ballistic power
PB = A0·S0/2 (area under the line), and the Fv-offset dFv.

## Scripts

Run them in order from this folder.

| Script | Does | Paper |
|---|---|---|
| `01_build_cache.py` | Reads raw exports once; saves each player's filtered speed per session | - |
| `02_pooled_profiles.py` | One pooled profile per player for matches and for training | Tables 1-2 |
| `03_insitu_vs_isolated.py` | In-situ profiles vs the first sprint of a repeated-sprint test | Table 3 |
| `04_minimum_data_load.py` | How R² improves as data accumulate; threshold and knee times | Table 4, Fig 3 |
| `05_in_match_trends.py` | 7- and 12-min sliding windows through each half | Table 5, Fig 4 |
| `06_quantile_sensitivity.py` | Ordinary vs quantile-regression fits (Miguens et al., 2024) | sensitivity check |

```bash
python 01_build_cache.py                       # demo data
python 01_build_cache.py --raw /path/exports --anonymise   # your data
python 02_pooled_profiles.py
python 03_insitu_vs_isolated.py --rsa ../ballistic_power/outputs/rsa_reps.csv
python 04_minimum_data_load.py
python 05_in_match_trends.py --windows 7 12
python 06_quantile_sensitivity.py
```

Results go to `outputs/`. Script 03 needs the per-repetition output of the
ballistic-power study (run that study first).

## Method in brief

1. Filter speed (1 Hz, 2nd order, zero-phase); acceleration by central differences.
2. Matches: 1st and 2nd half only. Training: the whole session.
3. Per session: keep positive accelerations at >= 3 m/s; in each 0.2 m/s
   speed bin keep the 2 highest accelerations (the "intensity frame").
4. Pool sessions, fit a line, drop points beyond 1.96 SD of the residuals, refit.
5. A pooled profile needs >= 5 sessions of that type.

## Notes

* Because in-situ profiles are straight lines, **PB = 2 · Pmax**: their % changes
  are identical. PB's extra sensitivity to fatigue in the ballistic-power study
  comes from the curved shape of an isolated sprint's A-S profile.
* In Table 3 the toolkit compares intercept with intercept (A0, S0) and reports
  observed peaks separately (Amax, Smax). See `docs/paper_vs_toolkit.md`.
* The demo data are simulated and give cleaner profiles (R² ≈ 0.99) than real
  data (R² ≈ 0.93-0.95).
