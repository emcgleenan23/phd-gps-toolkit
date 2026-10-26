"""
Step 3 - how do Fmax, Pmax and PB scale with sprint speed?

The paper fitted linear mixed-effects models in R (nlme):

    Fmax ~ vmax + (1 | player)
    Pmax ~ vmax + (1 | player)
    PB   ~ vmax + (1 | player)

Each player has their own intercept (random effect), because the 15
repetitions of one player are not independent. The R code was not kept with
this repository; this script fits the same models in Python (statsmodels,
REML), which should give practically identical fixed effects.

Reported per model: fixed-effect intercept and slope with 95 % CI, the
intraclass correlation (ICC = between-player variance / total variance), and
marginal / conditional R² (Nakagawa & Schielzeth, 2013).

Usage:  python 03_mixed_models.py [--reps outputs/rsa_reps.csv]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=Path, default=HERE / "outputs" / "rsa_reps.csv")
    args = ap.parse_args()
    d = pd.read_csv(args.reps)

    for y in ("Fmax", "Pmax", "PB"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"{y} ~ vmax", d, groups=d["player"]).fit(reml=True)
        ci = fit.conf_int()
        var_player = float(fit.cov_re.iloc[0, 0])
        var_resid = float(fit.scale)
        var_fixed = float(np.var(fit.fe_params["Intercept"] + fit.fe_params["vmax"] * d["vmax"]))
        total = var_fixed + var_player + var_resid
        print(f"{y} = {fit.fe_params['Intercept']:.2f} + {fit.fe_params['vmax']:.2f} · vmax"
              f"   slope 95 % CI [{ci.loc['vmax', 0]:.2f}, {ci.loc['vmax', 1]:.2f}]")
        print(f"   ICC = {var_player / (var_player + var_resid):.2f}   "
              f"R² marginal = {var_fixed / total:.2f}   conditional = {(var_fixed + var_player) / total:.2f}\n")


if __name__ == "__main__":
    main()
