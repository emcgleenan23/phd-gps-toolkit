"""
Statistics used across the studies.

Effect sizes
------------
* `hedges_g`      : standardised difference between two independent groups
                    (e.g. season A vs season B), with the small-sample
                    correction. Positive = the second group is higher.
* `paired_d`      : Cohen's d for paired data (same players measured twice),
                    mean difference / SD of the differences.
* `magnitude`     : Hopkins et al. (2009) words for an effect size.
* `bootstrap_ci`  : percentile bootstrap confidence interval for any
                    two-sample statistic.
* `bland_altman`  : bias and 95 % limits of agreement between two methods.

Models
------
* `season_contrasts` : linear mixed-effects model
                       metric ~ season + venue + opponent points + (1 | player)
                       used for the squad-level season contrasts in the
                       running-demands study.
* `match_context`    : match-level regression of team means on season, venue
                       and opponent strength (per +1 SD).

A note on SDs: the in-situ PFv paper reports "mean ± SD" using the
population SD (numpy's default, ddof=0). Use `np.std(x)` to reproduce those
tables and `np.std(x, ddof=1)` for the usual sample SD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Hopkins et al. (2009) thresholds for standardised effects.
HOPKINS_BANDS = [(0.2, "trivial"), (0.6, "small"), (1.2, "moderate"),
                 (2.0, "large"), (4.0, "very large"), (np.inf, "extremely large")]


def magnitude(effect: float) -> str:
    """Hopkins magnitude word for an absolute standardised effect size."""
    if not np.isfinite(effect):
        return "unclear"
    for upper, word in HOPKINS_BANDS:
        if abs(effect) < upper:
            return word
    return "extremely large"


def _clean(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[~np.isnan(x)]


def hedges_g(group_a, group_b) -> float:
    """
    Hedges' g = (mean_b - mean_a) / pooled SD, multiplied by the small-sample
    correction J = 1 - 3 / (4·(n_a + n_b - 2) - 1).
    """
    a, b = _clean(group_a), _clean(group_b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    j = 1 - 3 / (4 * (na + nb - 2) - 1)
    return float((b.mean() - a.mean()) / pooled * j)


def bootstrap_ci(group_a, group_b, statistic=hedges_g, n_boot: int = 5000,
                 level: float = 0.95, seed: int = 17) -> tuple[float, float]:
    """
    Percentile bootstrap CI: resample each group with replacement `n_boot`
    times, recompute the statistic, take the middle `level` of the results.
    A fixed seed makes the interval reproducible.
    """
    a, b = _clean(group_a), _clean(group_b)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = np.array([statistic(rng.choice(a, len(a)), rng.choice(b, len(b)))
                      for _ in range(n_boot)])
    stats = stats[np.isfinite(stats)]
    tail = (1 - level) / 2 * 100
    return (float(np.percentile(stats, tail)), float(np.percentile(stats, 100 - tail)))


def compare_groups(group_a, group_b, level: float = 0.95, n_boot: int = 5000) -> dict:
    """Means, raw and % change, Hedges' g with bootstrap CI, and its magnitude word."""
    a, b = _clean(group_a), _clean(group_b)
    g = hedges_g(a, b)
    lo, hi = bootstrap_ci(a, b, level=level, n_boot=n_boot)
    mean_a, mean_b = a.mean() if len(a) else np.nan, b.mean() if len(b) else np.nan
    return {
        "n_a": len(a), "n_b": len(b), "mean_a": mean_a, "mean_b": mean_b,
        "diff": mean_b - mean_a,
        "pct_change": (mean_b - mean_a) / mean_a * 100 if mean_a else np.nan,
        "g": g, "g_low": lo, "g_high": hi, "magnitude": magnitude(g),
    }


def paired_d(x, y) -> float:
    """Cohen's d for paired samples: mean(x - y) / SD(x - y) (population SD)."""
    diff = np.asarray(x, float) - np.asarray(y, float)
    sd = diff.std()
    return float(diff.mean() / sd) if sd > 0 else np.nan


def bland_altman(x, y) -> dict:
    """Bias (mean of x - y) and 95 % limits of agreement (bias ± 1.96·SD)."""
    diff = np.asarray(x, float) - np.asarray(y, float)
    bias, sd = diff.mean(), diff.std()
    return {"bias": float(bias), "loa_low": float(bias - 1.96 * sd),
            "loa_high": float(bias + 1.96 * sd)}


# ---------------------------------------------------------------------------
# Models for the running-demands study
# ---------------------------------------------------------------------------

def season_contrasts(df: pd.DataFrame, metric: str, *, season="Season",
                     player="Player", venue="Venue", opponent="opp_pts",
                     baseline: int | None = None) -> pd.DataFrame:
    """
    Linear mixed-effects model, fitted by REML with statsmodels:

        metric ~ C(season) + C(venue) + opponent + (1 | player)

    The random intercept for player stops repeated matches by the same player
    being treated as independent observations. Venue and opponent strength
    are held constant, so the season terms are "adjusted" season differences.

    Parameters
    ----------
    df : one row per player-match.
    metric : column to model.
    baseline : season used as the reference level. The returned contrasts are
        (each other season) - baseline. Fit once with baseline=2024 for
        2024->2025 and once with baseline=2025 for 2025->2026.

    Returns
    -------
    DataFrame with one row per non-baseline season: estimate, 95 % CI, SE and
    whether the optimiser converged cleanly (check any False rows).
    """
    import statsmodels.formula.api as smf  # imported here: only this function needs it

    data = df[[metric, season, player, venue, opponent]].dropna().rename(
        columns={metric: "y", season: "season", player: "player",
                 venue: "venue", opponent: "opp"})
    if baseline is None:
        baseline = int(data["season"].min())

    model = smf.mixedlm(f"y ~ C(season, Treatment({baseline})) + C(venue) + opp",
                        data, groups=data["player"])
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = model.fit(reml=True)
    # Small samples often trigger optimiser warnings; report them instead of printing.
    converged = bool(fit.converged) and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    ci = fit.conf_int()

    rows = []
    for name in fit.params.index:
        if name.startswith("C(season"):
            other = int(name.split("[T.")[1].rstrip("]"))
            rows.append({"metric": metric, "contrast": f"{baseline}->{other}",
                         "estimate": fit.params[name], "ci_low": ci.loc[name, 0],
                         "ci_high": ci.loc[name, 1], "se": fit.bse[name],
                         "n_obs": int(fit.nobs), "n_players": data["player"].nunique(),
                         "converged": converged})
    return pd.DataFrame(rows)


def match_context(df: pd.DataFrame, metric: str, *, match="Date", season="Season",
                  venue="Venue", opponent="opp_pts", home_label="Home") -> pd.DataFrame:
    """
    Effects of venue and opponent strength, estimated at the MATCH level.

    Venue and opponent are the same for every player in a match, so fitting
    them on player-match rows would overstate their precision. Instead we
    average each match to one team value and fit an ordinary regression:

        team_mean ~ C(season) + C(venue) + opponent_z

    opponent_z is opponent league points divided by the SD of opponent points
    across player-match rows, so the coefficient reads "per +1 SD".
    """
    import statsmodels.formula.api as smf

    sd_opp = df[opponent].std()
    team = (df.groupby(match)
              .agg(y=(metric, "mean"), season=(season, "first"),
                   venue=(venue, "first"), opp=(opponent, "first"))
              .reset_index())
    team["opp_z"] = team["opp"] / sd_opp  # centring does not change the slope

    fit = smf.ols(f"y ~ C(season) + C(venue, Treatment('{home_label}')) + opp_z", team).fit()
    ci = fit.conf_int()
    rows = []
    for name in fit.params.index:
        if "venue" in name or name == "opp_z":
            label = "away vs home" if "venue" in name else "per +1 SD opponent points"
            rows.append({"metric": metric, "term": label, "estimate": fit.params[name],
                         "ci_low": ci.loc[name, 0], "ci_high": ci.loc[name, 1],
                         "n_matches": len(team)})
    return pd.DataFrame(rows)
