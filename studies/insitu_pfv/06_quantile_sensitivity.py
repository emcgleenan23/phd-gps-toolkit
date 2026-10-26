"""
Step 6 (optional) - sensitivity check: ordinary vs quantile regression.

Miguens et al. (2024) fit the A-S line with quantile regression across a grid
of quantiles (0.05 to 0.95) and average the resulting lines. The spread of A0
and S0 across quantiles then gives a built-in uncertainty estimate.

This script fits both versions to the SAME pooled IF points (same selection,
same 1.96 SD outlier step); only the line-fitting method differs. The paper
reports that the differences were trivial, so the ordinary fit was kept.

Output: outputs/quantile_sensitivity.csv
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import common
from gpskit.profile import OUTLIER_SD, line_metrics, pooled_profile, select_if_points, session_if_points

QUANTILES = np.arange(0.05, 0.96, 0.01)


def quantile_line(x, y, q):
    """
    Straight line at quantile q (minimises the "pinball" loss), solved as a
    linear programme. Returns (slope, intercept).
    """
    n = len(x)
    # variables: slope+, slope-, intercept+, intercept-, residual above (n), residual below (n)
    cost = np.concatenate([[0, 0, 0, 0], np.full(n, q), np.full(n, 1 - q)])
    A = np.zeros((n, 4 + 2 * n))
    A[:, 0], A[:, 1], A[:, 2], A[:, 3] = x, -x, 1, -1
    A[np.arange(n), 4 + np.arange(n)] = 1
    A[np.arange(n), 4 + n + np.arange(n)] = -1
    res = linprog(cost, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    return (res.x[0] - res.x[1], res.x[2] - res.x[3]) if res.success else (np.nan, np.nan)


def quantile_profile(sp, ac):
    """Average the quantile-regression lines over QUANTILES (after the usual SD step)."""
    slope, intercept = np.polyfit(sp, ac, 1)
    resid = ac - (slope * sp + intercept)
    keep = np.abs(resid) <= OUTLIER_SD * np.std(resid)
    lines = [quantile_line(sp[keep], ac[keep], q) for q in QUANTILES]
    lines = [(m, b) for m, b in lines if np.isfinite(m) and m < 0 and b > 0]
    if len(lines) < 3:
        return None
    A0 = np.mean([b for _, b in lines])
    S0 = np.mean([-b / m for m, b in lines])
    out = line_metrics(np.mean([m for m, _ in lines]), A0)
    out.update(A0=A0, S0=S0, A0_spread=np.std([b for _, b in lines]),
               S0_spread=np.std([-b / m for m, b in lines]))
    return out


def main():
    rows = []
    for player, sessions in common.iter_players():
        for kind in ("match", "training"):
            pts = [session_if_points(s["speed"], common.FS) for s in sessions if s["type"] == kind]
            pts = [p for p in pts if len(p[0]) >= 2]
            if len(pts) < common.MIN_SESSIONS:
                continue
            ols = pooled_profile(pts, rebin=True)
            sp, ac = select_if_points(np.concatenate([p[0] for p in pts]),
                                      np.concatenate([p[1] for p in pts]))
            qr = quantile_profile(sp, ac)
            if ols is None or qr is None:
                continue
            rows.append({"player": player, "type": kind,
                         **{f"{m}_ols": getattr(ols, m) for m in ("A0", "S0", "PB")},
                         **{f"{m}_quantile": qr[m] for m in ("A0", "S0", "PB")},
                         "A0_spread": qr["A0_spread"], "S0_spread": qr["S0_spread"]})
    df = pd.DataFrame(rows)
    df.to_csv(common.output_path("quantile_sensitivity.csv"), index=False, float_format="%.4f")
    for kind, g in df.groupby("type"):
        print(f"{kind}: n = {len(g)}")
        for m in ("A0", "S0", "PB"):
            diff = g[f"{m}_quantile"] - g[f"{m}_ols"]
            print(f"   {m:3s} quantile - OLS = {diff.mean():+.3f} (SD {diff.std():.3f})")


if __name__ == "__main__":
    main()
