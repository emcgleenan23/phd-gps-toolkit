"""
Step 2 - match running demands for every eligible player-match.

1. Inclusion
   * a player-match counts if the player was on the pitch for >= 40 of the
     60 regulation minutes;
   * a player-season counts if it has >= 3 such matches (each season separately).

2. Seasonal references, per player-season (from eligible matches only)
   * MSS: maximal seasonal speed = the highest speed reached in at least 3
     separate efforts (3rd-highest effort peak). Players with fewer than
     4 halves or 3 efforts fall back to their highest peak and are flagged.
   * MSA / MSD: the same idea for acceleration and deceleration events.

3. Per player-match metrics
   * duration, total distance, distance per minute, maximum speed
   * for each speed band (absolute: 4.72 / 6.11 / 7.00 m/s; normalised:
     75 / 85 / 90 % of MSS): number of efforts, distance, distance per minute.
     Bands are cumulative (a sprint is also an HSR effort).
   * efforts above 95 % MSS
   * acceleration and deceleration events by zone, absolute (1-2, 2-3,
     >= 3 m/s²) and normalised (25-50, 50-75, >= 75 % of MSA / MSD), and how
     many began at or above HSR speed
   * the tangent-chord angle of every effort and event (for step 4)

Outputs (outputs/):
    match_metrics.csv          one row per player-match (input to steps 3 and 5)
    efforts.csv                one row per speed-band effort
    accel_events.csv           one row per acceleration / deceleration event
    seasonal_references.csv    MSS, MSA, MSD per player-season
"""

import numpy as np
import pandas as pd

import common
from gpskit.efforts import (accel_events, continuous_runs, effort_summary, effort_windows,
                            kth_highest, tangent_chord_angle, zone)

_trapz = getattr(np, "trapezoid", None) or np.trapz
FS = common.FS


def runs_of(g):
    """Continuous runs of one player-match: list of (half, speed, accel, lat, lon)."""
    out = []
    for (half, run), r in g.groupby(["half", "run"], sort=True):
        out.append((half, r["speed"].to_numpy(), r["accel"].to_numpy(),
                    r["lat"].to_numpy(), r["lon"].to_numpy()))
    return out


def eligible_player_matches(data: pd.DataFrame) -> pd.DataFrame:
    """Minutes played per player-match and the two inclusion rules."""
    halves = data.groupby(["season", "date", "player", "half"]).t.agg(lambda t: (t.max() - t.min()) / 60)
    minutes = halves.groupby(["season", "date", "player"]).sum().rename("minutes").reset_index()
    minutes = minutes[minutes.minutes >= common.MIN_MATCH_MINUTES]
    games = minutes.groupby(["season", "player"]).date.transform("nunique")
    return minutes[games >= common.MIN_GAMES_PER_SEASON]


def seasonal_references(matches: dict) -> pd.DataFrame:
    """MSS, MSA and MSD for each player-season."""
    peaks, halves, acc_peaks, dec_peaks = {}, {}, {}, {}
    for (season, date, player), runs in matches.items():
        key = (season, player)
        halves.setdefault(key, set()).update((date, h) for h, *_ in runs)
        for _, v, a, *_ in runs:
            for i, j in effort_windows(v, common.MSS_DETECT, fs=FS, exit_fraction=common.EXIT_FRACTION,
                                       min_s=common.MIN_EFFORT_S, max_s=common.MAX_EFFORT_S):
                if v[i:j + 1].max() <= common.MSS_PLAUSIBLE:
                    peaks.setdefault(key, []).append(v[i:j + 1].max())
            for ev in accel_events(v, a, fs=FS, floor=common.ACCEL_FLOOR, min_s=common.ACCEL_MIN_S):
                (acc_peaks if ev["direction"] == "acc" else dec_peaks).setdefault(key, []).append(
                    abs(ev["peak_accel"]))
    rows = []
    for key in sorted(halves):
        p = peaks.get(key, [])
        ok = len(p) >= common.MSS_K and len(halves[key]) >= common.MSS_MIN_HALVES
        rows.append({"season": key[0], "player": key[1],
                     "MSS": kth_highest(p, common.MSS_K) if ok else (max(p) if p else np.nan),
                     "MSS_flag": "" if ok else "low exposure: highest peak used",
                     "Smax": max(p) if p else np.nan,
                     "MSA": kth_highest(acc_peaks.get(key, []), common.MSS_K),
                     "MSD": kth_highest(dec_peaks.get(key, []), common.MSS_K),
                     "n_efforts": len(p), "n_halves": len(halves[key])})
    return pd.DataFrame(rows)


def match_row(season, date, player, minutes, runs, ref, effort_rows, event_rows):
    """All metrics for one player-match; appends effort / event detail rows."""
    row = {"Season": season, "Date": date, "Player": player, "Duration (min)": minutes}
    td = sum(_trapz(v, dx=1 / FS) for _, v, *_ in runs)
    row["Total Distance"] = td
    row["Distance Per Min"] = td / minutes
    row["Max Speed"] = max(v.max() for _, v, *_ in runs)

    bands = [("Abs", name, thr) for name, thr in common.ABSOLUTE_BANDS.items()]
    bands += [("Rel", name, frac * ref.MSS) for name, frac in common.NORMALISED_BANDS.items()]
    bands += [("Rel", name, frac * ref.MSS) for name, frac in common.SPEED_EXPOSURE.items()]
    for kind, name, entry in bands:
        count = dist = 0
        if np.isfinite(entry):
            for half, v, a, lat, lon in runs:
                for i, j in effort_windows(v, entry, fs=FS, exit_fraction=common.EXIT_FRACTION,
                                           min_s=common.MIN_EFFORT_S, max_s=common.MAX_EFFORT_S):
                    e = effort_summary(v, a, i, j, FS)
                    count += 1
                    dist += e["distance_m"]
                    effort_rows.append({"season": season, "date": date, "player": player, "half": half,
                                        "threshold": kind, "band": name, "entry_ms": entry, **e,
                                        "angle_deg": tangent_chord_angle(lat[i:j + 1], lon[i:j + 1],
                                                                         common.TANGENT_POINTS)})
        row[f"{name} Efforts ({kind})"] = count
        if name != "Speed 95":
            row[f"{name} Distance ({kind})"] = dist
            row[f"{name} Per Minute ({kind})"] = dist / minutes

    for half, v, a, lat, lon in runs:
        for ev in accel_events(v, a, fs=FS, floor=common.ACCEL_FLOOR, min_s=common.ACCEL_MIN_S):
            ref_mag = ref.MSA if ev["direction"] == "acc" else ref.MSD
            rel = ev["peak_accel"] / ref_mag if np.isfinite(ref_mag) else np.nan
            s, e = ev["start"], ev["stop"]
            event_rows.append({"season": season, "date": date, "player": player, "half": half, **ev,
                               "zone_abs": zone(ev["peak_accel"], common.ACCEL_ABS_EDGES),
                               "zone_rel": zone(rel, common.ACCEL_REL_EDGES) if np.isfinite(rel) else None,
                               "context": "hsr_plus" if ev["onset_speed"] >= common.SPEED_CONTEXT_MS
                               else "below_hsr",
                               "angle_deg": tangent_chord_angle(lat[s:e], lon[s:e], common.TANGENT_POINTS)})
    ev = pd.DataFrame([r for r in event_rows if r["date"] == date and r["player"] == player])
    for direction, label in (("acc", "Accel"), ("dec", "Decel")):
        d = ev[ev.direction == direction] if len(ev) else ev
        for z in ("low", "moderate", "high"):
            row[f"{label} {z.capitalize()} (Abs)"] = int((d.zone_abs == z).sum()) if len(d) else 0
            row[f"{label} {z.capitalize()} (Rel)"] = int((d.zone_rel == z).sum()) if len(d) else 0
    return row


def main():
    data = common.load_cache()
    keep = eligible_player_matches(data)
    data = data.merge(keep[["season", "date", "player"]], on=["season", "date", "player"])
    print(f"{len(keep)} eligible player-matches, {keep.player.nunique()} players, "
          f"{keep.date.nunique()} matches")

    matches = {k: runs_of(g) for k, g in data.groupby(["season", "date", "player"])}
    refs = seasonal_references(matches)
    refs.to_csv(common.output_path("seasonal_references.csv"), index=False, float_format="%.3f")
    ref_lookup = refs.set_index(["season", "player"])

    rows, effort_rows, event_rows = [], [], []
    minutes = keep.set_index(["season", "date", "player"]).minutes
    for (season, date, player), runs in matches.items():
        rows.append(match_row(season, date, player, minutes[(season, date, player)], runs,
                              ref_lookup.loc[(season, player)], effort_rows, event_rows))

    table = pd.DataFrame(rows)
    fixtures = pd.read_csv(common.FIXTURES_CSV, dtype={"Date": str})
    table = table.merge(fixtures, on="Date", how="left")
    if common.POSITIONS_CSV.exists():
        table = table.merge(pd.read_csv(common.POSITIONS_CSV), on="Player", how="left")
    table.to_csv(common.output_path("match_metrics.csv"), index=False, float_format="%.3f")
    pd.DataFrame(effort_rows).to_csv(common.output_path("efforts.csv"), index=False, float_format="%.3f")
    pd.DataFrame(event_rows).to_csv(common.output_path("accel_events.csv"), index=False, float_format="%.3f")

    print("\nSmax and MSS by season (mean ± SD, m/s)")
    for season, g in refs.groupby("season"):
        print(f"  {season}: Smax {g.Smax.mean():.2f} ± {g.Smax.std():.2f}   "
              f"MSS {g.MSS.mean():.2f} ± {g.MSS.std():.2f}   (n = {len(g)})")


if __name__ == "__main__":
    main()
