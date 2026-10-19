"""Build model-ready state tables.

Outputs (data/processed):
  chase_states.parquet  - one row per delivery of 2nd innings (+ one synthetic 'start' row per match)
                          describing the state AFTER that delivery.
  first_states.parquet  - same for 1st innings (used by the projected-score regression).
  venue_stats.parquet   - latest venue stats (used by the Streamlit what-if page).

Every history-based feature (Elo, venue averages) uses ONLY matches played before the match in question.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BALLS = 120
PRIOR_SCORE = 165.0  # shrinkage prior for venue first-innings average
PRIOR_N = 8
PRIOR_CHASE_N = 10
# Run-rate features are shrunk towards a typical IPL rate when few balls have been bowled. Without this, one boundary
# off the first ball gives a "current run rate" of 24 an over and the model badly over-reacts to early balls.
PRIOR_RPB = 8.0 / 6  # ~8 runs an over
PRIOR_BALLS = 24     # worth four overs of evidence


def smooth_crr(runs, balls):
    """Current run rate (per over), shrunk towards the typical rate. Equals 8.0 before the first ball."""
    return (runs + PRIOR_BALLS * PRIOR_RPB) * 6 / (balls + PRIOR_BALLS)


def match_level_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Pre-match Elo diff and venue stats, chronologically, with no look-ahead."""
    m = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    elo = {}
    venue_runs, venue_n, venue_chase_w, venue_chase_n = {}, {}, {}, {}
    last_year = {}
    out = []
    for r in m.itertuples():
        t1, t2 = r.team1, r.team2
        for t in (t1, t2):
            if t not in elo:
                elo[t] = 1500.0
            if last_year.get(t) is not None and last_year[t] != r.year:
                elo[t] = 1500 + 0.75 * (elo[t] - 1500)  # regress to mean between seasons
            last_year[t] = r.year
        v = r.venue
        n = venue_n.get(v, 0)
        v_avg = (venue_runs.get(v, 0.0) + PRIOR_N * PRIOR_SCORE) / (n + PRIOR_N)
        cn = venue_chase_n.get(v, 0)
        v_chase = (venue_chase_w.get(v, 0) + PRIOR_CHASE_N * 0.5) / (cn + PRIOR_CHASE_N)
        out.append(
            dict(match_id=r.match_id, elo1=elo[t1], elo2=elo[t2], venue_avg1=v_avg,
                 venue_chase_rate=v_chase, venue_n=n)
        )
        # ---- update AFTER recording features ----
        if r.winner is not None and r.result == "win":
            e1 = 1 / (1 + 10 ** ((elo[t2] - elo[t1]) / 400))
            s1 = 1.0 if r.winner == t1 else 0.0
            elo[t1] += 24 * (s1 - e1)
            elo[t2] -= 24 * (s1 - e1)
        if r.usable:
            venue_n[v] = n + 1
            venue_runs[v] = venue_runs.get(v, 0.0) + getattr(r, "first_total")
            venue_chase_n[v] = cn + 1
            venue_chase_w[v] = venue_chase_w.get(v, 0) + int(getattr(r, "chase_won"))
    return m.merge(pd.DataFrame(out), on="match_id")


def rolling_last(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Runs/wickets in the last `window` legal balls, per (match, innings). df must be in delivery order."""
    r12 = np.zeros(len(df))
    w12 = np.zeros(len(df))
    start = 0
    keys = (df.match_id + "_" + df.innings.astype(str)).values
    lb = df.legal_balls_so_far.values
    cr = df.cum_runs.values
    cw = df.cum_wkts.values
    n = len(df)
    while start < n:
        end = start
        while end < n and keys[end] == keys[start]:
            end += 1
        lbs, crs, cws = lb[start:end], cr[start:end], cw[start:end]
        idx = np.searchsorted(lbs, lbs - window, side="right") - 1
        base_r = np.where(idx >= 0, crs[np.clip(idx, 0, None)], 0)
        base_w = np.where(idx >= 0, cws[np.clip(idx, 0, None)], 0)
        # fewer than `window` balls bowled: fill the missing balls at the typical rate so early balls aren't distorted
        r12[start:end] = crs - base_r + (window - np.minimum(lbs, window)) * PRIOR_RPB
        w12[start:end] = cws - base_w
        start = end
    df["runs_l12"] = r12
    df["wkts_l12"] = w12
    return df


def state_features(df: pd.DataFrame) -> pd.DataFrame:
    df["balls_left"] = BALLS - df.legal_balls_so_far
    df["wickets_lost"] = df.cum_wkts
    df["crr"] = smooth_crr(df.cum_runs, df.legal_balls_so_far)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Smooth extra terms that let a linear-logit model capture the shape of a chase."""
    df["log_rrr"] = np.log1p(df.rrr.clip(lower=0))
    df["sqrt_bl"] = np.sqrt(df.balls_left.clip(lower=0))
    df["wk_x_bl"] = df.wickets_lost * df.balls_left / BALLS  # wickets in hand matter more with more balls left
    return df


def build(raw_dir="data/processed"):
    raw = Path(raw_dir)
    balls = pd.read_parquet(raw / "balls_raw.parquet")
    matches = pd.read_parquet(raw / "matches.parquet")
    balls = balls[balls.usable].copy()
    balls["legal_balls_so_far"] = balls.legal_balls_so_far.clip(upper=BALLS)
    balls["cum_runs"] = balls.groupby(["match_id", "innings"]).runs_total.cumsum()
    balls["cum_wkts"] = balls.groupby(["match_id", "innings"]).is_wicket.cumsum()
    balls = balls.reset_index(drop=True)

    # A chase must end naturally: target reached, 9+ wickets down (10th batter absent/injured counts as all out),
    # or all overs bowled. Anything else was curtailed in some way the data doesn't explain -> exclude.
    last2 = balls[balls.innings == 2].groupby("match_id").tail(1)
    natural = ((last2.target_runs - last2.cum_runs) <= 0) | (last2.cum_wkts >= 9) | (last2.legal_balls_so_far >= BALLS)
    excluded = set(last2.match_id[~natural])
    print("excluded (chase ended without a natural conclusion):", len(excluded))
    balls = balls[~balls.match_id.isin(excluded)].reset_index(drop=True)
    matches.loc[matches.match_id.isin(excluded), "usable"] = False
    balls["seq"] = balls.groupby(["match_id", "innings"]).cumcount() + 1

    totals = balls.groupby(["match_id", "innings"]).cum_runs.max().unstack()
    totals.columns = ["first_total", "second_total"]
    chasers = balls[balls.innings == 2].groupby("match_id").batting_team.first().rename("chasing_team")
    firsts = balls[balls.innings == 1].groupby("match_id").batting_team.first().rename("defending_team")
    matches = matches[matches.usable].copy()  # everything downstream uses only usable matches
    mm = matches.merge(totals, on="match_id", how="left").merge(chasers, on="match_id", how="left").merge(
        firsts, on="match_id", how="left")
    mm["chase_won"] = (mm.winner == mm.chasing_team).astype(int)
    mm = match_level_features(mm)
    # orient Elo to chasing / defending side
    mm["elo_chase"] = np.where(mm.chasing_team == mm.team1, mm.elo1, mm.elo2)
    mm["elo_def"] = np.where(mm.chasing_team == mm.team1, mm.elo2, mm.elo1)
    mm["elo_diff"] = mm.elo_chase - mm.elo_def
    mm.to_parquet(raw / "matches_feat.parquet", index=False)

    keep = mm[["match_id", "year", "date", "venue", "chasing_team", "defending_team", "first_total",
               "chase_won", "venue_avg1", "venue_chase_rate", "elo_diff"]]

    # ---------------- 2nd innings ----------------
    c = balls[balls.innings == 2].copy()
    c = rolling_last(c)
    c = state_features(c)
    c = c.merge(keep, on="match_id")
    c["target"] = c.target_runs.fillna(c.first_total + 1)
    c["runs_needed"] = c.target - c.cum_runs
    c["rrr"] = np.where(c.balls_left > 0, c.runs_needed * 6 / c.balls_left.clip(lower=1), np.where(c.runs_needed > 0, 36.0, 0.0))
    c["rrr"] = c.rrr.clip(-6, 36)
    c["rrr_minus_crr"] = c.rrr - c.crr
    is_last = c.groupby("match_id").cumcount(ascending=False) == 0
    c["terminal"] = ((c.runs_needed <= 0) | (c.wickets_lost >= 10) | (c.balls_left <= 0)
                     | (is_last & (c.wickets_lost >= 9))).astype(int)
    c["terminal_p"] = np.where(c.runs_needed <= 0, 1.0, 0.0)
    c["is_start"] = 0

    # synthetic start rows (state before the first ball)
    s = c.groupby("match_id").first().reset_index()
    s["runs_needed"] = s.target
    s["balls_left"] = BALLS
    s["wickets_lost"] = 0
    s["cum_runs"] = 0
    s["legal_balls_so_far"] = 0
    s["crr"] = smooth_crr(0, 0)
    s["rrr"] = s.target * 6 / BALLS
    s["rrr_minus_crr"] = s.rrr - s.crr
    s["runs_l12"] = 12 * PRIOR_RPB
    s["wkts_l12"] = 0.0
    s["terminal"] = 0
    s["terminal_p"] = np.nan
    s["is_start"] = 1
    s["seq"] = 0
    for col in ["batter", "bowler", "non_striker", "runs_batter", "runs_extras", "runs_total", "is_wicket",
                "wicket_kind", "player_out", "over", "is_wide", "is_noball"]:
        s[col] = np.nan
    c = pd.concat([s[c.columns], c], ignore_index=True).sort_values(["match_id", "seq"]).reset_index(drop=True)
    c.to_parquet(raw / "chase_states.parquet", index=False)

    # sanity: terminal outcome must agree with the recorded result
    last = c[c.is_start == 0].groupby("match_id").tail(1)
    bad = last[(last.terminal == 1) & (last.terminal_p != last.chase_won)]
    print("chase matches:", c.match_id.nunique(), "| terminal-vs-result mismatches:", len(bad))

    # ---------------- 1st innings ----------------
    f = balls[balls.innings == 1].copy()
    f = rolling_last(f)
    f = state_features(f)
    f = f.merge(keep, on="match_id")
    f = f[f.balls_left >= 0]
    f.to_parquet(raw / "first_states.parquet", index=False)

    g = mm[mm.usable].groupby("venue").agg(n=("match_id", "size"), runs=("first_total", "sum"), cw=("chase_won", "sum"))
    vs = pd.DataFrame({
        "venue": g.index,
        "venue_avg1": ((g.runs + PRIOR_N * PRIOR_SCORE) / (g.n + PRIOR_N)).values,
        "venue_chase_rate": ((g.cw + PRIOR_CHASE_N * 0.5) / (g.n + PRIOR_CHASE_N)).values,
        "n_matches": g.n.values,
    })
    vs.to_parquet(raw / "venue_stats.parquet", index=False)
    print("chase rows:", len(c), "| first-innings rows:", len(f))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
