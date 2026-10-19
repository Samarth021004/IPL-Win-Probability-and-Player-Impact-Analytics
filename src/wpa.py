"""Win Probability Added (WPA), Leverage Index and shrunk, bootstrapped clutch scores.

Definitions
-----------
WPA (ball)    = P(chasing side wins) after the ball - before the ball, using OUT-OF-FOLD model probabilities.
                Batter gets +WPA, bowler gets -WPA. Wides are credited to the bowler only (batter didn't face them).
Leverage (LI) = average |WPA| of balls in comparable situations (same win-probability band and phase of the chase),
                divided by the overall average |WPA|.  LI = 1.0 is an average ball; 2.0 is twice as swingy.
Clutch        = WPA / pLI  -  sum(WPA / LI)      (the Fangraphs-style formulation, in win units)
                pLI = the player's average leverage. Positive => produced more value when it mattered more
                than in a context-neutral world.
Shrinkage     = empirical Bayes. Sampling variance per player comes from a match-level bootstrap; the true
                between-player variance tau^2 is estimated as var(observed) - mean(sampling variance).
                If tau^2 ~ 0, clutch is statistically indistinguishable from luck - and we say so in the app.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_LI_SHRINK = 100


def compute_ball_wpa(proc="data/processed") -> pd.DataFrame:
    proc = Path(proc)
    st = pd.read_parquet(proc / "chase_states.parquet")
    pr = pd.read_parquet(proc / "chase_probs.parquet")
    d = st.merge(pr, on=["match_id", "seq"]).sort_values(["match_id", "seq"]).reset_index(drop=True)
    d["p_before"] = d.groupby("match_id").win_prob.shift(1)
    d = d[d.is_start == 0].copy()
    d["p_after"] = d.win_prob
    d["wpa"] = d.p_after - d.p_before  # from the chasing side's point of view

    # ---- leverage index ----
    d["pband"] = np.clip((d.p_before * 20).astype(int), 0, 19)
    d["stage"] = pd.cut(d.legal_balls_so_far - d.is_legal, [-1, 35, 71, 101, 121], labels=False)
    d["abs_wpa"] = d.wpa.abs()
    g_mean = d.abs_wpa.mean()
    marg = d.groupby("pband").abs_wpa.mean().rename("marg")
    cell = d.groupby(["pband", "stage"]).abs_wpa.agg(["sum", "size"]).reset_index().merge(marg, on="pband")
    cell["li"] = ((cell["sum"] + N_LI_SHRINK * cell.marg) / (cell["size"] + N_LI_SHRINK)) / g_mean
    d = d.merge(cell[["pband", "stage", "li"]], on=["pband", "stage"], how="left")
    d["li"] = d.li.fillna(1.0).clip(lower=0.05)

    keep = ["match_id", "seq", "innings", "batting_team", "over", "legal_balls_so_far", "is_legal", "is_wide", "batter",
            "bowler", "runs_batter", "runs_extras", "runs_total", "is_wicket", "wicket_kind", "player_out", "cum_runs",
            "wickets_lost", "balls_left", "target", "year", "p_before", "p_after", "wpa", "li"]
    return d[keep].reset_index(drop=True)


def player_match_table(bw: pd.DataFrame, role: str) -> pd.DataFrame:
    """Per (player, match) sums needed for the bootstrap: n balls, WPA, sum(WPA/LI), sum(LI), runs."""
    x = bw.copy()
    if role == "batter":
        x = x[x.is_wide == 0]
        x["pw"] = x.wpa
        x["player"] = x.batter
        x["runs"] = x.runs_batter
        x["wk"] = x.is_wicket  # dismissals while on strike (excludes non-striker run-outs)
        x.loc[x.player_out != x.batter, "wk"] = 0
    else:
        x["pw"] = -x.wpa
        x["player"] = x.bowler
        x["runs"] = x.runs_total
        x["wk"] = ((x.is_wicket == 1) & (~x.wicket_kind.isin(["run out", "retired out", "obstructing the field"]))).astype(int)
    x["pw_li"] = x.pw / x.li
    x["hi"] = (x.li >= 1.5).astype(int)
    pm = x.groupby(["player", "match_id"]).agg(
        year=("year", "first"), n=("pw", "size"), wpa=("pw", "sum"), wpa_li=("pw_li", "sum"), li_sum=("li", "sum"),
        runs=("runs", "sum"), wk=("wk", "sum"), n_hi=("hi", "sum")).reset_index()
    return pm


def leaderboard(pm: pd.DataFrame, min_balls: int = 150, B: int = 300, seed: int = 11) -> pd.DataFrame:
    """Aggregate per player, bootstrap by match for CIs, and apply empirical-Bayes shrinkage to clutch."""
    rng = np.random.default_rng(seed)
    rows = []
    for player, g in pm.groupby("player"):
        n = g.n.sum()
        if n < min_balls:
            continue
        a = g[["n", "wpa", "wpa_li", "li_sum"]].to_numpy(dtype=float)
        m = len(a)
        idx = rng.integers(0, m, size=(B, m))
        bs = a[idx].sum(axis=1)  # B x 4
        def clutch(v):
            return v[..., 1] / (v[..., 3] / v[..., 0]) - v[..., 2]
        tot = a.sum(axis=0)
        c_obs = clutch(tot)
        c_bs = clutch(bs)
        w_bs = bs[:, 1]
        rows.append(dict(
            player=player, matches=m, balls=int(n), runs=int(g.runs.sum()), wickets=int(g.wk.sum()),
            wpa=tot[1], wpa_per_100=100 * tot[1] / n,
            wpa_lo=np.percentile(100 * w_bs / bs[:, 0], 5), wpa_hi=np.percentile(100 * w_bs / bs[:, 0], 95),
            avg_leverage=tot[3] / n, hi_lev_balls=int(g.n_hi.sum()),
            clutch=c_obs, clutch_se=float(np.std(c_bs, ddof=1)),
        ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["clutch_per_100"] = 100 * out.clutch / out.balls
    out["clutch_se_per_100"] = 100 * out.clutch_se / out.balls
    obs_var = out.clutch_per_100.var(ddof=1)
    samp_var = (out.clutch_se_per_100 ** 2).mean()
    tau2 = max(0.0, obs_var - samp_var)
    k = tau2 / (tau2 + out.clutch_se_per_100 ** 2) if tau2 > 0 else 0.0
    out["clutch_shrunk_per_100"] = out.clutch_per_100 * k
    out["shrink_weight"] = k
    out.attrs["tau2"] = tau2
    out.attrs["obs_var"] = obs_var
    out.attrs["samp_var"] = samp_var
    out["clutch_ci_lo"] = out.clutch_per_100 - 1.645 * out.clutch_se_per_100
    out["clutch_ci_hi"] = out.clutch_per_100 + 1.645 * out.clutch_se_per_100
    return out.sort_values("wpa", ascending=False).reset_index(drop=True)


def main(proc="data/processed"):
    proc = Path(proc)
    bw = compute_ball_wpa(proc)
    bw.to_parquet(proc / "ball_wpa.parquet", index=False)

    # ---- integrity checks ----
    tot = bw.groupby("match_id").wpa.sum()
    st = pd.read_parquet(proc / "matches_feat.parquet").set_index("match_id")
    p0 = bw.groupby("match_id").p_before.first()
    resid = (tot - (st.loc[tot.index, "chase_won"] - p0)).abs().max()
    print(f"WPA telescopes to (result - starting win prob): max residual = {resid:.2e}")
    print(f"mean |WPA| per ball = {bw.wpa.abs().mean():.4f};  LI range {bw.li.min():.2f}-{bw.li.max():.2f},  mean {bw.li.mean():.2f}")

    bpm = player_match_table(bw, "batter")
    wpm = player_match_table(bw, "bowler")
    bpm.to_parquet(proc / "batter_match.parquet", index=False)
    wpm.to_parquet(proc / "bowler_match.parquet", index=False)

    for role, pm in (("batter", bpm), ("bowler", wpm)):
        lb = leaderboard(pm, min_balls=150)
        lb.to_parquet(proc / f"{role}_leaderboard.parquet", index=False)
        print(f"\n=== {role}s (>=150 balls): {len(lb)} players ===")
        print(f"clutch/100 balls: observed variance {lb.attrs['obs_var']:.5f}, mean sampling variance {lb.attrs['samp_var']:.5f}, "
              f"tau^2 (true spread) {lb.attrs['tau2']:.5f}")
        cols = ["player", "matches", "balls", "runs", "wpa", "wpa_per_100", "avg_leverage", "clutch_per_100", "clutch_shrunk_per_100"]
        print(lb[cols].head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
