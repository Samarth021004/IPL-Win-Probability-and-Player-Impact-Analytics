"""Pressure Index - who actually delivers when an IPL chase is on the line? (Streamlit app)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import engineer, smooth_crr  # noqa: E402
from wpa import leaderboard  # noqa: E402

PROC = ROOT / "data" / "processed"
CHASE, DEFEND = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the reference palette
MUTED, GRID = "rgba(128,128,128,0.9)", "rgba(128,128,128,0.25)"  # mid-grey: readable on light and dark

st.set_page_config(page_title="Pressure Index", page_icon="🏏", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    bw = pd.read_parquet(PROC / "ball_wpa.parquet")
    mm = pd.read_parquet(PROC / "matches_feat.parquet").set_index("match_id")
    f1 = pd.read_parquet(PROC / "first_states.parquet").merge(
        pd.read_parquet(PROC / "first_proj.parquet"), on=["match_id", "seq"])
    pms = {"Batter": pd.read_parquet(PROC / "batter_match.parquet"), "Bowler": pd.read_parquet(PROC / "bowler_match.parquet")}
    venues = pd.read_parquet(PROC / "venue_stats.parquet").set_index("venue")
    metrics = json.load(open(ROOT / "reports" / "metrics.json"))
    return bw, mm, f1, pms, venues, metrics


@st.cache_resource
def load_model():
    """Logistic-regression coefficients stored as plain JSON: no pickle, so no scikit-learn version issues."""
    m = json.load(open(ROOT / "models" / "win_prob.json"))
    return {k: (np.array(v) if isinstance(v, list) and k != "features" else v) for k, v in m.items()}


def predict_win_prob(mdl, df):
    x = df[mdl["features"]].to_numpy(dtype=float)
    z = mdl["intercept"] + ((x - mdl["mean"]) / mdl["scale"]) @ mdl["coef"]
    return 1.0 / (1.0 + np.exp(-z))


@st.cache_data
def cached_leaderboard(role, y0, y1, min_balls):
    pm = load()[3][role]
    pm = pm[(pm.year >= y0) & (pm.year <= y1)]
    lb = leaderboard(pm, min_balls=min_balls, B=200)
    return lb, dict(lb.attrs)


bw, mm, f1, PMS, venues, metrics = load()


def style(fig, h=380, yfmt=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13), hovermode="closest",
                      legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=yfmt)
    return fig


def chart_title(title, explain=None, where=st):
    """Chart heading + one-line plain-English explanation, rendered by Streamlit so it follows the light/dark theme."""
    where.markdown(f"**{title}**")
    if explain:
        where.caption(explain)


def ball_label(r):
    n = int(r.legal_balls_so_far)
    if r.is_legal:
        return f"{(n - 1) // 6}.{(n - 1) % 6 + 1}"
    return f"{n // 6}.{n % 6 + 1}{'wd' if r.is_wide else 'nb'}"


def outcome_text(r):
    if r.is_wicket:
        who = r.player_out if isinstance(r.player_out, str) else r.batter
        return f"WICKET ({r.wicket_kind}) - {who}"
    if r.is_wide:
        return "wide"
    if r.runs_batter == 6:
        return "SIX"
    if r.runs_batter == 4:
        return "FOUR"
    return f"{int(r.runs_total)} run{'s' if r.runs_total != 1 else ''}"


def moments(df, n=5, order=None, context=None, swing=None, swing_label="Swing"):
    """Top-n deliveries by `order` (Series aligned to df, descending); default is absolute swing.

    context: None, or the header for the extra target column ("Target chased" / "Total defended"); this also adds
             the match situation just before the ball.
    swing:   Series aligned to df to show in the Swing column (default: the chasing side's win-probability change).
    """
    key = df.wpa.abs() if order is None else order
    d = df.loc[key.sort_values(ascending=False).index].head(n)
    sw = d.wpa if swing is None else swing.loc[d.index]
    out = {
        "Date": d.match_id.map(mm.date).dt.date,
        "Match": d.match_id.map(lambda m: f"{mm.chasing_team[m]} v {mm.defending_team[m]}"),
        "Ball": [ball_label(r) for r in d.itertuples()],
    }
    if context:
        tgt = d.match_id.map(lambda m: int(mm.first_total[m]) + 1)  # runs needed to win
        out[context] = tgt if context == "Target chased" else tgt - 1  # a defended total is the score posted
        before_balls = (d.legal_balls_so_far - d.is_legal).astype(int)
        out["Situation (before ball)"] = [
            f"{int(r.cum_runs - r.runs_total)}/{int(r.wickets_lost - r.is_wicket)} ({nb // 6}.{nb % 6} ov)"
            for r, nb in zip(d.itertuples(), before_balls)]
    out["Moment"] = [f"{r.batter} vs {r.bowler}: {outcome_text(r)}" for r in d.itertuples()]
    out["Win prob (chasing side)"] = [f"{r.p_before:.0%} -> {r.p_after:.0%}" for r in d.itertuples()]
    out[swing_label] = [f"{v:+.1%}" for v in sw]
    return pd.DataFrame(out)


# ------------------------------------------------------------------ navigation
st.sidebar.title("🏏 Pressure Index")
st.sidebar.caption("Who actually delivers when an IPL chase is on the line?")
page = st.sidebar.radio("Section", ["Match Story", "Player Explorer", "Clutch Leaderboard", "Chase Simulator", "Model Card"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(
    f"{len(mm):,} IPL chases, {mm.year.min()}-{mm.year.max()} - {len(bw):,} deliveries.\n\n"
    "Ball-by-ball data: [Cricsheet](https://cricsheet.org), ODC-By licence.")

# ================================================================== Match Story
if page == "Match Story":
    st.title("Match Story")
    st.caption("The second innings, ball by ball: the chasing side's win probability, and the moments that swung it.")
    c1, c2 = st.columns([1, 3])
    year = c1.selectbox("Season", sorted(mm.year.unique(), reverse=True))
    ms = mm[mm.year == year].sort_values("date")
    labels = {m: f"{r.date:%d %b} - {r.chasing_team} v {r.defending_team} ({'chase won' if r.chase_won else 'chase lost'})"
              for m, r in ms.iterrows()}
    mid = c2.selectbox("Match", list(labels), format_func=labels.get)
    r = mm.loc[mid]
    d = bw[bw.match_id == mid].sort_values("seq")

    st.subheader(f"{r.chasing_team} chasing {int(r.first_total) + 1} at {r.venue}")
    last = d.iloc[-1]
    fl = f1[f1.match_id == mid].sort_values("seq").iloc[-1]
    n1, n2 = int(fl.legal_balls_so_far), int(last.legal_balls_so_far)
    if r.chase_won:
        w, spare = 10 - int(last.wickets_lost), 120 - n2
        margin = f"{w} wicket{'s' if w != 1 else ''} ({spare} ball{'s' if spare != 1 else ''} to spare)"
    else:
        runs_margin = int(r.first_total - last.cum_runs)
        margin = f"{runs_margin} run{'s' if runs_margin != 1 else ''}"
    st.success(f"**Result: {r.winner} won by {margin}**")
    st.markdown(f"{r.defending_team} (batted first): **{int(r.first_total)}/{int(fl.wickets_lost)}** ({n1 // 6}.{n1 % 6} ov)"
                f"  |  {r.chasing_team}: **{int(last.cum_runs)}/{int(last.wickets_lost)}** ({n2 // 6}.{n2 % 6} ov)")
    x = [0.0] + (d.legal_balls_so_far / 6).tolist()
    y = [d.p_before.iloc[0]] + d.p_after.tolist()
    top = d.reindex(d.wpa.abs().sort_values(ascending=False).index).head(5)
    fig = go.Figure()
    fig.add_hline(y=0.5, line_color=GRID, line_dash="dot")
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=CHASE, width=2, shape="hv"), name=f"{r.chasing_team} win probability",
                             hovertemplate="after %{x:.1f} overs<br>win prob %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=(top.legal_balls_so_far / 6), y=top.p_after, mode="markers+text", name="Biggest swings",
        marker=dict(size=11, color=DEFEND, line=dict(color="white", width=2)),
        text=[f"{i + 1}" for i in range(len(top))], textposition="top center",
        customdata=[f"{ball_label(t)} {t.batter} v {t.bowler}: {outcome_text(t)} ({t.wpa:+.0%})" for t in top.itertuples()],
        hovertemplate="%{customdata}<extra></extra>"))
    fig.update_xaxes(title="Overs completed", range=[0, 20.2])
    style(fig, 400, ".0%")
    fig.update_yaxes(range=[0, 1], title=f"P({r.chasing_team} win)")
    st.plotly_chart(fig, width="stretch")
    t = moments(top, 5)
    t.insert(0, "#", range(1, len(t) + 1))
    st.dataframe(t.drop(columns=["Date", "Match"]), hide_index=True, width="stretch", column_config={"Moment": st.column_config.TextColumn(width="large"), "Match": st.column_config.TextColumn(width="medium")})

    with st.expander(f"First innings: {r.defending_team}'s projected total as the innings unfolded"):
        f = f1[f1.match_id == mid].sort_values("seq")
        fig2 = go.Figure()
        fig2.add_hline(y=r.first_total, line_color=DEFEND, line_dash="dot",
                       annotation_text=f"Final: {int(r.first_total)}", annotation_font_color=MUTED)
        fig2.add_trace(go.Scatter(x=f.legal_balls_so_far / 6, y=f.proj_score, mode="lines", line=dict(color=CHASE, width=2),
                                  name="Projected total", hovertemplate="after %{x:.1f} overs: projected %{y:.0f}<extra></extra>"))
        fig2.update_xaxes(title="Overs completed")
        fig2.update_yaxes(title="Projected total")
        style(fig2, 320)
        st.plotly_chart(fig2, width="stretch")
        st.caption(f"Ridge regression on score, wickets, overs left, recent scoring, venue history and team strength. "
                   f"Test MAE {metrics['score_regression']['mae_ridge']:.1f} runs, against "
                   f"{metrics['score_regression']['mae_naive_run_rate']:.1f} for 'current run rate x overs left'.")

# ================================================================== Player Explorer
elif page == "Player Explorer":
    st.title("Player Explorer")
    st.caption("Win Probability Added (WPA) in IPL chases, in win-equivalents: +1.0 means one extra win's worth of value.")
    c1, c2 = st.columns([1, 3])
    role = c1.radio("Role", ["Batter", "Bowler"], horizontal=True)
    pm = PMS[role]
    tot = pm.groupby("player").n.sum()
    choices = tot[tot >= 100].sort_values(ascending=False).index.tolist()
    player = c2.selectbox("Player (100+ balls in chases)", choices)
    g = pm[pm.player == player]
    n = g.n.sum()
    li = g.li_sum.sum() / n
    a, b, c, d_ = st.columns(4)
    a.metric("Chase matches", f"{g.match_id.nunique()}")
    b.metric("Balls", f"{int(n):,}")
    c.metric("Total WPA", f"{g.wpa.sum():+.2f}")
    d_.metric("Avg leverage faced", f"{li:.2f}", help="1.0 = an average ball; >1 = he plays in tighter situations than usual.")

    col1, col2 = st.columns(2)
    by_year = g.groupby("year").wpa.sum()
    fig = go.Figure(go.Bar(x=by_year.index, y=by_year.values, marker_color=[CHASE if v >= 0 else DEFEND for v in by_year.values],
                           hovertemplate="%{x}: %{y:+.2f} wins<extra></extra>"))
    fig.update_xaxes(dtick=1, tickformat="d")
    style(fig, 320)
    chart_title("Value added in each season",
                f"Adds up how much this {role.lower()}'s balls in run chases moved his team's chance of winning, season by season. "
                "Blue = helped his team, orange = hurt it. A value of +1.0 is worth roughly one extra win.", where=col1)
    col1.plotly_chart(fig, width="stretch")

    col = "batter" if role == "Batter" else "bowler"
    pb = bw[(bw[col] == player)].copy()
    if role == "Batter":
        pb = pb[pb.is_wide == 0]
    pb["pv"] = pb.wpa if role == "Batter" else -pb.wpa  # value credited to this player
    pb["Phase"] = pd.cut(pb.over, [-1, 5, 14, 19], labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"])
    ph = pb.groupby("Phase", observed=False).agg(wpa=("pv", "sum"), balls=("pv", "size"))
    ph["per100"] = 100 * ph.wpa / ph.balls.clip(lower=1)
    fig = go.Figure(go.Bar(x=ph.index.astype(str), y=ph.per100, marker_color=[CHASE if v >= 0 else DEFEND for v in ph.per100],
                           customdata=ph.balls, hovertemplate="%{x}: %{y:+.2f} WPA/100 balls (%{customdata} balls)<extra></extra>"))
    style(fig, 320)
    chart_title("Value per 100 balls, by stage of the innings",
                "The same idea averaged per 100 balls, so stages can be compared fairly. Powerplay = overs 1-6, Middle = 7-15, "
                "Death = 16-20. The taller the blue bar, the more he adds per ball at that stage; orange means he costs his team.",
                where=col2)
    col2.plotly_chart(fig, width="stretch")

    ctx = "Target chased" if role == "Batter" else "Total defended"
    lab = f"Swing for {role.lower()}"
    st.subheader("Best moments")
    st.dataframe(moments(pb, 5, order=pb.pv, context=ctx, swing=pb.pv, swing_label=lab), hide_index=True, width="stretch",
                 column_config={"Moment": st.column_config.TextColumn(width="large"), "Match": st.column_config.TextColumn(width="medium")})
    st.subheader("Worst moments")
    st.dataframe(moments(pb, 5, order=-pb.pv, context=ctx, swing=pb.pv, swing_label=lab), hide_index=True, width="stretch",
                 column_config={"Moment": st.column_config.TextColumn(width="large"), "Match": st.column_config.TextColumn(width="medium")})
    st.caption("Match lists the chasing side first. "
               + ("Target chased = runs the chasing side needed to win. " if role == "Batter" else "Total defended = the score his side posted batting first. ")
               + "Situation is the chasing side's score/wickets just before the ball. Win prob is the chasing side's chance before -> after the ball; "
                 f"{lab} is that change from the {role.lower()}'s point of view (+ helped his side). "
                 "Batters get credit for balls they face; wides go to the bowler only.")

# ================================================================== Clutch Leaderboard
elif page == "Clutch Leaderboard":
    st.title("Clutch Leaderboard")
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    role = c1.radio("Role", ["Batter", "Bowler"])
    yrs = c2.slider("Seasons", 2008, int(mm.year.max()), (2008, int(mm.year.max())))
    min_balls = c3.slider("Minimum balls in chases", 50, 1000, 200, step=50)
    topn = c4.selectbox("Show top", [10, 15, 20, 30], index=1)
    lb, attrs = cached_leaderboard(role, yrs[0], yrs[1], min_balls)
    if lb.empty:
        st.warning("No players meet this filter.")
        st.stop()

    tab1, tab2 = st.tabs(["Value delivered (WPA)", "Clutch: does he rise when it matters?"])
    with tab1:
        t = lb.head(topn).iloc[::-1]
        fig = go.Figure(go.Scatter(
            x=t.wpa_per_100, y=t.player, mode="markers", marker=dict(size=10, color=CHASE if role == "Batter" else DEFEND),
            error_x=dict(type="data", symmetric=False, array=t.wpa_hi - t.wpa_per_100, arrayminus=t.wpa_per_100 - t.wpa_lo,
                         color=MUTED, thickness=1.5, width=0),
            customdata=np.c_[t.wpa, t.balls, t.wpa_lo, t.wpa_hi],
            hovertemplate="%{y}<br>%{x:.2f} WPA/100 balls (90% CI %{customdata[2]:.2f} to %{customdata[3]:.2f})<br>"
                          "total %{customdata[0]:+.2f} wins over %{customdata[1]:.0f} balls<extra></extra>"))
        style(fig, 60 + 30 * len(t))
        fig.update_xaxes(title="WPA per 100 balls (dot = estimate, bar = 90% interval from match-level bootstrap)", showgrid=True,
                         gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Ranked by total WPA (value over a whole career in these seasons). Where bars overlap, "
                   f"the order is not statistically meaningful.")
        show = lb.head(topn)[["player", "matches", "balls", "runs", "wpa", "wpa_per_100", "wpa_lo", "wpa_hi", "avg_leverage"]]
        st.dataframe(show.rename(columns={"player": "Player", "matches": "Matches", "balls": "Balls", "runs": "Runs", "wpa": "Total WPA",
                                          "wpa_per_100": "WPA/100", "wpa_lo": "CI low", "wpa_hi": "CI high", "avg_leverage": "Avg leverage"}),
                     hide_index=True, width="stretch")
    with tab2:
        obs, samp, tau2 = attrs["obs_var"], attrs["samp_var"], attrs["tau2"]
        share = 0 if obs == 0 else min(1.0, tau2 / obs)
        if share < 0.15:
            st.warning(f"**Mostly luck.** Of the spread in clutch scores among these {role.lower()}s, about **{share:.0%}** reflects real "
                       f"differences; the rest is sampling noise. Shrunk scores below are pulled strongly towards zero for that reason.")
        else:
            st.info(f"About **{share:.0%}** of the spread in clutch scores among these {role.lower()}s reflects real differences; "
                    f"the rest is noise. Shrunk scores account for this.")
        t = lb.reindex(lb.clutch_shrunk_per_100.abs().sort_values(ascending=False).index).head(topn).sort_values("clutch_shrunk_per_100")
        fig = go.Figure(go.Scatter(
            x=t.clutch_per_100, y=t.player, mode="markers", name="Raw clutch", marker=dict(size=10, color=CHASE),
            error_x=dict(type="data", symmetric=False, array=t.clutch_ci_hi - t.clutch_per_100, arrayminus=t.clutch_per_100 - t.clutch_ci_lo,
                         color=MUTED, thickness=1.5, width=0)))
        fig.add_trace(go.Scatter(x=t.clutch_shrunk_per_100, y=t.player, mode="markers", name="Shrunk (best estimate)",
                                 marker=dict(size=10, color=DEFEND, symbol="diamond", line=dict(color="white", width=1))))
        fig.add_vline(x=0, line_color=MUTED, line_dash="dot")
        style(fig, 80 + 30 * len(t))
        fig.update_xaxes(title="Clutch per 100 balls (win units). Bar: 90% interval on the raw score", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Clutch = WPA / avg leverage - sum(WPA / leverage). Positive means more value in high-leverage balls than in a "
                   "context-neutral world. Shown for the players with the largest shrunk scores.")

# ================================================================== Chase Simulator
elif page == "Chase Simulator":
    st.title("Chase Simulator")
    st.caption("Set a situation and see the model's win probability, and what the next ball does to it.")
    mdl = load_model()
    c1, c2, c3 = st.columns(3)
    target = c1.number_input("Target", 60, 300, 180)
    runs = c1.number_input("Runs scored", 0, 299, 90)
    balls_done = c2.slider("Legal balls bowled", 0, 119, 60)
    wkts = c2.slider("Wickets lost", 0, 9, 3)
    venue = c3.selectbox("Venue", sorted(venues.index), index=sorted(venues.index).index("Wankhede Stadium") if "Wankhede Stadium" in venues.index else 0)
    elo = c3.slider("Chasing side's pre-match strength edge (Elo points)", -150, 150, 0)
    crr0 = smooth_crr(runs, balls_done)
    r12 = c1.number_input("Runs in last 12 balls", 0, 100, int(round(min(crr0 * 2, 100))))
    w12 = c2.number_input("Wickets in last 12 balls", 0, 5, 0)

    def win_prob(target, runs, balls, wk, r12, w12):
        need = target - runs
        if need <= 0:
            return 1.0
        if wk >= 10 or balls >= 120:
            return 0.0
        left = 120 - balls
        row = dict(runs_needed=need, balls_left=left, wickets_lost=wk, rrr=min(need * 6 / left, 36), crr=smooth_crr(runs, balls),
                   target=target, venue_avg1=venues.loc[venue, "venue_avg1"], venue_chase_rate=venues.loc[venue, "venue_chase_rate"],
                   elo_diff=elo, runs_l12=r12, wkts_l12=w12)
        row["rrr_minus_crr"] = row["rrr"] - row["crr"]
        df = engineer(pd.DataFrame([row]))
        return float(predict_win_prob(mdl, df)[0])

    p = win_prob(target, runs, balls_done, wkts, r12, w12)
    need = target - runs
    m1, m2, m3 = st.columns(3)
    m1.metric("Chasing side win probability", f"{p:.0%}")
    m2.metric("Needed", f"{max(need, 0)} off {120 - balls_done}")
    m3.metric("Required rate", f"{max(need, 0) * 6 / max(120 - balls_done, 1):.1f} per over")

    rows = []
    for name, rr, w in [("Dot ball", 0, 0), ("Single", 1, 0), ("Four", 4, 0), ("Six", 6, 0), ("Wicket", 0, 1)]:
        p2 = win_prob(target, runs + rr, balls_done + 1, wkts + w, max(r12 + rr - r12 / 12, 0), max(w12 + w - w12 / 12, 0))
        rows.append(dict(Outcome=name, After=p2, Change=p2 - p))
    nb = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(x=nb.Outcome, y=nb.Change, marker_color=[CHASE if v >= 0 else DEFEND for v in nb.Change],
                           customdata=nb.After, hovertemplate="%{x}: %{y:+.1%} -> win prob %{customdata:.0%}<extra></extra>"))
    style(fig, 300, "+.0%")
    chart_title("What the next ball does", "Change in the chasing side's win probability after each possible outcome of the next ball.")
    st.plotly_chart(fig, width="stretch")

    ws = [win_prob(target, runs, balls_done, w, r12, 0) for w in range(0, 10)]
    fig = go.Figure(go.Scatter(x=list(range(10)), y=ws, mode="lines+markers", line=dict(color=CHASE, width=2), marker=dict(size=8),
                               hovertemplate="%{x} wickets down: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[wkts], y=[p], mode="markers", marker=dict(size=13, color=DEFEND, line=dict(color="white", width=2)),
                             name="Current", hovertemplate="current: %{y:.0%}<extra></extra>"))
    fig.update_xaxes(title="Wickets lost (same score and balls)", dtick=1)
    style(fig, 300, ".0%")
    fig.update_layout(showlegend=False)
    chart_title("What wickets in hand are worth here", "Same score and balls bowled, but with more or fewer wickets lost. The orange dot is the current situation.")
    st.plotly_chart(fig, width="stretch")
    st.caption("The last-12-balls momentum inputs are updated approximately for the next-ball table. This is a statistical model of "
               "past IPL chases, not a guarantee: it knows nothing about who is batting or bowling.")

# ================================================================== Model Card
else:
    st.title("Model Card")
    m = metrics
    st.markdown(
        "**Task.** Predict, after every delivery of an IPL second innings, the probability that the chasing side wins. "
        "**Data.** Cricsheet ball-by-ball IPL, 2008-2026; matches with a result, no DLS, and a natural conclusion are used "
        f"(**{len(mm):,}** chases, ~140k deliveries).")
    st.subheader("Which model? Chronological test on 2024-2026")
    st.caption(f"Trained on 2008-2022 ({m['split']['train']['matches']} matches), tested on {m['split']['test']['matches']} untouched matches. "
               "Lower log loss and Brier are better. Rows within a match are correlated, so treat differences as indicative.")
    tm = pd.DataFrame(m["test_metrics"]).T.rename(columns={"logloss": "Log loss", "brier": "Brier", "auc": "ROC AUC"})
    tm.index = [i.replace("_", " ") for i in tm.index]
    st.dataframe(tm.round(4), width="stretch")
    gc = pd.DataFrame(m["grouped_cv_all_seasons"]).T.rename(columns={"logloss": "Log loss", "brier": "Brier", "auc": "ROC AUC"})
    st.markdown("**Grouped 5-fold CV over all seasons** (a far larger sample than three test years):")
    gc.index = [i.replace("_", " ") for i in gc.index]
    st.dataframe(gc.round(4), width="stretch")
    st.info("**Simple beat complex.** With ~1,200 matches, a regularised logistic regression with three smooth engineered terms "
            "(log required rate, sqrt balls left, wickets x balls) beat gradient boosting. The relationship between match state and "
            "outcome is smooth, and 120 correlated balls per match means little independent signal for trees to exploit. "
            "Isotonic calibration fit on the 71-match 2023 validation set made things *worse*, so no post-hoc calibration is used; "
            "maximum-likelihood logistic probabilities are already well calibrated (below).")

    c1, c2 = st.columns(2)
    cal = pd.DataFrame(m["calibration"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, dash="dot"), name="Perfect"))
    fig.add_trace(go.Scatter(x=cal.mean_pred, y=cal.frac_win, mode="lines+markers", line=dict(color=CHASE, width=2), marker=dict(size=9),
                             customdata=cal.n, name="Model", hovertemplate="predicted %{x:.0%}, actual %{y:.0%} (%{customdata:,} balls)<extra></extra>"))
    style(fig, 340, ".0%")
    fig.update_xaxes(title="Predicted win probability", tickformat=".0%")
    fig.update_yaxes(title="Observed win rate")
    fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
    chart_title("Calibration (test years)", "When the model says X%, do about X% of those chases get won? Points on the dotted line = perfect.", where=c1)
    c1.plotly_chart(fig, width="stretch")
    yrs = pd.DataFrame(m["oof_by_year"]).T
    fig = go.Figure(go.Bar(x=yrs.index.astype(int), y=yrs.auc, marker_color=CHASE, hovertemplate="%{x}: AUC %{y:.3f}<extra></extra>"))
    style(fig, 340)
    fig.update_yaxes(range=[0.7, 1.0], title="ROC AUC (out-of-fold)")
    fig.update_xaxes(dtick=1, tickformat="d")
    chart_title("Some seasons are simply more predictable", "ROC AUC by season (1.0 = perfect ranking of winners, 0.5 = coin flip).", where=c2)
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Sharper as the chase goes on, as it should be")
    bs = pd.DataFrame(m["by_stage"]).rename(columns={"stage": "Stage", "n": "Balls", "auc": "AUC", "logloss": "Log loss", "mean_abs_conf": "Avg |P-0.5|"})
    st.dataframe(bs.round(3), hide_index=True, width="stretch")

    sr = m["score_regression"]
    st.subheader("First-innings projected total (regression)")
    st.markdown(f"Ridge regression on the *remaining* runs. Test MAE **{sr['mae_ridge']:.1f}** runs vs LightGBM {sr['mae_lgbm']:.1f} "
                f"vs naive run-rate projection {sr['mae_naive_run_rate']:.1f}. The simple model matched the complex one here too.")
    st.dataframe(pd.DataFrame(sr["by_stage"]).rename(columns={"stage": "Stage", "mae_lgbm": "MAE LightGBM", "mae_naive_run_rate": "MAE run-rate"}).round(1),
                 hide_index=True, width="stretch")

    st.subheader("How WPA, leverage and clutch are defined")
    st.markdown(
        "- **WPA** = win probability after a ball minus before it, using *out-of-fold* probabilities (no ball is scored by a model that saw its own match). "
        "Batter gets +WPA, bowler -WPA; wides count against the bowler only. Over a match WPA sums exactly to (result - starting probability).\n"
        "- **Leverage** = the average |WPA| of balls in comparable situations (same win-probability band and phase), divided by the overall average. 1.0 is an average ball.\n"
        "- **Clutch** = WPA / avg leverage - sum(WPA / leverage). Confidence intervals come from a bootstrap over matches; shrinkage is empirical-Bayes, "
        "with the true between-player variance estimated as observed variance minus average sampling variance.")
    st.subheader("Limitations")
    st.markdown(
        "- Only chases are scored, so first-innings batting and bowling earn no WPA here.\n"
        "- Credit is by delivery: fielders, run-out credit and partnerships are not separated; a batter on strike takes the whole swing.\n"
        "- The model does not know who is batting or bowling, so WPA credits players for outcomes relative to an *average* ball.\n"
        "- Rain-affected matches, ties and no-results (and 4 curtailed chases) are excluded.\n"
        "- Impact Player rules (2023 onward) and changing pitches/scoring rates mean eras differ; Elo and venue history are the only context features.")
