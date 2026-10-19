"""Flatten Cricsheet IPL JSON into clean ball-level and match-level tables.

Usage: python -I src/parse.py data/raw/ipl_json data/processed
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Deccan Chargers": "Sunrisers Hyderabad",  # different franchise, but kept separate below
}
# Deccan Chargers and Sunrisers are different franchises: keep them separate.
TEAM_MAP.pop("Deccan Chargers")

# venue name -> canonical (applied after stripping the ", City" suffix)
VENUE_MAP = {
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "PCA Stadium Mohali",
    "Punjab Cricket Association Stadium": "PCA Stadium Mohali",
    "Maharaja Yadavindra Singh International Cricket Stadium": "PCA Stadium Mohali",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Wankhede Stadium": "Wankhede Stadium",
    "Eden Gardens": "Eden Gardens",
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi Intl Stadium",
    "Narendra Modi Stadium": "Narendra Modi Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "Dr DY Patil Sports Academy": "Dr DY Patil Sports Academy",
    "Brabourne Stadium": "Brabourne Stadium",
    "Himachal Pradesh Cricket Association Stadium": "HPCA Stadium Dharamsala",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "ACA-VDCA Stadium Visakhapatnam",
    "Barabati Stadium": "Barabati Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium",  # Abu Dhabi, same ground
}


def norm_team(t):
    return TEAM_MAP.get(t, t)


def norm_venue(v):
    if v is None:
        return "Unknown"
    base = v.split(",")[0].strip()
    return VENUE_MAP.get(base, VENUE_MAP.get(v, base))


def parse_match(path):
    m = json.load(open(path))
    info = m["info"]
    outcome = info.get("outcome", {})
    date = pd.Timestamp(info["dates"][0])
    mid = Path(path).stem
    meta = dict(
        match_id=mid,
        date=date,
        year=date.year,  # derived from date: Cricsheet 'season' labels are inconsistent
        venue=norm_venue(info.get("venue")),
        city=info.get("city"),
        team1=norm_team(info["teams"][0]),
        team2=norm_team(info["teams"][1]),
        winner=norm_team(outcome["winner"]) if "winner" in outcome else None,
        result=outcome.get("result", "win" if "winner" in outcome else "?"),
        dls=int("method" in outcome),
        has_super_over=int(any(i.get("super_over") for i in m["innings"])),
        n_innings=len([i for i in m["innings"] if not i.get("super_over")]),
        overs=info.get("overs", 20),
        balls_per_over=info.get("balls_per_over", 6),
    )
    rows = []
    for inn_no, inn in enumerate(m["innings"], 1):
        if inn.get("super_over"):
            continue
        legal = 0
        tgt = inn.get("target", {})
        for ov in inn["overs"]:
            for d in ov["deliveries"]:
                ex = d.get("extras", {})
                is_legal = not ("wides" in ex or "noballs" in ex)
                if is_legal:
                    legal += 1
                wk = [w for w in d.get("wickets", []) if w["kind"] != "retired hurt"]
                rows.append(
                    dict(
                        match_id=mid,
                        innings=inn_no,
                        batting_team=norm_team(inn["team"]),
                        over=ov["over"],
                        legal_ball_no=legal if is_legal else None,
                        legal_balls_so_far=legal,  # legal balls bowled up to and including this delivery
                        is_legal=int(is_legal),
                        is_wide=int("wides" in ex),
                        is_noball=int("noballs" in ex),
                        batter=d["batter"],
                        non_striker=d["non_striker"],
                        bowler=d["bowler"],
                        runs_batter=d["runs"]["batter"],
                        runs_extras=d["runs"]["extras"],
                        runs_total=d["runs"]["total"],
                        is_wicket=int(len(wk) > 0),
                        wicket_kind=wk[0]["kind"] if wk else None,
                        player_out=wk[0]["player_out"] if wk else None,
                        target_runs=tgt.get("runs"),
                        target_overs=tgt.get("overs"),
                    )
                )
    return meta, rows


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metas, all_rows = [], []
    for f in sorted(glob.glob(f"{src}/*.json")):
        meta, rows = parse_match(f)
        metas.append(meta)
        all_rows.extend(rows)
    matches = pd.DataFrame(metas)
    balls = pd.DataFrame(all_rows)

    # ---- eligibility for the chase model ----
    matches["usable"] = (
        (matches.result == "win")
        & matches.winner.notna()
        & (matches.dls == 0)
        & (matches.n_innings == 2)
        & (matches.overs == 20)
    )
    print("matches total:", len(matches))
    print("  dropped (no result / tie):", int((~matches.result.eq("win")).sum()))
    print("  dropped (DLS):", int((matches.dls == 1).sum()))
    print("  usable for chase model:", int(matches.usable.sum()))

    balls = balls.merge(matches[["match_id", "usable"]], on="match_id")
    balls.to_parquet(out / "balls_raw.parquet", index=False)
    matches.to_parquet(out / "matches.parquet", index=False)
    print("balls rows:", len(balls))
    print("teams:", sorted(set(matches.team1) | set(matches.team2)))
    print("venues:", matches.venue.nunique())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
