# -*- coding: utf-8 -*-
"""
RAPM Model

Adaptation of Patrick Bacon's work in Python: https://medium.com/data-science/a-comprehensive-guide-to-using-regression-rapm-to-evaluate-nhl-skaters-with-source-code-ad76d1e1b8da
    
Builds stints from shifts + PBP (regular season periods 1–3), then fits a single-lambda
ridge RAPM per situation (Even Strength, Power Play, Penalty Kill).

Includes fitting of baseline OLS model and paired t-test comparison in fit_comparison file

NOTE: Had AI assist with some of the issues I ran into with translating Bacon's approach into Python, mainly surrounding my
L2 regularization not firing properly. Also used it to clean up my script and make it neater and remove redundant code.
"""
# Import
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, hstack
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold
import math
from scipy.stats import ttest_rel

# Config
base_directory = Path(__file__).resolve().parent.parent
data_root = base_directory / "Data" / "Clean Data"

start_year = 2015
end_year = 2024
random_seed = 17

situation_list = ("Even Strength", "Power Play", "Penalty Kill")
rebuild_stint_toggle = False

keep_strength_by_situation = {
    "Even Strength": {"5v5"},
    "Power Play": {"5v4", "6v5"},
    "Penalty Kill": {"4v5", "5v6"},
    }

lambda_grid = [0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] 

# Minutes-aware shrinkage
goalie_L2_multiplier = 2.0
minute_floor_for_scaler = 120.0
clear_fit_meta_toggle = True

# Helpers
def season_label(year):
    return f"{year}-{year+1}"

def season_dir(s):
    return data_root / s

def coeffs_dir(s):
    return season_dir(s) / "model coefficients"

def missing_dir(s):
    return season_dir(s) / "missing logs"

def stints_path(s):
    return season_dir(s) / f"stints_{s}.csv"

def fitmeta_path(s):
    return coeffs_dir(s) / f"RAPM_fit_meta_{s}.csv"

def season_player_outfile(s, sit):
    return season_dir(s) / f"RAPM_xGF_{sit.replace(' ','_')}_{s}.csv"

def ensure_dirs(season):
    season_dir(season).mkdir(parents=True, exist_ok=True)
    coeffs_dir(season).mkdir(parents=True, exist_ok=True)
    missing_dir(season).mkdir(parents=True, exist_ok=True)

def parse_date_col(s):
    try:
        return pd.to_datetime(s).normalize()
    except Exception:
        return pd.NaT
    
def period_end_seconds(p):
    return 1200.0 if p in [1, 2, 3] else 0.0

def fitcomparison_path(s):
    return coeffs_dir(s) / f"RAPM_fit_comparison_{s}.csv"

# Data Loading
def load_player_game_stats(season):
    file_path = season_dir(season) / f"player_game_stats_{season}.csv"
    player_games = pd.read_csv(file_path, low_memory=False)

    required = {"gameId", "playerId", "skaterFullName", "isGoalie"}
    missing = required - set(player_games.columns)
    if missing:
        raise ValueError(f"player_game_stats missing columns: {sorted(missing)}")

    player_games["playerId"] = pd.to_numeric(player_games["playerId"], errors="coerce")
    player_games = player_games.dropna(subset=["playerId"]).copy()
    player_games["playerId"] = player_games["playerId"].astype(int)

    is_goalie_by_row = player_games["isGoalie"].astype(str).str.upper().eq("TRUE")
    is_goalie_by_pid = is_goalie_by_row.groupby(player_games["playerId"]).any()

    goalie_ids = set(is_goalie_by_pid[is_goalie_by_pid].index.astype(int))
    all_ids = set(is_goalie_by_pid.index.astype(int))
    skater_ids = all_ids - goalie_ids

    def pick_name(series):
        for x in series.astype(str):
            if x.strip():
                return x
        return "Unknown"

    pid_to_name = player_games.groupby("playerId")["skaterFullName"].apply(pick_name).to_dict()
    return skater_ids, goalie_ids, pid_to_name

def load_shifts(season):
    file_path = season_dir(season) / f"shifts_{season}.csv"
    df = pd.read_csv(file_path)
    df = df[df["period"].isin([1, 2, 3])].copy()

    for c in ["startSec", "endSec", "durSec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["startSec", "endSec"])
    df = df[df["endSec"] >= df["startSec"]].copy()

    df["playerId"] = pd.to_numeric(df["playerId"], errors="coerce")
    df = df.dropna(subset=["playerId"]).copy()
    df["playerId"] = df["playerId"].astype(int)

    df["teamId"] = pd.to_numeric(df["teamId"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["teamId"]).copy()
    df["teamId"] = df["teamId"].astype(int)
    return df

def load_pbp(season):
    file_path = season_dir(season) / f"pbp_events_{season}.csv"
    df = pd.read_csv(file_path)
    df = df[df["period"].isin([1, 2, 3])].copy()
    df["eventSec"] = pd.to_numeric(df["eventSec"], errors="coerce")
    df = df.dropna(subset=["eventSec"])
    pbp_events = df.copy()
    pbp_fenwick = df[df["isFenwick"] == True].copy()
    pbp_fenwick["teamId"] = pbp_fenwick["eventTeamId"]
    return pbp_events, pbp_fenwick

def load_teamgames(season):
    file_path = season_dir(season) / f"TeamGames_{season}.csv"
    tg = pd.read_csv(file_path)
    tg["gameDate"] = tg["gameDate"].apply(parse_date_col)
    home_team_by_game, away_team_by_game = {}, {}
    for gid, sub in tg.groupby("gameId"):
        h = sub[sub["home_or_away"] == "home"]["teamId"].astype(int).tolist()
        a = sub[sub["home_or_away"] == "away"]["teamId"].astype(int).tolist()
        if h:
            home_team_by_game[int(gid)] = int(h[0])
        if a:
            away_team_by_game[int(gid)] = int(a[0])

    b2b_map = {}
    for tid, sub in tg.groupby("teamId"):
        sub = sub.sort_values("gameDate")
        prev = None
        for _, r in sub.iterrows():
            gid = int(r["gameId"])
            d = r["gameDate"]
            b2b_map[(gid, int(tid))] = 1 if (pd.notna(d) and pd.notna(prev) and (d - prev).days == 1) else 0
            prev = d

    return b2b_map, home_team_by_game, away_team_by_game

def filter_games_with_no_shifts(season, pbp_events, pbp_fenwick, shifts):
    with_shifts = set(shifts["gameId"].astype(int).unique().tolist())
    pbp_games = set(pbp_events["gameId"].astype(int).unique().tolist())
    dropped = sorted(list(pbp_games - with_shifts))
    if dropped:
        ensure_dirs(season)
        out = missing_dir(season) / "games_dropped_no_shifts.csv"
        pd.DataFrame({"gameId": dropped}).to_csv(out, index=False)
        print(f"Dropped {len(dropped)} games without shifts")
    keep = list(with_shifts)
    events = pbp_events[pbp_events["gameId"].astype(int).isin(keep)].copy()
    fenwick = pbp_fenwick[pbp_fenwick["gameId"].astype(int).isin(keep)].copy()
    return events, fenwick

def build_indices_for_period(gid, per, pbp_events, pbp_fenwick, shifts, home_tid, away_tid):
    events = pbp_events[(pbp_events["gameId"] == gid) & (pbp_events["period"] == per)].copy()
    fenwick = pbp_fenwick[(pbp_fenwick["gameId"] == gid) & (pbp_fenwick["period"] == per)].copy()
    home_shifts = shifts[(shifts["gameId"] == gid) & (shifts["period"] == per) & (shifts["teamId"] == home_tid)].copy()
    away_shifts = shifts[(shifts["gameId"] == gid) & (shifts["period"] == per) & (shifts["teamId"] == away_tid)].copy()

    grid = sorted(set(
        [0.0, period_end_seconds(per)]
        + events["eventSec"].tolist()
        + home_shifts["startSec"].tolist() + home_shifts["endSec"].tolist()
        + away_shifts["startSec"].tolist() + away_shifts["endSec"].tolist()
        ))
    if len(grid) < 2:
        return grid, None

    types_by_t = defaultdict(set)
    for t, gsub in events.groupby("eventSec"):
        types_by_t[float(t)] = set(gsub["type"].tolist())

    stopp = set(float(t) for t, ts in types_by_t.items() if "STOPPAGE" in ts)
    fac = set(float(t) for t, ts in types_by_t.items() if "FACEOFF" in ts)
    goal_at_t = set(float(t) for t, ts in types_by_t.items() if "GOAL" in ts)

    fo_start_at_t = set()
    for t in fac:
        if (t in stopp) or ((t - 1.0) in stopp):
            fo_start_at_t.add(t)

    zone_by_t = {}
    if "faceoff_zone_home" in events.columns:
        fo_rows = events[events["type"] == "FACEOFF"][["eventSec", "faceoff_zone_home"]]
        for _, r in fo_rows.iterrows():
            tt = float(r["eventSec"])
            if tt in fo_start_at_t:
                z = r["faceoff_zone_home"]
                zone_by_t[tt] = "N" if pd.isna(z) else str(z)

    events_sort = events.sort_values("eventSec")
    times = events_sort["eventSec"].to_numpy()
    HS = events_sort["home_score"].fillna(0).astype(int).to_numpy()
    AS = events_sort["away_score"].fillna(0).astype(int).to_numpy()

    def last_idx_leq(t):
        return np.searchsorted(times, t, side="right") - 1

    score_at_left = {}
    for j in range(1, len(grid)):
        t0 = grid[j - 1]
        idx = last_idx_leq(t0)
        score_at_left[t0] = (int(HS[idx]), int(AS[idx])) if idx >= 0 else (0, 0)

    fenwick_sorted = fenwick.sort_values("eventSec")
    fenwick_times = fenwick_sorted["eventSec"].to_numpy()
    fenwick_xg = fenwick_sorted["xG"].fillna(0.0).to_numpy()
    fteam = fenwick_sorted["teamId"].astype(int).to_numpy()
    home_mask = (fteam == home_tid)
    home_xg_cum = np.cumsum(np.where(home_mask, fenwick_xg, 0.0))
    away_xg_cum = np.cumsum(np.where(~home_mask, fenwick_xg, 0.0))

    def xg_sum(a, b):
        if len(fenwick_times) == 0:
            return 0.0, 0.0
        r = np.searchsorted(fenwick_times, b, side="right") - 1
        l = np.searchsorted(fenwick_times, a, side="right") - 1
        if r < 0:
            return 0.0, 0.0
        h = home_xg_cum[r] - (home_xg_cum[l] if l >= 0 else 0.0)
        aw = away_xg_cum[r] - (away_xg_cum[l] if l >= 0 else 0.0)
        return float(h), float(aw)

    def make_sweeper(sh_df):
        s = sh_df.sort_values("startSec")[["playerId", "startSec"]].to_numpy()
        e = sh_df.sort_values("endSec")[["playerId", "endSec"]].to_numpy()
        return s, e, 0, 0, defaultdict(int)

    home_starts, home_ends, home_start_idx, home_end_idx, home_active = make_sweeper(home_shifts)
    away_starts, away_ends, away_start_idx, away_end_idx, away_active = make_sweeper(away_shifts)

    def update_active(t, s_arr, e_arr, si, ei, active):
        while si < len(s_arr) and float(s_arr[si][1]) <= t:
            pid = int(s_arr[si][0])
            active[pid] += 1
            si += 1
        while ei < len(e_arr) and float(e_arr[ei][1]) <= t:
            pid = int(e_arr[ei][0])
            if active.get(pid, 0) > 1:
                active[pid] -= 1
            else:
                active.pop(pid, None)
            ei += 1
        return si, ei, active

    return grid, dict(
        fo_start_at_t=fo_start_at_t,
        zone_by_t=zone_by_t,
        goal_at_t=goal_at_t,
        score_at_left=score_at_left,
        xg_sum=xg_sum,
        home_starts=home_starts, home_ends=home_ends, home_start_idx=home_start_idx, home_end_idx=home_end_idx, home_active=home_active,
        away_starts=away_starts, away_ends=away_ends, away_start_idx=away_start_idx, away_end_idx=away_end_idx, away_active=away_active,
        update_active=update_active
        )

def encode_score_buckets(diff):
    d = dict(score_plus1=0, score_minus1=0, score_plus2=0, score_minus2=0, score_plus3p=0, score_minus3p=0)
    if diff == 1: d["score_plus1"] = 1
    elif diff == -1: d["score_minus1"] = 1
    elif diff == 2: d["score_plus2"] = 1
    elif diff == -2: d["score_minus2"] = 1
    elif diff >= 3: d["score_plus3p"] = 1
    elif diff <= -3: d["score_minus3p"] = 1
    return d

def row_strength_from_counts(rn, on):
    if rn > on: return "Power Play"
    if rn < on: return "Penalty Kill"
    return "Even Strength"

def ppx_pkx_for_next_stint(prev_row_strength, now_row_strength, is_fo_start):
    if is_fo_start and now_row_strength == "Even Strength" and prev_row_strength in ["Power Play", "Penalty Kill"]:
        return (1 if prev_row_strength == "Power Play" else 0,
                1 if prev_row_strength == "Penalty Kill" else 0)
    return (0, 0)

# Stint Builder
def build_stints(season, skater_ids, pbp_events, pbp_fenwick, shifts, b2b_map, home_team_by_game, away_team_by_game):
    rows = []
    games = sorted(set(pbp_events["gameId"].astype(int)).intersection(set(shifts["gameId"].astype(int))))
    for gid in games:
        home_tid = int(home_team_by_game.get(gid, -1))
        away_tid = int(away_team_by_game.get(gid, -1))
        if home_tid == -1 or away_tid == -1:
            continue

        p_per = set(pd.to_numeric(pbp_events.loc[pbp_events["gameId"] == gid, "period"], errors="coerce").dropna().astype(int))
        s_per = set(pd.to_numeric(shifts.loc[shifts["gameId"] == gid, "period"], errors="coerce").dropna().astype(int))
        periods = sorted(list((p_per | s_per) & {1, 2, 3}))
        for per in periods:
            grid, idx = build_indices_for_period(gid, per, pbp_events, pbp_fenwick, shifts, home_tid, away_tid)
            if idx is None or len(grid) < 2:
                continue

            home_starts, home_ends, home_start_idx, home_end_idx, home_active = idx["home_starts"], idx["home_ends"], idx["home_start_idx"], idx["home_end_idx"], idx["home_active"]
            away_starts, away_ends, away_start_idx, away_end_idx, away_active = idx["away_starts"], idx["away_ends"], idx["away_start_idx"], idx["away_end_idx"], idx["away_active"]
            update_active = idx["update_active"]

            def active_skaters(act):
                return tuple(sorted([p for p in act.keys() if p in skater_ids]))

            open_start = grid[0]
            home_start_idx, home_end_idx, home_active = update_active(grid[1], home_starts, home_ends, home_start_idx, home_end_idx, home_active)
            away_start_idx, away_end_idx, away_active = update_active(grid[1], away_starts, away_ends, away_start_idx, away_end_idx, away_active)
            hs, as_ = idx["score_at_left"].get(open_start, (0, 0))
            open_state = (active_skaters(home_active), active_skaters(away_active), 1, 1, hs - as_)
            zone_off, zone_def, zone_neu = 0, 0, 0
            PPxH, PKxH = 0, 0
            PPxA, PKxA = 0, 0
            ev_after_H = ev_after_A = 0
            pp_onfly_H = pp_onfly_A = 0
            if grid[0] in idx["fo_start_at_t"]:
                z = idx["zone_by_t"].get(grid[0], "N")
                zone_off, zone_def, zone_neu = (1, 0, 0) if z == "O" else ((0, 1, 0) if z == "D" else (0, 0, 1))

            for j in range(1, len(grid)):
                t0e = grid[j - 1]
                t1e = grid[j]
                home_start_idx, home_end_idx, home_active = update_active(t1e, home_starts, home_ends, home_start_idx, home_end_idx, home_active)
                away_start_idx, away_end_idx, away_active = update_active(t1e, away_starts, away_ends, away_start_idx, away_end_idx, away_active)
                home_set = active_skaters(home_active)
                away_set = active_skaters(away_active)
                hs, as_ = idx["score_at_left"].get(t0e, (0, 0))
                score_diff = hs - as_
                state = (home_set, away_set, 1, 1, score_diff)
                cut_now = (state != open_state) or (t1e in idx["goal_at_t"]) or (t1e in idx["fo_start_at_t"])
                if cut_now:
                    minutes = (t1e - open_start) / 60.0
                    if minutes > 0:
                        hx, ax = idx["xg_sum"](open_start, t1e)
                        home_skaters = len(open_state[0])
                        away_skaters = len(open_state[1])
                        sitH = "Power Play" if home_skaters > away_skaters else ("Penalty Kill" if home_skaters < away_skaters else "Even Strength")
                        sitA = "Power Play" if away_skaters > home_skaters else ("Penalty Kill" if away_skaters < home_skaters else "Even Strength")
                        home_score_flags = encode_score_buckets(open_state[4])
                        away_score_flags = encode_score_buckets(-open_state[4])
                        per2 = 1 if per == 2 else 0
                        per3 = 1 if per == 3 else 0
                        rows.append(dict(
                            gameId=gid, period=per, teamId=home_tid,
                            startSec=open_start, endSec=t1e, minutes=minutes,
                            xGF=hx, opp_xGF=ax, Situation=sitH,
                            players=list(open_state[0]), opp_players=list(open_state[1]),
                            is_home=1,
                            zone_off=zone_off, zone_def=zone_def, zone_neu=zone_neu,
                            ev_after_pp_on_fly=ev_after_H, pp_started_on_fly=pp_onfly_H,
                            PPx=PPxH, PKx=PKxH,
                            score_plus1=home_score_flags["score_plus1"], score_minus1=home_score_flags["score_minus1"],
                            score_plus2=home_score_flags["score_plus2"], score_minus2=home_score_flags["score_minus2"],
                            score_plus3p=home_score_flags["score_plus3p"], score_minus3p=home_score_flags["score_minus3p"],
                            row_b2b=int(b2b_map.get((gid, home_tid), 0)),
                            opp_b2b=int(b2b_map.get((gid, away_tid), 0)),
                            row_n_skaters=home_skaters, opp_n_skaters=away_skaters,
                            per2=per2, per3=per3,
                            strength_key=f"{home_skaters}v{away_skaters}"
                            ))
                        rows.append(dict(
                            gameId=gid, period=per, teamId=away_tid,
                            startSec=open_start, endSec=t1e, minutes=minutes,
                            xGF=ax, opp_xGF=hx, Situation=sitA,
                            players=list(open_state[1]), opp_players=list(open_state[0]),
                            is_home=0,
                            zone_off=zone_def, zone_def=zone_off, zone_neu=zone_neu,
                            ev_after_pp_on_fly=ev_after_A, pp_started_on_fly=pp_onfly_A,
                            PPx=PPxA, PKx=PKxA,
                            score_plus1=away_score_flags["score_plus1"], score_minus1=away_score_flags["score_minus1"],
                            score_plus2=away_score_flags["score_plus2"], score_minus2=away_score_flags["score_minus2"],
                            score_plus3p=away_score_flags["score_plus3p"], score_minus3p=away_score_flags["score_minus3p"],
                            row_b2b=int(b2b_map.get((gid, away_tid), 0)),
                            opp_b2b=int(b2b_map.get((gid, home_tid), 0)),
                            row_n_skaters=away_skaters, opp_n_skaters=home_skaters,
                            per2=per2, per3=per3,
                            strength_key=f"{away_skaters}v{home_skaters}"
                            ))

                    prevH = row_strength_from_counts(len(open_state[0]), len(open_state[1]))
                    prevA = row_strength_from_counts(len(open_state[1]), len(open_state[0]))
                    nowH = row_strength_from_counts(len(state[0]), len(state[1]))
                    nowA = row_strength_from_counts(len(state[1]), len(state[0]))
                    is_fo = 1 if (t1e in idx["fo_start_at_t"]) else 0
                    if is_fo:
                        z = idx["zone_by_t"].get(t1e, "N")
                        zone_off, zone_def, zone_neu = (1, 0, 0) if z == "O" else ((0, 1, 0) if z == "D" else (0, 0, 1))
                    else:
                        zone_off, zone_def, zone_neu = 0, 0, 0

                    PPxH, PKxH = ppx_pkx_for_next_stint(prevH, nowH, is_fo)
                    PPxA, PKxA = ppx_pkx_for_next_stint(prevA, nowA, is_fo)
                    ev_after_H = int(prevH in ["Power Play", "Penalty Kill"] and nowH == "Even Strength" and not is_fo)
                    ev_after_A = int(prevA in ["Power Play", "Penalty Kill"] and nowA == "Even Strength" and not is_fo)
                    pp_onfly_H = int(prevH == "Even Strength" and nowH in ["Power Play", "Penalty Kill"] and not is_fo)
                    pp_onfly_A = int(prevA == "Even Strength" and nowA in ["Power Play", "Penalty Kill"] and not is_fo)
                    open_start = t1e
                    open_state = state

            if open_start < grid[-1]:
                minutes = (grid[-1] - open_start) / 60.0
                if minutes > 0:
                    hx, ax = idx["xg_sum"](open_start, grid[-1])
                    home_skaters = len(open_state[0])
                    away_skaters = len(open_state[1])
                    sitH = "Power Play" if home_skaters > away_skaters else ("Penalty Kill" if home_skaters < away_skaters else "Even Strength")
                    sitA = "Power Play" if away_skaters > home_skaters else ("Penalty Kill" if away_skaters < home_skaters else "Even Strength")
                    home_score_flags = encode_score_buckets(open_state[4])
                    away_score_flags = encode_score_buckets(-open_state[4])
                    per2 = 1 if per == 2 else 0
                    per3 = 1 if per == 3 else 0
                    rows.append(dict(
                        gameId=gid, period=per, teamId=home_tid,
                        startSec=open_start, endSec=grid[-1], minutes=minutes,
                        xGF=hx, opp_xGF=ax, Situation=sitH,
                        players=list(open_state[0]), opp_players=list(open_state[1]),
                        is_home=1,
                        zone_off=zone_off, zone_def=zone_def, zone_neu=zone_neu,
                        ev_after_pp_on_fly=ev_after_H, pp_started_on_fly=pp_onfly_H,
                        PPx=PPxH, PKx=PKxH,
                        score_plus1=home_score_flags["score_plus1"], score_minus1=home_score_flags["score_minus1"],
                        score_plus2=home_score_flags["score_plus2"], score_minus2=home_score_flags["score_minus2"],
                        score_plus3p=home_score_flags["score_plus3p"], score_minus3p=home_score_flags["score_minus3p"],
                        row_b2b=int(b2b_map.get((gid, home_tid), 0)),
                        opp_b2b=int(b2b_map.get((gid, away_tid), 0)),
                        row_n_skaters=home_skaters, opp_n_skaters=away_skaters,
                        per2=per2, per3=per3,
                        strength_key=f"{home_skaters}v{away_skaters}"
                        ))
                    rows.append(dict(
                        gameId=gid, period=per, teamId=away_tid,
                        startSec=open_start, endSec=grid[-1], minutes=minutes,
                        xGF=ax, opp_xGF=hx, Situation=sitA,
                        players=list(open_state[1]), opp_players=list(open_state[0]),
                        is_home=0,
                        zone_off=zone_def, zone_def=zone_off, zone_neu=zone_neu,
                        ev_after_pp_on_fly=ev_after_A, pp_started_on_fly=pp_onfly_A,
                        PPx=PPxA, PKx=PKxA,
                        score_plus1=away_score_flags["score_plus1"], score_minus1=away_score_flags["score_minus1"],
                        score_plus2=away_score_flags["score_plus2"], score_minus2=away_score_flags["score_minus2"],
                        score_plus3p=away_score_flags["score_plus3p"], score_minus3p=away_score_flags["score_minus3p"],
                        row_b2b=int(b2b_map.get((gid, away_tid), 0)),
                        opp_b2b=int(b2b_map.get((gid, home_tid), 0)),
                        row_n_skaters=away_skaters, opp_n_skaters=home_skaters,
                        per2=per2, per3=per3,
                        strength_key=f"{away_skaters}v{home_skaters}"
                        ))

    print(f"Built {len(rows):,} Stint rows for {season}")
    return pd.DataFrame(rows)

def ensure_stints(season, skater_ids, pbp_ctx, pbp_fen, shifts, b2b_map, home_team_by_game, away_team_by_game):
    ensure_dirs(season)
    sp = stints_path(season)
    if sp.exists() and not rebuild_stint_toggle:
        return pd.read_csv(sp)
    st = build_stints(season, skater_ids, pbp_ctx, pbp_fen, shifts, b2b_map, home_team_by_game, away_team_by_game)
    st["players"] = st["players"].apply(lambda vals: ";".join(str(int(x)) for x in vals) if isinstance(vals, list) else str(vals))
    st["opp_players"] = st["opp_players"].apply(lambda vals: ";".join(str(int(x)) for x in vals) if isinstance(vals, list) else str(vals))
    st.to_csv(sp, index=False)
    return st

def parse_players_field(val):
    if isinstance(val, list):
        return val
    if val is None:
        return []
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return []
    if ";" in s:
        parts = [p.strip() for p in s.split(";")]
    else:
        parts = s.replace("[", "").replace("]", "").replace(",", " ").split()
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out

# Design Matrix
def filter_rows_for_situation(df, situation):
    base = df[df["period"].isin([1, 2, 3])].copy()

    if "strength_key" not in base.columns:
        a = base["row_n_skaters"].astype(int)
        b = base["opp_n_skaters"].astype(int)
        base["strength_key"] = a.astype(str) + "v" + b.astype(str)

    if situation == "Even Strength":
        sub = base[base["row_n_skaters"] == base["opp_n_skaters"]].copy()
    elif situation == "Power Play":
        sub = base[base["row_n_skaters"] > base["opp_n_skaters"]].copy()
    else:
        sub = base[base["row_n_skaters"] < base["opp_n_skaters"]].copy()

    keep = keep_strength_by_situation.get(situation, set())
    sub = sub[sub["strength_key"].isin(keep)].copy()
    return sub.reset_index(drop=True)

def prepare_design(stints, situation):
    stints_subset = filter_rows_for_situation(stints, situation)

    player_id_set = set()
    for lst in stints_subset["players"]:
        player_id_set.update(parse_players_field(lst))
    for lst in stints_subset["opp_players"]:
        player_id_set.update(parse_players_field(lst))
    player_ids = sorted(player_id_set)
    player_index = {pid: i for i, pid in enumerate(player_ids)}
    number_players = len(player_ids)

    base_ctx = [
        "is_home",
        "zone_off", "zone_def", "zone_neu",
        "ev_after_pp_on_fly", "pp_started_on_fly", "PPx", "PKx",
        "score_plus1", "score_minus1", "score_plus2", "score_minus2", "score_plus3p", "score_minus3p",
        "row_b2b", "opp_b2b",
        "per2", "per3",
        "State_5v5", "State_4v4", "State_3v3"
        ]
    stints_subset["State_5v5"] = (stints_subset["strength_key"] == "5v5").astype(int)
    stints_subset["State_4v4"] = (stints_subset["strength_key"] == "4v4").astype(int)
    stints_subset["State_3v3"] = (stints_subset["strength_key"] == "3v3").astype(int)

    for c in base_ctx:
        if c not in stints_subset.columns:
            stints_subset[c] = 0

    n = len(stints_subset)
    X_off = lil_matrix((n, number_players), dtype=float)
    X_def = lil_matrix((n, number_players), dtype=float)
    X_ctx = np.zeros((n, len(base_ctx)), dtype=float)
    y_rate = np.zeros(n, dtype=float)
    y_raw = np.zeros(n, dtype=float)
    w = np.zeros(n, dtype=float)

    for i, row in stints_subset.iterrows():
        rps = parse_players_field(row["players"])
        ops = parse_players_field(row["opp_players"])
        scale_off = 1.0 / max(1, len(rps))
        scale_def = 1.0 / max(1, len(ops))
        
        for pid in rps:
            j = player_index.get(pid)
            if j is not None:
                X_off[i, j] += scale_off
        for pid in ops:
            j = player_index.get(pid)
            if j is not None:
                X_def[i, j] += scale_def

        X_ctx[i, :] = [float(row[c]) for c in base_ctx]
        mins = float(row["minutes"])
        w[i] = mins
        xgf = float(row["xGF"])
        y_raw[i] = xgf
        y_rate[i] = 60.0 * xgf / mins if mins > 0 else 0.0

    avg_stint_minutes = float(stints_subset["minutes"].mean()) if len(stints_subset) else float("nan")
    return X_off.tocsr(), X_def.tocsr(), csr_matrix(X_ctx), y_rate, y_raw, w, player_ids, stints_subset, base_ctx, avg_stint_minutes

# Minutes-aware scaling
def build_minutes_scalers(ev, player_ids, goalie_ids):
    mins_by_pid = defaultdict(float)
    for _, r in ev.iterrows():
        m = float(r["minutes"])
        for pid in parse_players_field(r["players"]):
            mins_by_pid[pid] += m
    scalers = {}
    for pid in player_ids:
        m = max(minute_floor_for_scaler, float(mins_by_pid.get(pid, 0.0)))
        s = 1.0 / math.sqrt(m)
        if pid in goalie_ids:
            s *= goalie_L2_multiplier
        scalers[pid] = s
    return scalers

def apply_scalers_to_blocks(X_off, X_def, player_ids, scalers):
    Xo = X_off.tocsc(copy=True)
    Xd = X_def.tocsc(copy=True)
    for j, pid in enumerate(player_ids):
        s = float(scalers[pid])
        Xo.data[Xo.indptr[j]:Xo.indptr[j+1]] *= s
        Xd.data[Xd.indptr[j]:Xd.indptr[j+1]] *= s
    return Xo.tocsr(), Xd.tocsr()

# Cross Validation and Model Fit
def minutes_weighted_mse(y_true, y_pred, w):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    w = np.asarray(w, float)
    w = np.where(np.isfinite(w), w, 0.0)
    denom = max(1e-9, w.sum())
    return float((w * (y_true - y_pred) ** 2).sum() / denom)

def choose_lambda(X, y_rate, y_raw, w, groups, seed):
    # grouped CV by gameId
    gkf = GroupKFold(n_splits=5)
    rows = []
    best_lambda = None
    best_mse_rate = float("inf")

    # store per-fold RMSE for paired  t-test vs OLS
    fold_rmse_rate = {}
    fold_rmse_raw = {}

    for lam in lambda_grid:
        rmse_rate_folds = []
        rmse_raw_folds = []
        beta_norms = []

        for tr, va in gkf.split(X, groups=groups):
            if lam == 0.0:
                model = LinearRegression(fit_intercept=True)
                model.fit(X[tr], y_rate[tr], sample_weight=w[tr])
            else:
                model = Ridge(alpha=float(lam), fit_intercept=True, random_state=seed)
                model.fit(X[tr], y_rate[tr], sample_weight=w[tr])

            # per-60 prediction (training target)
            p_rate = model.predict(X[va]) 
            p_raw = p_rate * (w[va] / 60.0)
            mse_rate = minutes_weighted_mse(y_rate[va], p_rate, w[va])
            mse_raw = minutes_weighted_mse(y_raw[va], p_raw, w[va])
            rmse_rate_folds.append(float(np.sqrt(mse_rate)))
            rmse_raw_folds.append(float(np.sqrt(mse_raw)))

            try:
                beta_norms.append(float(np.linalg.norm(model.coef_.ravel(), ord=2)))
            except Exception:
                beta_norms.append(float("nan"))

        fold_rmse_rate[lam] = rmse_rate_folds
        fold_rmse_raw[lam] = rmse_raw_folds

        mean_mse_rate = float(np.mean(np.array(rmse_rate_folds) ** 2))
        mean_mse_raw = float(np.mean(np.array(rmse_raw_folds) ** 2))
        mean_rmse_rate = float(np.mean(rmse_rate_folds))
        mean_rmse_raw = float(np.mean(rmse_raw_folds))
        beta_l2_mean = float(np.nanmean(beta_norms)) if len(beta_norms) else float("nan")

        rows.append({
            "lambda": lam,
            "cv_mse_rate": mean_mse_rate,
            "cv_rmse_rate": mean_rmse_rate,
            "cv_mse_raw": mean_mse_raw,
            "per_stint_xG_RMSE": mean_rmse_raw,
            "folds": len(rmse_rate_folds),
            "beta_l2_mean": beta_l2_mean,
            "paired_t_rate": np.nan,
            "paired_p_rate": np.nan,
            "paired_t_raw": np.nan,
            "paired_p_raw": np.nan,
            "mean_delta_rate": np.nan,
            "mean_delta_raw": np.nan
            })

        if mean_mse_rate < best_mse_rate:
            best_mse_rate = mean_mse_rate
            best_lambda = lam

    # fill paired tests vs OLS into the best-lambda row
    if 0.0 in fold_rmse_rate and best_lambda is not None:
        ols_r = np.array(fold_rmse_rate[0.0], dtype=float)
        ols_x = np.array(fold_rmse_raw[0.0], dtype=float)
        bst_r = np.array(fold_rmse_rate[best_lambda], dtype=float)
        bst_x = np.array(fold_rmse_raw[best_lambda], dtype=float)
        d_rate = bst_r - ols_r
        d_raw = bst_x - ols_x
        t_rate, p_rate = ttest_rel(bst_r, ols_r, nan_policy="omit")
        t_raw, p_raw = ttest_rel(bst_x, ols_x, nan_policy="omit")

        for r in rows:
            if r["lambda"] == best_lambda:
                r["paired_t_rate"] = float(t_rate)
                r["paired_p_rate"] = float(p_rate)
                r["paired_t_raw"] = float(t_raw)
                r["paired_p_raw"] = float(p_raw)
                r["mean_delta_rate"] = float(d_rate.mean())
                r["mean_delta_raw"] = float(d_raw.mean())
                break

    return best_lambda, best_mse_rate, rows

def fit(season, situation, X_off, X_def, X_ctx, y_rate, y_raw, w, player_ids, ev, base_ctx, avg_stint_minutes, pid_to_name, goalie_ids):
    groups = ev["gameId"].astype(int).to_numpy()
    scalers = build_minutes_scalers(ev, player_ids, goalie_ids)
    Xo_s, Xd_s = apply_scalers_to_blocks(X_off, X_def, player_ids, scalers)

    # design = [off block | def block | context]
    X = hstack([Xo_s, Xd_s, X_ctx], format="csr")
    lam, cv_mse_rate, cv_rows = choose_lambda(X, y_rate, y_raw, w, groups, seed=random_seed)
    model = Ridge(alpha=lam, fit_intercept=True, random_state=random_seed)
    model.fit(X, y_rate, sample_weight=w)
    coef_all = model.coef_.ravel()
    nP = len(player_ids)
    b_off_scaled = coef_all[0:nP]
    b_def_scaled = coef_all[nP:2*nP]
    beta_off = np.zeros(nP, float)
    b_def = np.zeros(nP, float)
    for j, pid in enumerate(player_ids):
        s = float(scalers[pid])
        beta_off[j] = b_off_scaled[j] / s
        b_def[j] = b_def_scaled[j] / s

    off_xGF60 = {pid: float(beta_off[i]) for i, pid in enumerate(player_ids)}
    def_xGF60 = {pid: float(-b_def[i]) for i, pid in enumerate(player_ids)}  # positive = better defense
    tot_xGF60 = {pid: off_xGF60[pid] + def_xGF60[pid] for pid in player_ids}

    # minutes by player
    minutes = defaultdict(float)
    for _, r in ev.iterrows():
        m = float(r["minutes"])
        for pid in parse_players_field(r["players"]):
            minutes[pid] += m

    # write player output
    player_rows = []
    for pid in player_ids:
        player_rows.append(dict(
            playerId=pid,
            playerName=pid_to_name.get(pid, "Unknown"),
            minutes=float(minutes.get(pid, 0.0)),
            off_xGF60_raw=float(off_xGF60.get(pid, 0.0)),
            def_xGF60_raw=float(def_xGF60.get(pid, 0.0)),
            total_xGF60_raw=float(tot_xGF60.get(pid, 0.0))
            ))
    out = season_player_outfile(season, situation)
    pd.DataFrame(player_rows).to_csv(out, index=False)

    # append CV table to fit_meta
    ensure_dirs(season)
    meta_path = fitmeta_path(season)
    if not meta_path.exists():
        header = pd.DataFrame(columns=[
            "season","situation","lambda","lambda_off",
            "cv_minutes_weighted_MSE","RMSE_per60",
            "per_stint_xG_RMSE","avg_stint_minutes",
            "n_rows","n_players","n_ctx",
            "intercept_per60","is_best","folds","beta_l2_mean"
            ])
        header.to_csv(meta_path, index=False)

    df = pd.DataFrame(cv_rows)
    df["season"] = season
    df["situation"] = situation
    df["lambda_off"] = df["lambda"]
    df["cv_minutes_weighted_MSE"] = df["cv_mse_rate"]
    df["RMSE_per60"] = df["cv_rmse_rate"]
    df["avg_stint_minutes"] = float(avg_stint_minutes)
    df["n_rows"] = int(X.shape[0])
    df["n_players"] = int(nP)
    df["n_ctx"] = int(len(base_ctx))
    df["intercept_per60"] = float(model.intercept_)
    df["is_best"] = 0
    if len(df):
        df.loc[df["cv_mse_rate"].astype(float).idxmin(), "is_best"] = 1

    df = df[[
        "season","situation","lambda","lambda_off",
        "cv_minutes_weighted_MSE","RMSE_per60",
        "per_stint_xG_RMSE","avg_stint_minutes",
        "n_rows","n_players","n_ctx",
        "intercept_per60","is_best","folds","beta_l2_mean"
        ]]
    df.to_csv(meta_path, mode="a", index=False, header=False)
    comp_path = fitcomparison_path(season)
    write_header = not comp_path.exists()
    best_row = min(cv_rows, key=lambda r: r["cv_mse_rate"])
    ols_row = None
    for r in cv_rows:
        if abs(float(r["lambda"])) <= 1e-12:
            ols_row = r
            break

    comp_rows = []
    comp_rows.append({
        "season": season,
        "situation": situation,
        "model": "Ridge",
        "lambda": float(best_row["lambda"]),
        "cv_rmse_rate": float(best_row["cv_rmse_rate"]),
        "per_stint_xG_RMSE": float(best_row["per_stint_xG_RMSE"]),
        "paired_t_rate": float(best_row.get("paired_t_rate", np.nan)),
        "paired_p_rate": float(best_row.get("paired_p_rate", np.nan)),
        "paired_t_raw": float(best_row.get("paired_t_raw", np.nan)),
        "paired_p_raw": float(best_row.get("paired_p_raw", np.nan)),
        "mean_delta_rate": float(best_row.get("mean_delta_rate", np.nan)),
        "mean_delta_raw": float(best_row.get("mean_delta_raw", np.nan))
        })
    if ols_row is not None:
        comp_rows.append({
            "season": season,
            "situation": situation,
            "model": "OLS",
            "lambda": 0.0,
            "cv_rmse_rate": float(ols_row["cv_rmse_rate"]),
            "per_stint_xG_RMSE": float(ols_row["per_stint_xG_RMSE"]),
            "paired_t_rate": np.nan,
            "paired_p_rate": np.nan,
            "paired_t_raw": np.nan,
            "paired_p_raw": np.nan,
            "mean_delta_rate": np.nan,
            "mean_delta_raw": np.nan
            })

    pd.DataFrame(comp_rows).to_csv(comp_path, mode="a", index=False, header=write_header)

# Driver
def run_one_season(year):
    season = season_label(year)
    print(f"Running RAPM for {season}")
    ensure_dirs(season)

    if clear_fit_meta_toggle:
        meta_path = fitmeta_path(season)
        header = pd.DataFrame(columns=[
            "season","situation","lambda","lambda_off",
            "cv_minutes_weighted_MSE","RMSE_per60",
            "per_stint_xG_RMSE","avg_stint_minutes",
            "n_rows","n_players","n_ctx",
            "intercept_per60","is_best","folds","beta_l2_mean"
            ])
        header.to_csv(meta_path, index=False)

    skater_ids, goalie_ids, pid_to_name = load_player_game_stats(season)
    shifts = load_shifts(season)
    pbp_ctx, pbp_fen = load_pbp(season)
    
    b2b_map, home_team_by_game, away_team_by_game = load_teamgames(season)
    pbp_ctx, pbp_fen = filter_games_with_no_shifts(season, pbp_ctx, pbp_fen, shifts)

    stints = ensure_stints(season, skater_ids, pbp_ctx, pbp_fen, shifts, b2b_map, home_team_by_game, away_team_by_game)
    
    duration_sec = pd.to_numeric(stints["endSec"], errors="coerce") - pd.to_numeric(stints["startSec"], errors="coerce")
    stints = stints[duration_sec.fillna(0.0) >= 10.0].copy()

    stints["players"] = stints["players"].apply(parse_players_field)
    stints["opp_players"] = stints["opp_players"].apply(parse_players_field)

    for situation in situation_list:
        X_off, X_def, X_ctx, y_rate, y_raw, w, P, ev, base_ctx, avg_stint_minutes = prepare_design(stints, situation)
        if X_off.shape[0] == 0:
            print("No rows; skipping")
            continue
        fit(season, situation, X_off, X_def, X_ctx, y_rate, y_raw, w, P, ev, base_ctx, avg_stint_minutes, pid_to_name, goalie_ids)

def main():
    for y in range(start_year, end_year + 1):
        run_one_season(y)

if __name__ == "__main__":
    main()