# -*- coding: utf-8 -*-
"""
GAR/WAR/SPAR Calculation File

Calcualtes GAR using RAPM results and NST data, defining the replacement pool with team-based cutoffs. 
GAR -> WAR/SPAR done via Goals per Win/Standing Point table sourced from EvolvingHockey and filled for recent seasons.

NOTE: I used AI to help clean up my script a bit, as I had some extraneous code left over from previous attempts at getting it working
AI also helped with adding a deployment gate for if players were being deployed often in the defensive zone (this leads to a higher xGA for that player)
"""
# Imports
import pandas as pd
import numpy as np
from pathlib import Path

# Config
script_dir = Path(__file__).resolve().parent
data_root = script_dir.parent / "Data" / "Clean Data"
start_year = 2015
end_year = 2024

goal_table_file = "NHL_GOAL_TABLE_MASTER.csv"
nst_file_map = {
    "EV": "NST_player_master_ES_{season}.csv",
    "PP": "NST_player_master_PP_{season}.csv",
    "PK": "NST_player_master_PK_{season}.csv",
    }

# Replacement pool cutoffs
top_forwards_es = 12
top_defenders_es = 8
top_forwards_pp = 5
top_defenders_pp = 2
top_forwards_pk = 4
top_defenders_pk = 4

# Replacement Baseline configs
min_games_for_role = 15
min_games_for_base = 10
min_toi_for_base = 100.0
trim_q_def_low = 0.10
trim_q_off_high = 0.05

k_shrink = {"EV": 300.0, "PP": 120.0, "PK": 1200.0}
k_shrink_es_by_pos = {"F": 300.0, "D": 260.0}
lambda_decay = 0.55
usage_beta_def_zone = 0.10
usage_clip_min = 0.92
usage_clip_max = 1.08

pct_gp_gate = 9
pct_pp_toi_min = 60.0
pct_pk_toi_min = 60.0

# Role buckets (by expected TOI/G). Used to assign RoleBucket for percentiles.
forward_role_buckets = [
    ("4th Line", 9.0),
    ("Bottom-6", 12.0),
    ("Middle-6", 15.5),
    ("Top-6", 18.5),
    ("1st Line", 20.0),
    ]
defender_role_buckets = [
    ("3rd Pair", 15.0),
    ("Bottom-4", 18.0),
    ("Top-4", 20.0),
    ("1st Pair", 21.5),
    ]

possible_gp_cols = ["GP", "Games Played"]
final_output_columns = [
    "Player","PlayerID","TeamStd_Primary","PosBucket",
    "GP","TOI_EV","TOI_PP","TOI_PK","TOI_all","TOI_perG",
    "ES_xGF60_shr","ES_xGA60_shr","ES_xGF60_blend","ES_xGA60_blend",
    "PP_GF60_shr","PK_GA60_shr",
    "GAR_ES_off","GAR_ES_def","GAR_PP","GAR_PK",
    "GAR_off","GAR_def","GAR_total",
    "WAR","SPAR",
    "RoleBucket",
    "pct_ES_pos_role","pct_PP_pos_role","pct_PK_pos_role",
    "pct_off_pos_role","pct_def_pos_role","pct_total_pos_role",
    ]

# Helpers
def season_label(year):
    return f"{year}-{year+1}"

def pos_bucket(pos_text):
    if not isinstance(pos_text, str):
        return "F"
    p = pos_text.upper()
    if ("LD" in p) or ("RD" in p) or (p == "D") or (" D" in p) or p.startswith("D"):
        return "D"
    if any(x in p for x in ["C","LW","RW","F"]):
        return "F"
    return "F"

def to_float(x, default=0.0):
    return pd.to_numeric(x, errors="coerce").fillna(default).astype(float)

def read_gp(df):
    for c in possible_gp_cols:
        if c in df.columns:
            return to_float(df[c], 0.0)
    return pd.Series(np.zeros(len(df), dtype=float), index=df.index)

def choose_team_col(df):
    for c in ["TeamStd_Primary","Team","TeamStd"]:
        if c in df.columns:
            return c
    return None

def load_nst(root, situation, season):
    path = root / season / nst_file_map[situation].format(season=season)
    df = pd.read_csv(path)
    df["Player"] = df["Player"].astype(str).str.strip()
    pid = pd.to_numeric(df.get("playerId", df.get("PlayerID")), errors="coerce").astype("Int64")
    df["playerId"] = pid
    df["PosBucket"] = df.get("Position", "F").apply(pos_bucket)
    tcol = choose_team_col(df)
    df["TeamStd_Primary"] = df[tcol].astype(str) if tcol else ""

    def alias(out_name, candidates):
        if out_name in df.columns:
            return
        for c in candidates:
            if c in df.columns:
                df[out_name] = pd.to_numeric(df[c], errors="coerce")
                return
        df[out_name] = np.nan

    alias("xGF_per60", ["xGF/60","xGF60","xGF_per60"])
    alias("xGA_per60", ["xGA/60","xGA60","xGA_per60"])
    alias("GF_per60", ["GF/60","GF60","GF_per60"])
    alias("GA_per60", ["GA/60","GA60","GA_per60"])
    df["TOI"] = pd.to_numeric(df.get("TOI"), errors="coerce").fillna(0.0)
    df["GP"] = read_gp(df)

    for c in ["Off. Zone Start Pct","PDO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    return df

def shrink_weight_ev_by_pos(toi_min, pos_series):
    toi = np.maximum(0.0, np.asarray(toi_min, dtype=float))
    pos = pos_series.astype(str).fillna("F")
    k_vec = np.array([k_shrink_es_by_pos.get(p, k_shrink["EV"]) for p in pos], dtype=float)
    return np.where(toi > 0.0, toi / (toi + k_vec), 0.0)

def shrink_weight(toi_min, k_val):
    toi = np.maximum(0.0, np.asarray(toi_min, dtype=float))
    return np.where(toi > 0.0, toi / (toi + float(k_val)), 0.0)

def weighted_trimmed_mean(values, weights, drop_low_q=None, drop_high_q=None, default_val=np.nan):
    v = pd.to_numeric(values, errors="coerce").to_numpy()
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    mask = np.isfinite(v) & (w > 0)
    if mask.sum() == 0 or w[mask].sum() <= 0:
        return default_val

    v_use, w_use = v[mask], w[mask]
    lo, hi = -np.inf, np.inf
    if drop_low_q is not None and 0.0 < drop_low_q < 1.0:
        lo = np.quantile(v_use, drop_low_q)
    if drop_high_q is not None and 0.0 < drop_high_q < 1.0:
        hi = np.quantile(v_use, 1.0 - drop_high_q)

    keep = (v_use >= lo) & (v_use <= hi)
    if keep.sum() == 0 or w_use[keep].sum() <= 0:
        return float(np.average(v_use, weights=w_use + 1e-9))
    return float(np.average(v_use[keep], weights=w_use[keep] + 1e-9))

# Replacement Baseline by position
def league_replacement_by_pos(nst_ev, nst_pp, nst_pk):
    def pool_from_sit(df, top_f, top_d):
        if df is None or len(df) == 0:
            return {"F": pd.DataFrame(), "D": pd.DataFrame()}
        t = df.copy()
        t["rank_team_pos"] = t.groupby(["TeamStd_Primary","PosBucket"])["TOI"].rank(method="first", ascending=False)
        cutoff = np.where(t["PosBucket"] == "D", top_d, top_f)
        t["is_rep"] = t["rank_team_pos"] > cutoff
        elig = t["is_rep"] & (to_float(t["GP"]) >= min_games_for_base) & (to_float(t["TOI"]) >= min_toi_for_base)
        return {
            "F": t.loc[elig & (t["PosBucket"]=="F")].copy(),
            "D": t.loc[elig & (t["PosBucket"]=="D")].copy()
            }

    ev_pool = pool_from_sit(nst_ev, top_forwards_es, top_defenders_es)
    pp_pool = pool_from_sit(nst_pp, top_forwards_pp, top_defenders_pp)
    pk_pool = pool_from_sit(nst_pk, top_forwards_pk, top_defenders_pk)

    def mean_trim(df, val_col, drop_low_q=None, drop_high_q=None):
        if df is None or len(df) == 0 or val_col not in df.columns:
            return np.nan
        v = pd.to_numeric(df[val_col], errors="coerce")
        w = to_float(df.get("TOI", 0.0), 0.0)
        return weighted_trimmed_mean(v, w, drop_low_q=drop_low_q, drop_high_q=drop_high_q, default_val=np.nan)

    repl_xGF60_ES_pos = {}
    repl_xGA60_ES_pos = {}
    repl_GF60_PP_pos = {}
    repl_GA60_PK_pos = {}

    for pos in ["F","D"]:
        repl_xGF60_ES_pos[pos] = mean_trim(ev_pool[pos], "xGF_per60", drop_high_q=trim_q_off_high)
        repl_xGA60_ES_pos[pos] = mean_trim(ev_pool[pos], "xGA_per60", drop_low_q=trim_q_def_low)
        repl_GF60_PP_pos[pos] = mean_trim(pp_pool[pos], "GF_per60", drop_high_q=trim_q_off_high)
        repl_GA60_PK_pos[pos] = mean_trim(pk_pool[pos], "GA_per60", drop_low_q=trim_q_def_low)

    for pos in ["F","D"]:
        other = "D" if pos == "F" else "F"
        if not np.isfinite(repl_xGF60_ES_pos[pos]): repl_xGF60_ES_pos[pos] = repl_xGF60_ES_pos.get(other, np.nan)
        if not np.isfinite(repl_xGA60_ES_pos[pos]): repl_xGA60_ES_pos[pos] = repl_xGA60_ES_pos.get(other, np.nan)
        if not np.isfinite(repl_GF60_PP_pos[pos]): repl_GF60_PP_pos[pos] = repl_GF60_PP_pos.get(other, np.nan)
        if not np.isfinite(repl_GA60_PK_pos[pos]): repl_GA60_PK_pos[pos] = repl_GA60_PK_pos.get(other, np.nan)

    return {
        "repl_xGF60_ES_pos": repl_xGF60_ES_pos,
        "repl_xGA60_ES_pos": repl_xGA60_ES_pos,
        "repl_GF60_PP_pos": repl_GF60_PP_pos,
        "repl_GA60_PK_pos": repl_GA60_PK_pos,
        }

def usage_factor_from_ozpct(series_ozpct):
    oz = to_float(series_ozpct, 50.0)
    dz_share = 100.0 - oz
    dz_centered = dz_share - float(dz_share.mean())
    fac = 1.0 + usage_beta_def_zone * (dz_centered / 25.0)
    return np.clip(fac, usage_clip_min, usage_clip_max)

# Percentile Buckets
def add_percentiles_pos_role(m):
    out = m.copy()
    for c in ["GAR_ES_off","GAR_ES_def","GAR_PP","GAR_PK","GAR_off","GAR_def","GAR_total",
              "TOI_EV","TOI_PP","TOI_PK","TOI_all","GP"]:
        out[c] = pd.to_numeric(out.get(c, 0.0), errors="coerce").fillna(0.0)

    out["GAR_ES_total"] = out["GAR_ES_off"] + out["GAR_ES_def"]
    if "RoleBucket" not in out.columns:
        out["RoleBucket"] = "Unassigned"

    mask_es = (out["GP"] >= pct_gp_gate) & (out["TOI_EV"] > 0)
    mask_offd = (out["GP"] >= pct_gp_gate) & (out["TOI_all"] > 0)
    mask_pp = (out["TOI_PP"] >= pct_pp_toi_min)
    mask_pk = (out["TOI_PK"] >= pct_pk_toi_min)

    def rank_into(col, newc, elig_mask):
        out[newc] = np.nan
        sub = out.loc[elig_mask, ["PosBucket","RoleBucket", col]].copy()
        if not sub.empty:
            ranks = (sub.groupby(["PosBucket","RoleBucket"], dropna=False)[col].rank(method="average", pct=True) * 100.0).round(1)
            out.loc[elig_mask, newc] = ranks.values

    rank_into("GAR_ES_total", "pct_ES_pos_role", mask_es)
    rank_into("GAR_PP", "pct_PP_pos_role", mask_pp)
    rank_into("GAR_PK", "pct_PK_pos_role", mask_pk)
    rank_into("GAR_off", "pct_off_pos_role", mask_offd)
    rank_into("GAR_def", "pct_def_pos_role", mask_offd)
    rank_into("GAR_total", "pct_total_pos_role", mask_offd)

    return out

# Priors
def compute_ev_shrunk_for_one_season(root, season):
    if not season:
        return None
    nst_ev = load_nst(root, "EV", season)
    nst_pp = load_nst(root, "PP", season)
    nst_pk = load_nst(root, "PK", season)
    repl_pos = league_replacement_by_pos(nst_ev, nst_pp, nst_pk)
    pos = nst_ev["PosBucket"].fillna("F").astype(str).values
    repl_xgf_vec = np.array([repl_pos["repl_xGF60_ES_pos"].get(p, np.nan) for p in pos], dtype=float)
    repl_xga_vec = np.array([repl_pos["repl_xGA60_ES_pos"].get(p, np.nan) for p in pos], dtype=float)
    w_ev = shrink_weight_ev_by_pos(nst_ev["TOI"], nst_ev["PosBucket"])
    xgf = pd.to_numeric(nst_ev["xGF_per60"], errors="coerce").to_numpy()
    xga = pd.to_numeric(nst_ev["xGA_per60"], errors="coerce").to_numpy()
    xgf = np.where(np.isnan(xgf), repl_xgf_vec, xgf)
    xga = np.where(np.isnan(xga), repl_xga_vec, xga)
    es_xgf_shr = w_ev * xgf + (1.0 - w_ev) * repl_xgf_vec
    es_xga_shr = w_ev * xga + (1.0 - w_ev) * repl_xga_vec
    out = pd.DataFrame({
        "playerId": nst_ev["playerId"].astype("Int64"),
        "ES_xGF60_shr_prev": es_xgf_shr,
        "ES_xGA60_shr_prev": es_xga_shr,
        "TOI_EV_prev": to_float(nst_ev["TOI"], 0.0)
        })

    def mw_avg_ev(g):
        w = to_float(g["TOI_EV_prev"], 0.0) + 1e-9
        return pd.Series({
            "ES_xGF60_shr_prev": float(np.average(g["ES_xGF60_shr_prev"], weights=w)),
            "ES_xGA60_shr_prev": float(np.average(g["ES_xGA60_shr_prev"], weights=w)),
            "TOI_EV_prev": float(g["TOI_EV_prev"].sum())
            })

    out = out.groupby("playerId", as_index=False).apply(mw_avg_ev, include_groups=False).reset_index(drop=True)
    out["playerId"] = out["playerId"].astype("Int64")
    return out

def compute_st_priors_for_one_season(root, season, which):
    if not season:
        return None
    df = load_nst(root, which, season)

    if which == "PP":
        rate_col = "xGF_per60" if "xGF_per60" in df.columns else "GF_per60"
        out = pd.DataFrame({
            "playerId": df["playerId"].astype("Int64"),
            f"{which}_rate_prev": pd.to_numeric(df.get(rate_col), errors="coerce"),
            f"TOI_{which}_prev": to_float(df["TOI"], 0.0)
            })
    else:
        rate_col = "xGA_per60" if "xGA_per60" in df.columns else "GA_per60"
        out = pd.DataFrame({
            "playerId": df["playerId"].astype("Int64"),
            f"{which}_rate_prev": pd.to_numeric(df.get(rate_col), errors="coerce"),
            f"TOI_{which}_prev": to_float(df["TOI"], 0.0)
            })

    def mw(g):
        wtoi = to_float(g[f"TOI_{which}_prev"], 0.0) + 1e-9
        return pd.Series({
            f"{which}_rate_prev": float(np.average(g[f"{which}_rate_prev"], weights=wtoi)),
            f"TOI_{which}_prev": float(g[f"TOI_{which}_prev"].sum())
            })

    out = out.groupby("playerId", as_index=False).apply(mw, include_groups=False).reset_index(drop=True)
    out["playerId"] = out["playerId"].astype("Int64")
    return out

# Builder
def build_one_season(y):
    season = season_label(y)
    out_dir = data_root / season
    out_dir.mkdir(parents=True, exist_ok=True)

    goals_tab = pd.read_csv(data_root / goal_table_file)
    goals_tab["Season"] = goals_tab["Season"].astype(str).str.strip()
    row = goals_tab[goals_tab["Season"] == season]
    if row.empty:
        raise RuntimeError("Missing conversion row in goal table for " + season)
    goals_per_win = float(row.iloc[0]["GOALS_TO_WIN"])
    goals_per_standing_point = float(row.iloc[0]["GOALS_TO_STANDING_POINT"])

    nst_ev = load_nst(data_root, "EV", season)
    nst_pp = load_nst(data_root, "PP", season)
    nst_pk = load_nst(data_root, "PK", season)

    repl_pos = league_replacement_by_pos(nst_ev, nst_pp, nst_pk)
    ev_df = nst_ev.copy()
    ev_df["playerId"] = ev_df["playerId"].astype("Int64")
    w_ev = shrink_weight_ev_by_pos(ev_df["TOI"], ev_df["PosBucket"])
    pos_vec = ev_df["PosBucket"].fillna("F").astype(str).values
    repl_xgf_pos_vec = np.array([repl_pos["repl_xGF60_ES_pos"].get(p, np.nan) for p in pos_vec], dtype=float)
    repl_xga_pos_vec = np.array([repl_pos["repl_xGA60_ES_pos"].get(p, np.nan) for p in pos_vec], dtype=float)

    xgf_raw = pd.to_numeric(ev_df["xGF_per60"], errors="coerce").to_numpy()
    xga_raw = pd.to_numeric(ev_df["xGA_per60"], errors="coerce").to_numpy()
    xgf_raw = np.where(np.isnan(xgf_raw), repl_xgf_pos_vec, xgf_raw)
    xga_raw = np.where(np.isnan(xga_raw), repl_xga_pos_vec, xga_raw)
    ev_df["ES_xGF60_shr"] = w_ev * xgf_raw + (1.0 - w_ev) * repl_xgf_pos_vec
    ev_df["ES_xGA60_shr"] = w_ev * xga_raw + (1.0 - w_ev) * repl_xga_pos_vec

    season_prev1 = season_label(y-1) if y-1 >= start_year else None
    season_prev2 = season_label(y-2) if y-2 >= start_year else None
    ev_prev1 = compute_ev_shrunk_for_one_season(data_root, season_prev1) if season_prev1 else None
    ev_prev2 = compute_ev_shrunk_for_one_season(data_root, season_prev2) if season_prev2 else None
    pp_prev1 = compute_st_priors_for_one_season(data_root, season_prev1, "PP") if season_prev1 else None
    pp_prev2 = compute_st_priors_for_one_season(data_root, season_prev2, "PP") if season_prev2 else None
    pk_prev1 = compute_st_priors_for_one_season(data_root, season_prev1, "PK") if season_prev1 else None
    pk_prev2 = compute_st_priors_for_one_season(data_root, season_prev2, "PK") if season_prev2 else None

    ev_table = ev_df[["playerId","Player","PosBucket","TeamStd_Primary","GP","TOI",
                "ES_xGF60_shr","ES_xGA60_shr","Off. Zone Start Pct"]].copy()
    ev_table["playerId"] = ev_table["playerId"].astype("Int64")

    if ev_prev1 is not None:
        ev_table = ev_table.merge(
            ev_prev1.rename(columns={
                "ES_xGF60_shr_prev":"ES_xGF60_shr_prev1",
                "ES_xGA60_shr_prev":"ES_xGA60_shr_prev1",
                "TOI_EV_prev":"TOI_EV_prev1"
                }), on="playerId", how="left")
    else:
        ev_table["ES_xGF60_shr_prev1"] = np.nan
        ev_table["ES_xGA60_shr_prev1"] = np.nan
        ev_table["TOI_EV_prev1"] = 0.0

    if ev_prev2 is not None:
        ev_table = ev_table.merge(
            ev_prev2.rename(columns={
                "ES_xGF60_shr_prev":"ES_xGF60_shr_prev2",
                "ES_xGA60_shr_prev":"ES_xGA60_shr_prev2",
                "TOI_EV_prev":"TOI_EV_prev2"
                }), on="playerId", how="left")
    else:
        ev_table["ES_xGF60_shr_prev2"] = np.nan
        ev_table["ES_xGA60_shr_prev2"] = np.nan
        ev_table["TOI_EV_prev2"] = 0.0

    gp_current = to_float(ev_table.get("GP", 0.0), 0.0).to_numpy()
    toi_current = to_float(ev_table.get("TOI", 0.0), 0.0).to_numpy()
    toi_per_game_ev = np.where(gp_current > 0.0, toi_current / gp_current, 0.0)
    gp_gate = float(pct_gp_gate)
    gp_fac = np.where(gp_current + gp_gate > 0.0, gp_current / (gp_current + gp_gate), 0.0)
    pos_for_k = ev_table["PosBucket"].astype(str).fillna("F").values
    k_ev_forward = k_shrink_es_by_pos.get("F", k_shrink["EV"]) / gp_gate
    k_ev_defense = k_shrink_es_by_pos.get("D", k_shrink["EV"]) / gp_gate
    k_ev_by_row = np.where(pos_for_k == "D", k_ev_defense, k_ev_forward)
    ev_current_weight = np.where(k_ev_by_row > 0.0, (toi_per_game_ev / k_ev_by_row) * gp_fac, 0.0)
    ev_current_weight = np.clip(ev_current_weight, 0.0, None)

    k_prev_ev = 240.0
    toi_ev_prev1 = to_float(ev_table.get("TOI_EV_prev1"), 0.0).to_numpy()
    toi_ev_prev2 = to_float(ev_table.get("TOI_EV_prev2"), 0.0).to_numpy()
    w_curr = ev_current_weight
    w_prev1 = np.where(toi_ev_prev1 > 0, toi_ev_prev1 / (toi_ev_prev1 + k_prev_ev), 0.0)
    w_prev2 = lambda_decay * np.where(toi_ev_prev2 > 0, toi_ev_prev2 / (toi_ev_prev2 + k_prev_ev), 0.0)
    denom = w_curr + w_prev1 + w_prev2
    denom = np.where(denom > 0.0, denom, 1.0)
    weight_curr, weight_prev1, weight_prev2 = w_curr/denom, w_prev1/denom, w_prev2/denom

    curr_xgf = pd.to_numeric(ev_table["ES_xGF60_shr"], errors="coerce").to_numpy()
    curr_xga = pd.to_numeric(ev_table["ES_xGA60_shr"], errors="coerce").to_numpy()
    curr_xgf = np.where(np.isnan(curr_xgf), repl_xgf_pos_vec, curr_xgf)
    curr_xga = np.where(np.isnan(curr_xga), repl_xga_pos_vec, curr_xga)
    p1_xgf = pd.to_numeric(ev_table["ES_xGF60_shr_prev1"], errors="coerce").to_numpy()
    p1_xga = pd.to_numeric(ev_table["ES_xGA60_shr_prev1"], errors="coerce").to_numpy()
    p1_xgf = np.where(np.isnan(p1_xgf), repl_xgf_pos_vec, p1_xgf)
    p1_xga = np.where(np.isnan(p1_xga), repl_xga_pos_vec, p1_xga)
    p2_xgf = pd.to_numeric(ev_table["ES_xGF60_shr_prev2"], errors="coerce").to_numpy()
    p2_xga = pd.to_numeric(ev_table["ES_xGA60_shr_prev2"], errors="coerce").to_numpy()
    p2_xgf = np.where(np.isnan(p2_xgf), repl_xgf_pos_vec, p2_xgf)
    p2_xga = np.where(np.isnan(p2_xga), repl_xga_pos_vec, p2_xga)

    ev_table["ES_xGF60_blend"] = weight_curr*curr_xgf + weight_prev1*p1_xgf + weight_prev2*p2_xgf
    ev_table["ES_xGA60_blend"] = weight_curr*curr_xga + weight_prev1*p1_xga + weight_prev2*p2_xga
    ev_table = ev_table.rename(columns={"TOI":"TOI_EV"})
    repl_xgf_pos_map = {"F": repl_pos["repl_xGF60_ES_pos"]["F"], "D": repl_pos["repl_xGF60_ES_pos"]["D"]}
    repl_xga_pos_map = {"F": repl_pos["repl_xGA60_ES_pos"]["F"], "D": repl_pos["repl_xGA60_ES_pos"]["D"]}
    repl_xgf_byrow = np.array([repl_xgf_pos_map.get(p, np.nan) for p in ev_table["PosBucket"].astype(str)], dtype=float)
    repl_xga_byrow = np.array([repl_xga_pos_map.get(p, np.nan) for p in ev_table["PosBucket"].astype(str)], dtype=float)
    ev_table["GAR_ES_off"] = (ev_table["ES_xGF60_blend"] - repl_xgf_byrow) * (ev_table["TOI_EV"] / 60.0)
    usage_fac = usage_factor_from_ozpct(ev_table["Off. Zone Start Pct"])
    delta_def_per60 = (repl_xga_byrow - ev_table["ES_xGA60_blend"])
    ev_table["GAR_ES_def"] = (delta_def_per60 * usage_fac) * (ev_table["TOI_EV"] / 60.0)

    pp_df = nst_pp.copy()
    pp_df["playerId"] = pp_df["playerId"].astype("Int64")
    w_pp = shrink_weight(pp_df["TOI"], k_shrink["PP"])
    pos_pp = pp_df["PosBucket"].fillna("F").astype(str).values
    repl_gf_pp_vec = np.array([repl_pos["repl_GF60_PP_pos"].get(p, np.nan) for p in pos_pp], dtype=float)
    pp_gf_raw = pd.to_numeric(pp_df.get("xGF_per60", pp_df.get("GF_per60")), errors="coerce").to_numpy()
    pp_gf_raw = np.where(np.isnan(pp_gf_raw), repl_gf_pp_vec, pp_gf_raw)
    pp_df["PP_GF60_shr"] = w_pp * pp_gf_raw + (1.0 - w_pp) * repl_gf_pp_vec
    pp_table = pp_df[["playerId","Player","PosBucket","TeamStd_Primary","GP","TOI","PP_GF60_shr"]].rename(columns={"TOI":"TOI_PP"})

    if pp_prev1 is not None:
        pp_table = pp_table.merge(pp_prev1.rename(columns={"PP_rate_prev":"PP_GF60_prev1", "TOI_PP_prev":"TOI_PP_prev1"}), on="playerId", how="left")
    else:
        pp_table["PP_GF60_prev1"] = np.nan
        pp_table["TOI_PP_prev1"] = 0.0
    if pp_prev2 is not None:
        pp_table = pp_table.merge(pp_prev2.rename(columns={"PP_rate_prev":"PP_GF60_prev2", "TOI_PP_prev":"TOI_PP_prev2"}), on="playerId", how="left")
    else:
        pp_table["PP_GF60_prev2"] = np.nan
        pp_table["TOI_PP_prev2"] = 0.0

    k_prev_pp = 120.0
    toi_pp_prev1 = to_float(pp_table.get("TOI_PP_prev1"), 0.0).to_numpy()
    toi_pp_prev2 = to_float(pp_table.get("TOI_PP_prev2"), 0.0).to_numpy()
    w_pp_prev1 = np.where(toi_pp_prev1 > 0, toi_pp_prev1 / (toi_pp_prev1 + k_prev_pp), 0.0)
    w_pp_prev2 = lambda_decay * np.where(toi_pp_prev2 > 0, toi_pp_prev2 / (toi_pp_prev2 + k_prev_pp), 0.0)
    pp_current_weight = shrink_weight(pp_table["TOI_PP"], k_shrink["PP"])
    denom_pp = pp_current_weight + w_pp_prev1 + w_pp_prev2
    denom_pp = np.where(denom_pp > 0, denom_pp, 1.0)
    weight_pp_curr, weight_pp_prev1, weight_pp_prev2 = pp_current_weight/denom_pp, w_pp_prev1/denom_pp, w_pp_prev2/denom_pp

    repl_gf_pp_byrow = np.array([repl_pos["repl_GF60_PP_pos"].get(p, np.nan) for p in pp_table["PosBucket"].astype(str)], dtype=float)
    pp_cur = pd.to_numeric(pp_table["PP_GF60_shr"],  errors="coerce").fillna(pd.Series(repl_gf_pp_byrow, index=pp_table.index)).to_numpy()
    pp_p1 = pd.to_numeric(pp_table["PP_GF60_prev1"], errors="coerce").fillna(pd.Series(repl_gf_pp_byrow, index=pp_table.index)).to_numpy()
    pp_p2 = pd.to_numeric(pp_table["PP_GF60_prev2"], errors="coerce").fillna(pd.Series(repl_gf_pp_byrow, index=pp_table.index)).to_numpy()
    pp_table["PP_GF60_blend"] = weight_pp_curr*pp_cur + weight_pp_prev1*pp_p1 + weight_pp_prev2*pp_p2
    pp_table["GAR_PP"] = (pp_table["PP_GF60_blend"] - repl_gf_pp_byrow) * (pp_table["TOI_PP"] / 60.0)

    pk_df = nst_pk.copy()
    pk_df["playerId"] = pk_df["playerId"].astype("Int64")
    w_pk = shrink_weight(pk_df["TOI"], k_shrink["PK"])
    pos_pk = pk_df["PosBucket"].fillna("F").astype(str).values
    repl_ga_pk_vec = np.array([repl_pos["repl_GA60_PK_pos"].get(p, np.nan) for p in pos_pk], dtype=float)
    pk_ga_raw = pd.to_numeric(pk_df.get("xGA_per60", pk_df.get("GA_per60")), errors="coerce").to_numpy()
    pk_ga_raw = np.where(np.isnan(pk_ga_raw), repl_ga_pk_vec, pk_ga_raw)
    pk_df["PK_GA60_shr"] = w_pk * pk_ga_raw + (1.0 - w_pk) * repl_ga_pk_vec
    pk_table = pk_df[["playerId","Player","PosBucket","TeamStd_Primary","GP","TOI","PK_GA60_shr"]].rename(columns={"TOI":"TOI_PK"})

    if pk_prev1 is not None:
        pk_table = pk_table.merge(pk_prev1.rename(columns={"PK_rate_prev":"PK_GA60_prev1", "TOI_PK_prev":"TOI_PK_prev1"}), on="playerId", how="left")
    else:
        pk_table["PK_GA60_prev1"] = np.nan
        pk_table["TOI_PK_prev1"] = 0.0
    if pk_prev2 is not None:
        pk_table = pk_table.merge(pk_prev2.rename(columns={"PK_rate_prev":"PK_GA60_prev2", "TOI_PK_prev":"TOI_PK_prev2"}), on="playerId", how="left")
    else:
        pk_table["PK_GA60_prev2"] = np.nan
        pk_table["TOI_PK_prev2"] = 0.0

    k_prev_pk = 600.0
    toi_pk_prev1 = to_float(pk_table.get("TOI_PK_prev1"), 0.0).to_numpy()
    toi_pk_prev2 = to_float(pk_table.get("TOI_PK_prev2"), 0.0).to_numpy()
    w_pk_prev1 = np.where(toi_pk_prev1 > 0, toi_pk_prev1 / (toi_pk_prev1 + k_prev_pk), 0.0)
    w_pk_prev2 = lambda_decay * np.where(toi_pk_prev2 > 0, toi_pk_prev2 / (toi_pk_prev2 + k_prev_pk), 0.0)
    pk_current_weight = shrink_weight(pk_table["TOI_PK"], k_shrink["PK"])
    denom_pk = pk_current_weight + w_pk_prev1 + w_pk_prev2
    denom_pk = np.where(denom_pk > 0, denom_pk, 1.0)
    weight_pk_curr, weight_pk_prev1, weight_pk_prev2 = pk_current_weight/denom_pk, w_pk_prev1/denom_pk, w_pk_prev2/denom_pk

    repl_ga_pk_byrow = np.array([repl_pos["repl_GA60_PK_pos"].get(p, np.nan) for p in pk_table["PosBucket"].astype(str)], dtype=float)
    pk_cur = pd.to_numeric(pk_table["PK_GA60_shr"], errors="coerce").fillna(pd.Series(repl_ga_pk_byrow, index=pk_table.index)).to_numpy()
    pk_p1 = pd.to_numeric(pk_table["PK_GA60_prev1"], errors="coerce").fillna(pd.Series(repl_ga_pk_byrow, index=pk_table.index)).to_numpy()
    pk_p2 = pd.to_numeric(pk_table["PK_GA60_prev2"], errors="coerce").fillna(pd.Series(repl_ga_pk_byrow, index=pk_table.index)).to_numpy()
    pk_table["PK_GA60_blend"] = weight_pk_curr*pk_cur + weight_pk_prev1*pk_p1 + weight_pk_prev2*pk_p2
    pk_table["GAR_PK"] = (repl_ga_pk_byrow - pk_table["PK_GA60_blend"]) * (pk_table["TOI_PK"] / 60.0)

    key = ["playerId","Player","PosBucket","TeamStd_Primary","GP"]
    ev_cols = key + ["TOI_EV", "ES_xGF60_shr", "ES_xGA60_shr", "ES_xGF60_blend", "ES_xGA60_blend", "GAR_ES_off", "GAR_ES_def", "Off. Zone Start Pct",]
    pp_cols = key + ["TOI_PP", "PP_GF60_shr", "GAR_PP"]
    pk_cols = key + ["TOI_PK", "PK_GA60_shr", "GAR_PK"]
    merged = ev_table[ev_cols].merge(pp_table[pp_cols], on=key, how="outer")
    merged = merged.merge(pk_table[pk_cols], on=key, how="outer")

    for c in ["TOI_EV","TOI_PP","TOI_PK","GAR_ES_off","GAR_ES_def","GAR_PP","GAR_PK"]:
        merged[c] = pd.to_numeric(merged.get(c, 0.0), errors="coerce").fillna(0.0)

    merged["TOI_all"] = merged["TOI_EV"] + merged["TOI_PP"] + merged["TOI_PK"]
    merged["GAR_off"] = merged["GAR_ES_off"] + merged["GAR_PP"]
    merged["GAR_def"] = merged["GAR_ES_def"] + merged["GAR_PK"]
    merged["GAR_total"]= merged["GAR_off"] + merged["GAR_def"]
    merged["WAR"] = np.where(goals_per_win > 0, merged["GAR_total"] / goals_per_win, np.nan)
    merged["SPAR"] = np.where(goals_per_standing_point > 0, merged["GAR_total"] / goals_per_standing_point, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        merged["TOI_perG"] = np.where(merged["GP"] > 0.0, merged["TOI_all"] / merged["GP"], 0.0)

    def choose_role_label(pos, toi_per_g):
        buckets = defender_role_buckets if pos == "D" else forward_role_buckets
        current = float(toi_per_g) if np.isfinite(toi_per_g) else 0.0
        label = buckets[-1][0]
        for name, t in buckets:
            label = name
            if current <= t:
                break
        return label

    roles = []
    for pos, gp_i, tpg in zip(merged["PosBucket"], merged["GP"], merged["TOI_perG"]):
        if float(gp_i) < float(min_games_for_role):
            roles.append("Other")
        else:
            roles.append(choose_role_label(pos, tpg))
    merged["RoleBucket"] = roles
    merged = add_percentiles_pos_role(merged)
    merged = merged.rename(columns={"playerId": "PlayerID"})
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.sort_values(["GAR_total", "WAR"], ascending=False).reset_index(drop=True)
    out = merged.copy()
    for c in final_output_columns:
        if c not in out.columns:
            out[c] = np.nan

    out = out[final_output_columns]
    out_path = out_dir / ("Skater_GAR_WAR_" + season + ".csv")
    out.to_csv(out_path, index=False)
    print(season + ": WAR/SPAR complete")

def main():
    for year in range(start_year, end_year + 1):
        build_one_season(year)
        
if __name__ == "__main__":
    main()
