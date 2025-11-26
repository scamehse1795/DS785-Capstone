# -*- coding: utf-8 -*-
"""
Contract Projections with KNN + EH blended comps. Also projects out skater value for the next 9 seasons (including 2025-2026)

NOTE: I had AI assistance with fixing up some of the broken sections of code, as well as balancing my EvolvingHockey blend into the
KNN-based comps. Mainly AI was used for script cleaning and ensuring different sections were not falling into inefficient loops

"""
# Import
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Config
script_dir = Path(__file__).resolve().parent
clean_dir = script_dir.parent / "Data" / "Clean Data"
target_year = 2025
season_str = f"{target_year}-{target_year+1}"
out_dir = clean_dir / season_str

contracts_master_path = clean_dir / "NHL_CONTRACTS_MASTER.csv"
age_curve_path = clean_dir / "AGE_CURVE_GAR_BY_AGE.csv"

k_neighbors = 100
k_min_per_term = 5
distance_scale = 0.85
kernel_power = 0.85
block_weights = {"USAGE":2.0, "IMPACT":1.2, "VALUE":1.5, "AGE":2.0, "DEMO":0.6, "CTX":1.2}

evolv_hockey_blend_base = 0.50
evolv_hockey_blend_strong = 0.80
evolv_hockey_blend_cap = 0.75
evolv_hockey_blend_alpha = 2.0

tail_decay_mult = 1.10
baseline_with_last = 0.60
baseline_with_M1 = 0.30
baseline_with_M2 = 0.10
linear_switch_GAR = 0.2

# Helpers
def to_float(x):
    try:
        return float(pd.to_numeric(x, errors="coerce"))
    except:
        return np.nan

def parse_height_inches(hstr):
    if pd.isna(hstr): 
        return np.nan
    s = str(hstr).strip()
    m = re.match(r"^\s*(\d+)\s*['-]\s*(\d+)", s) or re.match(r"^\s*(\d+)\s*'?(\d+)", s)
    return float(m.group(1))*12.0 + float(m.group(2)) if m else np.nan

def parse_start_year(val):
    if pd.isna(val): 
        return np.nan
    s = str(val).strip()
    m = re.match(r"^\s*(\d{4})", s)
    return int(m.group(1)) if m else to_float(s)

def clause_bucket(raw):
    if pd.isna(raw) or str(raw).strip()=="": 
        return "None"
    t = str(raw).upper()
    if "NMC" in t: 
        return "NMC"
    if "NTC" in t: 
        return "NTC"
    return "Other"

def status_group(signing_status):
    if pd.isna(signing_status): return "UFA"
    return "RFA" if "RFA" in str(signing_status).upper() else "UFA"

def normalize_level(level_raw, signing_age):
    lev = "" if pd.isna(level_raw) else str(level_raw).upper()
    if "ELC" in lev: 
        return "ELC"
    if "35" in lev: 
        return "35_PLUS"
    try:
        if float(signing_age) >= 35: 
            return "35_PLUS"
    except:
        pass
    return "STD"

def encode_shot(series):
    vals = series.astype(str).str.upper().str.strip()
    return vals.map({"R":1.0,"L":0.0}).where(vals.isin(["R","L"]), np.nan)

def season_str_from_year(y):
    return f"{y}-{y+1}"

def broad_pos_from_posbucket(pb):
    return "D" if str(pb).upper()=="D" else "F"

def map_role_tier(posbucket, rolebucket):
    p = str(posbucket).strip()
    r = str(rolebucket).strip()
    if p == "F":
        return {"1st Line":5,"Top-6":4,"Middle-6":3,"Bottom-6":2,"4th Line":1,"Other":0}.get(r, 0)
    if p == "D":
        return {"1st Pair":5,"Top-4":4,"Bottom-4":2,"3rd Pair":1,"Other":0}.get(r, 0)
    return 0

def age_band(a):
    if pd.isna(a): return "NA"
    x = float(a)
    if x <= 21: return "U22" # The "U" syntax is common in hockey for denoting age groupings of players UNDER a certain threshold (i.e. U22 means <=21, just in a neater way)
    if x <= 24: return "22-24"
    if x <= 27: return "25-27"
    if x <= 30: return "28-30"
    if x <= 33: return "31-33"
    if x <= 36: return "34-36"
    return "35+"

def kernel_similarity(dist, distance_scale, kernel_power):
    s = np.exp(-distance_scale * np.maximum(dist, 0.0))
    return s if kernel_power is None else np.power(s, kernel_power)

def ensure_engineered_columns(df):
    df = df.copy()
    if "Shot" in df.columns and "shot_num" not in df.columns:
        df["shot_num"] = encode_shot(df["Shot"])
    return df

def ensure_meta_cols(df, cols):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def load_contracts(path):
    df = pd.read_csv(path, low_memory=False)
    req = ["Skaters","Pos","Shot","W(lbs)","H(f)","Length","Level",
           "Cap Hit","Start Yr Cap %","Structure","Clauses","Start Year",
           "Signing Age","Signing Status","Expiry Year","Expiry Status",
           "Signing Agent","Signing GM","Signing Season","playerId"]
    for c in req:
        if c not in df.columns:
            raise RuntimeError(f"[FATAL] contracts missing column: {c}")

    out = df.copy()
    out["playerId"] = pd.to_numeric(out["playerId"], errors="coerce").astype("Int64")
    out["Signing_Age"] = pd.to_numeric(out["Signing Age"], errors="coerce")
    out["Start_Year"] = out["Start Year"].apply(parse_start_year).astype("Int64")
    out["Length"] = pd.to_numeric(out["Length"], errors="coerce").astype("Int64")
    out["Start_Yr_Cap_Pct"] = pd.to_numeric(out["Start Yr Cap %"], errors="coerce")
    out["Weight_lb"] = pd.to_numeric(out["W(lbs)"], errors="coerce")
    out["Height_in"] = out["H(f)"].apply(parse_height_inches)
    out["Shot"] = out["Shot"].astype(str).str.strip()
    out["Skaters"] = out["Skaters"].astype(str).str.strip()
    out["Pos"] = out["Pos"].astype(str).str.strip()
    out["Signing_Status"] = out["Signing Status"].astype(str).str.strip()
    out["Expiry_Status"] = out["Expiry Status"].astype(str).str.strip()
    out["Level_raw"] = out["Level"].astype(str).str.strip()
    out["Clauses_raw"] = out["Clauses"].astype(str).str.strip()
    out["Structure"] = out["Structure"].astype(str).str.strip()
    out["Signing_Season"] = out["Signing Season"].astype(str).str.strip()
    out["Signing_GM"] = out["Signing GM"].astype(str).str.strip()
    out["status_group"] = out["Signing_Status"].apply(status_group)
    out["level_clean"] = out.apply(lambda r: normalize_level(r["Level_raw"], r["Signing_Age"]), axis=1)
    out["clause_bucket"]= out["Clauses_raw"].apply(clause_bucket)
    out["age_band"] = out["Signing_Age"].apply(age_band)
    return out

def load_gar_for_year(clean_dir, stats_year):
    season = season_str_from_year(stats_year)
    fpath = clean_dir / season / f"Skater_GAR_WAR_{season}.csv"
    g = pd.read_csv(fpath, low_memory=False)

    keep = [
        "Player","PlayerID","PosBucket","RoleBucket",
        "GP","TOI_EV","TOI_PP","TOI_PK","TOI_all",
        "GAR_total","WAR","SPAR",
        "ES_xGF60_blend","ES_xGA60_blend",
        "ES_xGF60_shr","ES_xGA60_shr","Team"
        ]
    have = [c for c in keep if c in g.columns]
    g = g[have].copy()

    g["playerId"] = pd.to_numeric(g.get("PlayerID"), errors="coerce").astype("Int64")
    g["Stats_Year"] = int(stats_year)

    for c in ["GP","TOI_EV","TOI_PP","TOI_PK","TOI_all","GAR_total","WAR","SPAR",
              "ES_xGF60_blend","ES_xGA60_blend","ES_xGF60_shr","ES_xGA60_shr"]:
        if c in g.columns: 
            g[c] = pd.to_numeric(g[c], errors="coerce")

    toi_all = g["TOI_all"].replace(0, np.nan)
    g["PP_share"] = g["TOI_PP"] / toi_all
    g["PK_share"] = g["TOI_PK"] / toi_all
    g["TOI_total"] = g["TOI_all"] / 60.0

    if "ES_xGF60_blend" in g.columns and "ES_xGA60_blend" in g.columns:
        g["xGF_per60_ES"] = g["ES_xGF60_blend"] 
        g["xGA_per60_ES"] = g["ES_xGA60_blend"]
    elif "ES_xGF60_shr" in g.columns and "ES_xGA60_shr" in g.columns:
        g["xGF_per60_ES"] = g["ES_xGF60_shr"]
        g["xGA_per60_ES"] = g["ES_xGA60_shr"]

    g["role_tier"] = g.apply(lambda r: map_role_tier(r.get("PosBucket",""), r.get("RoleBucket","")), axis=1)
    if "Team" not in g.columns: 
        g["Team"] = ""
    g["TOI_all_value"] = g["TOI_all"]
    return g

def build_prev_season_bank(clean_dir, start_min, start_max):
    frames = []
    for year in range(start_min, start_max+1):
        season_df = load_gar_for_year(clean_dir, year)
        if len(season_df): 
            frames.append(season_df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

def load_age_curve(path):
    ac_raw = pd.read_csv(path, low_memory=False)
    cols_lower = {c.lower(): c for c in ac_raw.columns}
    def pick(*cands):
        for c in cands:
            if c in cols_lower: 
                return cols_lower[c]
        return None
    col_pos = pick("posbucket","pos_bucket","pos","position","posbucket ")
    col_age = pick("age")
    col_avg = pick("avg","mean","curve_mean","curve avg","curve")
    col_lo = pick("lower","lo","curve_lo","low")
    col_hi = pick("upper","hi","curve_hi","high")

    missing = [name for name, col in
               [("posBucket", col_pos), ("age", col_age),
                ("avg", col_avg), ("lower", col_lo), ("upper", col_hi)]
               if col is None]
    if missing:
        raise RuntimeError("[FATAL] age curve missing columns: " + ", ".join(missing))

    def norm_pos(v):
        if pd.isna(v): 
            return np.nan
        s = str(v).strip().upper()
        if s in {"F","FORWARD","FORWARDS","C","L","R","W","LW","RW"}: 
            return "F"
        if s in {"D","DEFENSE","DEFENCE","DEFENSEMAN","DEFENSEMEN"}: 
            return "D"
        return "F"

    ac = (ac_raw.rename(columns={
            col_pos:"posBucket", col_age:"age", col_avg:"avg", col_lo:"lower", col_hi:"upper"})
          [["posBucket","age","avg","lower","upper"]].copy())
    ac["posBucket"] = ac["posBucket"].apply(norm_pos)
    for c in ["age","avg","lower","upper"]: ac[c] = pd.to_numeric(ac[c], errors="coerce")
    return ac.dropna(subset=["posBucket","age","avg"]).sort_values(["posBucket","age"]).reset_index(drop=True)

def solid_tail_step(val_m, val_lo, val_hi, tail_rm, tail_rlo, tail_rhi):
    def tail_ratio(r):
        try:
            r = float(r)
        except Exception:
            return 1.0
        if r <= 0.0:
            return 1.0
        r_tail = r ** tail_decay_mult
        return max(0.3, min(0.95, r_tail))

    rm = tail_ratio(tail_rm)
    rlo = tail_ratio(tail_rlo)
    rhi = tail_ratio(tail_rhi)
    new_m = val_m * rm
    new_lo = val_lo * rlo
    new_hi = val_hi * rhi

    return new_m, new_lo, new_hi

def three_year_gar_baseline(g1, g2, g3):
    vals, wts = [], []
    for w, g in [(baseline_with_last, g1), (baseline_with_M1, g2), (baseline_with_M2, g3)]:
        if not pd.isna(g):
            vals.append(float(g))
            wts.append(float(w))
    if not wts:
        return np.nan
    return float(np.average(vals, weights=wts))

def build_ratio_tables(ac):
    tables = {}
    for pb, g in ac.groupby("posBucket"):
        gg = g.sort_values("age").reset_index(drop=True)
        ages = gg["age"].astype(int).tolist()
        m = gg["avg"].astype(float).tolist()
        lo = gg["lower"].astype(float).tolist()
        hi = gg["upper"].astype(float).tolist()
        r_m = [m[i+1] / m[i] for i in range(len(m) - 1)]
        r_lo = [lo[i+1] / lo[i] for i in range(len(lo) - 1)]
        r_hi = [hi[i+1] / hi[i] for i in range(len(hi) - 1)]

        if len(r_m):
            tail_rm = r_m[-1]
        else:
            tail_rm = 1.0
        if len(r_lo):
            tail_rlo = r_lo[-1]
        else:
            tail_rlo = tail_rm
        if len(r_hi):
            tail_rhi = r_hi[-1]
        else:
            tail_rhi = tail_rm

        tables[str(pb)] = {
            "ages": ages,
            "m": m,
            "lo": lo,
            "hi": hi,
            "r_m": r_m,
            "r_lo": r_lo,
            "r_hi": r_hi,
            "max_age": max(ages),
            "tail_rm": tail_rm,
            "tail_rlo": tail_rlo,
            "tail_rhi": tail_rhi,
            }
    return tables

def project_gar_series(posbucket, age0, base_t, ratio_tbl, horizon=9):
    tb = ratio_tbl.get(str(posbucket), None)
    if tb is None or pd.isna(base_t):
        return {}, {}

    base_t = float(base_t)

    ages = tb["ages"]
    r_m = tb["r_m"]
    r_lo = tb.get("r_lo", [])
    r_hi = tb.get("r_hi", [])
    tail_rm = tb["tail_rm"]
    tail_rlo = tb["tail_rlo"]
    tail_rhi = tb["tail_rhi"]

    out, out_lo, out_hi = {}, {}, {}
    cur_m = base_t
    cur_lo = base_t
    cur_hi = base_t
    a_now = int(round(age0))

    mode = "curve" # "curve" = use age ratios; "linear" = subtractive
    linear_step = 0.0

    def clean_ratio(r):
        try:
            r = float(r)
        except Exception:
            return 1.0
        if r <= 0.0:
            return 1.0
        # Clamp to avoid wild jumps even if the curve is noisy
        return min(1.15, max(0.6, r))

    for k in range(1, horizon + 1):
        a_from = a_now
        prev_m, prev_lo, prev_hi = cur_m, cur_lo, cur_hi

        # If in curve mode and GAR has dropped below the linear 
        # switch threshold, switch to linear decay.
        if mode == "curve" and prev_m < linear_switch_GAR:
            remaining_steps = horizon - (k - 1)
            if remaining_steps <= 0:
                remaining_steps = 1 # avoid divide-by-zero
            linear_step = prev_m / remaining_steps
            mode = "linear"

        if mode == "linear":
            # Subtractive decay: same step every year, goes to 0 at horizon
            cur_m = prev_m - linear_step
            cur_lo = prev_lo - linear_step
            cur_hi = prev_hi - linear_step
        else:
            # Age-curve multiplicative step
            if a_from in ages:
                idx = ages.index(a_from)
                if idx < len(r_m):
                    ratio_m = clean_ratio(r_m[idx])
                    ratio_lo = clean_ratio(r_lo[idx]) if idx < len(r_lo) else ratio_m
                    ratio_hi = clean_ratio(r_hi[idx]) if idx < len(r_hi) else ratio_m
                    cur_m = prev_m * ratio_m
                    cur_lo = prev_lo * ratio_lo
                    cur_hi = prev_hi * ratio_hi
                else:
                    cur_m, cur_lo, cur_hi = solid_tail_step(
                        prev_m, prev_lo, prev_hi,
                        tail_rm, tail_rlo, tail_rhi,
                        )
            else:
                cur_m, cur_lo, cur_hi = solid_tail_step(
                    prev_m, prev_lo, prev_hi,
                    tail_rm, tail_rlo, tail_rhi,
                    )

        out[f"proj_age_year{k}"] = a_from + 1
        out[f"proj_GAR_total_year{k}"] = cur_m
        out_lo[f"proj_GAR_total_low_year{k}"] = cur_lo
        out_hi[f"proj_GAR_total_high_year{k}"] = cur_hi
        a_now = a_from + 1

    out.update(out_lo)
    out.update(out_hi)
    return out, {"age0": int(round(age0)), "base_GAR3": base_t}

def cap_era_label_from_year(y):
    y = int(y)
    if 2015 <= y <= 2020: 
        return "Pre-COVID"
    if 2021 <= y <= 2023: 
        return "COVID Plateau"
    if y == 2024: 
        return "Lift"
    if 2025 <= y <= 2027: 
        return "Post-COVID Jump"
    if y >= 2028: 
        return "Future"
    return "Unknown"

def build_training_master(contracts_all, gar_bank):
    contracts_all = contracts_all.copy()
    contracts_all["cap_era"] = contracts_all["Start_Year"].apply(cap_era_label_from_year)
    contracts_all = contracts_all.loc[contracts_all["level_clean"] != "ELC"].copy()
    rows = []
    last_demo = (contracts_all.sort_values(["playerId", "Start_Year"]).groupby("playerId")
        .tail(1)[["playerId", "Shot", "Weight_lb", "Height_in", "Signing_Age"]]).set_index("playerId")

    for row in contracts_all.itertuples(index=False):
        pid, syear = row.playerId, row.Start_Year
        if pd.isna(pid) or pd.isna(syear): continue
        prev = gar_bank.loc[(gar_bank["playerId"]==pid) & (gar_bank["Stats_Year"]==(int(syear)-1))]
        if len(prev)==0: continue
        p2 = prev.sort_values("TOI_all", ascending=False).head(1).iloc[0]

        shot_val = row.Shot
        wt_val = row.Weight_lb
        ht_val = row.Height_in
        sa_val = row.Signing_Age

        if pid in last_demo.index:
            dd = last_demo.loc[pid]
            if pd.isna(wt_val):
                wt_val = dd.get("Weight_lb", np.nan)
            if pd.isna(ht_val):
                ht_val = dd.get("Height_in", np.nan)
            if str(shot_val).strip() == "":
                shot_val = dd.get("Shot", "")
            if pd.isna(sa_val):
                sa_val = dd.get("Signing_Age", np.nan)

        rows.append({
            "playerId": pid, "PlayerName": row.Skaters, "Pos": row.Pos,
            "Shot": shot_val, "Weight_lb": wt_val, "Height_in": ht_val,
            "Signing_Age": sa_val,
            "Start_Year": syear, "Length": row.Length,
            "Start_Yr_Cap_Pct": row.Start_Yr_Cap_Pct, "Signing_Status": row.Signing_Status,
            "Expiry_Status": row.Expiry_Status, "Structure": row.Structure,
            "level_clean": row.level_clean, "clause_bucket": row.clause_bucket,
            "Clauses_raw": row.Clauses_raw, "status_group": row.status_group,
            "age_band": row.age_band, "Signing_GM": row.Signing_GM,
            "cap_era": getattr(row, "cap_era", cap_era_label_from_year(syear)),
            "Stats_Year": int(p2["Stats_Year"]), "PosBucket": p2.get("PosBucket",""),
            "RoleBucket": p2.get("RoleBucket",""), "role_tier": p2.get("role_tier", np.nan),
            "TOI_EV": p2.get("TOI_EV", np.nan), "TOI_PP": p2.get("TOI_PP", np.nan),
            "TOI_PK": p2.get("TOI_PK", np.nan), "TOI_all_value": p2.get("TOI_all", np.nan),
            "TOI_total": p2.get("TOI_total", np.nan), "PP_share": p2.get("PP_share", np.nan),
            "PK_share": p2.get("PK_share", np.nan), "xGF_per60_ES": p2.get("xGF_per60_ES", np.nan),
            "xGA_per60_ES": p2.get("xGA_per60_ES", np.nan), "GAR_total": p2.get("GAR_total", np.nan),
            "WAR": p2.get("WAR", np.nan), "SPAR": p2.get("SPAR", np.nan), "GP": p2.get("GP", np.nan)
            })

    tm = pd.DataFrame(rows)
    if len(tm):
        tm = (tm.sort_values(["playerId","Start_Year","TOI_all_value"], ascending=[True, True, False])
                .drop_duplicates(subset=["playerId","Start_Year"], keep="first").copy())
    return tm

def attach_gar_projections(df_rows, gar_bank, ratio_tables, horizon=9):
    g_by_pid = gar_bank.sort_values(["playerId","Stats_Year"]).groupby("playerId")
    out_rows = []

    for row in df_rows.itertuples(index=False):
        pid, syear, signing_age = row.playerId, int(row.Start_Year), row.Signing_Age
        stats_year = syear - 1
        if pd.isna(signing_age) or pd.isna(pid):
            out_rows.append({"playerId": pid, "Start_Year": syear})
            continue

        base_t = np.nan
        pos_base = "F"
        if pid in g_by_pid.groups:
            gar_grp = g_by_pid.get_group(pid)
            gar_grp = gar_grp[gar_grp["Stats_Year"] <= stats_year]
            if len(gar_grp):
                last3 = gar_grp.sort_values("Stats_Year").tail(3)
                vals = last3["GAR_total"].astype(float).tolist()
                toiv = last3["TOI_all"].astype(float).tolist() if "TOI_all" in last3.columns else [np.nan]*len(vals)
                while len(vals) < 3: 
                    vals.insert(0, np.nan)
                    toiv.insert(0, np.nan)
                base_t = three_year_gar_baseline(vals[-1], vals[-2], vals[-3])
                pos_base = last3.iloc[-1].get("PosBucket","F")

        if pd.isna(base_t):
            out_rows.append({"playerId": pid, "Start_Year": syear})
            continue

        proj, _ = project_gar_series(broad_pos_from_posbucket(pos_base), int(round(signing_age)), float(base_t), ratio_tables, horizon=horizon)
        row = {"playerId": pid, "Start_Year": syear, "baseline_Tstar": base_t}
        row.update(proj)
        out_rows.append(row)

    proj_df = pd.DataFrame(out_rows)
    if len(proj_df):
        for t in range(1, 9):
            cols = [f"proj_GAR_total_year{k}" for k in range(1, t+1) if f"proj_GAR_total_year{k}" in proj_df.columns]
            proj_df[f"ProjValue_term{t}"] = proj_df[cols].sum(axis=1, skipna=True) if cols else np.nan
    return df_rows.merge(proj_df, on=["playerId","Start_Year"], how="left")

def add_context_onehots(train_df, target_df):
    target_df = target_df.copy()
    train_df = train_df.copy()
    for col, pref in [("status_group","sg"), ("level_clean","lvl"),
                      ("clause_bucket","cl"), ("cap_era","era"), ("age_band","ab"),
                      ("PosBucket","pos"), ("RoleBucket","role")]:
        if col not in train_df.columns: continue
        cats = train_df[col].fillna("").astype(str).unique().tolist()
        for c in cats:
            name = f"{pref}_{c}"
            train_df[name] = (train_df[col].astype(str) == c).astype(float)
            target_df[name] = (target_df[col].astype(str) == c).astype(float) if col in target_df.columns else 0.0
    return train_df, target_df

def get_block_defs(target_df):
    block_defs = {
        "USAGE": ["TOI_all_value","TOI_total","PP_share","PK_share","TOI_EV","TOI_PP","TOI_PK"],
        "IMPACT": ["xGF_per60_ES","xGA_per60_ES"],
        "VALUE": ["GAR_total","WAR","SPAR","baseline_Tstar","ProjValue_term3","ProjValue_term5"],
        "AGE": ["Signing_Age"],
        "DEMO": ["Weight_lb","Height_in","shot_num"],
        "CTX": [c for c in target_df.columns if c.startswith(("sg_","lvl_","cl_","era_","ab_","pos_","role_"))]
        }
    return block_defs

def fit_block_scalers_and_covs(train_df, block_cols):
    train_df = ensure_engineered_columns(train_df.copy())
    scalers, inv_covs, used_cols = {}, {}, {}

    for bname, cols in block_cols.items():
        use = [c for c in cols if c in train_df.columns]
        used_cols[bname] = use
        if len(use) == 0:
            scalers[bname] = None 
            inv_covs[bname] = None
            continue
        sub = train_df[use].astype(float) 
        mask = ~sub.isna().any(axis=1) 
        X = sub[mask].to_numpy()
        if X.shape[0] == 0: 
            scalers[bname] = None
            inv_covs[bname] = None
            continue
        sc = StandardScaler() 
        sc.fit(X) 
        scalers[bname] = sc

        if bname != "CTX" and X.shape[0] >= 3 and X.shape[1] >= 2:
            Z = sc.transform(X) 
            cov = np.cov(Z, rowvar=False)
            cov = cov + 1e-6 * np.eye(cov.shape[0])
            try: 
                inv_covs[bname] = np.linalg.inv(cov)
            except Exception: 
                inv_covs[bname] = None
        else:
            inv_covs[bname] = None
    return scalers, inv_covs, used_cols

def build_block_arrays(df, used_cols, scalers):
    df = ensure_engineered_columns(df.copy())
    arrays, valid = {}, {}
    for bname, use in used_cols.items():
        if len(use) == 0:
            arrays[bname] = np.zeros((len(df), 0))
            valid[bname] = np.zeros(len(df), dtype=bool) 
            continue
        sub = df[use].astype(float)
        mask = ~sub.isna().any(axis=1)
        arr = np.full((len(df), len(use)), np.nan)
        sc = scalers.get(bname, None)
        if sc is not None and mask.any(): 
            arr[mask, :] = sc.transform(sub[mask].to_numpy())
        arrays[bname] = np.where(np.isnan(arr), 0.0, arr) 
        valid[bname] = mask.to_numpy()
    return arrays, valid

def blockwise_distance(idx_t, idx_cands, targ_arrays, train_arrays, targ_valid, train_valid, inv_covs, block_weights):
    total = None
    for bname, w in block_weights.items():
        Xt = targ_arrays[bname][idx_t:idx_t+1, :]
        Xc = train_arrays[bname][idx_cands, :]
        vt = targ_valid[bname][idx_t]
        vc = train_valid[bname][idx_cands]
        if Xc.shape[1] == 0:
            d = np.zeros(len(idx_cands))
        else:
            usable = vt & vc
            if bname == "CTX" or inv_covs.get(bname, None) is None:
                diff = Xc - Xt
                d_all = np.sqrt(np.sum(diff * diff, axis=1))
            else:
                VI = inv_covs[bname]
                diff = Xc - Xt
                d_all = np.sqrt(np.sum(diff.dot(VI) * diff, axis=1))
            d = np.where(usable, d_all, 0.0)
        comp = w * d
        total = comp if total is None else total + comp
    return total

# KNN + EH Ridge Neighbors
def build_term_model_matrix(df):
    base = [
        "TOI_all_value","TOI_total","PP_share","PK_share","TOI_EV","TOI_PP","TOI_PK",
        "xGF_per60_ES","xGA_per60_ES",
        "GAR_total","WAR","SPAR","baseline_Tstar","ProjValue_term3","ProjValue_term5",
        "Signing_Age","Weight_lb","Height_in","shot_num"
        ]
    ctx = [c for c in df.columns if c.startswith(("sg_","lvl_","cl_","era_","ab_","pos_","role_"))]
    cols = [c for c in base if c in df.columns] + ctx
    X = df[cols].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]): 
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        else:
            s = X[c].astype(object) 
            X[c] = s.where(s.notna(), "")
    return X, cols

def fit_term_models(train_df, valid_terms, alpha=2.0):
    models, meta = {}, {}
    for t in valid_terms:
        sub = train_df.loc[(train_df["Length"].astype("Int64") == t) & train_df["Start_Yr_Cap_Pct"].notna()].copy()
        if not len(sub):
            models[t] = None
            meta[t] = {"n": 0, "cols": []}
            continue

        X, cols = build_term_model_matrix(sub)
        y = pd.to_numeric(sub["Start_Yr_Cap_Pct"], errors="coerce").fillna(0.0).values
        mdl = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
        mdl.fit(X, y)
        models[t] = mdl
        meta[t] = {"n": int(len(sub)), "cols": cols}

    return models, meta


def predict_term_model_cap_pct(row_df, t, model, meta):
    if model is None or not meta.get("cols"): 
        return np.nan
    cols = meta["cols"]
    X = row_df.reindex(columns=cols, fill_value=0.0).copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]): 
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        else:
            s = X[c].astype(object)
            X[c] = s.where(s.notna(), "")
    try:
        return float(model.predict(X)[0])
    except Exception:
        return np.nan

def build_candidates(train_df, row, k):
    same_pos = (train_df["PosBucket"].astype(str) == str(row.get("PosBucket","")))
    idx = np.where(same_pos.values)[0]
    if idx.size < k: 
        idx = np.arange(len(train_df))
    return idx

def compute_knn_and_terms(train_df, target_df, block_weights, k_neighbors, distance_scale, kernel_power, kmin_per_term,
                          term_models, term_meta, eh_blend_base, eh_blend_strong, eh_blend_cap):
    valid_terms = list(range(1, 9))

    tr, tg = add_context_onehots(train_df, target_df)
    block_defs = get_block_defs(pd.concat([tr, tg], ignore_index=True, sort=False))
    scalers, inv_covs, used_cols = fit_block_scalers_and_covs(tr, block_defs)
    train_blk, train_valid = build_block_arrays(tr, used_cols, scalers)
    targ_blk, targ_valid = build_block_arrays(tg, used_cols, scalers)
    out_rows, comps_rows = [], []

    def block_meta_for_target(idx_t):
        info = {}
        for b in ["USAGE","IMPACT","VALUE","AGE","DEMO","CTX"]:
            has_b = bool(targ_valid[b][idx_t])
            ncols = int(len(used_cols.get(b, [])))
            info[f"block_has_{b}"] = int(has_b)
            info[f"block_cols_{b}"] = ncols
            info[f"block_w_{b}"] = float(block_weights.get(b, 0.0))
        return info

    for idx_t, row in tg.reset_index(drop=True).iterrows():
        pid, sY = row.get("playerId", np.nan), row.get("Start_Year", np.nan)
        idx_pool = build_candidates(tr, row, k_neighbors)
        d = blockwise_distance(idx_t, idx_pool, targ_blk, train_blk, targ_valid, train_valid, inv_covs, block_weights)
        take = min(k_neighbors, len(idx_pool))
        order = np.argsort(d)[:take]
        top_idx, d_top = idx_pool[order], d[order]
        sim_kernel = kernel_similarity(d_top, distance_scale, kernel_power)
        ssum = float(sim_kernel.sum())
        if ssum > 0:
            w_norm = sim_kernel / ssum
        else:
            w_norm = np.ones_like(sim_kernel) / max(len(sim_kernel), 1)

        if len(d_top) == 0:
            sim_pct = np.zeros_like(d_top, dtype=float)
        else:
            d_top_safe = np.where(np.isfinite(d_top), d_top, np.nan)
            if np.all(np.isnan(d_top_safe)):
                sim_pct = np.zeros_like(d_top_safe, dtype=float)
            else:
                d_max = float(np.nanmax(d_top_safe))
                if d_max <= 0.0:
                    sim_pct = np.full_like(d_top_safe, 100.0, dtype=float)
                else:
                    raw = 1.0 - (d_top_safe / d_max)
                    raw = np.clip(raw, 0.0, 1.0)
                    sim_pct = 100.0 * raw
                    sim_pct = np.round(sim_pct).astype(float)

        neigh = tr.iloc[top_idx].copy().reset_index(drop=True)
        neigh["knn_dist"] = d_top
        neigh["knn_sim"] = sim_kernel
        neigh["knn_weight"] = w_norm
        neigh["knn_sim_pct"] = sim_pct
        cap_knn, iqr_knn, n_term, eff_n_term, cap_eh, w_model_term, cap_blend = {}, {}, {}, {}, {}, {}, {}
        row_X, _ = build_term_model_matrix(row.to_frame().T)
        for t in valid_terms:
            # KNN stats
            mask_t = neigh["Length"].astype("Int64").eq(t).values
            n_t = int(mask_t.sum())
            n_term[t] = n_t
            if n_t == 0:
                cap_knn[t], iqr_knn[t], eff_n_term[t] = np.nan, np.nan, np.nan
            else:
                sub = neigh.loc[mask_t]
                vals = pd.to_numeric(sub["Start_Yr_Cap_Pct"], errors="coerce")
                wts = sub["knn_weight"].to_numpy()
                if vals.notna().sum() == 0:
                    cap_knn[t], iqr_knn[t], eff_n_term[t] = np.nan, np.nan, np.nan
                else:
                    cap_knn[t] = float(np.nansum(vals.to_numpy()*wts) / max(np.nansum(wts), 1e-12))
                    vclean = vals.dropna().to_numpy()
                    iqr_knn[t] = float(np.quantile(vclean, 0.75) - np.quantile(vclean, 0.25)) if vclean.size >= 4 else np.nan
                    wn = wts / max(wts.sum(), 1e-12)
                    eff_n_term[t] = float(1.0 / max((wn**2).sum(), 1e-12))

            # EH per-term ridge prediction
            cap_eh[t] = predict_term_model_cap_pct(row_X, t, term_models.get(t), term_meta.get(t, {}))

            # Blending EH and KNN
            if n_t == 0:
                w_model = eh_blend_strong
            elif n_t < kmin_per_term:
                w_model = eh_blend_base
            else:
                iq = iqr_knn[t]
                w_model = eh_blend_base if pd.isna(iq) else min(eh_blend_cap, eh_blend_base + 0.10*(iq/0.05))

            w_model_term[t] = w_model
            ck, ce = cap_knn[t], cap_eh[t]
            if pd.isna(ck) and pd.isna(ce): 
                cap_blend[t] = np.nan
            elif pd.isna(ck):
                cap_blend[t] = ce
            elif pd.isna(ce):
                cap_blend[t] = ck
            else:
                cap_blend[t] = float(w_model*ce + (1.0 - w_model)*ck)

        row_out = {
            "playerId": pid, "Start_Year": sY, "neighbor_count": int(len(neigh)),
            "PosBucket": row.get("PosBucket",""), "status_group": row.get("status_group",""),
            "cap_era": row.get("cap_era",""), "role_tier": row.get("role_tier", np.nan)
            }
        row_out.update(block_meta_for_target(idx_t))

        for t in valid_terms:
            row_out[f"capPct_len{t}"] = cap_blend[t]
            row_out[f"capPct_len{t}_knn"] = cap_knn[t]
            row_out[f"capPct_len{t}_eh"]  = cap_eh[t]
            row_out[f"capPct_len{t}_iqr"] = iqr_knn[t]
            row_out[f"n_len{t}"] = int(n_term[t])
            row_out[f"eff_n_len{t}"] = float(eff_n_term[t]) if not pd.isna(eff_n_term[t]) else np.nan
            row_out[f"w_model_len{t}"] = float(w_model_term[t])
        out_rows.append(row_out)

        # top-5 comps (for dashboard)
        order_by_w = np.argsort(neigh["knn_weight"].to_numpy())[::-1]
        seen_pids = set()
        rank = 1
        for ni in order_by_w:
            if rank > 5:
                break

            nr = neigh.iloc[ni]
            comp_pid = nr.get("playerId", np.nan)

            if pd.isna(comp_pid):
                continue

            try:
                if not pd.isna(pid) and int(comp_pid) == int(pid):
                    continue
            except Exception:
                continue

            try:
                key_pid = int(comp_pid)
            except Exception:
                key_pid = comp_pid

            if key_pid in seen_pids:
                continue

            seen_pids.add(key_pid)
            comps_rows.append({
                "target_playerId": pid,
                "comp_rank": rank,
                "comp_playerId": comp_pid,
                "comp_PlayerName": nr.get("PlayerName", ""),
                "comp_Pos": nr.get("Pos", ""),
                "comp_PosBucket": nr.get("PosBucket", ""),
                "comp_RoleBucket": nr.get("RoleBucket", ""),
                "comp_Start_Year": nr.get("Start_Year", np.nan),
                "comp_Length": nr.get("Length", np.nan),
                "comp_CapPct": nr.get("Start_Yr_Cap_Pct", np.nan),
                "comp_weight_norm": nr.get("knn_weight", np.nan),
                "comp_dist": nr.get("knn_dist", np.nan),
                "comp_sim_pct": nr.get("knn_sim_pct", np.nan),
                })
            rank += 1
    return pd.DataFrame(out_rows), pd.DataFrame(comps_rows)

# Evaluation
def evaluate_per_true_term(pred_df, target_df):
    join_keys = ["playerId", "Start_Year"]
    meta_cols = ["PosBucket", "status_group", "cap_era", "role_tier"]
    td = ensure_meta_cols(target_df, meta_cols)

    base = td[["playerId","Start_Year","Length","Start_Yr_Cap_Pct"] + meta_cols].copy()
    base["Length"] = pd.to_numeric(base["Length"], errors="coerce").astype("Int64")

    pred_df = pred_df.drop(columns=meta_cols, errors="ignore")
    df = base.merge(pred_df, on=join_keys, how="left")

    cap_err, pred_cap, n_true, effn_true, wmodel_true = [], [], [], [], []
    for i, L in enumerate(df["Length"].astype("Int64").tolist()):
        if pd.isna(L):
            cap_err.append(np.nan)
            pred_cap.append(np.nan)
            n_true.append(np.nan) 
            effn_true.append(np.nan)
            wmodel_true.append(np.nan) 
            continue
        cap_col = f"capPct_len{int(L)}"
        ncol, encol, wmcol = f"n_len{int(L)}", f"eff_n_len{int(L)}", f"w_model_len{int(L)}"

        if cap_col not in df.columns or pd.isna(df.at[i, cap_col]) or pd.isna(df.at[i, "Start_Yr_Cap_Pct"]):
            cap_err.append(np.nan)
            pred_cap.append(np.nan)
        else:
            cap_err.append(abs(float(df.at[i, cap_col]) - float(df.at[i, "Start_Yr_Cap_Pct"])))
            pred_cap.append(float(df.at[i, cap_col]))

        n_true.append(float(df.at[i, ncol]) if ncol in df.columns else np.nan)
        effn_true.append(float(df.at[i, encol]) if encol in df.columns else np.nan)
        wmodel_true.append(float(df.at[i, wmcol]) if wmcol in df.columns else np.nan)

    df["abs_err_capPct"] = cap_err
    df["pred_capPct_at_true_term"] = pred_cap
    df["n_neighbors_true_term"] = n_true
    df["eff_n_true_term"] = effn_true
    df["w_model_true_term"] = wmodel_true

    def get_mean(x):
        vals = pd.to_numeric(x, errors="coerce")
        return float(vals.mean(skipna=True)) if vals.notna().any() else np.nan

    def get_median(x):
        vals = pd.to_numeric(x, errors="coerce")
        return float(vals.median(skipna=True)) if vals.notna().any() else np.nan

    rows = []
    for t in range(1, 9):
        mask = df["Length"].astype("Int64").eq(t)
        if not mask.any(): 
            continue
        sub = df.loc[mask]
        rows.append({
            "term": t,
            "n_eval": int(mask.sum()),
            "MAE_cap_pct": get_mean(sub["abs_err_capPct"]),
            "median_abs_error": get_median(sub["abs_err_capPct"])
            })
    by_term = pd.DataFrame(rows)
    return df, by_term

def error_breakdowns(expiring_eval_df):
    dims = ["PosBucket", "status_group", "cap_era", "role_tier"]
    df = ensure_meta_cols(expiring_eval_df, dims).copy()
    if "abs_err_capPct" not in df.columns:
        df["abs_err_capPct"] = np.nan

    parts = []
    for d in dims:
        cols = [c for c in [d, "abs_err_capPct"] if c in df.columns]
        if d not in cols:
            continue
        g = (df[cols].groupby(d, dropna=False, observed=True)["abs_err_capPct"].agg(n="count",
                 MAE_cap_pct=lambda x: pd.to_numeric(x, errors="coerce").mean(skipna=True),
                 median_abs_error=lambda x: pd.to_numeric(x, errors="coerce").median(skipna=True))
            .reset_index())
        g.insert(0, "group_by", d)
        parts.append(g)

    if parts:
        return pd.concat(parts, ignore_index=True, sort=False)
    return pd.DataFrame(columns=["group_by", "n", "MAE_cap_pct", "median_abs_error"])

def widen_top5_comps(comps_df):
    if comps_df is None or len(comps_df) == 0:
        return pd.DataFrame(columns=["playerId"])

    gdf = comps_df.copy()
    gdf = gdf.sort_values(["target_playerId", "comp_rank", "comp_weight_norm"], ascending=[True, True, False])
    out_rows = []
    for pid, grp in gdf.groupby("target_playerId", dropna=False):
        row_dict = {"playerId": pid}
        grp = grp.drop_duplicates(subset=["comp_playerId"], keep="first")
        grp = grp.head(5).reset_index(drop=True)

        for idx, row in enumerate(grp.itertuples(index=False), start=1):
            row_dict[f"comp{idx}_playerId"] = getattr(row, "comp_playerId", np.nan)
            row_dict[f"comp{idx}_PlayerName"] = getattr(row, "comp_PlayerName", "")
            row_dict[f"comp{idx}_Pos"] = getattr(row, "comp_Pos", "")
            row_dict[f"comp{idx}_PosBucket"] = getattr(row, "comp_PosBucket", "")
            row_dict[f"comp{idx}_RoleBucket"] = getattr(row, "comp_RoleBucket", "")
            row_dict[f"comp{idx}_Start_Year"] = getattr(row, "comp_Start_Year", np.nan)
            row_dict[f"comp{idx}_Length"] = getattr(row, "comp_Length", np.nan)
            row_dict[f"comp{idx}_CapPct"] = getattr(row, "comp_CapPct", np.nan)
            row_dict[f"comp{idx}_weight_norm"] = getattr(row, "comp_weight_norm", np.nan)
            row_dict[f"comp{idx}_dist"] = getattr(row, "comp_dist", np.nan)
            row_dict[f"comp{idx}_sim_pct"] = getattr(row, "comp_sim_pct", np.nan)

        out_rows.append(row_dict)

    wide = pd.DataFrame(out_rows)
    if "playerId" not in wide.columns:
        wide["playerId"] = pd.Series(dtype="Int64")
    return wide

def write_results(out_dir, season_str, expiring_df, whatif_df, pred_df, comps_df):
    union_base = pd.concat([expiring_df, whatif_df], ignore_index=True, sort=False)
    join_keys = ["playerId","Start_Year"]
    union_pred = union_base.merge(pred_df, on=join_keys, how="left")

    keep_base = [
        "playerId","PlayerName","Pos","Shot","Weight_lb","Height_in","Signing_Age",
        "Signing_Status","Expiry_Status","Structure",
        "status_group","level_clean","clause_bucket","Clauses_raw",
        "age_band","Signing_GM",
        "PosBucket","RoleBucket","role_tier","cap_era",
        "Start_Year","Length","Start_Yr_Cap_Pct"
        ]
    proj_cols = []
    for k in range(1,10):
        proj_cols += [f"proj_age_year{k}", f"proj_GAR_total_year{k}", f"proj_GAR_total_low_year{k}", f"proj_GAR_total_high_year{k}"]
    proj_cols += [f"ProjValue_term{t}" for t in range(1,9)]
    per_term_cols = []
    for t in range(1,9):
        per_term_cols += [f"capPct_len{t}", f"capPct_len{t}_knn", f"capPct_len{t}_eh", f"capPct_len{t}_iqr", f"n_len{t}", f"eff_n_len{t}", f"w_model_len{t}"]

    keep = [c for c in keep_base + per_term_cols + proj_cols if c in union_pred.columns]
    res = union_pred[keep].copy()
    comps_wide = widen_top5_comps(comps_df)
    res = res.merge(comps_wide, on="playerId", how="left")
    res = (res.sort_values(["playerId","Start_Year"], ascending=[True, True])
              .drop_duplicates(subset=["playerId","Start_Year"], keep="first").copy())

    out_results = out_dir / f"contract_results_{season_str}.csv"
    res.to_csv(out_results, index=False)
    return res

def write_model_diagnostics(out_dir, season_str, by_term_df):
    cols = ["term","n_eval","MAE_cap_pct","median_abs_error"]
    bt = by_term_df.copy()
    for c in cols:
        if c not in bt.columns: 
            bt[c] = np.nan
    bt = bt.sort_values("term")
    out_file = out_dir / f"model_diagnostics_{season_str}.csv"
    bt[cols].to_csv(out_file, index=False)

# Main
def main():
    contracts_all = load_contracts(contracts_master_path)
    contracts_all["cap_era"] = contracts_all["Start_Year"].apply(cap_era_label_from_year)
    gar_bank = build_prev_season_bank(clean_dir, 2015, target_year-1)
    contracts_cut = contracts_all.loc[contracts_all["Start_Year"] <= target_year].copy()
    training_master = build_training_master(contracts_cut, gar_bank)
    age_curve = load_age_curve(age_curve_path)
    age_ratio_tables = build_ratio_tables(age_curve)
    training_master = attach_gar_projections(training_master, gar_bank, age_ratio_tables, horizon=9)

    train_df = training_master.loc[training_master["Start_Year"] < target_year].copy()
    expiring_df = training_master.loc[training_master["Start_Year"] == target_year].copy()

    gar_prev = gar_bank.loc[gar_bank["Stats_Year"] == target_year-1].copy()
    base = (gar_prev.sort_values(["playerId","TOI_all"], ascending=[True, False]).drop_duplicates(subset=["playerId"]).copy())
    base["PlayerName"] = base["Player"]
    base["Pos"] = base["PosBucket"].map(lambda x: "D" if str(x).upper()=="D" else "F")

    cc = contracts_all.copy().sort_values(["playerId","Start_Year"])
    last = cc.groupby("playerId").tail(1)[["playerId","Signing_Age","Signing_Season",
                                           "Signing_Status","Level_raw","Clauses_raw",
                                           "clause_bucket","level_clean","Shot","Weight_lb","Height_in",
                                           "cap_era","age_band","Signing_GM"]].copy()
    def parse_season_start(ss):
        if pd.isna(ss): 
            return np.nan
        m = re.match(r"^\s*(\d{4})", str(ss))
        return int(m.group(1)) if m else np.nan
        
    last["Signing_Season_StartYr"] = last["Signing_Season"].apply(parse_season_start).astype("Int64")
    def extrap_age(row):
        a, y0 = row["Signing_Age"], row["Signing_Season_StartYr"]
        if pd.isna(a) or pd.isna(y0): 
            return np.nan
        return float(a) + float(target_year - int(y0))
    
    last["Signing_Age_now"] = last.apply(extrap_age, axis=1)
    wf = base.merge(last, on="playerId", how="left")
    wf["Signing_Age"] = wf["Signing_Age_now"]
    wf = wf.loc[wf["level_clean"].fillna("STD") != "ELC"].copy()
    wf["Start_Year"] = target_year
    wf["Length"] = np.nan
    wf["Start_Yr_Cap_Pct"] = np.nan
    wf["Signing_Status"] = wf["Signing_Status"].fillna("")
    wf["Expiry_Status"] = ""
    wf["Structure"] = ""
    wf["status_group"] = wf["Signing_Status"].apply(status_group)
    wf["clause_bucket"] = wf["clause_bucket"].fillna("None")
    wf["Clauses_raw"] = wf["Clauses_raw"].fillna("")
    wf["level_clean"] = wf["level_clean"].fillna("STD")
    wf["cap_era"] = wf["cap_era"].fillna(cap_era_label_from_year(target_year))
    wf["age_band"] = wf["Signing_Age"].apply(age_band)
    wf["Signing_GM"] = wf["Signing_GM"].fillna("")

    keep = [
        "playerId","PlayerName","Pos",
        "Signing_Age","Start_Year","Length","Start_Yr_Cap_Pct",
        "Signing_Status","Expiry_Status","Structure","level_clean","clause_bucket","Clauses_raw","status_group",
        "cap_era","age_band","Signing_GM",
        "Stats_Year","PosBucket","RoleBucket","role_tier",
        "TOI_EV","TOI_PP","TOI_PK","TOI_all","TOI_all_value","TOI_total",
        "PP_share","PK_share","xGF_per60_ES","xGA_per60_ES",
        "GAR_total","WAR","SPAR","GP","Shot","Weight_lb","Height_in"
        ]
    for c in keep:
        if c not in wf.columns: 
            wf[c] = np.nan
        
    whatif_df = wf[keep].copy()
    whatif_df = attach_gar_projections(whatif_df, gar_bank, age_ratio_tables, horizon=9)
    expiring_ids = set(expiring_df["playerId"].dropna().astype("Int64"))
    whatif_df = whatif_df.loc[~whatif_df["playerId"].astype("Int64").isin(expiring_ids)].copy()
    expiring_df["src"] = "E"
    whatif_df["src"] = "W"
    target_df = (pd.concat([expiring_df, whatif_df], ignore_index=True, sort=False).sort_values(["playerId","Start_Year","src"], ascending=[True, True, True])
                   .drop_duplicates(subset=["playerId","Start_Year"], keep="first").drop(columns="src").copy())

    K_opt = k_neighbors
    W_opt = block_weights
    dm_opt = distance_scale
    uq_opt = kernel_power
    eh_alpha_opt, eh_b_opt, eh_s_opt, eh_cap_opt = evolv_hockey_blend_alpha, evolv_hockey_blend_base, evolv_hockey_blend_strong, evolv_hockey_blend_cap

    term_models, term_meta = fit_term_models(train_df, list(range(1,9)), alpha=eh_alpha_opt)
    pred_df, comps_df = compute_knn_and_terms(train_df, target_df, W_opt, K_opt, dm_opt, uq_opt, k_min_per_term,
                                              term_models, term_meta, eh_blend_base=eh_b_opt, eh_blend_strong=eh_s_opt, eh_blend_cap=eh_cap_opt)

    base_targ = target_df.loc[target_df["Start_Year"].astype("Int64")==target_year].copy()
    eval_joined, by_term = evaluate_per_true_term(pred_df, base_targ)
    write_results(out_dir, season_str, expiring_df, whatif_df, pred_df, comps_df)
    write_model_diagnostics(out_dir, season_str, by_term)

    err_br = error_breakdowns(eval_joined)
    err_br_file = out_dir / f"model_error_breakdowns_{season_str}.csv"
    err_br.to_csv(err_br_file, index=False)
    print("Predictions Complete for 2025/2026 Season")

if __name__ == "__main__":
    main()
    
