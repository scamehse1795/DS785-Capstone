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

# K-nearest-neighbors config values
k_neighbors = 100
k_min_per_term = 5
distance_scale = 0.85
kernel_power = 0.85
block_weights = {"USAGE":2.0, "IMPACT":1.2, "VALUE":1.5, "AGE":2.0, "DEMO":0.6, "CTX":1.2}

# EvolvingHockey Ridge model blend weights
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

# Role tiers are based on industry language; maps role tiers to numeric values
def map_role_tier(posbucket, rolebucket):
    p = str(posbucket).strip()
    r = str(rolebucket).strip()
    if p == "F":
        return {"1st Line":5, "Top-6":4, "Middle-6":3, "Bottom-6":2, "4th Line":1, "Other":0}.get(r, 0)
    if p == "D":
        return {"1st Pair":5, "Top-4":4, "Bottom-4":2, "3rd Pair":1, "Other":0}.get(r, 0)
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
    similarity_raw = np.exp(-distance_scale * np.maximum(dist, 0.0))
    return similarity_raw if kernel_power is None else np.power(similarity_raw, kernel_power)

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

# Load contact data and make sure columns are standardized
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

# Load GAR values for given year
def load_gar_for_year(clean_dir, stats_year):
    season = season_str_from_year(stats_year)
    fpath = clean_dir / season / f"Skater_GAR_WAR_{season}.csv"
    gar_df = pd.read_csv(fpath, low_memory=False)

    keep = [
        "Player","PlayerID","PosBucket","RoleBucket",
        "GP","TOI_EV","TOI_PP","TOI_PK","TOI_all",
        "GAR_total","WAR","SPAR",
        "ES_xGF60_blend","ES_xGA60_blend",
        "ES_xGF60_shr","ES_xGA60_shr","Team"
        ]
    have = [c for c in keep if c in gar_df.columns]
    gar_df = gar_df[have].copy()

    gar_df["playerId"] = pd.to_numeric(gar_df.get("PlayerID"), errors="coerce").astype("Int64")
    gar_df["Stats_Year"] = int(stats_year)

    for c in ["GP","TOI_EV","TOI_PP","TOI_PK","TOI_all","GAR_total","WAR","SPAR",
              "ES_xGF60_blend","ES_xGA60_blend","ES_xGF60_shr","ES_xGA60_shr"]:
        if c in gar_df.columns: 
            gar_df[c] = pd.to_numeric(gar_df[c], errors="coerce")

    toi_all = gar_df["TOI_all"].replace(0, np.nan)
    gar_df["PP_share"] = gar_df["TOI_PP"] / toi_all
    gar_df["PK_share"] = gar_df["TOI_PK"] / toi_all
    gar_df["TOI_total"] = gar_df["TOI_all"] / 60.0

    if "ES_xGF60_blend" in gar_df.columns and "ES_xGA60_blend" in gar_df.columns:
        gar_df["xGF_per60_ES"] = gar_df["ES_xGF60_blend"] 
        gar_df["xGA_per60_ES"] = gar_df["ES_xGA60_blend"]
    elif "ES_xGF60_shr" in gar_df.columns and "ES_xGA60_shr" in gar_df.columns:
        gar_df["xGF_per60_ES"] = gar_df["ES_xGF60_shr"]
        gar_df["xGA_per60_ES"] = gar_df["ES_xGA60_shr"]

    gar_df["role_tier"] = gar_df.apply(lambda r: map_role_tier(r.get("PosBucket",""), r.get("RoleBucket","")), axis=1)
    if "Team" not in gar_df.columns: 
        gar_df["Team"] = ""
    gar_df["TOI_all_value"] = gar_df["TOI_all"]
    return gar_df

# Loop over a range of seasons and stacks the per-year GAR files into a "bank" of GAR values
def build_prev_season_bank(clean_dir, start_min, start_max):
    frames = []
    for year in range(start_min, start_max+1):
        season_df = load_gar_for_year(clean_dir, year)
        if len(season_df): 
            frames.append(season_df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

# Read in age curve file and normalize labels/column names
def load_age_curve(path):
    age_curve_raw = pd.read_csv(path, low_memory=False)
    cols_lower = {c.lower(): c for c in age_curve_raw.columns}
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
        raise RuntimeError("Age curve missing columns: " + ", ".join(missing))

    def norm_pos(raw_pos):
        if pd.isna(raw_pos): 
            return np.nan
        pos_str = str(raw_pos).strip().upper()
        if pos_str in {"F","C","L","R","W","LW","RW"}: 
            return "F"
        if pos_str in {"D","LD", "RD"}: 
            return "D"
        return "F"

    age_curve = (age_curve_raw.rename(columns={
            col_pos:"posBucket", col_age:"age", col_avg:"avg", col_lo:"lower", col_hi:"upper"})
          [["posBucket","age","avg","lower","upper"]].copy())
    age_curve["posBucket"] = age_curve["posBucket"].apply(norm_pos)
    for c in ["age","avg","lower","upper"]: 
        age_curve[c] = pd.to_numeric(age_curve[c], errors="coerce")
    return age_curve.dropna(subset=["posBucket","age","avg"]).sort_values(["posBucket","age"]).reset_index(drop=True)

# Apply ratios after ages in curve to prevent extreme growth/decline in older ages
def solid_tail_step(current_mean, current_low, current_high, tail_ratio_mean, tail_ratio_low, tail_ratio_high):
    def tail_ratio(raw_ratio):
        try:
            raw_ratio = float(raw_ratio)
        except Exception:
            return 1.0
        if raw_ratio <= 0.0:
            return 1.0
        decay_ratio = raw_ratio ** tail_decay_mult
        return max(0.3, min(0.95, decay_ratio))

    ratio_mean_decay = tail_ratio(tail_ratio_mean)
    ratio_low_decay = tail_ratio(tail_ratio_low)
    ratio_high_decay = tail_ratio(tail_ratio_high)
    new_mean = current_mean * ratio_mean_decay
    new_low = current_low * ratio_low_decay
    new_high = current_high * ratio_high_decay

    return new_mean, new_low, new_high

# Builds a baseline GAR estimate per player by taking a weighted average of the last three seasons of GAR, weighting most recent years more heavily
def three_year_gar_baseline(gar_last, gar_minus1, gar_minus2):
    gar_values = [] 
    gar_weights = []
    for w, g in [(baseline_with_last, gar_last), (baseline_with_M1, gar_minus1), (baseline_with_M2, gar_minus2)]:
        if not pd.isna(g):
            gar_values.append(float(g))
            gar_weights.append(float(w))
    if not gar_weights:
        return np.nan
    return float(np.average(gar_values, weights=gar_weights))

# For each position, convert the age curve into year-over-year multiplicative ratios
def build_ratio_tables(age_curve):
    tables = {}
    for posbucket, group_df in age_curve.groupby("posBucket"):
        group_sorted = group_df.sort_values("age").reset_index(drop=True)
        ages = group_sorted["age"].astype(int).tolist()
        mean_gar = group_sorted["avg"].astype(float).tolist()
        lower_gar = group_sorted["lower"].astype(float).tolist()
        upper_gar = group_sorted["upper"].astype(float).tolist()
        ratios_mean = [mean_gar[i+1] / mean_gar[i] for i in range(len(mean_gar) - 1)]
        ratios_low = [lower_gar[i+1] / lower_gar[i] for i in range(len(lower_gar) - 1)]
        ratios_high = [upper_gar[i+1] / upper_gar[i] for i in range(len(upper_gar) - 1)]

        if len(ratios_mean):
            tail_mean_ratio = ratios_mean[-1]
        else:
            tail_mean_ratio = 1.0
        if len(ratios_low):
            tail_low_ratio = ratios_low[-1]
        else:
            tail_low_ratio = tail_mean_ratio
        if len(ratios_high):
            tail_high_ratio = ratios_high[-1]
        else:
            tail_high_ratio = tail_mean_ratio

        tables[str(posbucket)] = {
            "ages": ages,
            "mean_gar": mean_gar,
            "lower_gar": lower_gar,
            "upper_gar": upper_gar,
            "ratios_mean": ratios_mean,
            "ratios_low": ratios_low,
            "ratios_high": ratios_high,
            "max_age": max(ages),
            "tail_mean_ratio": tail_mean_ratio,
            "tail_low_ratio": tail_low_ratio,
            "tail_high_ratio": tail_high_ratio,
            }
    return tables

# Given a starting age and baseline GAR, uses the age-curve ratios (and a linear tail if GAR gets very small) to project total GAR for up to N future seasons (default is 9)
def project_gar_series(posbucket, starting_age, baseline_gar, position_ratio_tables, horizon=9):
    pos_ratio_table = position_ratio_tables.get(str(posbucket), None)
    if pos_ratio_table is None or pd.isna(baseline_gar):
        return {}, {}

    baseline_gar = float(baseline_gar)

    curve_ages = pos_ratio_table["curve_ages"]
    ratios_mean = pos_ratio_table["ratios_mean"]
    ratios_low = pos_ratio_table.get("ratios_low", [])
    ratios_high = pos_ratio_table.get("ratios_high", [])
    tail_mean_ratio = pos_ratio_table["tail_mean_ratio"]
    tail_low_ratio = pos_ratio_table["tail_low_ratio"]
    tail_high_ratio = pos_ratio_table["tail_high_ratio"]

    proj_mean_dict = {}
    proj_low_dict = {} 
    proj_high_dic = {}
    current_mean = baseline_gar
    current_low = baseline_gar
    current_high = baseline_gar
    current_age = int(round(starting_age))

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

    for year_offset in range(1, horizon + 1):
        age_this_year = current_age
        prev_mean = current_mean
        prev_low = current_low
        prev_high = current_high

        # If in curve mode and GAR has dropped below the linear switch threshold, switch to linear decay.
        if mode == "curve" and prev_mean < linear_switch_GAR:
            remaining_years = horizon - (year_offset - 1)
            if remaining_years <= 0:
                remaining_years = 1 # avoid divide-by-zero
            linear_step = prev_mean / remaining_years
            mode = "linear"

        if mode == "linear":
            # Subtractive decay: same step every year, goes to 0 at horizon
            current_mean = prev_mean - linear_step
            current_low = prev_low - linear_step
            current_high = prev_high - linear_step
        else:
            # Age-curve multiplicative step
            if age_this_year in curve_ages:
                idx = curve_ages.index(age_this_year)
                if idx < len(ratios_mean):
                    ratio_m = clean_ratio(ratios_mean[idx])
                    ratio_lo = clean_ratio(ratios_low[idx]) if idx < len(ratios_low) else ratio_m
                    ratio_hi = clean_ratio(ratios_high[idx]) if idx < len(ratios_high) else ratio_m
                    current_mean = prev_mean * ratio_m
                    current_low = prev_low * ratio_lo
                    current_high = prev_high * ratio_hi
                else:
                    current_mean, current_low, current_high = solid_tail_step(
                        prev_mean, prev_low, prev_high,
                        tail_mean_ratio, tail_low_ratio, tail_high_ratio,
                        )
            else:
                current_mean, current_low, current_high = solid_tail_step(
                    prev_mean, prev_low, prev_high,
                    tail_mean_ratio, tail_low_ratio, tail_high_ratio,
                    )

        proj_mean_dict[f"proj_age_year{year_offset}"] = age_this_year + 1
        proj_mean_dict[f"proj_GAR_total_year{year_offset}"] = current_mean
        proj_low_dict[f"proj_GAR_total_low_year{year_offset}"] = current_low
        proj_high_dic[f"proj_GAR_total_high_year{year_offset}"] = current_high
        current_age = age_this_year + 1

    proj_mean_dict.update(proj_low_dict)
    proj_mean_dict.update(proj_high_dic)
    return proj_mean_dict, {"starting_age": int(round(starting_age)), "base_GAR3": baseline_gar}

# Era labels for future contract learning on how contract values shift based on era
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
    # Tag each contract with a simple cap-era bucket for context and drop ELCs
    contracts_all = contracts_all.copy()
    contracts_all["cap_era"] = contracts_all["Start_Year"].apply(cap_era_label_from_year)
    contracts_all = contracts_all.loc[contracts_all["level_clean"] != "ELC"].copy()
    
    # Grab demographic row from latest contract for each player
    rows = []
    last_demo = (contracts_all.sort_values(["playerId", "Start_Year"]).groupby("playerId").tail(1)[["playerId", "Shot", "Weight_lb", "Height_in", "Signing_Age"]]).set_index("playerId")

    # Iterate contract rows and attach prior-year GAR/WAR and usage
    for row in contracts_all.itertuples(index=False):
        pid, syear = row.playerId, row.Start_Year
        # Skip rows without playerIds (players that signed contracts but have no NHL stats)
        if pd.isna(pid) or pd.isna(syear): 
            continue
        
        # Pull prior season GAR/WAR rows
        prev = gar_bank.loc[(gar_bank["playerId"]==pid) & (gar_bank["Stats_Year"]==(int(syear)-1))]
        if len(prev)==0: 
            continue
        p2 = prev.sort_values("TOI_all", ascending=False).head(1).iloc[0]
        
        # Grab contract demographics
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
                
        # Build the combined contract + prior-value row
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
        # Prefer higher TOI rows when deduplicating
        tm = (tm.sort_values(["playerId","Start_Year","TOI_all_value"], ascending=[True, True, False]).drop_duplicates(subset=["playerId","Start_Year"], keep="first").copy())
    return tm

# Attach GAR projections and 3-year baseline
def attach_gar_projections(df_rows, gar_bank, ratio_tables, horizon=9):
    g_by_pid = gar_bank.sort_values(["playerId","Stats_Year"]).groupby("playerId")
    out_rows = []

    for row in df_rows.itertuples(index=False):
        pid, syear, signing_age = row.playerId, int(row.Start_Year), row.Signing_Age
        stats_year = syear - 1
        
        # Can't project without signing age or playerId
        if pd.isna(signing_age) or pd.isna(pid):
            out_rows.append({"playerId": pid, "Start_Year": syear})
            continue

        base_t = np.nan # 3-year baseline
        pos_base = "F" # default to forwards if no position can be infered
        
        # Compute 3-year baseline for player (using only immediately preceeding seasons)
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
                # Weighted 3-year GAR baseline (heavier weight on the most recent season)
                base_t = three_year_gar_baseline(vals[-1], vals[-2], vals[-3])
                pos_base = last3.iloc[-1].get("PosBucket","F")

        if pd.isna(base_t):
            out_rows.append({"playerId": pid, "Start_Year": syear})
            continue
        
        # Project future GAR using age curve ratios and a linear tail if needed
        proj, _ = project_gar_series(broad_pos_from_posbucket(pos_base), int(round(signing_age)), float(base_t), ratio_tables, horizon=horizon)
        row = {"playerId": pid, "Start_Year": syear, "baseline_Tstar": base_t}
        row.update(proj)
        out_rows.append(row)

    proj_df = pd.DataFrame(out_rows)
    if len(proj_df):
        # For each possible term length, sum the projected GAR values
        for t in range(1, 9):
            cols = [f"proj_GAR_total_year{k}" for k in range(1, t+1) if f"proj_GAR_total_year{k}" in proj_df.columns]
            proj_df[f"ProjValue_term{t}"] = proj_df[cols].sum(axis=1, skipna=True) if cols else np.nan
    return df_rows.merge(proj_df, on=["playerId","Start_Year"], how="left")

# Add context columns (status_group, level, clauses, cap era, age band, position bucket, role bucket)
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

# Fit a StandardScaler on rows with complete data in each feature block, and compute a covariance matrix and its inverse (for Mahalanobis distance)
def fit_block_scalers_and_covs(train_df, block_cols):
    train_df = ensure_engineered_columns(train_df.copy())
    scalers = {}
    inv_covs = {}
    used_cols = {}

    for block_name, cols in block_cols.items():
        use = [c for c in cols if c in train_df.columns]
        used_cols[block_name] = use
        if len(use) == 0:
            scalers[block_name] = None 
            inv_covs[block_name] = None
            continue
        block_data = train_df[use].astype(float) 
        mask = ~block_data.isna().any(axis=1) 
        numeric_block_values = block_data[mask].to_numpy()
        if numeric_block_values.shape[0] == 0: 
            scalers[block_name] = None
            inv_covs[block_name] = None
            continue
        
        # Standardize each block
        block_scaler = StandardScaler() 
        block_scaler.fit(numeric_block_values) 
        scalers[block_name] = block_scaler

        # Use Mahalanobis distance for deature blocks (except CTX due to high-dimensional sparcity; fall back to Euclidean)
        if block_name != "CTX" and numeric_block_values.shape[0] >= 3 and numeric_block_values.shape[1] >= 2:
            standardized_block_values = block_scaler.transform(numeric_block_values) 
            block_covariance = np.block_covariance(standardized_block_values, rowvar=False)
            block_covariance = block_covariance + 1e-6 * np.eye(block_covariance.shape[0])
            try: 
                inv_covs[block_name] = np.linalg.inv(block_covariance)
            except Exception: 
                inv_covs[block_name] = None
        else:
            inv_covs[block_name] = None
    return scalers, inv_covs, used_cols

def build_block_arrays(df, used_cols, scalers):
    df = ensure_engineered_columns(df.copy())
    arrays = {}
    valid = {}
    for block_name, use in used_cols.items():
        if len(use) == 0:
            arrays[block_name] = np.zeros((len(df), 0))
            valid[block_name] = np.zeros(len(df), dtype=bool) 
            continue
        block_data = df[use].astype(float)
        mask = ~block_data.isna().any(axis=1)
        block_array = np.full((len(df), len(use)), np.nan)
        block_scaler = scalers.get(block_name, None)
        if block_scaler is not None and mask.any(): 
            block_array[mask, :] = block_scaler.transform(block_data[mask].to_numpy())
        
        # Replace NaNs with zeros
        arrays[block_name] = np.where(np.isnan(block_array), 0.0, block_array) 
        valid[block_name] = mask.to_numpy()
    return arrays, valid

# Compute a weighted sum of per-block distances between a single target row and a set of candidate training rows using Mahalanobis Distance (if possible, otherwise Euclidean)
def blockwise_distance(target_index, candidate_indices, targ_arrays, train_arrays, targ_valid, train_valid, block_inv_covariances, block_weights):
    total = None
    for block_name, block_weight in block_weights.items():
        target_block_row = targ_arrays[block_name][target_index:target_index+1, :]
        candidate_block_rows = train_arrays[block_name][candidate_indices, :]
        target_block_has_data = targ_valid[block_name][target_index]
        candidate_block_has_data = train_valid[block_name][candidate_indices]
        if candidate_block_rows.shape[1] == 0:
            block_distance_contrib = np.zeros(len(candidate_indices))
        else:
            usable = target_block_has_data & candidate_block_has_data
            if block_name == "CTX" or block_inv_covariances.get(block_name, None) is None:
                diff = candidate_block_rows - target_block_row
                block_distances = np.sqrt(np.sum(diff * diff, axis=1))
            else:
                block_inv_cov = block_inv_covariances[block_name]
                diff = candidate_block_rows - target_block_row
                block_distances = np.sqrt(np.sum(diff.dot(block_inv_cov) * diff, axis=1))
            block_distance_contrib = np.where(usable, block_distances, 0.0)
        comp = block_weight * block_distance_contrib
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
    feature_matrix = df[cols].copy()
    # Ensure numeric columns are numeric, and fill missing values with 0
    for c in feature_matrix.columns:
        if pd.api.types.is_numeric_dtype(feature_matrix[c]): 
            feature_matrix[c] = pd.to_numeric(feature_matrix[c], errors="coerce").fillna(0.0)
        else:
            s = feature_matrix[c].astype(object) 
            feature_matrix[c] = s.where(s.notna(), "")
    return feature_matrix, cols

def fit_term_models(train_df, valid_terms, alpha=2.0):
    models = {}
    meta = {}
    for term_length in valid_terms:
        # Only include contracts that actually have this term length and a known cap %
        term_train_df = train_df.loc[(train_df["Length"].astype("Int64") == term_length) & train_df["Start_Yr_Cap_Pct"].notna()].copy()
        if not len(term_train_df):
            models[term_length] = None
            meta[term_length] = {"n": 0, "cols": []}
            continue

        term_features, cols = build_term_model_matrix(term_train_df)
        term_cap_pct = pd.to_numeric(term_train_df["Start_Yr_Cap_Pct"], errors="coerce").fillna(0.0).values
        ridge_model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
        ridge_model.fit(term_features, term_cap_pct)
        models[term_length] = ridge_model
        meta[term_length] = {"n": int(len(term_train_df)), "cols": cols}

    return models, meta

# Apply fitted per-term ridge
def predict_term_model_cap_pct(row_df, term_length, model, meta):
    if model is None or not meta.get("cols"): 
        return np.nan
    cols = meta["cols"]
    row_features = row_df.reindex(columns=cols, fill_value=0.0).copy()
    for c in row_features.columns:
        if pd.api.types.is_numeric_dtype(row_features[c]): 
            row_features[c] = pd.to_numeric(row_features[c], errors="coerce").fillna(0.0)
        else:
            s = row_features[c].astype(object)
            row_features[c] = s.where(s.notna(), "")
    try:
        return float(model.predict(row_features)[0])
    except Exception:
        return np.nan

def build_candidates(train_df, row, k):
    same_pos = (train_df["PosBucket"].astype(str) == str(row.get("PosBucket","")))
    candidate_indices = np.where(same_pos.values)[0]
    if candidate_indices.size < k: 
        candidate_indices = np.arange(len(train_df))
    return candidate_indices

def compute_knn_and_terms(train_df, target_df, block_weights, k_neighbors, distance_scale, kernel_power, kmin_per_term,
                          term_models, term_meta, eh_blend_base, eh_blend_strong, eh_blend_cap):
    valid_terms = list(range(1, 9))
    
    # Expand context and build feature blocks
    train_with_context, target_with_context = add_context_onehots(train_df, target_df)
    block_defs = get_block_defs(pd.concat([train_with_context, target_with_context], ignore_index=True, sort=False))
    scalers, inv_covs, used_cols = fit_block_scalers_and_covs(train_with_context, block_defs)
    train_blk, train_valid = build_block_arrays(train_with_context, used_cols, scalers)
    targ_blk, targ_valid = build_block_arrays(target_with_context, used_cols, scalers)
    out_rows = []
    comps_rows = []

    def block_meta_for_target(target_index):
        block_meta = {}
        for block_name in ["USAGE","IMPACT","VALUE","AGE","DEMO","CTX"]:
            has_b = bool(targ_valid[block_name][target_index])
            ncols = int(len(used_cols.get(block_name, [])))
            block_meta[f"block_has_{block_name}"] = int(has_b)
            block_meta[f"block_cols_{block_name}"] = ncols
            block_meta[f"block_w_{block_name}"] = float(block_weights.get(block_name, 0.0))
        return block_meta

    for target_index, row in target_with_context.reset_index(drop=True).iterrows():
        target_player_id = row.get("playerId", np.nan)
        start_year = row.get("Start_Year", np.nan)
        
        # Candidate pool: same-pos neighbors, with fallback to all train rows
        candidate_indices = build_candidates(train_with_context, row, k_neighbors)
        
        # Full blockwise distance vector for target vs candidate pool and keep closest candidates
        distances = blockwise_distance(target_index, candidate_indices, targ_blk, train_blk, targ_valid, train_valid, inv_covs, block_weights)
        take = min(k_neighbors, len(candidate_indices))
        sorted_indices = np.argsort(distances)[:take]
        top_candidate_indices = candidate_indices[sorted_indices]
        top_distances = distances[sorted_indices]
        sim_kernel = kernel_similarity(top_distances, distance_scale, kernel_power)
        kernel_sum = float(sim_kernel.sum())
        if kernel_sum > 0:
            normalized_kernel_weights = sim_kernel / kernel_sum
        else:
            normalized_kernel_weights = np.ones_like(sim_kernel) / max(len(sim_kernel), 1)

        # Produce a percentage similarity score (0–100) based on relative distances for dashboard
        if len(top_distances) == 0:
            sim_pct = np.zeros_like(top_distances, dtype=float)
        else:
            distances_safe = np.where(np.isfinite(top_distances), top_distances, np.nan)
            if np.all(np.isnan(distances_safe)):
                sim_pct = np.zeros_like(distances_safe, dtype=float)
            else:
                max_distance = float(np.nanmax(distances_safe))
                if max_distance <= 0.0:
                    sim_pct = np.full_like(distances_safe, 100.0, dtype=float)
                else:
                    raw = 1.0 - (distances_safe / max_distance)
                    raw = np.clip(raw, 0.0, 1.0)
                    sim_pct = 100.0 * raw
                    sim_pct = np.round(sim_pct).astype(float)
        
        # Build a neighbor DataFrame with distances and KNN weights
        neighbor_df = train_with_context.iloc[top_candidate_indices].copy().reset_index(drop=True)
        neighbor_df["knn_dist"] = top_distances
        neighbor_df["knn_sim"] = sim_kernel
        neighbor_df["knn_weight"] = normalized_kernel_weights
        neighbor_df["knn_sim_pct"] = sim_pct
        cap_pct_knn = {} 
        cap_pct_iqr = {} 
        neighbors_per_term = {} 
        effective_neighbors_per_term = {} 
        cap_pct_model = {} 
        model_weight_per_term = {}
        cap_pct_blend = {}
        row_X, _ = build_term_model_matrix(row.to_frame().T)
        for term_length in valid_terms:
            # KNN stats
            neighbors_term_mask = neighbor_df["Length"].astype("Int64").eq(term_length).values
            neighbor_count_term = int(neighbors_term_mask.sum())
            neighbors_per_term[term_length] = neighbor_count_term
            if neighbor_count_term == 0:
                cap_pct_knn[term_length] = np.nan
                cap_pct_iqr[term_length] = np.nan
                effective_neighbors_per_term[term_length] = np.nan
            else:
                neighbors_for_term = neighbor_df.loc[neighbors_term_mask]
                neighbor_cap_values = pd.to_numeric(neighbors_for_term["Start_Yr_Cap_Pct"], errors="coerce")
                neighbor_weights = neighbors_for_term["knn_weight"].to_numpy()
                if neighbor_cap_values.notna().sum() == 0:
                    cap_pct_knn[term_length] = np.nan
                    cap_pct_iqr[term_length] = np.nan
                    effective_neighbors_per_term[term_length] = np.nan
                else:
                    cap_pct_knn[term_length] = float(np.nansum(neighbor_cap_values.to_numpy()*neighbor_weights) / max(np.nansum(neighbor_weights), 1e-12))
                    neighbor_cap_nonmissing = neighbor_cap_values.dropna().to_numpy()
                    cap_pct_iqr[term_length] = float(np.quantile(neighbor_cap_nonmissing, 0.75) - np.quantile(neighbor_cap_nonmissing, 0.25)) if neighbor_cap_nonmissing.size >= 4 else np.nan
                    normalized_neighbor_weights = neighbor_weights / max(neighbor_weights.sum(), 1e-12)
                    effective_neighbors_per_term[term_length] = float(1.0 / max((normalized_neighbor_weights**2).sum(), 1e-12))

            # EH per-term ridge prediction
            cap_pct_model[term_length] = predict_term_model_cap_pct(row_X, term_length, term_models.get(term_length), term_meta.get(term_length, {}))

            # Blending EH and KNN
            if neighbor_count_term == 0:
                model_weight = eh_blend_strong
            elif neighbor_count_term < kmin_per_term:
                model_weight = eh_blend_base
            else:
                iqr_value = cap_pct_iqr[term_length]
                model_weight = eh_blend_base if pd.isna(iqr_value) else min(eh_blend_cap, eh_blend_base + 0.10*(iqr_value/0.05))

            model_weight_per_term[term_length] = model_weight
            cap_knn_term, cap_eh_term = cap_pct_knn[term_length], cap_pct_model[term_length]
            if pd.isna(cap_knn_term) and pd.isna(cap_eh_term): 
                cap_pct_blend[term_length] = np.nan
            elif pd.isna(cap_knn_term):
                cap_pct_blend[term_length] = cap_eh_term
            elif pd.isna(cap_eh_term):
                cap_pct_blend[term_length] = cap_knn_term
            else:
                cap_pct_blend[term_length] = float(model_weight*cap_eh_term + (1.0 - model_weight)*cap_knn_term)

        row_out = {
            "playerId": target_player_id, "Start_Year": start_year, "neighbor_count": int(len(neighbor_df)),
            "PosBucket": row.get("PosBucket",""), "status_group": row.get("status_group",""),
            "cap_era": row.get("cap_era",""), "role_tier": row.get("role_tier", np.nan)
            }
        row_out.update(block_meta_for_target(target_index))

        for term_length in valid_terms:
            row_out[f"capPct_len{term_length}"] = cap_pct_blend[term_length]
            row_out[f"capPct_len{term_length}_knn"] = cap_pct_knn[term_length]
            row_out[f"capPct_len{term_length}_eh"]  = cap_pct_model[term_length]
            row_out[f"capPct_len{term_length}_iqr"] = cap_pct_iqr[term_length]
            row_out[f"n_len{term_length}"] = int(neighbors_per_term[term_length])
            row_out[f"eff_n_len{term_length}"] = float(effective_neighbors_per_term[term_length]) if not pd.isna(effective_neighbors_per_term[term_length]) else np.nan
            row_out[f"w_model_len{term_length}"] = float(model_weight_per_term[term_length])
        out_rows.append(row_out)

        # top-5 comps (for dashboard)
        order_by_weight_desc = np.argsort(neighbor_df["knn_weight"].to_numpy())[::-1]
        seen_pids = set()
        rank = 1
        for neighbor_idx in order_by_weight_desc:
            if rank > 5:
                break

            neighbor_row = neighbor_df.iloc[neighbor_idx]
            comp_pid = neighbor_row.get("playerId", np.nan)

            if pd.isna(comp_pid):
                continue

            try:
                if not pd.isna(target_player_id) and int(comp_pid) == int(target_player_id):
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
                "target_playerId": target_player_id,
                "comp_rank": rank,
                "comp_playerId": comp_pid,
                "comp_PlayerName": neighbor_row.get("PlayerName", ""),
                "comp_Pos": neighbor_row.get("Pos", ""),
                "comp_PosBucket": neighbor_row.get("PosBucket", ""),
                "comp_RoleBucket": neighbor_row.get("RoleBucket", ""),
                "comp_Start_Year": neighbor_row.get("Start_Year", np.nan),
                "comp_Length": neighbor_row.get("Length", np.nan),
                "comp_CapPct": neighbor_row.get("Start_Yr_Cap_Pct", np.nan),
                "comp_weight_norm": neighbor_row.get("knn_weight", np.nan),
                "comp_dist": neighbor_row.get("knn_dist", np.nan),
                "comp_sim_pct": neighbor_row.get("knn_sim_pct", np.nan),
                })
            rank += 1
    return pd.DataFrame(out_rows), pd.DataFrame(comps_rows)

# Evaluation against real-signed terms
def evaluate_per_true_term(pred_df, target_df):
    join_keys = ["playerId", "Start_Year"]
    meta_cols = ["PosBucket", "status_group", "cap_era", "role_tier"]
    target_with_meta = ensure_meta_cols(target_df, meta_cols)

    base = target_with_meta[["playerId","Start_Year","Length","Start_Yr_Cap_Pct"] + meta_cols].copy()
    base["Length"] = pd.to_numeric(base["Length"], errors="coerce").astype("Int64")

    pred_df = pred_df.drop(columns=meta_cols, errors="ignore")
    df = base.merge(pred_df, on=join_keys, how="left")
    cap_err = []
    pred_cap = []
    neighbor_count_list = []
    effective_neighbors_list = []
    true_model_weight_list = []
    
    # Loop by row, reading off the true term and then pulling the matching prediction
    for i, true_term_length in enumerate(df["Length"].astype("Int64").tolist()):
        if pd.isna(true_term_length):
            cap_err.append(np.nan)
            pred_cap.append(np.nan)
            neighbor_count_list.append(np.nan) 
            effective_neighbors_list.append(np.nan)
            true_model_weight_list.append(np.nan) 
            continue
        cap_column_name = f"capPct_len{int(true_term_length)}"
        neighbor_count_column, effective_neighbors_column, model_weight_column = f"n_len{int(true_term_length)}", f"eff_n_len{int(true_term_length)}", f"w_model_len{int(true_term_length)}"

        if cap_column_name not in df.columns or pd.isna(df.at[i, cap_column_name]) or pd.isna(df.at[i, "Start_Yr_Cap_Pct"]):
            cap_err.append(np.nan)
            pred_cap.append(np.nan)
        else:
            cap_err.append(abs(float(df.at[i, cap_column_name]) - float(df.at[i, "Start_Yr_Cap_Pct"])))
            pred_cap.append(float(df.at[i, cap_column_name]))

        neighbor_count_list.append(float(df.at[i, neighbor_count_column]) if neighbor_count_column in df.columns else np.nan)
        effective_neighbors_list.append(float(df.at[i, effective_neighbors_column]) if effective_neighbors_column in df.columns else np.nan)
        true_model_weight_list.append(float(df.at[i, model_weight_column]) if model_weight_column in df.columns else np.nan)

    df["abs_err_capPct"] = cap_err
    df["pred_capPct_at_true_term"] = pred_cap
    df["n_neighbors_true_term"] = neighbor_count_list
    df["eff_n_true_term"] = effective_neighbors_list
    df["w_model_true_term"] = true_model_weight_list

    def get_mean(x):
        vals = pd.to_numeric(x, errors="coerce")
        return float(vals.mean(skipna=True)) if vals.notna().any() else np.nan

    def get_median(x):
        vals = pd.to_numeric(x, errors="coerce")
        return float(vals.median(skipna=True)) if vals.notna().any() else np.nan

    # Aggregate error metrics by contract term
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
    group_dimensions = ["PosBucket", "status_group", "cap_era", "role_tier"]
    df = ensure_meta_cols(expiring_eval_df, group_dimensions).copy()
    if "abs_err_capPct" not in df.columns:
        df["abs_err_capPct"] = np.nan

    parts = []
    for dimension in group_dimensions:
        group_columns = [c for c in [dimension, "abs_err_capPct"] if c in df.columns]
        if dimension not in group_columns:
            continue
        group_stats = (df[group_columns].groupby(dimension, dropna=False, observed=True)["abs_err_capPct"].agg(n="count",
                 MAE_cap_pct=lambda x: pd.to_numeric(x, errors="coerce").mean(skipna=True),
                 median_abs_error=lambda x: pd.to_numeric(x, errors="coerce").median(skipna=True))
            .reset_index())
        group_stats.insert(0, "group_by", dimension)
        parts.append(group_stats)

    if parts:
        return pd.concat(parts, ignore_index=True, sort=False)
    return pd.DataFrame(columns=["group_by", "n", "MAE_cap_pct", "median_abs_error"])

def widen_top5_comps(comps_df):
    if comps_df is None or len(comps_df) == 0:
        return pd.DataFrame(columns=["playerId"])

    sorted_comps = comps_df.copy()
    sorted_comps = sorted_comps.sort_values(["target_playerId", "comp_rank", "comp_weight_norm"], ascending=[True, True, False])
    out_rows = []
    for target_id, target_group in sorted_comps.groupby("target_playerId", dropna=False):
        row_dict = {"playerId": target_id}
        target_group = target_group.drop_duplicates(subset=["comp_playerId"], keep="first")
        target_group = target_group.head(5).reset_index(drop=True)

        for comp_index, row in enumerate(target_group.itertuples(index=False), start=1):
            row_dict[f"comp{comp_index}_playerId"] = getattr(row, "comp_playerId", np.nan)
            row_dict[f"comp{comp_index}_PlayerName"] = getattr(row, "comp_PlayerName", "")
            row_dict[f"comp{comp_index}_Pos"] = getattr(row, "comp_Pos", "")
            row_dict[f"comp{comp_index}_PosBucket"] = getattr(row, "comp_PosBucket", "")
            row_dict[f"comp{comp_index}_RoleBucket"] = getattr(row, "comp_RoleBucket", "")
            row_dict[f"comp{comp_index}_Start_Year"] = getattr(row, "comp_Start_Year", np.nan)
            row_dict[f"comp{comp_index}_Length"] = getattr(row, "comp_Length", np.nan)
            row_dict[f"comp{comp_index}_CapPct"] = getattr(row, "comp_CapPct", np.nan)
            row_dict[f"comp{comp_index}_weight_norm"] = getattr(row, "comp_weight_norm", np.nan)
            row_dict[f"comp{comp_index}_dist"] = getattr(row, "comp_dist", np.nan)
            row_dict[f"comp{comp_index}_sim_pct"] = getattr(row, "comp_sim_pct", np.nan)

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
    results_df = union_pred[keep].copy()
    comps_wide = widen_top5_comps(comps_df)
    results_df = results_df.merge(comps_wide, on="playerId", how="left")
    results_df = (results_df.sort_values(["playerId","Start_Year"], ascending=[True, True]).drop_duplicates(subset=["playerId","Start_Year"], keep="first").copy())

    out_results = out_dir / f"contract_results_{season_str}.csv"
    results_df.to_csv(out_results, index=False)
    return results_df

def write_model_diagnostics(out_dir, season_str, by_term_df):
    cols = ["term","n_eval","MAE_cap_pct","median_abs_error"]
    diag_df = by_term_df.copy()
    for c in cols:
        if c not in diag_df.columns: 
            diag_df[c] = np.nan
    diag_df = diag_df.sort_values("term")
    diagnostics_path = out_dir / f"model_diagnostics_{season_str}.csv"
    diag_df[cols].to_csv(diagnostics_path, index=False)

# Main
def main():
    # Load contracts and build GAR bank
    contracts_all = load_contracts(contracts_master_path)
    contracts_all["cap_era"] = contracts_all["Start_Year"].apply(cap_era_label_from_year)
    gar_bank = build_prev_season_bank(clean_dir, 2015, target_year-1)
    contracts_cut = contracts_all.loc[contracts_all["Start_Year"] <= target_year].copy()
    # Build per-contract training rows linking to prior-year GAR/WAR and age curve projections
    training_master = build_training_master(contracts_cut, gar_bank)
    age_curve = load_age_curve(age_curve_path)
    age_ratio_tables = build_ratio_tables(age_curve)
    training_master = attach_gar_projections(training_master, gar_bank, age_ratio_tables, horizon=9)

    train_df = training_master.loc[training_master["Start_Year"] < target_year].copy()
    expiring_df = training_master.loc[training_master["Start_Year"] == target_year].copy()
    
    # Base population of skaters who played last year and signed contracts this year
    gar_prev = gar_bank.loc[gar_bank["Stats_Year"] == target_year-1].copy()
    base = (gar_prev.sort_values(["playerId","TOI_all"], ascending=[True, False]).drop_duplicates(subset=["playerId"]).copy())
    base["PlayerName"] = base["Player"]
    base["Pos"] = base["PosBucket"].map(lambda x: "D" if str(x).upper()=="D" else "F")

    cc = contracts_all.copy().sort_values(["playerId","Start_Year"])
    last = cc.groupby("playerId").tail(1)[["playerId","Signing_Age","Signing_Season", "Signing_Status","Level_raw","Clauses_raw",
                                           "clause_bucket","level_clean","Shot","Weight_lb","Height_in", "cap_era","age_band","Signing_GM"]].copy()
    
    def parse_season_start(ss):
        if pd.isna(ss): 
            return np.nan
        m = re.match(r"^\s*(\d{4})", str(ss))
        return int(m.group(1)) if m else np.nan
        
    last["Signing_Season_StartYr"] = last["Signing_Season"].apply(parse_season_start).astype("Int64")
    
    # Extrapolate the player's "current" age at the target year based on last known signing
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
    
    # Attach projections for what-if players
    whatif_df = wf[keep].copy()
    whatif_df = attach_gar_projections(whatif_df, gar_bank, age_ratio_tables, horizon=9)
    expiring_ids = set(expiring_df["playerId"].dropna().astype("Int64"))
    whatif_df = whatif_df.loc[~whatif_df["playerId"].astype("Int64").isin(expiring_ids)].copy()
    expiring_df["src"] = "E"
    whatif_df["src"] = "W"
    target_df = (pd.concat([expiring_df, whatif_df], ignore_index=True, sort=False).sort_values(["playerId","Start_Year","src"], ascending=[True, True, True])
                   .drop_duplicates(subset=["playerId","Start_Year"], keep="first").drop(columns="src").copy())

    # Fit KNN and Ridge models for each term using historical contracts
    K_opt = k_neighbors
    W_opt = block_weights
    dm_opt = distance_scale
    uq_opt = kernel_power
    eh_alpha_opt = evolv_hockey_blend_alpha
    eh_b_opt = evolv_hockey_blend_base
    eh_s_opt = evolv_hockey_blend_strong
    eh_cap_opt = evolv_hockey_blend_cap
    term_models, term_meta = fit_term_models(train_df, list(range(1,9)), alpha=eh_alpha_opt)
    pred_df, comps_df = compute_knn_and_terms(train_df, target_df, W_opt, K_opt, dm_opt, uq_opt, k_min_per_term,
                                              term_models, term_meta, eh_blend_base=eh_b_opt, eh_blend_strong=eh_s_opt, eh_blend_cap=eh_cap_opt)

    # Evaluate only on contracts starting in target year
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
    
