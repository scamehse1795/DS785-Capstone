"""
Helper to expand Goals per Win/Standing Point table to include most recent seasons
Initial table/fill strategy found here: https://evolving-hockey.com/glossary/goals-above-replacement/
"""
# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import re

project_root = Path(__file__).resolve().parents[2]
hockey_reference_dir = project_root / "Data" / "Raw Data" / "HockeyReference"
hockey_reference_template = "HR League Standings *.csv"
moneypuck_dir = project_root / "Data" / "Raw Data" / "MoneyPuck" / "Team Level"
moneypuck_template = "MP Team Level *.csv"
out_path = project_root / "Data" / "Clean Data" / "NHL_GOAL_TABLE_MASTER.csv"
 
hr_to_mp = {
    "Anaheim Ducks": "ANA","Arizona Coyotes": "ARI","Boston Bruins": "BOS","Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY","Carolina Hurricanes": "CAR","Chicago Blackhawks": "CHI","Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ","Dallas Stars": "DAL","Detroit Red Wings": "DET","Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA","Los Angeles Kings": "LAK","Minnesota Wild": "MIN","Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH","New Jersey Devils": "NJD","New York Islanders": "NYI","New York Rangers": "NYR",
    "Ottawa Senators": "OTT","Philadelphia Flyers": "PHI","Pittsburgh Penguins": "PIT","San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA","St. Louis Blues": "STL","Tampa Bay Lightning": "TBL","Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN","Vegas Golden Knights": "VGK","Washington Capitals": "WSH","Winnipeg Jets": "WPG",
    "Utah Hockey Club": "UTA"
    }

mp_tricode_normalization = {"T.B": "TBL", "N.J": "NJD", "L.A": "LAK", "S.J": "SJS"}
recompute_from = 2020

# Helpers
def norm_season(season_raw):
    season_raw = str(season_raw).strip()
    if season_raw == "":
        return season_raw
    match_year_short = re.match(r"^(\d{4})\season_raw*[-/]\season_raw*(\d{2})$", season_raw)
    match_year_long = re.match(r"^(\d{4})\season_raw*[-/]\season_raw*(\d{4})$", season_raw)
    match_year_concatenated = re.match(r"^(\d{4})(\d{4})$", season_raw)
    match_single_year = re.match(r"^(\d{4})$", season_raw)
    if match_year_short:
        start_year = int(match_year_short.group(1))
        return f"{start_year}-{start_year+1}"
    if match_year_long:
        start_year = int(match_year_long.group(1))
        end_year = int(match_year_long.group(2))
        return f"{start_year}-{end_year}"
    if match_year_concatenated:
        start_year = int(match_year_concatenated.group(1))
        end_year = int(match_year_concatenated.group(2))
        return f"{start_year}-{end_year}"
    if match_single_year:
        start_year = int(match_single_year.group(1))
        return f"{start_year}-{start_year+1}"
    return season_raw

def season_from_filename(path):
    filename = path.stem
    season_match = re.search(r"(20\d{2}\s*[-/]\s*20\d{2}|20\d{2}\s*[-/]\s*\d{2}|20\d{2}20\d{2}|20\d{2})", filename)
    return norm_season(season_match.group(1)) if season_match else ""

def first_year(season_label):
    try:
        return int(str(season_label).split("-")[0])
    except Exception:
        return -1

def parse_overall_w_l_otl(record_str):
    record_str = str(record_str).strip()
    match_w_l_otl = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*$", record_str)
    if match_w_l_otl:
        return (int(match_w_l_otl.group(1)), int(match_w_l_otl.group(2)), int(match_w_l_otl.group(3)))

    match_w_l = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", record_str)
    if match_w_l:
        return int(match_w_l.group(1)), int(match_w_l.group(2)), 0

    return None, None, None

def wls_slope_y_on_x(y_values, x_values, weights):
    valid_mask = ((~pd.isna(y_values)) & (~pd.isna(x_values)) & (~pd.isna(weights)) & (weights > 0))
    if valid_mask.sum() < 3:
        return np.nan

    X_design = np.column_stack([np.ones(valid_mask.sum()), x_values[valid_mask].values.astype(float)])
    y_clean = y_values[valid_mask].values.astype(float)
    weights_sqrt = np.sqrt(weights[valid_mask].values.astype(float))
    X_weighted = X_design * weights_sqrt[:, None]
    y_weighted = y_clean * weights_sqrt
    beta, *_ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    return float(beta[1])

def fit_window_for_target_season(merged, season_label, window=3):
    years = sorted({first_year(s) for s in merged["Season"].unique() if str(s).strip()})
    target_year = first_year(season_label)
    pool = [f"{y}-{y+1}" for y in years if target_year - (window-1) <= y <= target_year]
    if len(pool) == 0:
        return np.nan, np.nan
    
    window_data = merged[merged["Season"].isin(pool)].copy()
    window_data["recency_w"] = window_data["Season"].apply(lambda s: 1 + max(0, first_year(s) - (target_year - (window-1))))
    slope_goals_per_win = wls_slope_y_on_x(window_data["GDPG"], window_data["WPG"], window_data["recency_w"])
    slope_goals_per_point = wls_slope_y_on_x(window_data["GDPG"], window_data["PTSG"], window_data["recency_w"])
    return slope_goals_per_win, slope_goals_per_point

def load_hr_all():
    files = sorted(hockey_reference_dir.glob(hockey_reference_template))
    if not files:
        return pd.DataFrame(columns=["Season","TeamMP","Points","Wins","ROW_HR","SOW_HR","GP_HR"])

    parts = []
    for csv_path in files:
        df = pd.read_csv(csv_path)
        if "Team" not in df.columns or "Overall" not in df.columns:
            continue
        parsed = df["Overall"].apply(parse_overall_w_l_otl)
        W = pd.to_numeric(parsed.apply(lambda x: x[0] if x else None), errors="coerce").fillna(0).astype(int)
        L = pd.to_numeric(parsed.apply(lambda x: x[1] if x else None), errors="coerce").fillna(0).astype(int)
        OTL = pd.to_numeric(parsed.apply(lambda x: x[2] if x else 0),   errors="coerce").fillna(0).astype(int)
        PTS = 2*W + OTL
        GP = W + L + OTL
        team_mp = (
            df["Team"].astype(str)
              .str.replace(r"[\*\u2020\u2021†]+","", regex=True)
              .str.strip()
              .map(hr_to_mp)
              .astype(str)
              .replace(mp_tricode_normalization)
            )
        parts.append(pd.DataFrame({
            "Season": norm_season(season_from_filename(csv_path)),
            "TeamMP": team_mp,
            "Points": PTS,
            "Wins": W,
            "GP_HR": GP
            }))

    out = pd.concat(parts, ignore_index=True)
    out = out[(out["TeamMP"].notna()) & (out["TeamMP"].str.len() > 0)]
    out["Season"] = out["Season"].astype(str).apply(norm_season)
    return out

def norm_season_from_mp(season_value):
    try:
        season_year = int(season_value) 
        return f"{season_year}-{season_year+1}"
    except Exception:
        return norm_season(season_value)

def load_mp_goal_diff_for(season_list):
    files = sorted(moneypuck_dir.glob(moneypuck_template))
    if not files:
        return pd.DataFrame(columns=["Season","TeamMP","GoalDiff","GP_MP","SOW_MP"])

    frames = []
    for csv_path in files:
        df = pd.read_csv(csv_path)
        team_col = "team"
        season_col = "season"
        sit_col = "situation"
        gf_col = "goalsFor"
        ga_col = "goalsAgainst"
        gp_col = "games_played"
        needed = {team_col, season_col, sit_col, gf_col, ga_col}
        if None in needed:
            continue

        cols = [team_col, season_col, gf_col, ga_col]
        if gp_col:
            cols.append(gp_col)

        season_slice = df.loc[df[sit_col] == "all", cols].copy()
        season_slice["TeamMP"] = season_slice[team_col].astype(str).str.upper().str.strip().replace(mp_tricode_normalization)
        season_slice["Season"] = season_slice[season_col].apply(norm_season_from_mp)
        season_slice["goalsFor"] = pd.to_numeric(season_slice[gf_col], errors="coerce").fillna(0.0)
        season_slice["goalsAgainst"] = pd.to_numeric(season_slice[ga_col], errors="coerce").fillna(0.0)
        season_slice["GoalDiff"] = season_slice["goalsFor"] - season_slice["goalsAgainst"]
        season_slice["GP_MP"] = pd.to_numeric(season_slice[gp_col], errors="coerce") if gp_col else np.nan
        keep = season_slice[["Season","TeamMP","GoalDiff","GP_MP"]].drop_duplicates(subset=["Season","TeamMP"], keep="last")
        frames.append(keep)
        
    if not frames:
        return pd.DataFrame(columns=["Season","TeamMP","GoalDiff","GP_MP"])

    goal_diff_all = pd.concat(frames, ignore_index=True)
    if season_list:
        goal_diff_all = goal_diff_all[goal_diff_all["Season"].isin(season_list)].copy()
        
    goal_diff_all["Season"] = goal_diff_all["Season"].astype(str).apply(norm_season_from_mp)
    goal_diff_all = goal_diff_all.dropna(subset=["Season", "TeamMP"]).reset_index(drop=True)
    return goal_diff_all

def build_merged_with_rates(hr_df, mp_df, season_list=None):
    merged = hr_df.merge(mp_df, on=["Season","TeamMP"], how="inner")
    merged["GP_pref"] = np.where(merged["GP_HR"].notna() & (merged["GP_HR"] > 0), merged["GP_HR"], merged["GP_MP"])
    merged = merged[pd.to_numeric(merged["GP_pref"], errors="coerce") > 0].copy()
    row_from_hr = pd.to_numeric(merged.get("ROW_HR"), errors="coerce")
    wins_from_hr = pd.to_numeric(merged["Wins"], errors="coerce")
    wins_row = row_from_hr.copy()
    wins_row = np.where(pd.isna(wins_row), wins_from_hr, wins_row)
    merged["Wins_ROW"] = wins_row
    merged["GDPG"] = pd.to_numeric(merged["GoalDiff"], errors="coerce") / merged["GP_pref"]
    merged["WPG"] = pd.to_numeric(merged["Wins_ROW"], errors="coerce") / merged["GP_pref"]
    merged["PTSG"] = pd.to_numeric(merged["Points"],   errors="coerce") / merged["GP_pref"]
    merged = merged.dropna(subset=["GDPG","WPG","PTSG"])
    if season_list:
        merged = merged[merged["Season"].isin(season_list)].copy()
    return merged

def load_master():
    cols = ["Season","GOALS_TO_WIN","GOALS_TO_STANDING_POINT"]
    if not out_path.exists():
        return pd.DataFrame(columns=cols)
    goal_table = pd.read_csv(out_path)
    goal_table.columns = [str(c).replace("\ufeff","").strip() for c in goal_table.columns]
    mask = ~goal_table.columns.duplicated(keep="first")
    goal_table = goal_table.loc[:, mask]
    rename_map = {}
    for c in goal_table.columns:
        lc = c.lower()
        if lc == "season" and c != "Season": rename_map[c] = "Season"
        if lc in ("goals_to_win","goalsperwin","gpw") and c != "GOALS_TO_WIN": rename_map[c] = "GOALS_TO_WIN"
        if lc in ("goals_to_standing_point","goalsperstandingpoint","gpsp") and c != "GOALS_TO_STANDING_POINT": rename_map[c] = "GOALS_TO_STANDING_POINT"
    if rename_map:
        goal_table = goal_table.rename(columns=rename_map)
    goal_table["GOALS_TO_WIN"] = pd.to_numeric(goal_table["GOALS_TO_WIN"], errors="coerce")
    goal_table["GOALS_TO_STANDING_POINT"] = pd.to_numeric(goal_table["GOALS_TO_STANDING_POINT"], errors="coerce")
    goal_table = goal_table[["Season","GOALS_TO_WIN","GOALS_TO_STANDING_POINT"]].copy()
    goal_table["Season"] = goal_table["Season"].astype(str).apply(norm_season)
    goal_table = goal_table.dropna(subset=["Season"]).drop_duplicates(subset=["Season"], keep="last").reset_index(drop=True)
    return goal_table

# Main
def main():
    hr_standings = load_hr_all()
    goal_diff = load_mp_goal_diff_for(sorted(hr_standings["Season"].dropna().unique().tolist()))
    seasons_to_recompute = sorted(set(hr_standings["Season"]).intersection(goal_diff["Season"]))
    seasons_to_recompute = [season_label for season_label in seasons_to_recompute if first_year(season_label) >= recompute_from]
    merged = build_merged_with_rates(hr_standings, goal_diff, season_list=seasons_to_recompute)
    master = load_master()
    master_map = (
        master.set_index("Season")[["GOALS_TO_WIN","GOALS_TO_STANDING_POINT"]].to_dict("index")
        if not master.empty else {})
    
    def is_finite_pair(a, b):
        return np.isfinite(a) and np.isfinite(b)

    rows = []
    for season_label in seasons_to_recompute:
        if season_label in master_map:
            gtw_m = master_map[season_label].get("GOALS_TO_WIN")
            gtsp_m = master_map[season_label].get("GOALS_TO_STANDING_POINT")
            if is_finite_pair(gtw_m, gtsp_m):
                gtw, gtsp = float(gtw_m), float(gtsp_m)
            else:
                gtw, gtsp = fit_window_for_target_season(merged, season_label, window=3)
        else:
            gtw, gtsp = fit_window_for_target_season(merged, season_label, window=3)

        rows.append({
            "Season": season_label,
            "GOALS_TO_WIN": round(float(gtw), 4),
            "GOALS_TO_STANDING_POINT": round(float(gtsp), 4)
            })

    updates = pd.DataFrame(rows, columns=["Season","GOALS_TO_WIN","GOALS_TO_STANDING_POINT"])
    if not master.empty:
        keep_master = master[~master["Season"].isin(updates["Season"])]
        updated = pd.concat([keep_master, updates], ignore_index=True)
    else:
        updated = updates

    updated = updated[["Season","GOALS_TO_WIN","GOALS_TO_STANDING_POINT"]]
    updated["Season"] = updated["Season"].astype(str).apply(norm_season)
    updated = updated.drop_duplicates(subset=["Season"], keep="last").sort_values("Season").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
