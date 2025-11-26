# -*- coding: utf-8 -*-
"""
NST and API Game ID Mapper

Connect NST 'Games' exports to NHL TeamGames by matching (date, team pair),
then merge per-situation team metrics (ES/PP/PK) into TeamGames for each season.
"""
# Imports
import re
from pathlib import Path
import pandas as pd

# Config
start_year = 2015
end_year = 2024

base_dir = Path(__file__).resolve().parent.parent
data_root = base_dir / "Data"
raw_NST_root = data_root / "Raw Data" / "NaturalStatTrick" / "Games"
clean_data_root = data_root / "Clean Data"

es_counts_path_template = raw_NST_root / "Counts" / "Even Strength" / "NST Games Even Strength Counts {season}.csv"
es_rates_path_template = raw_NST_root / "Rates" / "Even Strength" / "NST Games Even Strength Rates {season}.csv"
pp_counts_path_template = raw_NST_root / "Counts" / "Power Play" / "NST Games Power Play Counts {season}.csv"
pp_rates_path_template = raw_NST_root / "Rates" / "Power Play" / "NST Games Power Play Rates {season}.csv"
pk_counts_path_template = raw_NST_root / "Counts" / "Penalty Kill" / "NST Games Penalty Kill Counts {season}.csv"
pk_rates_path_template = raw_NST_root / "Rates" / "Penalty Kill" / "NST Games Penalty Kill Rates {season}.csv"

team_code_map = {
    "Anaheim Ducks":"ANA","Boston Bruins":"BOS","Buffalo Sabres":"BUF","Calgary Flames":"CGY",
    "Carolina Hurricanes":"CAR","Chicago Blackhawks":"CHI","Colorado Avalanche":"COL","Columbus Blue Jackets":"CBJ",
    "Dallas Stars":"DAL","Detroit Red Wings":"DET","Edmonton Oilers":"EDM","Florida Panthers":"FLA",
    "Los Angeles Kings":"LAK","Minnesota Wild":"MIN","Montreal Canadiens":"MTL","Nashville Predators":"NSH",
    "New Jersey Devils":"NJD","New York Islanders":"NYI","New York Rangers":"NYR","Ottawa Senators":"OTT",
    "Philadelphia Flyers":"PHI","Pittsburgh Penguins":"PIT","San Jose Sharks":"SJS","Seattle Kraken":"SEA",
    "St Louis Blues":"STL","Tampa Bay Lightning":"TBL","Toronto Maple Leafs":"TOR","Vancouver Canucks":"VAN",
    "Vegas Golden Knights":"VGK","Washington Capitals":"WSH","Winnipeg Jets":"WPG","Utah Hockey Club":"UTA",
    "T.B":"TBL","S.J":"SJS","L.A":"LAK","N.J":"NJD",
    "Arizona Coyotes":"ARI"
    }

known_tris = {
    "ANA","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM","FLA",
    "LAK","MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT","SJS","SEA",
    "STL","TBL","TOR","VAN","VGK","WSH","WPG","UTA","ARI"
    }

# Helpers
def season_label(year):
    return f"{year}-{year+1}"

def strip_bad_whitespace(s):
    if s is None:
        return ""
    s = str(s).replace("\xa0", " ").replace("Â", " ")
    return re.sub(r"\s+", " ", s).strip()

def to_tricode(team_full_or_abbrev):
    raw = strip_bad_whitespace(team_full_or_abbrev)
    if not raw:
        return None
    if raw in team_code_map:
        return team_code_map[raw]
    u = raw.upper()
    if u in team_code_map:
        return team_code_map[u]
    if len(u) == 3 and u in known_tris:
        return u
    u_nodots = u.replace(".", "")
    if u_nodots in team_code_map:
        return team_code_map[u_nodots]
    if len(u_nodots) == 3 and u_nodots in known_tris:
        return u_nodots
    return None

def mmss_to_seconds(time):
    time_string = str(time).strip() if pd.notna(time) else ""
    if not time_string or ":" not in time_string:
        return pd.to_numeric(time, errors="coerce")
    parts = time_string.split(":")
    try:
        if len(parts) == 2:
            m, sec = int(parts[0]), int(parts[1])
            return m*60 + sec
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            return h*3600 + m*60 + sec
    except Exception:
        return pd.NA
    return pd.NA

def coerce_numeric_cols(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (out[c].astype(str).str.replace("'", "", regex=False).str.replace(",", "", regex=False))
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# NST Parsing
def parse_nst_games_grouped(nst_path):
    df = pd.read_csv(nst_path)
    game_col = None
    team_col = None
    for c in df.columns:
        if c.strip().lower() == "game":
            game_col = c
        if c.strip().lower() == "team":
            team_col = c
    if game_col is None or team_col is None:
        raise ValueError(f"{Path(nst_path).name}: expected columns 'Game' and 'Team'.")

    games = df[[game_col, team_col]].copy()
    games["Game"] = games[game_col].astype(str).map(strip_bad_whitespace)
    games["Team"] = games[team_col].astype(str).map(strip_bad_whitespace)

    date_pat = re.compile(r"^(\s*\d{4}-\d{2}-\d{2})")
    games["Date"] = pd.to_datetime(
        games["Game"].str.extract(date_pat)[0].str.strip(),
        errors="coerce"
        ).dt.normalize()

    grouped = (games.dropna(subset=["Date"])
             .groupby("Game", as_index=False)
             .agg({"Date":"first", "Team": lambda s: sorted(set(s))}))

    grouped = grouped[grouped["Team"].apply(lambda lst: isinstance(lst, list) and len(lst) == 2)].copy()
    grouped["teamA_full"] = grouped["Team"].str[0]
    grouped["teamB_full"] = grouped["Team"].str[1]

    grouped["teamA"] = grouped["teamA_full"].map(to_tricode)
    grouped["teamB"] = grouped["teamB_full"].map(to_tricode)

    bad = grouped[grouped["teamA"].isna() | grouped["teamB"].isna()]
    if len(bad) > 0:
        print("Some NST team names did not map to tricodes; sample:")
        print(bad[["Date","teamA_full","teamB_full"]].head(8).to_string(index=False))

    grouped = grouped.dropna(subset=["teamA","teamB"]).copy()

    def pair_key(a, b):
        aa, bb = sorted([a, b])
        return aa + "|" + bb

    grouped["pair_key"] = grouped.apply(lambda r: pair_key(r["teamA"], r["teamB"]), axis=1)
    return grouped[["Date","pair_key","teamA","teamB"]].copy()

# TeamGames parsing
def build_schedule_wide(teamgames_csv):
    teamgames_df = pd.read_csv(teamgames_csv)
    required_cols = ["gameId", "Team", "home_or_away", "gameDate"]
    missing = [c for c in required_cols if c not in teamgames_df.columns]
    if missing:
        raise ValueError(f"ERROR: TeamGames missing required columns: {missing}")

    schedule_df = teamgames_df[["gameId", "Team", "home_or_away", "gameDate"]].copy()
    schedule_df.rename(columns={"Team": "TeamRaw", "home_or_away": "hoa"}, inplace=True)
    schedule_df["Team"] = schedule_df["TeamRaw"].map(to_tricode)
    schedule_df["Date"] = pd.to_datetime(schedule_df["gameDate"], errors="coerce").dt.normalize()
    home = schedule_df[schedule_df["hoa"].astype(str).str.lower().eq("home")][["gameId","Team","Date"]].rename(columns={"Team":"homeTeam"})
    away = schedule_df[schedule_df["hoa"].astype(str).str.lower().eq("away")][["gameId","Team","Date"]].rename(columns={"Team":"awayTeam"})
    schedule_wide = home.merge(away, on="gameId", how="inner")
    schedule_wide["Date"] = schedule_wide["Date_x"].fillna(schedule_wide["Date_y"])
    schedule_wide.drop(columns=["Date_x","Date_y"], inplace=True, errors="ignore")

    def pair_key(a, b):
        aa, bb = sorted([a, b])
        return aa + "|" + bb

    schedule_wide["pair_key"] = schedule_wide.apply(lambda r: pair_key(r["homeTeam"], r["awayTeam"]), axis=1)
    return schedule_wide[["gameId","Date","homeTeam","awayTeam","pair_key"]]

# Match NST to GameIDs
def match_pairs_by_date_and_pair(nst_pairs, sched_wide, allow_tol=True):
    merged_pairs = nst_pairs.merge(sched_wide, on=["Date","pair_key"], how="left", suffixes=("_nst",""))
    unmatched = merged_pairs[merged_pairs["gameId"].isna()].copy()

    if allow_tol and len(unmatched) > 0:
        for delta in (-1, +1):
            trial = unmatched.copy()
            trial["Date_shift"] = trial["Date"] + pd.Timedelta(days=delta)
            trial = trial.drop(columns=["gameId","homeTeam","awayTeam"], errors="ignore")
            trial = trial.merge(
                sched_wide.rename(columns={"Date":"Date_shift"}),
                on=["Date_shift","pair_key"], how="left")
            fill_mask = trial["gameId"].notna()
            if fill_mask.any():
                for c in ["gameId","homeTeam","awayTeam","Date"]:
                    merged_pairs.loc[unmatched.index[fill_mask], c] = trial.loc[fill_mask, c].values
                unmatched = merged_pairs[merged_pairs["gameId"].isna()].copy()
                if len(unmatched) == 0:
                    break

    matched = merged_pairs[merged_pairs["gameId"].notna()].copy()
    still_unmatched = merged_pairs[merged_pairs["gameId"].isna()].copy()
    return matched, still_unmatched

def extract_nst_stats_for_situation(counts_path, rates_path, situation_label):
    counts_exists = Path(counts_path).exists()
    rates_exists = Path(rates_path).exists()
    if not counts_exists and not rates_exists:
        return pd.DataFrame()

    base = pd.read_csv(counts_path) if counts_exists else pd.read_csv(rates_path)
    rates_df = pd.read_csv(rates_path) if rates_exists else None
    game_col = "Game"
    team_col = "Team"
    toi_col  = "TOI"
    xgf_cnt  = "xGF"
    xga_cnt  = "xGA"
    xgf60_rate = "xGF/60"
    xga60_rate = "xGA/60"
    if game_col is None or team_col is None:
        return pd.DataFrame()

    df = base[[game_col, team_col]].copy()
    df.rename(columns={game_col:"Game", team_col:"Team_full"}, inplace=True)
    date_pat = re.compile(r"^(\s*\d{4}-\d{2}-\d{2})")
    df["Date"] = pd.to_datetime(df["Game"].astype(str).str.extract(date_pat)[0].str.strip(), errors="coerce").dt.normalize()
    df["Team"] = df["Team_full"].map(to_tricode)

    if toi_col and toi_col in base.columns:
        df["TOI_seconds"] = base[toi_col].apply(mmss_to_seconds)
    else:
        df["TOI_seconds"] = pd.NA
    if xgf_cnt and xgf_cnt in base.columns:
        df["xGF"] = pd.to_numeric(base[xgf_cnt], errors="coerce")
    else:
        df["xGF"] = pd.NA
    if xga_cnt and xga_cnt in base.columns:
        df["xGA"] = pd.to_numeric(base[xga_cnt], errors="coerce")
    else:
        df["xGA"] = pd.NA
    if xgf60_rate:
        src = rates_df if rates_df is not None else base
        df["xGF_per60"] = pd.to_numeric(src[xgf60_rate], errors="coerce")
    else:
        df["xGF_per60"] = pd.NA
    if xga60_rate:
        src = rates_df if rates_df is not None else base
        df["xGA_per60"] = pd.to_numeric(src[xga60_rate], errors="coerce")
    else:
        df["xGA_per60"] = pd.NA

    df["Situation"] = situation_label
    return df.dropna(subset=["Date","Team"])

def ensure_teamgames_columns(df, season):
    required_cols = [
        "Season","Team","TeamStd","Opponent","OppStd","Situation",
        "TOI_seconds","xGF","xGA","xGF_per60","xGA_per60",
        "gameId","home_or_away","gameDate"
        ]
    out = df.copy()
    if "Season" not in out.columns:
        out["Season"] = season
    for c in required_cols:
        if c not in out.columns:
            out[c] = pd.NA
    ordered = required_cols + [c for c in out.columns if c not in required_cols]
    return out[ordered]

# Merge NST into TeamGames
def update_teamgames_with_mapping_and_stats(teamgames_path, season, matched_pairs, stats_es, stats_pp, stats_pk):
    tg_path = Path(teamgames_path)
    if tg_path.exists():
        try:
            teamgames = pd.read_csv(tg_path)
        except Exception:
            teamgames = pd.DataFrame()
    else:
        teamgames = pd.DataFrame()

    team_id_map = pd.DataFrame(columns=["Season", "Team", "teamId"])
    if not teamgames.empty and ("teamId" in teamgames.columns):
        tmp = teamgames[["Season", "Team", "teamId"]].dropna(subset=["Team", "teamId"]).copy()
        tmp["Team"] = tmp["Team"].astype(str).str.strip().str.upper()
        team_id_map = tmp.drop_duplicates(subset=["Season", "Team"])

    map_min = matched_pairs[["gameId", "Date", "homeTeam", "awayTeam"]].drop_duplicates()

    def expand_with_opponents(stats_df):
        if stats_df.empty:
            return pd.DataFrame()
        s = stats_df.copy()
        s["Team"] = s["Team"].map(to_tricode)

        s_home = s.merge(map_min, left_on=["Date", "Team"], right_on=["Date", "homeTeam"], how="inner")
        s_home["home_or_away"] = "home"
        s_home["OppStd"] = s_home["awayTeam"]

        s_away = s.merge(map_min, left_on=["Date", "Team"], right_on=["Date", "awayTeam"], how="inner")
        s_away["home_or_away"] = "away"
        s_away["OppStd"] = s_away["homeTeam"]

        out = pd.concat([s_home, s_away], ignore_index=True, sort=False)
        out["TeamStd"] = out["Team"] # both are tricodes
        out["gameDate"] = out["Date"].dt.date.astype(str)
        out["Opponent"] = out["OppStd"]
        keep = ["gameId","gameDate","home_or_away","Team","TeamStd","OppStd","Opponent","Situation",
                "TOI_seconds","xGF","xGA","xGF_per60","xGA_per60"]
        return out[keep].drop_duplicates()

    es_long = expand_with_opponents(stats_es)
    pp_long = expand_with_opponents(stats_pp)
    pk_long = expand_with_opponents(stats_pk)
    all_long = pd.concat([es_long, pp_long, pk_long], ignore_index=True)
    all_long["Season"] = season

    if not all_long.empty and not team_id_map.empty:
        all_long = all_long.merge(team_id_map, how="left", on=["Season", "Team"])

    if teamgames.empty:
        new_teamgames = ensure_teamgames_columns(all_long.copy(), season)
        numeric_cols = ["TOI_seconds", "xGF", "xGA", "xGF_per60", "xGA_per60"]
        new_teamgames = coerce_numeric_cols(new_teamgames, numeric_cols)
        new_teamgames.to_csv(tg_path, index=False)
        return new_teamgames

    teamgames = ensure_teamgames_columns(teamgames, season)
    teamgames["Team"] = teamgames["Team"].astype(str).str.strip().str.upper()
    teamgames["home_or_away"] = teamgames["home_or_away"].astype(str).str.strip().str.lower()
    teamgames["Situation"] = teamgames["Situation"].astype("string")

    key_cols = ["gameId", "home_or_away", "Team", "Situation"]
    merged = teamgames.merge(all_long, how="outer", on=key_cols, suffixes=("", "_new"))

    fill_cols = ["gameDate", "TeamStd", "OppStd", "Opponent",
                 "TOI_seconds", "xGF", "xGA", "xGF_per60", "xGA_per60", "Season", "teamId"]
    for c in fill_cols:
        c_new = c + "_new"
        if c in merged.columns and c_new in merged.columns:
            merged[c] = merged[c].where(merged[c].notna(), merged[c_new])
            merged.drop(columns=[c_new], inplace=True, errors="ignore")

    baseline_mask = merged["Situation"].isna()
    situation_keys = (
        merged.loc[~merged["Situation"].isna(), ["gameId", "home_or_away", "Team"]]
        .drop_duplicates()
        .assign(_has_sit=True))
    merged = merged.merge(situation_keys, how="left", on=["gameId", "home_or_away", "Team"])
    drop_mask = baseline_mask & merged["_has_sit"].fillna(False)
    merged = merged.loc[~drop_mask].drop(columns=["_has_sit"])

    merged = ensure_teamgames_columns(merged, season)
    numeric_cols = ["TOI_seconds", "xGF", "xGA", "xGF_per60", "xGA_per60"]
    merged = coerce_numeric_cols(merged, numeric_cols)

    merged.to_csv(tg_path, index=False)
    return merged

# Main Driver
def process_season(year):
    season = season_label(year)
    print(f"Mapping NST/API for {season}")

    nst_counts_path = Path(str(es_counts_path_template).format(season=season))
    nst_pairs = parse_nst_games_grouped(nst_counts_path)
    tg_dir = clean_data_root / season
    tg_file = tg_dir / f"TeamGames_{season}.csv"
    sched = build_schedule_wide(tg_file)
    matched, unmatched = match_pairs_by_date_and_pair(nst_pairs, sched, allow_tol=True)

    # Read ES/PP/PK stats and merge into TeamGames
    es_counts = Path(str(es_counts_path_template).format(season=season))
    es_rates = Path(str(es_rates_path_template).format(season=season))
    pp_counts = Path(str(pp_counts_path_template).format(season=season))
    pp_rates = Path(str(pp_rates_path_template).format(season=season))
    pk_counts = Path(str(pk_counts_path_template).format(season=season))
    pk_rates = Path(str(pk_rates_path_template).format(season=season))

    es_stats = extract_nst_stats_for_situation(es_counts, es_rates, "Even Strength")
    pp_stats = extract_nst_stats_for_situation(pp_counts, pp_rates, "Power Play")
    pk_stats = extract_nst_stats_for_situation(pk_counts, pk_rates, "Penalty Kill")
    updated_tg = update_teamgames_with_mapping_and_stats(str(tg_file), season, matched, es_stats, pp_stats, pk_stats)

    miss_dir = tg_dir / "missing logs"
    miss_dir.mkdir(parents=True, exist_ok=True)
    out_unmatched_path = miss_dir / f"unmatched_nst_{season}.csv"
    unmatched[["Date","pair_key"]].drop_duplicates().to_csv(out_unmatched_path, index=False)

    print(f"Updated {updated_tg.shape[0]} TeamGames rows")

def main():
    for year in range(start_year, end_year + 1):
        process_season(year)

if __name__ == "__main__":
    main()
