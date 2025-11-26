# -*- coding: utf-8 -*-
"""
Per-season scrape of NHL per-game skater/goaltender stats with schedule mapping.
Also pulls shift-level segments and Play-by-play (PBP) events for each game and 
calulates shift xGF and xGA for every player's shift for RAPM.

*NOTE: I had some AI assistance setting up the API calls in this script, as the NHL
API does not have a lot of documentation and is incredibly vast in the data it can
return. AI helped me isolate the endpoints to focus on and the skeleton for setting
up the API call itself so that I could get it to work correctly, as I kept running
into errors connecting to the API.

I also used AI to help clean up my script, as I had some redundant code and some very inefficient blocks that AI helped
me with fixing up and making the script a bit faster.
"""
# Imports
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.linear_model import LogisticRegression

# Config 
base_directory = Path(__file__).resolve().parent.parent
output_root = base_directory / "Data" / "Clean Data"
start_year = 2015
end_year = 2025

connection_timeout = 8
read_timeout = 30
maximum_retries = 4
backoff_factor = 0.8
page_limit = 100

stats_rest_base = "https://api.nhle.com/stats/rest"
web_base = "https://api-web.nhle.com"

# Rink geometry (NHL uses coordinates to track the puck on the ice, these mark home/away goals)
GOAL_X = 89.0
GOAL_Y = 0.0

# Required Columns for TeamGames file
teamgames_required_columns = [
    "Season", "Team", "TeamStd", "Opponent", "OppStd", "Situation",
    "TOI_seconds", "xGF", "xGA", "xGF_per60", "xGA_per60",
    "gameId", "home_or_away", "gameDate"
    ]

# Helpers
def format_season_code(year):
    return f"{year}{year+1}"

def format_season_folder(year):
    return f"{year}-{year+1}"

def list_seasons(start_year, end_year):
    return list(range(start_year, end_year + 1))

def build_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "NHL-Stats-Scraper/1.0",
        "Accept": "application/json",
        })
    retry = Retry(
        total = maximum_retries,
        backoff_factor = backoff_factor,
        status_forcelist = (429, 500, 502, 503, 504),
        allowed_methods = frozenset(["GET"]),
        raise_on_status = False,
        respect_retry_after_header = True,
        )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def get_json(session, url, params=None):
    r = session.get(url, params=params, timeout=(connection_timeout, read_timeout))
    if r.status_code != 200:
        r.raise_for_status()
    return r.json()

def parse_date(raw_date):
    if pd.isna(raw_date):
        return None
    timestamp = pd.to_datetime(str(raw_date).strip(), utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat() 

def fetch_player_bios(session, player_type, season_code, game_type_id=2, limit=page_limit):
    url = f"{stats_rest_base}/en/{player_type}/bios"
    all_rows = []
    start = 0
    page_size = int(limit)
    total = None

    if player_type == "goalie":
        report_name = "goaliebios"
        sort_property = "goalieFullName"
    else:
        report_name = "skaterbios"
        sort_property = "skaterFullName"

    base_params = {
        "isAggregate": "false",
        "reportType": "season",
        "isGame": "false",
        "reportName": report_name,
        "cayenneExp": f"seasonId={season_code} and gameTypeId={game_type_id}",
        "sort": f'[{{"property":"{sort_property}","direction":"ASC"}}]',
        "limit": str(page_size),
        }

    while True:
        params = dict(base_params)
        params["start"] = str(start)

        data = get_json(session, url, params=params)
        rows = data.get("data") or []

        if total is None:
            t = data.get("total")
            l = data.get("limit")
            if t is not None:
                total = int(t)
            if l is not None and int(l) < page_size:
                page_size = int(l)
                base_params["limit"] = str(page_size)

        if not rows:
            break

        all_rows.extend(rows)
        start += len(rows)
        if total is not None and start >= total:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.json_normalize(all_rows)
    df = df.assign(seasonId=df.get("seasonId", season_code))
    return df

def collect_bios_for_season(session, season_code, game_type_id=2):
    frames = []

    skater_bios = fetch_player_bios(session, "skater", season_code, game_type_id=game_type_id)
    if not skater_bios.empty:
        frames.append(skater_bios)

    goalie_bios = fetch_player_bios(session, "goalie", season_code, game_type_id=game_type_id)
    if not goalie_bios.empty:
        frames.append(goalie_bios)

    if not frames:
        return pd.DataFrame(columns=["playerId"])

    bios = pd.concat(frames, ignore_index=True)

    if "playerId" not in bios.columns:
        return pd.DataFrame(columns=["playerId"])

    bios["playerId"] = pd.to_numeric(bios["playerId"], errors="coerce").astype("Int64")
    bios = bios.dropna(subset=["playerId"])

    demo_cols = [
        "skaterFullName", "goalieFullName",
        "currentTeamAbbrev", "positionCode", "shootsCatches",
        "birthDate", "birthCity", "birthStateProvinceCode", "birthCountryCode",
        "nationalityCode", "height", "weight",
        "draftYear", "draftRound", "draftOverall",
        "seasonId",
        ]

    keep_cols = ["playerId"] + [c for c in demo_cols if c in bios.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    bios = bios[keep_cols].drop_duplicates(subset=["playerId"], keep="last")

    return bios

def mmss_to_seconds(time):
    if time is None or (isinstance(time, float) and pd.isna(time)):
        return pd.NA
    time_string = str(time).strip()
    if not time_string:
        return pd.NA
    if time_string.isdigit():
        v = pd.to_numeric(time_string, errors="coerce")
        if pd.notna(v):
            return int(v) 
        else:
            return pd.NA
    if ":" in time_string:
        parts = time_string.split(":")
        if len(parts) == 2:
            m = int(parts[0])
            sec = int(parts[1])
            return (m * 60 + sec)
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = int(parts[2])
            return (h * 3600 + m * 60 + sec)
    v = pd.to_numeric(time_string, errors="coerce")
    if pd.notna(v):
        return int(v)
    else:
        return pd.NA

def list_teams(session):
    data = get_json(session, f"{stats_rest_base}/en/team")
    df = pd.json_normalize(data, "data")
    return df[["id", "triCode"]].dropna().drop_duplicates()

def ensure_teamgames_columns(df, season_folder_name):
    out = df.copy()
    out = out.assign(Season=out.get("Season", season_folder_name))
    for c in teamgames_required_columns:
        if c not in out.columns:
            out[c] = pd.NA
    
    remaining_cols = [c for c in out.columns if c not in teamgames_required_columns]
    ordered = teamgames_required_columns + remaining_cols
    return out[ordered]

# Parses through schedule data and builds team games list and data
def build_teamgames_from_schedules(session, season_code, season_folder_name):
    teams = list_teams(session)
    schedule_frames = []
    for tri in teams["triCode"].unique():
        url = f"{web_base}/v1/club-schedule-season/{tri}/{season_code}"
        try:
            json_data = get_json(session, url)
        except Exception:
            print(f"Schedule fetch failed for {tri}/{season_code}")
            continue

        schedule_df = pd.json_normalize(json_data, "games")

        if "id" not in schedule_df.columns:
            continue
        if "gameType" in schedule_df.columns:
            schedule_df = schedule_df[schedule_df["gameType"] == 2]
        schedule_frames.append(schedule_df)

    if not schedule_frames:
        print(f"No schedules returned for {season_code}")
        return pd.DataFrame(columns=["id", "gameDate"]), None

    schedule_all = (pd.concat(schedule_frames, ignore_index=True).drop_duplicates(subset=["id"]))

    # Build simple map w/ two rows per game (home/away)
    rows = []
    for _, row in schedule_all.iterrows():
        gameid = int(row["id"])
        gamedate_raw = row.get("gameDate")
        gamedate = parse_date(gamedate_raw)
        home_abbr = None
        for k in ("homeTeam.abbrev", "homeTeam.triCode", "homeTeam.abbreviation"):
            if k in row and pd.notna(row[k]):
                home_abbr = row[k]
                break
        
        home_id = None
        for k in ("homeTeam.id", "homeTeam.teamId"):
            if k in row and pd.notna(row[k]):
                home_id = row[k]
                break
        
        away_abbr = None
        for k in ("awayTeam.abbrev", "awayTeam.triCode", "awayTeam.abbreviation"):
            if k in row and pd.notna(row[k]):
                away_abbr = row[k]
                break
        
        away_id = None
        for k in ("awayTeam.id", "awayTeam.teamId"):
            if k in row and pd.notna(row[k]):
                away_id = row[k]
                break
        rows.append({
            "gameId": gameid,
            "gameDate": gamedate,
            "home_or_away": "home",
            "Team": home_abbr,
            "Opponent": away_abbr,
            "teamId": home_id,
            "Season": season_folder_name})
        
        rows.append({
            "gameId": gameid,
            "gameDate": gamedate,
            "home_or_away": "away",
            "Team": away_abbr,
            "Opponent": home_abbr,
            "teamId": away_id,
            "Season": season_folder_name})

    tg_map = pd.DataFrame(rows).drop_duplicates(subset=["gameId", "home_or_away", "Team"])
    tg_map["teamId"] = pd.to_numeric(tg_map["teamId"], errors="coerce").astype("Int64")
    tg_dir = Path(output_root) / season_folder_name
    tg_dir.mkdir(parents=True, exist_ok=True)
    tg_path = tg_dir / f"TeamGames_{season_folder_name}.csv"

    if tg_path.exists():
        existing = pd.read_csv(tg_path)
        merged = existing.merge(tg_map[["gameId", "home_or_away", "gameDate", "Team", "teamId", "Season"]], how="left", on=["gameId", "home_or_away"], suffixes=("", "_new"))

        if "gameDate_new" in merged.columns:
            merged["gameDate"] = merged["gameDate"].fillna(merged["gameDate_new"])
            merged = merged.drop(columns=["gameDate_new"])
    
        if "Team_new" in merged.columns:
            merged["Team"] = merged["Team"].fillna(merged["Team_new"])
            merged = merged.drop(columns=["Team_new"])
    
        if "teamId_new" in merged.columns:
            merged["teamId"] = merged["teamId"].fillna(merged["teamId_new"])
            merged = merged.drop(columns=["teamId_new"])
    
        if "Season_new" in merged.columns:
            merged["Season"] = merged["Season"].fillna(merged["Season_new"])
            merged = merged.drop(columns=["Season_new"])

        merged = ensure_teamgames_columns(merged, season_folder_name)
        merged.to_csv(tg_path, index=False)
    else:
        minimal = tg_map.copy()
        minimal["TeamStd"] = pd.NA
        minimal["Opponent"] = pd.NA
        minimal["OppStd"] = pd.NA
        minimal["Situation"] = pd.NA
        minimal["TOI_seconds"] = pd.NA
        minimal["xGF"] = pd.NA
        minimal["xGA"] = pd.NA
        minimal["xGF_per60"] = pd.NA
        minimal["xGA_per60"] = pd.NA
        minimal = ensure_teamgames_columns(minimal, season_folder_name)
        minimal.to_csv(tg_path, index=False)

    return schedule_all[["id", "gameDate"]], tg_path

# Paginated fetch of data by team tricode (game tpye id of 2 specifies regular season in the API)
def fetch_paginated_team_data(session, path, season_code, game_type_id=2, limit=page_limit):
    all_rows = []
    teams_df = list_teams(session).dropna(subset=["id"]).copy()
    teams_df["id"] = teams_df["id"].astype(int)

    url = f"{stats_rest_base}/en/{path}"
    page_size = int(limit)

    for _, team_row in teams_df.iterrows():
        team_id = int(team_row["id"])
        start = 0
        total = None
        params_base = {
            "isGame": "true",
            "cayenneExp": f"seasonId={season_code} and gameTypeId={game_type_id} and teamId={team_id}",
            "limit": str(page_size),
            "sort": '[{"property":"gameId","direction":"ASC"},{"property":"playerId","direction":"ASC"}]',
            }

        while True:
            params = dict(params_base)
            params["start"] = str(start)
            json_data = get_json(session, url, params=params)
            page_rows = json_data.get("data") or []
            if total is None:
                t = json_data.get("total")
                l = json_data.get("limit")
                if t is not None:
                    total = int(t)
                if l is not None and int(l) < page_size:
                    page_size = int(l)
                    params_base["limit"] = str(page_size)

            num_rows = len(page_rows)
            if num_rows == 0:
                break

            all_rows.extend(page_rows)
            start += num_rows
            if total is not None and start >= total:
                break

    if all_rows:
        return pd.json_normalize(all_rows)
    else:
        return pd.DataFrame()
    
# Fetching helpers (game type id of 2 specifies regular season in the API)
def fetch_skater_toi(session, season_code, game_type_id=2):
    return fetch_paginated_team_data(session, "skater/timeonice", season_code, game_type_id, limit=page_limit)

def fetch_skater_summary(session, season_code, game_type_id=2):
    return fetch_paginated_team_data(session, "skater/summary", season_code, game_type_id, limit=page_limit)

def fetch_goalie_summary(session, season_code, game_type_id=2):
    return fetch_paginated_team_data(session, "goalie/summary", season_code, game_type_id, limit=page_limit)

# Helpers for cleaning IDs
def cast_id_columns(df):
    out = df.copy()
    if "gameId" in out.columns:
        out["gameId"] = pd.to_numeric(out["gameId"], errors="coerce").astype("Int64")
    if "playerId" in out.columns:
        out["playerId"] = pd.to_numeric(out["playerId"], errors="coerce").astype("Int64")
    return out

def dedupe_game_player(df):
    if not {"gameId", "playerId"}.issubset(df.columns):
        return df
    return df.sort_values(["gameId", "playerId"]).drop_duplicates(subset=["gameId", "playerId"], keep="last")

def sort_columns_readable(df):
    core = ["seasonId", "gameId", "gameDate", "playerId", "playerFullName", "skaterFullName", "teamAbbrev", "opponentTeamAbbrev"]
    front = [c for c in core if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]

def normalize_goalie_summary(df, season_code):
    g = df.copy()
    if "seasonId" not in g.columns:
        g["seasonId"] = season_code
    g = cast_id_columns(g).dropna(subset=["gameId", "playerId"])
    g = dedupe_game_player(g)

    if "playerFullName" not in g.columns:
        if "goalieFullName" in g.columns:
            g["playerFullName"] = g["goalieFullName"]
        elif "skaterFullName" in g.columns:
            g["playerFullName"] = g["skaterFullName"]
        else:
            g["playerFullName"] = pd.NA

    g["positionCode"] = "G"
    g["isGoalie"] = True

    core = [
        "seasonId", "gameId", "gameDate", "playerId", "playerFullName",
        "teamAbbrev", "opponentTeamAbbrev", "positionCode", "isGoalie",
        "timeOnIce", "shotsAgainst", "saves", "goalsAgainst", "savePct",
        "goalsAgainstAverage", "wins", "losses", "otLosses", "shutouts"
        ]
    front = [c for c in core if c in g.columns]
    rest = [c for c in g.columns if c not in front]
    return g[front + rest]

# I/O helpers
def ensure_paths_for_season_folder(season_folder_name, root):
    sdir = Path(root) / season_folder_name
    sdir.mkdir(parents=True, exist_ok=True)
    final_csv = sdir / f"player_game_stats_{season_folder_name}.csv"
    return final_csv

def left_join_unique(left, right, on_keys):
    right_dedup_cols = on_keys + [c for c in right.columns if c not in left.columns and c not in on_keys]
    right_trimmed = right[right_dedup_cols].copy()
    merged = left.merge(right_trimmed, on=on_keys, how="left")
    return merged

# Player Master helpers
master_file = Path(output_root) / "NHL_PLAYERS_MASTER.csv"

def pos_to_bucket(code):
    if not code:
        return None
    c = str(code).upper()
    if c == "D":
        return "D"
    if c in {"C", "L", "R"}:
        return "F"
    return c

def build_players_from_frames(season_code, skaters_df, goalies_df):
    cols = ["playerId", "fullName", "positionCode", "PosBucket", "seasonId"]
    out_rows = []

    if skaters_df is not None and not skaters_df.empty:
        for _, row in skaters_df.iterrows():
            pid = row.get("playerId")
            if pd.isna(pid):
                continue
            name = row.get("skaterFullName")
            pos = row.get("positionCode")
            out_rows.append({
                "playerId": int(pid),
                "fullName": name,
                "positionCode": pos,
                "PosBucket": pos_to_bucket(pos),
                "seasonId": season_code
                })

    if goalies_df is not None and not goalies_df.empty:
        for _, row in goalies_df.iterrows():
            pid = row.get("playerId")
            if pd.isna(pid):
                continue
            name = row.get("goalieFullName")
            pos = row.get("positionCode", "G")
            if pd.isna(pos):
                pos = "G"
            out_rows.append({
                "playerId": int(pid),
                "fullName": name,
                "positionCode": pos,
                "PosBucket": pos_to_bucket(pos),
                "seasonId": season_code
                })

    players = pd.DataFrame(out_rows, columns=cols)
    if not players.empty:
        players = players.dropna(subset=["playerId"]).drop_duplicates(subset=["playerId"], keep="last")
    return players

def write_players_master_snapshot(players_df_list, bios_df_list=None, master_path=master_file):
    master_path.parent.mkdir(parents=True, exist_ok=True)

    if players_df_list:
        players_df = pd.concat(players_df_list, ignore_index=True)
    else:
        players_df = pd.DataFrame(columns=["playerId", "fullName", "positionCode", "PosBucket", "seasonId"])

    if not players_df.empty:
        players_df["seasonId"] = players_df["seasonId"].astype(str)
        players_df = (
            players_df
            .sort_values(["playerId", "seasonId"])
            .drop_duplicates(subset=["playerId"], keep="last")
            )

    if bios_df_list:
        bios_df = pd.concat(bios_df_list, ignore_index=True)

        if "playerId" in bios_df.columns:
            bios_df["playerId"] = pd.to_numeric(bios_df["playerId"], errors="coerce").astype("Int64")
            bios_df = bios_df.dropna(subset=["playerId"])
            bios_df = bios_df.drop_duplicates(subset=["playerId"], keep="last")
            players_df["playerId"] = pd.to_numeric(players_df["playerId"], errors="coerce").astype("Int64")

            allowed_bio_cols = {
                "playerId",
                "fullName",
                "positionCode",
                "PosBucket",
                "seasonId",
                "currentTeamAbbrev",
                "shootsCatches",
                "birthDate",
                "height",
                "weight",
                "nationalityCode",
                }

            bio_extra_cols = [
                c for c in bios_df.columns
                if c != "playerId"
                and c in allowed_bio_cols
                and c not in players_df.columns
                ]

            bios_keep_cols = ["playerId"] + bio_extra_cols
            bios_df = bios_df[bios_keep_cols]
            players_df["playerId"] = pd.to_numeric(players_df["playerId"], errors="coerce").astype("Int64")
            players_df = players_df.merge(bios_df, on="playerId", how="left")

    base_cols = ["playerId", "fullName", "positionCode", "PosBucket", "seasonId"]
    extra_cols = [c for c in players_df.columns if c not in base_cols]
    players_df = players_df.reindex(columns=base_cols + extra_cols)

    players_df.to_csv(master_path, index=False)

# Main scraper
def scrape_one_regular_season(start_year, out_root=output_root):
    season_code = format_season_code(start_year)
    season_folder = format_season_folder(start_year)
    print(f"Scraping API for {season_folder}")

    session = build_session()
    _, tg_path = build_teamgames_from_schedules(session, season_code, season_folder)
    skater_df = fetch_skater_toi(session, season_code, game_type_id=2)
    skater_summary = fetch_skater_summary(session, season_code, game_type_id=2)

    skater_df = skater_df.assign(seasonId=skater_df.get("seasonId", season_code))
    skater_summary = skater_summary.assign(seasonId=skater_summary.get("seasonId", season_code))

    skater_df = cast_id_columns(skater_df).dropna(subset=["gameId", "playerId"])
    skater_summary = cast_id_columns(skater_summary).dropna(subset=["gameId", "playerId"])
    skater_df = dedupe_game_player(skater_df)
    skater_summary = dedupe_game_player(skater_summary)

    for col in ("evTimeOnIce", "ppTimeOnIce", "shTimeOnIce", "timeOnIce",
                "evTimeOnIcePerGame", "ppTimeOnIcePerGame", "shTimeOnIcePerGame",
                "timeOnIcePerShift"):
        if col in skater_df.columns:
            skater_df[col] = skater_df[col].apply(mmss_to_seconds)

    skaters_merged = left_join_unique(skater_df, skater_summary, on_keys=["gameId", "playerId"])
    
    if "playerFullName" not in skaters_merged.columns and "skaterFullName" in skaters_merged.columns:
        skaters_merged["playerFullName"] = skaters_merged["skaterFullName"]
    if "positionCode" not in skaters_merged.columns:
        skaters_merged["positionCode"] = pd.NA
    skaters_merged["isGoalie"] = False
    skaters_merged = sort_columns_readable(skaters_merged)

    # Goalies (not modeled but included in RAPM)
    goalie_summary_raw = fetch_goalie_summary(session, season_code, game_type_id=2)
    goalies_merged = normalize_goalie_summary(goalie_summary_raw, season_code)

    # Players master segments for this season
    players_this_season = build_players_from_frames(season_code, skaters_merged, goalies_merged)
    bios_this_season = collect_bios_for_season(session, season_code, game_type_id=2)

    # Write csv outputs
    final_csv_path = ensure_paths_for_season_folder(season_folder, out_root)
    all_cols = sorted(set(skaters_merged.columns).union(set(goalies_merged.columns)))
    combined = pd.concat([skaters_merged.reindex(columns=all_cols), goalies_merged.reindex(columns=all_cols)],
        ignore_index=True, sort=False)
    combined.to_csv(final_csv_path, index=False)

    return {
        "player_game_stats_csv": str(final_csv_path),
        "team_games_csv": str(tg_path) if tg_path else None,
        "players_df": players_this_season,
        "bios_df": bios_this_season,
        }

# Shift/PBP Helpers
def load_skater_position_map(out_root, season_folder_name):
    path = Path(out_root) / season_folder_name / f"player_game_stats_{season_folder_name}.csv"
    positions = {}
    skater_ids = set()

    if not path.exists():
        return positions, skater_ids

    try:
        pg = pd.read_csv(path, low_memory=False)
    except Exception:
        return positions, skater_ids

    if "playerId" not in pg.columns:
        return positions, skater_ids
    pid_col = "playerId"
    
    pos_col = "positionCode" if "positionCode" in pg.columns else None

    pg["pid_tmp"] = pd.to_numeric(pg[pid_col], errors="coerce").astype("Int64")
    if pos_col is not None:
        pg["pos_tmp"] = pg[pos_col].astype(str).str.upper()
    else:
        pg["pos_tmp"] = pd.NA

    pg = pg.dropna(subset=["pid_tmp"]).sort_values(["pid_tmp"])
    dedup = pg.drop_duplicates(subset=["pid_tmp"], keep="last")

    for _, r in dedup.iterrows():
        pid = int(r["pid_tmp"])
        pos = r["pos_tmp"] if pd.notna(r["pos_tmp"]) else None
        if pos in ("L", "R", "C", "D"):
            positions[pid] = pos
            skater_ids.add(pid)
        elif pos == "G":
            positions[pid] = "G"
        else:
            positions[pid] = None
            skater_ids.add(pid)

    return positions, skater_ids

def parse_situation_code(s, default_home=5, default_away=5):
    if s is None:
        return default_home, default_away
    s = str(s).strip()
    if len(s) == 4 and s.isdigit():
        try:
            return int(s[1]), int(s[2])
        except Exception:
            pass
    sU = s.upper()
    if sU in ("PP", "POWERPLAY"):
        return default_home + 1, default_away - 1
    if sU in ("SH", "SHORTHANDED"):
        return default_home - 1, default_away + 1
    return default_home, default_away

def zone_from_faceoff(details):
    if not isinstance(details, dict):
        return None
    z = None
    for k in ("zoneCode", "zone", "rinkSide"):
        if details.get(k) not in (None, ""):
            z = str(details.get(k)).strip().upper()
            break
    if not z:
        return None
    if z in ("O", "OZ", "OFFENSIVE"):
        return "O"
    if z in ("D", "DZ", "DEFENSIVE"):
        return "D"
    return "N"

def choose_first(d, keys):
    for k in keys:
        if k in d and d[k] is not None and str(d[k]) != "":
            return d[k]
    return None

def to_int64_scalar(x):
    if x is None:
        return pd.NA
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return pd.NA
    try:
        return int(v)
    except Exception:
        return pd.NA

def infer_faceoff_zone_home(row):
    try:
        if str(row["type"]).upper() != "FACEOFF":
            return row.get("faceoff_zone_home", pd.NA)
        x = float(row["x"]) if pd.notna(row.get("x")) else 0.0
        p = int(row["period"])
    except Exception:
        return pd.NA
    if abs(x) < 25.0:
        return "N"
    home_attacks_right = (p in (1, 3))
    if home_attacks_right:
        return "O" if x > 0 else "D"
    else:
        return "O" if x < 0 else "D"
    
def infer_faceoff_winner_side(events_g, home_id, away_id):
    if events_g.empty:
        return events_g

    ev = events_g.sort_values(["period", "eventSec", "type"]).reset_index(drop=True)
    ev["fo_winner_side"] = pd.NA

    next_team_idx = [-1] * len(ev)
    last_with_team = -1
    for i in reversed(range(len(ev))):
        team = ev.at[i, "eventTeamId"]
        next_team_idx[i] = last_with_team
        if pd.notna(team):
            last_with_team = i

    for i, r in ev.iterrows():
        if str(r["type"]).upper() != "FACEOFF":
            continue
        j = next_team_idx[i]
        if j == -1:
            continue
        team = ev.at[j, "eventTeamId"]
        if pd.isna(team):
            continue
        try:
            team = int(team)
            if team == int(home_id):
                ev.at[i, "fo_winner_side"] = "H"
            elif team == int(away_id):
                ev.at[i, "fo_winner_side"] = "A"
        except Exception:
            pass

    return ev

def fetch_game_pbp(session, game_id):
    url = f"{web_base}/v1/gamecenter/{int(game_id)}/play-by-play"
    data = get_json(session, url, params=None)
    return data if isinstance(data, dict) else {}

def parse_pbp_events(game_id, pbp_json):
    event_rows = []
    items = []
    if isinstance(pbp_json.get("plays"), list):
        items = pbp_json["plays"]
    elif isinstance(pbp_json.get("items"), list):
        items = pbp_json["items"]

    home = pbp_json.get("homeTeam") or {}
    away = pbp_json.get("awayTeam") or {}
    home_id = home.get("id") or home.get("teamId")
    away_id = away.get("id") or away.get("teamId")

    for p in items:
        etype = (p.get("typeDescKey") or p.get("typeDesc") or "").upper()
        period = p.get("periodDescriptor", {}).get("number") or p.get("period")
        tstr = p.get("timeInPeriod") or p.get("timeRemaining") or p.get("periodTime")
        ev_sec = mmss_to_seconds(tstr)
        details = p.get("details", {}) if isinstance(p.get("details"), dict) else {}
        team_abbr = p.get("teamAbbrev") or (p.get("team", {}) or {}).get("abbrev") or details.get("eventOwnerTeamId")
        team_id = p.get("teamId") or (p.get("team", {}) or {}).get("id") or details.get("eventOwnerTeamId")
        x = p.get("xCoord", details.get("xCoord"))
        y = p.get("yCoord", details.get("yCoord"))
        is_rebound = bool(details.get("isRebound")) if "isRebound" in details else False
        is_rush = bool(details.get("isRush")) if "isRush" in details else False
        strength_raw = details.get("strength") or details.get("situationCode") or p.get("situationCode")
        owner_side = details.get("eventOwnerTeamId")
        try:
            owner_side = int(owner_side)
            if owner_side not in (1, 2):
                owner_side = None
        except Exception:
            owner_side = None

        event_team_id = None
        if owner_side == 1 and home_id is not None:
            event_team_id = int(home_id)
        elif owner_side == 2 and away_id is not None:
            event_team_id = int(away_id)

        faceoff_zone = zone_from_faceoff(details)
        event_rows.append({
            "gameId": int(game_id),
            "period": pd.to_numeric(period, errors="coerce"),
            "eventSec": ev_sec,
            "type": etype,
            "teamAbbrev": team_abbr,
            "teamId": pd.to_numeric(team_id, errors="coerce"),
            "ownerSide": owner_side,
            "eventTeamId": pd.to_numeric(event_team_id, errors="coerce"),
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
            "isRebound": is_rebound,
            "isRush": is_rush,
            "strength": strength_raw,
            "faceoff_zone_home": faceoff_zone
            })

    df = pd.DataFrame(event_rows)
    if df.empty:
        return df

    df["type"] = df["type"].astype(str).str.upper()
    # Use the Fenwick appraoch for calculating xG (disregard blocked shots)
    df["isFenwick"] = df["type"].isin(["SHOT-ON-GOAL", "MISSED-SHOT", "GOAL"])

    def shooter_is_home(row):
        if pd.notna(row.get("ownerSide")):
            try:
                return int(row["ownerSide"]) == 1
            except Exception:
                pass
        if pd.notna(row.get("eventTeamId")) and pd.notna(home_id):
            try:
                return int(row["eventTeamId"]) == int(home_id)
            except Exception:
                pass
        if pd.isna(row.get("x")) or pd.isna(row.get("period")):
            return True
        try:
            x = float(row["x"])
            p = int(row["period"])
            return (x > 0.0) if p in (1, 3) else (x < 0.0)
        except Exception:
            return True

    def att_right(row):
        if pd.isna(row["period"]):
            return True
        p = int(row["period"])
        return (p in (1, 3)) if shooter_is_home(row) else (p in (2, 4, 5))

    df["attRight"] = df.apply(att_right, axis=1)

    shot_like = ["SHOT-ON-GOAL", "MISSED-SHOT", "GOAL"]
    mask_shot = df["type"].isin(shot_like)

    df["xG"] = 0.0
    df["dist"] = np.nan
    df["angle"] = np.nan

    df["x"] = pd.to_numeric(df["x"], errors="coerce").astype("float64")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype("float64")
    df["attRight"] = df["attRight"].fillna(False).astype(bool)

    xs = df.loc[mask_shot, "x"].to_numpy(dtype="float64")
    ys = df.loc[mask_shot, "y"].to_numpy(dtype="float64")
    rs = df.loc[mask_shot, "attRight"].to_numpy(dtype=bool)
    xx = np.where(rs, xs, -xs)
    dx = GOAL_X - xx
    dy = GOAL_Y - ys
    dist = np.hypot(dx, dy)
    ang = np.degrees(np.abs(np.arctan2(dy, dx)))
    df.loc[mask_shot, "dist"] = dist
    df.loc[mask_shot, "angle"] = ang

    def fill_event_team_id(row):
        if pd.notna(row.get("eventTeamId")):
            return row["eventTeamId"]
        try:
            return int(home_id) if shooter_is_home(row) else int(away_id)
        except Exception:
            return pd.NA

    df["eventTeamId"] = df.apply(fill_event_team_id, axis=1)

    home_str, away_str = [], []
    for s in df["strength"].tolist():
        h, a = parse_situation_code(s, 5, 5)
        home_str.append(h)
        away_str.append(a)
    df["home_skaters"] = pd.to_numeric(home_str, errors="coerce")
    df["away_skaters"] = pd.to_numeric(away_str, errors="coerce")
    df["manpower_str"] = df["home_skaters"].astype("Int64").astype(str) + "v" + df["away_skaters"].astype("Int64").astype(str)

    df["pulled_home"] = (df["home_skaters"] >= 6)
    df["pulled_away"] = (df["away_skaters"] >= 6)

    df = df.sort_values(["period", "eventSec"], kind="mergesort").reset_index(drop=True)

    home_score = 0
    away_score = 0
    hs_arr, as_arr = [], []
    sd_home_arr, sd_for_evt_arr = [], []

    for _, r in df.iterrows():
        if r["type"] == "GOAL":
            if shooter_is_home(r):
                home_score += 1
            else:
                away_score += 1
        hs_arr.append(home_score)
        as_arr.append(away_score)
        sd_home_arr.append(home_score - away_score)
        if pd.notna(r["eventTeamId"]):
            if int(r["eventTeamId"]) == int(home_id):
                sd_for_evt_arr.append(home_score - away_score)
            else:
                sd_for_evt_arr.append(away_score - home_score)
        else:
            sd_for_evt_arr.append(pd.NA)

    df["home_score"] = hs_arr
    df["away_score"] = as_arr
    df["score_diff_home"] = sd_home_arr
    df["score_diff_for_event_team"] = sd_for_evt_arr

    def strength_bucket_event_team(row):
        if pd.isna(row.get("home_skaters")) or pd.isna(row.get("away_skaters")) or pd.isna(row.get("eventTeamId")):
            return "EV"
        hs = int(row["home_skaters"])
        as_ = int(row["away_skaters"])
        is_event_home = int(row["eventTeamId"]) == int(home_id)
        pulled_h = bool(row.get("pulled_home"))
        pulled_a = bool(row.get("pulled_away"))

        if hs > 5 and as_ == 5 and pulled_h:
            return "EV"
        if hs == 5 and as_ > 5 and pulled_a:
            return "EV"

        if is_event_home:
            if hs > as_:
                return "PP"
            if hs < as_:
                return "SH"
            return "EV"
        else:
            if as_ > hs:
                return "PP"
            if as_ < hs:
                return "SH"
            return "EV"

    df["strengthBucket"] = df.apply(strength_bucket_event_team, axis=1)

    return df

def normalize_shift_row(game_id, r, season_code):
    player_id = r.get("playerId")
    team_id = r.get("teamId")
    shift_no = r.get("shiftNumber")

    first_name = r.get("firstName")
    last_name = r.get("lastName")
    full_name = r.get("playerName") or r.get("playerFullName")
    team_abbrev = r.get("teamAbbrev") or r.get("teamAbbreviation") or r.get("teamTricode")

    period = r.get("period")
    start_str = choose_first(r, ["startTime", "startTimeInPeriod", "shiftStartTime", "periodTime"])
    end_str = choose_first(r, ["endTime", "endTimeInPeriod", "shiftEndTime"])
    dur_str = r.get("duration")

    start_sec = mmss_to_seconds(start_str)
    end_sec = mmss_to_seconds(end_str)
    dur_sec = mmss_to_seconds(dur_str)

    if pd.isna(dur_sec) and pd.notna(start_sec) and pd.notna(end_sec):
        diff = int(end_sec) - int(start_sec)
        dur_sec = diff if diff >= 0 else np.nan

    out = {
        "seasonId": season_code,
        "gameId": int(game_id),
        "playerId": to_int64_scalar(player_id),
        "teamId": to_int64_scalar(team_id),
        "shiftNumber": to_int64_scalar(shift_no),
        "period": to_int64_scalar(period),
        "startTime": start_str,
        "endTime": end_str,
        "startSec": start_sec,
        "endSec": end_sec,
        "durSec": dur_sec,
        "playerFullName": full_name,
        "firstName": first_name,
        "lastName": last_name,
        "teamAbbrev": team_abbrev
        }
    return out

def coerce_and_dedupe_shifts(df):
    out = df.copy()
    for c in ("gameId", "playerId", "teamId", "shiftNumber", "period", "startSec", "endSec", "durSec"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    if {"gameId", "playerId", "teamId", "shiftNumber"}.issubset(out.columns) and out["shiftNumber"].notna().any():
        out = out.sort_values(["gameId", "playerId", "teamId", "shiftNumber"]).drop_duplicates(
            subset=["gameId", "playerId", "teamId", "shiftNumber"], keep="last"
            )
    else:
        keys = [c for c in ["gameId", "playerId", "teamId", "startSec", "endSec"] if c in out.columns]
        out = out.sort_values(keys).drop_duplicates(subset=keys, keep="last")

    front = [
        "seasonId", "gameId", "playerId", "teamId", "shiftNumber",
        "period", "startTime", "endTime", "startSec", "endSec", "durSec",
        "playerFullName", "firstName", "lastName", "teamAbbrev"
        ]
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]

def ensure_event_columns(df):
    base_cols = [
        "gameId", "period", "eventSec", "type", "teamAbbrev", "teamId", "ownerSide", "eventTeamId",
        "x", "y", "isRebound", "isRush", "strength",
        "faceoff_zone_home", "fo_winner_side",
        "isFenwick", "attRight", "dist", "angle",
        "home_skaters", "away_skaters", "manpower_str", "pulled_home", "pulled_away",
        "home_score", "away_score", "score_diff_home", "score_diff_for_event_team",
        "strengthBucket"
        ]
    out = df.copy()
    defaults = {
        "faceoff_zone_home": pd.NA,
        "fo_winner_side": pd.NA,
        "home_skaters": 0,
        "away_skaters": 0,
        "manpower_str": "5v5",
        "pulled_home": False,
        "pulled_away": False,
        "home_score": 0,
        "away_score": 0,
        "score_diff_home": 0,
        "score_diff_for_event_team": 0,
        "strengthBucket": "EV",
        }
    for c in base_cols:
        if c not in out.columns:
            out[c] = defaults.get(c, pd.NA)
    return out[base_cols]

def fetch_shiftcharts_for_game(session, game_id):
    params = {"cayenneExp": f"gameId={int(game_id)}"}
    try:
        data = get_json(session, f"{stats_rest_base}/en/shiftcharts", params=params)
    except Exception:
        data = None

    segments = data.get("data") if isinstance(data, dict) else None
    if isinstance(segments, list) and segments:
        return segments

    fallback_url = f"{web_base}/v1/shiftcharts/{int(game_id)}"
    try:
        alt = get_json(session, fallback_url, params=None)
        if isinstance(alt, list) and alt:
            return alt
        if isinstance(alt, dict):
            alt_rows = alt.get("data") or alt.get("shifts") or []
            if isinstance(alt_rows, list) and alt_rows:
                return alt_rows
    except Exception:
        pass

    return []

# Basic xG model (Basic Logistic Regression: adheres to industry standard (source: https://evolving-hockey.com/glossary/general-terms/))
def xg_design_matrix(df):
    use = df.copy()
    use["PP"] = (use["strengthBucket"].astype(str).str.upper() == "PP").astype(int)
    use["SH"] = (use["strengthBucket"].astype(str).str.upper() == "SH").astype(int)
    X = np.column_stack([
        pd.to_numeric(use["dist"], errors="coerce").astype(float),
        pd.to_numeric(use["angle"], errors="coerce").astype(float),
        use["isRebound"].astype(int),
        use["isRush"].astype(int),
        use["PP"].astype(int),
        use["SH"].astype(int),
        ])
    names = ["dist", "angle", "isRebound", "isRush", "PP", "SH"]
    return X, names

def fit_xg_model(events_df):
    fen = events_df[events_df["isFenwick"] == True].copy()
    if fen.empty:
        return None, None

    fen = fen.dropna(subset=["dist", "angle"])
    fen["isGoal"] = (fen["type"].astype(str).str.upper() == "GOAL").astype(int)

    if fen["isGoal"].sum() == 0:
        return None, None

    X, names = xg_design_matrix(fen)
    y = fen["isGoal"].to_numpy()
    model = LogisticRegression(penalty="l2", C=10.0, solver="lbfgs", max_iter=2000, n_jobs=None)
    model.fit(X, y)
    coef = pd.DataFrame({"feature": names + ["intercept"], "coef": list(model.coef_[0]) + [model.intercept_[0]]})
    return model, coef

def apply_xg_model(events_df, model):
    out = events_df.copy()
    out["xG"] = 0.0
    mask = (out["isFenwick"] == True) & out["dist"].notna() & out["angle"].notna()
    if mask.any():
        X, _ = xg_design_matrix(out.loc[mask])
        out.loc[mask, "xG"] = model.predict_proba(X)[:, 1]
    return out

def attribute_xg_to_shifts(shifts_g, events_g):
    if shifts_g.empty:
        shifts_g["shift_xGF"] = 0.0
        shifts_g["shift_xGA"] = 0.0
        shifts_g["fenwickF"] = 0
        shifts_g["fenwickA"] = 0
        return shifts_g

    shifts = shifts_g.copy()
    shifts["shift_xGF"] = 0.0
    shifts["shift_xGA"] = 0.0
    shifts["fenwickF"] = 0
    shifts["fenwickA"] = 0
    if events_g.empty:
        return shifts

    shot_events = events_g[events_g["isFenwick"] == True].copy()
    if shot_events.empty:
        return shifts
    
   # Loop by game, then by period
    for gid, ev_g in shot_events.groupby("gameId", sort=False):
        if pd.isna(gid):
            continue

        shifts_game = shifts[shifts["gameId"] == gid]
        if shifts_game.empty:
            continue

        for p in sorted(ev_g["period"].dropna().astype(int).unique()):
            events_period = ev_g[ev_g["period"] == p]
            shifts_period = shifts_game[shifts_game["period"] == p]
            if shifts_period.empty or events_period.empty:
                continue

            for _, event in events_period.iterrows():
                es = event["eventSec"]
                if pd.isna(es):
                    continue

                on_ice = shifts_period[(shifts_period["startSec"].le(es)) & (shifts_period["endSec"].gt(es))]
                if on_ice.empty:
                    continue

                team_id_event = event.get("eventTeamId")
                if team_id_event is None or pd.isna(team_id_event):
                    continue

                try:
                    team_id_event = int(team_id_event)
                except Exception:
                    continue

                shooters = on_ice[on_ice["teamId"] == team_id_event]
                defenders = on_ice[on_ice["teamId"] != team_id_event]

                if not shooters.empty:
                    shifts.loc[shooters.index, "shift_xGF"] += float(event["xG"])
                    shifts.loc[shooters.index, "fenwickF"] += 1
                if not defenders.empty:
                    shifts.loc[defenders.index, "shift_xGA"] += float(event["xG"])
                    shifts.loc[defenders.index, "fenwickA"] += 1

    return shifts

# Builder
def build_onice_timelines(shifts_g, home_id, away_id, skater_positions=None, skater_ids=None):
    if shifts_g.empty:
        return {}

    if skater_positions is None:
        skater_positions = {}
    if skater_ids is None:
        skater_ids = set()

    team_groups = {}
    for tid, sub in shifts_g.groupby("teamId", sort=False):
        tid = int(tid)
        team_groups[tid] = sub.sort_values(["period", "startSec", "endSec"])

    def build_period_segments(team_df):
        out = {}
        for p, seg in team_df.groupby("period", sort=False):
            p = int(p)
            period_end = 1200.0 if p in (1, 2, 3) else 600.0

            if seg.empty:
                out[p] = []
                continue

            bounds = {0.0, period_end}
            if "startSec" in seg.columns:
                bounds.update(seg["startSec"].dropna().astype(float).tolist())
            if "endSec" in seg.columns:
                bounds.update(seg["endSec"].dropna().astype(float).tolist())
            tgrid = sorted(x for x in bounds if 0.0 <= x <= period_end)

            pid = seg["playerId"].astype("Int64").to_numpy()
            s0 = seg["startSec"].astype(float).to_numpy()
            s1 = seg["endSec"].astype(float).to_numpy()

            segments = []
            for i in range(len(tgrid) - 1):
                a, b = tgrid[i], tgrid[i + 1]
                mask = (s0 < b) & (s1 > a)

                current_pids = [int(x) for x in pid[mask] if pd.notna(x)]

                skaters_set = set()
                goalie_present = False
                for pid_now in current_pids:
                    pos = skater_positions.get(pid_now)
                    is_goalie = (pid_now not in skater_ids) or (pos == "G")
                    if not is_goalie:
                        skaters_set.add(pid_now)
                    else:
                        goalie_present = True

                segments.append((a, b, skaters_set, goalie_present))
            out[p] = segments
        return out

    home_timelines = build_period_segments(team_groups.get(home_id, shifts_g.iloc[0:0]))
    away_timelines = build_period_segments(team_groups.get(away_id, shifts_g.iloc[0:0]))

    periods = sorted(set(home_timelines.keys()) | set(away_timelines.keys()))
    merged = {}
    for p in periods:
        merged[p] = {"home": home_timelines.get(p, []), "away": away_timelines.get(p, [])}
    return merged

def annotate_events_with_onice(events_g, timelines, home_id, away_id):
    if events_g.empty:
        for c in ("home_skaters", "away_skaters", "manpower_str", "pulled_home", "pulled_away"):
            events_g[c] = 0 if c in ("home_skaters", "away_skaters") else ("5v5" if c == "manpower_str" else False)
        return events_g

    ev = events_g.copy()
    ev["home_skaters"] = 0
    ev["away_skaters"] = 0
    ev["pulled_home"] = False
    ev["pulled_away"] = False
    ev["manpower_str"] = "5v5"

    def lookup_segment(seg_list, t):
        for (a, b, sk_set, gk) in seg_list:
            if a <= t < b or (t == 0 and a == 0):
                return len(sk_set), bool(gk)
        return 0, False

    for idx, r in ev.iterrows():
        try:
            p = int(r.get("period"))
            t = float(r.get("eventSec"))
        except Exception:
            continue
        block = timelines.get(p)
        if not block:
            continue
        h_cnt, h_g = lookup_segment(block["home"], t)
        a_cnt, a_g = lookup_segment(block["away"], t)

        ev.at[idx, "home_skaters"] = int(h_cnt)
        ev.at[idx, "away_skaters"] = int(a_cnt)
        ev.at[idx, "pulled_home"] = (not h_g) and (h_cnt >= 6)
        ev.at[idx, "pulled_away"] = (not a_g) and (a_cnt >= 6)
        ev.at[idx, "manpower_str"] = f"{h_cnt}v{a_cnt}"

    return ev

def scrape_shifts_one_regular_season(start_year, out_root=output_root):
    season_code = format_season_code(start_year)
    season_folder = format_season_folder(start_year)
    print(f"Scraping Shifts/PBP for {season_folder}")

    session = build_session()
    skater_positions, skater_ids = load_skater_position_map(out_root, season_folder)
    sched_df, tg_path = build_teamgames_from_schedules(session, season_code, season_folder)

    game_ids = []
    if isinstance(sched_df, pd.DataFrame) and "id" in sched_df.columns:
        try:
            tmp = pd.to_numeric(sched_df["id"], errors="coerce").dropna().astype(int).tolist()
            game_ids = sorted(set(tmp))
        except Exception:
            game_ids = []

    if not game_ids and tg_path:
        try:
            tg_df = pd.read_csv(tg_path)
            tmp = pd.to_numeric(tg_df["gameId"], errors="coerce").dropna().astype(int).tolist()
            game_ids = sorted(set(tmp))
        except Exception:
            game_ids = []

    if not game_ids:
        print(f"No gameIds found for {season_code}")
        return {"shifts_csv": None, "events_csv": None}

    segments = []
    event_rows = []
    empty_pbp_game_ids = []
    empty_shift_game_ids = []

    for game_id in game_ids:
        raw_rows = fetch_shiftcharts_for_game(session, game_id)
        
        if raw_rows:
            game_shifts = pd.DataFrame([normalize_shift_row(game_id, r, season_code) for r in raw_rows])
            game_shifts = coerce_and_dedupe_shifts(game_shifts)
            segments.extend(game_shifts.to_dict(orient="records"))
        else:
            game_shifts = pd.DataFrame()
            empty_shift_game_ids.append(int(game_id))

        pbp_json = fetch_game_pbp(session, game_id)
        game_events = parse_pbp_events(game_id, pbp_json)

        home_id = (pbp_json.get("homeTeam") or {}).get("id") or (pbp_json.get("homeTeam") or {}).get("teamId")
        away_id = (pbp_json.get("awayTeam") or {}).get("id") or (pbp_json.get("awayTeam") or {}).get("teamId")

        if game_events is not None and not game_events.empty and pd.notna(home_id) and pd.notna(away_id):
            if not game_shifts.empty:
                timelines = build_onice_timelines(game_shifts, int(home_id), int(away_id), skater_positions=skater_positions, skater_ids=skater_ids)
                game_events = annotate_events_with_onice(game_events, timelines, int(home_id), int(away_id))
            game_events["faceoff_zone_home"] = game_events.apply(infer_faceoff_zone_home, axis=1)
            game_events = infer_faceoff_winner_side(game_events, int(home_id), int(away_id))
            game_events = ensure_event_columns(game_events)

        if game_events is not None and not game_events.empty:
            event_rows.append(game_events)
        else:
            empty_pbp_game_ids.append(int(game_id))

    # Missing logs
    if empty_shift_game_ids:
        miss_path = Path(out_root) / season_folder / "missing logs" / f"missing_shiftcharts_{season_folder}.csv"
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gameId": empty_shift_game_ids}).to_csv(miss_path, index=False)

    # Build final shifts dataframe
    if segments:
        shifts_df = pd.DataFrame(segments)
        shifts_df = coerce_and_dedupe_shifts(shifts_df)
    else:
        shifts_df = pd.DataFrame(columns=[
            "seasonId", "gameId", "playerId", "teamId", "shiftNumber", "period",
            "startTime", "endTime", "startSec", "endSec", "durSec",
            "playerFullName", "firstName", "lastName", "teamAbbrev"
            ])

    events_base_cols = [
        "gameId","period","eventSec","type","teamAbbrev","teamId","ownerSide","eventTeamId",
        "x","y","isRebound","isRush","strength",
        "faceoff_zone_home","fo_winner_side",
        "isFenwick","attRight","dist","angle",
        "home_skaters","away_skaters","manpower_str","pulled_home","pulled_away",
        "home_score","away_score","score_diff_home","score_diff_for_event_team",
        "strengthBucket"
        ]

    def is_effectively_empty(df):
        if df is None or not isinstance(df, pd.DataFrame):
            return True
        if df.empty:
            return True
        non_na = int(df.notna().to_numpy().sum())
        return non_na == 0

    clean_rows = []
    for df in event_rows:
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        if df.empty:
            continue

        df2 = df.dropna(axis=1, how="all")
        df2 = df2.dropna(axis=0, how="all")
        if df2.empty:
            continue

        if int(df2.notna().to_numpy().sum()) == 0:
            continue

        clean_rows.append(df2)

    if clean_rows:
        events_df = pd.concat(clean_rows, ignore_index=True, sort=False)
        events_df = events_df.reindex(columns=events_base_cols, fill_value=pd.NA)
    else:
        events_df = pd.DataFrame(columns=events_base_cols)


    if not events_df.empty:
        xg_model, xg_coef = fit_xg_model(events_df)
        events_df["xG"] = 0.0
        if xg_model is not None:
            events_df = apply_xg_model(events_df, xg_model)
            coef_path = Path(out_root) / season_folder / "model coefficients" / f"xG_model_coeffs_{season_folder}.csv"
            coef_path.parent.mkdir(parents=True, exist_ok=True)
            xg_coef.to_csv(coef_path, index=False)
        else:
            events_df["xG"] = 0.0
    else:
        events_df["xG"] = 0.0

    # Attribute xG to shifts
    if not shifts_df.empty and not events_df.empty:
        shifts_df = attribute_xg_to_shifts(shifts_df, events_df)

    sdir = Path(out_root) / season_folder
    sdir.mkdir(parents=True, exist_ok=True)

    shifts_path = sdir / f"shifts_{season_folder}.csv"
    shifts_df.to_csv(shifts_path, index=False)

    events_cols_final = events_base_cols + ["xG"]
    events_df = events_df.reindex(columns=events_cols_final)
    events_path = sdir / f"pbp_events_{season_folder}.csv"
    events_df.to_csv(events_path, index=False)

    return {"shifts_csv": str(shifts_path), "events_csv": str(events_path)}

# Season Driver
def scrape_regular_seasons_range(start_year, end_year, out_root=output_root):
    results = {}
    all_players_this_run = []
    all_bios_this_run = []

    for season_start in list_seasons(start_year, end_year):
        season_result = scrape_one_regular_season(season_start, out_root=out_root)
        results[season_start] = season_result

        players_df = season_result.get("players_df")
        if isinstance(players_df, pd.DataFrame) and not players_df.empty:
            all_players_this_run.append(players_df)

        bios_df = season_result.get("bios_df")
        if isinstance(bios_df, pd.DataFrame) and not bios_df.empty:
            all_bios_this_run.append(bios_df)

    write_players_master_snapshot(all_players_this_run, bios_df_list=all_bios_this_run, master_path=master_file)
    print("API data saved.")
    return results

def scrape_shifts_range(start_year, end_year, out_root=output_root):
    results = {}
    for season_start in list_seasons(start_year, end_year):
        try:
            season_result = scrape_shifts_one_regular_season(season_start, out_root=out_root)
            results[season_start] = season_result
        except Exception as e:
            print(f"Season {season_start}/{season_start+1} failed: {e}")
    
    print("Shift/PBP data saved.")
    return results

# Main
def main():
    run_api = True
    run_shifts = False
    if run_api:
        scrape_regular_seasons_range(start_year, end_year)
        
    if run_shifts:
        scrape_shifts_range(start_year, end_year, out_root=output_root)

if __name__ == "__main__":
    main()