# -*- coding: utf-8 -*-
"""
Clean raw NST and Contract data, fixing mojibake naming errors and standardizing names. Also maps
API playerIds to NST master files to match player production with specific players, and connects them in the contract
data as well.

NOTE: I had AI assistance with the mojibake character cleaning, as it was not something I have ever
encountered before.
"""
# Imports
import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

# Config
script_dir = Path(__file__).resolve().parent
data_root = script_dir.parent / "Data"
raw_root = data_root / "Raw Data"
clean_root = data_root / "Clean Data"

start_year = 2015
end_year = 2025

players_master_file = clean_root / "NHL_PLAYERS_MASTER.csv"
contracts_master_file = clean_root / "NHL_CONTRACTS_MASTER.csv"

long_to_std = {
    "Anaheim Ducks":"ANA","Boston Bruins":"BOS","Buffalo Sabres":"BUF","Calgary Flames":"CGY",
    "Carolina Hurricanes":"CAR","Chicago Blackhawks":"CHI","Colorado Avalanche":"COL","Columbus Blue Jackets":"CBJ",
    "Dallas Stars":"DAL","Detroit Red Wings":"DET","Edmonton Oilers":"EDM","Florida Panthers":"FLA",
    "Los Angeles Kings":"LAK","Minnesota Wild":"MIN","Montreal Canadiens":"MTL","Nashville Predators":"NSH",
    "New Jersey Devils":"NJD","New York Islanders":"NYI","New York Rangers":"NYR","Ottawa Senators":"OTT",
    "Philadelphia Flyers":"PHI","Pittsburgh Penguins":"PIT","San Jose Sharks":"SJS","Seattle Kraken":"SEA",
    "St Louis Blues":"STL","Tampa Bay Lightning":"TBL","Toronto Maple Leafs":"TOR","Vancouver Canucks":"VAN",
    "Vegas Golden Knights":"VGK","Washington Capitals":"WSH","Winnipeg Jets":"WPG","Utah Hockey Club":"UTA",
    "Arizona Coyotes":"ARI",
    "T.B":"TBL","S.J":"SJS","L.A":"LAK","N.J":"NJD","T.B.":"TBL","S.J.":"SJS","L.A.":"LAK","N.J.":"NJD"
    }

onice_totals = ["GF","GA","xGF","xGA","CF","CA","FF","FA","SF","SA","HDCF","HDCA","SCF","SCA","HDGF","HDGA"]
indiv_totals = ["iCF","iFF","iSF","iXG","iHDCF","iSCF","iG","iP","iA1","iA2"]

min_toi_for_rates_seconds = 300.0
es_max = 10.0
pp_max = 25.0
pk_max = 25.0

suffix_list = {"jr","jr.","sr","sr.","ii","iii","iv","v"}

full_name_replacement_list = {
    "mat?j blumel":"matej blumel",
    "mat j blumel":"matej blumel",
    "alex nylander":"alexander nylander",
    "alex petrovic":"alexander petrovic",
    "frederick gaudreau":"freddy gaudreau",
    "john jack roslovic":"jack roslovic",
    "john roslovic":"jack roslovic",
    "joe veleno":"joseph veleno",
    "josh norris":"joshua norris",
    "joshua mahura":"josh mahura",
    "luke glendening":"lucas glendening",
    "matt grzelcyk":"matthew grzelcyk",
    "maxwell crozier":"max crozier",
    "mitch marner":"mitchell marner",
    "tommy novak":"thomas novak",
    "vinnie hinostroza":"vincent hinostroza",
    "zac jones":"zachary jones",
    "zack bolduc":"zachary bolduc",
    "matt dumba":"mathew dumba",
    "christopher tanev":"chris tanev",
    "matt benning":"matthew benning",
    "pat maroon":"patrick maroon",
    "louie belpedio":"louis belpedio",
    "zach sanford":"zachary sanford",
    "william borgen":"will borgen",
    "nicholas merkley":"nick merkley",
    "callan foote":"cal foote",
    "alex kerfoot":"alexander kerfoot",
    "alexei toropchenko":"alexey toropchenko",
    "axel sandin pellikka":"axel sandin-pellikka",
    "c j suess":"cj suess",
    "charles-alexis legault":"charles alexis legault",
    "chris didomenico":"christopher didomenico",
    "danil yurtaikin":"danil yurtaykin",
    "fyodor svechkov":"fedor svechkov",
    "gabe chiarot":"ben chiarot",
    "j t brown":"jt brown",
    "jeremie poirier":"emile poirier",
    "joe labate":"joseph labate",
    "josh brown":"joshua brown",
    "joshua dunne":"josh dunne",
    "lucas pettersson":"elias pettersson",
    "matthew beniers":"matty beniers",
    "nikita okhotyuk":"nikita okhotiuk",
    "nikolay prokhorkin":"nikolai prokhorkin",
    "philip kemp":"phil kemp",
    "sammy blais":"samuel blais",
    "samuel poulin":"sam poulin",
    "samuel walker":"sammy walker",
    "sergei kalinin":"sergey kalinin",
    "t j brennan":"tj brennan",
    "timothy gettinger":"tim gettinger",
    "tj tynan":"t j tynan",
    "vasiliy ponomarev":"vasily ponomarev",
    "vladimir tkachyov":"vladimir tkachev",
    "will bitten":"william bitten",
    "yegor korshkov":"egor korshkov",
    "yegor zamula":"egor zamula",
    "michael matheson":"mike matheson",
    "mike cammalleri":"michael cammalleri",
    "nick shore":"nicholas shore",
    "anthony deangelo":"tony deangelo",
    "cristoval nieves":"boo nieves",
    "daniel o'regan":"danny o'regan",
    "nicholas baptiste":"nick baptiste",
    "jacob middleton":"jake middleton",
    "max comtois":"maxime comtois",
    "gerald mayhew":"gerry mayhew",
    "alex wennberg":"alexander wennberg",
    "alexander chmelevski":"sasha chmelevski",
    "janis moser":"j j moser",
    "nathan smith":"nate smith",
    "nicholas paul":"nick paul",
    "jacob lucchini":"jake lucchini",
    "nick abruzzese":"nicholas abruzzese",
    "cameron atkinson":"cam atkinson",
    "benoit-olivier groulx":"bo groulx",
    "cameron hillis":"cam hillis",
    "zach hayes":"zack hayes",
    "zachary hayes":"zack hayes",
    "janis jerome moser": "j j moser",
    "nicklaus perbix": "nick perbix",
    "cameron york": "cam york",
    "matthew stienburg": "matt stienburg",
    "john-jason peterka": "jj peterka",
    "cal burke": "callahan burke",
    "cameron lund": "cam lund",
    "matthew savoie": "matt savoie",
    "artyom zub": "artem zub",
    }

# Helpers
def repair_mojibake(x):
    try:
        return str(x).encode("latin1").decode("utf-8")
    except Exception:
        return str(x)

def strip_accents(x):
    s = unicodedata.normalize("NFKD", str(x))
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def canonical_name(raw):
    if pd.isna(raw):
        return ""
    s = repair_mojibake(str(raw))
    s = s.replace("\u2013","-").replace("\u2014","-").replace("\u2212","-")
    s = s.replace("\u2018","'").replace("\u2019","'")
    s = re.sub(r"\s+", " ", s).strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            s = parts[1] + " " + parts[0] + (" " + " ".join(parts[2:]) if len(parts) > 2 else "")
            s = re.sub(r"\s+", " ", s).strip()
    s = strip_accents(s)
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = full_name_replacement_list.get(s, s)
    toks = s.split()
    while toks and toks[-1].strip(".") in suffix_list:
        toks = toks[:-1]
    return " ".join(toks)

def title_from_canon(canon):
    out = []
    for tok in canon.split():
        parts = []
        for chunk in re.split(r"([\-'])", tok):
            parts.append(chunk if chunk in ("-","'") else chunk.capitalize())
        out.append("".join(parts))
    return " ".join(out)

def split_first_last(full):
    tokens = str(full).split()
    if len(tokens) >= 2:
        return tokens[0], " ".join(tokens[1:])
    if len(tokens) == 1:
        return tokens[0], ""
    return "", ""

def find_nst_seasons():
    roots = [
        raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Even Strength" / "Counts",
        raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Power Play" / "Counts",
        raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Penalty Kill" / "Counts",
        ]
    found = set()
    pat = re.compile(r"(\d{4}-\d{4})")
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            m = pat.search(path.name)
            if m:
                found.add(m.group(1))
    return sorted(found)


def parse_season_start(s):
    m = re.search(r"(\d{4})", str(s))
    return int(m.group(1)) if m else None

def filter_seasons(seasons, y0, y1):
    out = []
    for s in seasons:
        k = parse_season_start(s)
        if k is not None and y0 <= k <= y1:
            out.append(s)
    return sorted(set(out), key=parse_season_start)

def nst_paths(season):
    return {
        "ONICE_EV_COUNTS": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Even Strength" / "Counts" / f"NST Skaters On-Ice Even Strength Counts {season}.csv"),
        "ONICE_EV_RATES": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Even Strength" / "Rates" / f"NST Skaters On-Ice Even Strength Rates {season}.csv"),
        "ONICE_EV_REL": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Even Strength" / "Relative" / f"NST Skaters On-Ice Even Strength Relative {season}.csv"),
        "INDIV_EV_COUNTS": (raw_root / "NaturalStatTrick" / "Skaters" / "Individual" / "Even Strength" / "Counts" / f"NST Skaters Individual Even Strength Counts {season}.csv"),
        "INDIV_EV_RATES": (raw_root / "NaturalStatTrick" / "Skaters" / "Individual" / "Even Strength" / "Rates" / f"NST Skaters Individual Even Strength Rates {season}.csv" ),

        "ONICE_PP_COUNTS": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Power Play" / "Counts" / f"NST Skaters On-Ice Power Play Counts {season}.csv"),
        "ONICE_PP_RATES": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Power Play" / "Rates" / f"NST Skaters On-Ice Power Play Rates {season}.csv"),
        "ONICE_PP_REL": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Power Play" / "Relative" / f"NST Skaters On-Ice Power Play Relative {season}.csv"),
        "INDIV_PP_COUNTS": (raw_root / "NaturalStatTrick" / "Skaters" / "Individual" / "Power Play" / "Counts" / f"NST Skaters Individual Power Play Counts {season}.csv"),
        "INDIV_PP_RATES": (raw_root / "NaturalStatTrick" / "Skaters" / "Individual" / "Power Play" / "Rates" / f"NST Skaters Individual Power Play Rates {season}.csv"),

        "ONICE_PK_COUNTS": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Penalty Kill" / "Counts" / f"NST Skaters On-Ice Penalty Kill Counts {season}.csv"),
        "ONICE_PK_RATES": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Penalty Kill" / "Rates" / f"NST Skaters On-Ice Penalty Kill Rates {season}.csv"),
        "ONICE_PK_REL": (raw_root / "NaturalStatTrick" / "Skaters" / "On-Ice" / "Penalty Kill" / "Relative" / f"NST Skaters On-Ice Penalty Kill Relative {season}.csv"),
        "INDIV_PK_COUNTS": (raw_root / "NaturalStatTrick" / "Skaters" / "Individual" / "Penalty Kill" / "Counts" / f"NST Skaters Individual Penalty Kill Counts {season}.csv"),
        "INDIV_PK_RATES": (raw_root / "NaturalStatTrick" / "Skaters" / "Individual" / "Penalty Kill" / "Rates" / f"NST Skaters Individual Penalty Kill Rates {season}.csv"),

        "OUT_DIR": clean_root / season,
        "PGS": (clean_root / season) / f"player_game_stats_{season}.csv",
        }

def clean_nst_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(r"/60\b", "60", regex=True)
        .str.replace("%", "Pct", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(" per 60", "60", regex=False)
        .str.replace(" Per 60", "60", regex=False)
        .str.replace(" per60", "60", regex=False)
        .str.replace(" Rel", "_Rel", regex=False)
        .str.strip()
        )
    drop_me = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_me:
        df = df.drop(columns=drop_me)
    return df

def pick_rate_cols(df, keep_prefix=""):
    if df is None:
        return []
    cols = []
    for c in df.columns:
        lc = c.lower()
        if lc.startswith(keep_prefix):
            if lc.endswith("60"):
                cols.append(c)
    return cols

def try_read_csv(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    df = pd.read_csv(path)
    df = clean_nst_columns(df)
    if "Player" in df.columns:
        df["Player_canon"] = df["Player"].apply(canonical_name)
    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str)
    if "Position" in df.columns:
        df["Position"] = df["Position"].astype(str)
    return df

def normalize_team_code(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in long_to_std:
        return long_to_std[s]
    s2 = s.replace(" .",".").replace(". ", ".").strip()
    if s2 in long_to_std:
        return long_to_std[s2]
    sc = s.upper()
    if len(sc) in (2,3):
        return sc
    return s

def mmss_to_seconds(time):
    try:
        time_string = str(time).strip()
        if ":" in time_string:
            p = time_string.split(":")
            if len(p) == 2:
                mm, ss = p
                return int(mm)*60 + int(ss)
            if len(p) == 3:
                hh, mm, ss = p
                return int(hh)*3600 + int(mm)*60 + int(ss)
        return float(time_string)
    except Exception:
        return np.nan

def parse_toi_seconds(series):
    s = series.astype(str)
    is_clock = (s.str.contains(":", na=False)).mean() >= 0.3
    if is_clock:
        return s.apply(mmss_to_seconds)
    return pd.to_numeric(series, errors="coerce") * 60.0

def rename_relative_columns(df):
    df = df.copy()
    new_cols = {}
    for c in df.columns:
        lc = c.lower()
        if lc.endswith("_rel") or lc.startswith("rel_") or " rel" in lc:
            base = c.replace(" rel","",1).replace(" Rel","",1)
            if not base.lower().startswith("rel_") and not base.lower().endswith("_rel"):
                new_cols[c] = f"rel_{base}"
    if new_cols:
        df = df.rename(columns=new_cols)
    return df

def drop_all_na_and_constant(df, skip=()):
    if df is None or len(df) == 0:
        return df
    na_cols = [c for c in df.columns if df[c].isna().all()]
    if na_cols:
        df = df.drop(columns=na_cols)
    const = []
    for c in df.columns:
        if c in skip:
            continue
        vals = df[c].dropna().unique()
        if len(vals) <= 1:
            const.append(c)
    if const:
        df = df.drop(columns=const)
    return df

def unique_in_order(values):
    seen, out = set(), []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

# Name cleaning in player_game_stats
def clean_player_game_stats(pgs_path):
    pgs_path = Path(pgs_path)
    if not pgs_path.exists():
        return
    df = pd.read_csv(pgs_path, low_memory=False)
    name_col = next((c for c in ["skaterFullName","playerFullName"] if c in df.columns), None)
    if not name_col:
        return
    raw = df[name_col].astype(str).map(lambda s: strip_accents(repair_mojibake(s))).str.strip()
    canon = raw.map(canonical_name)
    disp = canon.map(title_from_canon)
    for c in ["skaterFullName","playerFullName"]:
        if c in df.columns:
            df[c] = disp
    if "firstName" in df.columns or "lastName" in df.columns:
        f, l = zip(*disp.map(split_first_last))
        if "firstName" in df.columns: df["firstName"] = list(f)
        if "lastName" in df.columns: df["lastName"] = list(l)
    if "playerId" in df.columns:
        df["playerId"] = pd.to_numeric(df["playerId"], errors="coerce").astype("Int64")
    df.to_csv(pgs_path, index=False)

def build_name_to_id_map(pgs_path):
    pgs_path = Path(pgs_path)
    if not pgs_path.exists():
        return
    pgs = pd.read_csv(pgs_path, low_memory=False)
    if {"firstName","lastName"}.issubset(pgs.columns):
        names = (pgs["firstName"].fillna("").astype(str) + " " + pgs["lastName"].fillna("").astype(str)).str.strip()
    else:
        name_col = next((c for c in ["skaterFullName","playerFullName"] if c in pgs.columns), None)
        if not name_col:
            return {}
        names = pgs[name_col].astype(str)
    keys = names.apply(canonical_name)
    ids = pd.to_numeric(pgs.get("playerId"), errors="coerce")
    out = {}
    for k, pid in zip(keys, ids):
        if k and pd.notna(pid) and k not in out:
            out[k] = int(pid)
    return out

# Aggregation
def aggregate_counts(df, situation):
    if df is None:
        return None
    df = df[df.get("Position","").str.upper().ne("G") | df["Position"].isna()].copy()
    sec = parse_toi_seconds(df["TOI"]) if "TOI" in df.columns else pd.Series(np.nan, index=df.index)
    df["TOI_seconds"] = sec
    df["TeamStd"] = df["Team"].apply(normalize_team_code) if "Team" in df.columns else np.nan
    for c in df.columns:
        if c not in ["Player","Player_canon","Team","Position","TeamStd"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    grp = df.groupby(["Player","Player_canon","Position"], as_index=False).agg(
        {c:"sum" for c in df.columns if c not in ["Player","Player_canon","Team","Position","TeamStd"]})
    teams = (df.groupby(["Player","Player_canon","Position"])["TeamStd"].apply(lambda s: ",".join(unique_in_order(s.map(normalize_team_code).tolist()))).reset_index(name="Teams_List"))
    def first_team(s):
        if not isinstance(s,str) or not s: return np.nan
        parts = [t.strip() for t in s.split(",") if t.strip()]
        return parts[0] if parts else np.nan
    teams["TeamStd_Primary"] = teams["Teams_List"].apply(first_team)
    out = grp.merge(teams, on=["Player","Player_canon","Position"], how="left")
    out["Situation"] = situation
    return out

def aggregate_relative(df_rel, df_counts, situation):
    if df_rel is None:
        return None
    df_rel = df_rel[df_rel.get("Position","").str.upper().ne("G") | df_rel["Position"].isna()].copy()
    if "Position" not in df_rel.columns and df_counts is not None and "Position" in df_counts.columns:
        pos_map = (df_counts.groupby("Player")["Position"].agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else np.nan))
        df_rel["Position"] = df_rel["Player"].map(pos_map)
    elif "Position" not in df_rel.columns:
        return None
    sec = parse_toi_seconds(df_rel["TOI"]) if "TOI" in df_rel.columns else pd.Series(np.nan, index=df_rel.index)
    df_rel["TOI_seconds_rel"] = sec
    df_rel = rename_relative_columns(df_rel)
    rel_cols = [c for c in df_rel.columns if c.lower().startswith("rel_")]
    need = ["Player","Player_canon","Position","TOI_seconds_rel"] + rel_cols
    df_rel = df_rel[need].copy()
    for c in ["TOI_seconds_rel"] + rel_cols:
        df_rel[c] = pd.to_numeric(df_rel[c], errors="coerce")
    def wavg(g):
        tot = g["TOI_seconds_rel"].sum()
        out = {}
        for c in rel_cols:
            v, w = g[c], g["TOI_seconds_rel"]
            out[c] = (v*w).sum()/tot if pd.notna(tot) and tot>0 else v.mean(skipna=True)
        return pd.Series(out)
    rel = df_rel.groupby(["Player","Player_canon","Position"])[["TOI_seconds_rel"]+rel_cols].apply(wavg).reset_index()
    rel["Situation"] = situation
    return rel

def aggregate_individual(df_ind, situation):
    if df_ind is None:
        return None
    df_ind = df_ind[df_ind.get("Position","").str.upper().ne("G") | df_ind["Position"].isna()].copy()
    sec = parse_toi_seconds(df_ind["TOI"]) if "TOI" in df_ind.columns else pd.Series(np.nan, index=df_ind.index)
    df_ind["TOI_seconds_indiv"] = sec
    key = ["Player","Player_canon","Team","Position"]
    for c in df_ind.columns:
        if c not in key:
            df_ind[c] = pd.to_numeric(df_ind[c], errors="coerce")
    agg_cols = [c for c in df_ind.columns if c not in key]
    grp = df_ind.groupby(["Player","Player_canon","Position"], as_index=False)[agg_cols].sum(min_count=1)
    grp["Situation"] = situation
    return grp

def per60(df, toi_col, names):
    if df is None or toi_col not in df.columns:
        return df
    df = df.copy()
    hrs = (df[toi_col].astype(float) / 3600.0).replace(0, np.nan)
    for c in names:
        if c in df.columns:
            base = pd.to_numeric(df[c], errors="coerce")
            df[f"{c}_per60"] = base / hrs
    return df

# Situation Block
def build_situation_master(paths, season, pref, label, name_to_id):
    on_counts = try_read_csv(paths[f"ONICE_{pref}_COUNTS"], required=True)
    on_rel = try_read_csv(paths.get(f"ONICE_{pref}_REL",""))
    ind_cnt = try_read_csv(paths.get(f"INDIV_{pref}_COUNTS",""))
    on_rates = try_read_csv(paths.get(f"ONICE_{pref}_RATES",""))
    ind_rates = try_read_csv(paths.get(f"INDIV_{pref}_RATES",""))

    a = aggregate_counts(on_counts, label)
    b = aggregate_relative(on_rel, a, label)
    c = aggregate_individual(ind_cnt, label)

    m = a.copy()
    if b is not None:
        rel_cols = [x for x in b.columns if x not in ["Player","Player_canon","Position","Situation"]]
        m = m.merge(b[["Player","Player_canon","Position"] + rel_cols],
                    on=["Player","Player_canon","Position"], how="left")
    if c is not None:
        keep = [x for x in c.columns if x != "Situation"]
        m = m.merge(c[keep], on=["Player","Player_canon","Position"], how="left", suffixes=("","_ind"))

    if on_rates is not None:
        join_cols = ["Player","Player_canon","Position"]
        if "Player" in on_rates.columns and "Player_canon" in on_rates.columns:
            rate_cols = pick_rate_cols(on_rates)
            if rate_cols:
                m = m.merge(on_rates[join_cols + rate_cols], on=join_cols, how="left", suffixes=("",""))
        else:
            on_rates = on_rates.copy()
            if "Player" in on_rates.columns and "Player_canon" not in on_rates.columns:
                on_rates["Player_canon"] = on_rates["Player"].apply(canonical_name)
            if "Position" not in on_rates.columns and "Position" in m.columns:
                pos_map = m.set_index("Player")["Position"] if "Player" in m.columns else None
                if pos_map is not None:
                    on_rates["Position"] = on_rates["Player"].map(pos_map)
            if {"Player","Player_canon","Position"}.issubset(on_rates.columns):
                rate_cols = pick_rate_cols(on_rates)
                if rate_cols:
                    m = m.merge(on_rates[["Player","Player_canon","Position"] + rate_cols],
                                on=["Player","Player_canon","Position"], how="left")

    if ind_rates is not None:
        join_cols = ["Player","Player_canon","Position"]
        if "Player" in ind_rates.columns and "Player_canon" in ind_rates.columns:
            rate_cols = pick_rate_cols(ind_rates)
            if rate_cols:
                ind_block = ind_rates[join_cols + rate_cols].copy()
                ind_block.columns = [*join_cols] + [f"{c}_ind" if c in m.columns else c for c in rate_cols]
                m = m.merge(ind_block, on=join_cols, how="left")
        else:
            ind_rates = ind_rates.copy()
            if "Player" in ind_rates.columns and "Player_canon" not in ind_rates.columns:
                ind_rates["Player_canon"] = ind_rates["Player"].apply(canonical_name)
            if "Position" not in ind_rates.columns and "Position" in m.columns:
                pos_map = m.set_index("Player")["Position"] if "Player" in m.columns else None
                if pos_map is not None:
                    ind_rates["Position"] = ind_rates["Player"].map(pos_map)
            if {"Player","Player_canon","Position"}.issubset(ind_rates.columns):
                rate_cols = pick_rate_cols(ind_rates)
                if rate_cols:
                    ind_block = ind_rates[["Player","Player_canon","Position"] + rate_cols].copy()
                    ind_block.columns = ["Player","Player_canon","Position"] + [
                        f"{c}_ind" if c in m.columns else c for c in rate_cols
                    ]
                    m = m.merge(ind_block, on=["Player","Player_canon","Position"], how="left")

    need_onice = [c for c in onice_totals if f"{c}_per60" not in m.columns and f"{c}60" not in m.columns]
    need_indiv = [c for c in indiv_totals if f"{c}_per60" not in m.columns and f"{c}60" not in m.columns]
    if need_onice:
        m = per60(m, "TOI_seconds", need_onice)
    if need_indiv:
        m = per60(m, "TOI_seconds_indiv", need_indiv)

    m = m[m["Position"].str.upper().ne("G") | m["Position"].isna()].copy()
    if name_to_id:
        m["playerId"] = m["Player_canon"].map(name_to_id).astype("Int64")
    m = drop_all_na_and_constant(m, skip={"Player","Player_canon","Position","Situation","playerId","TOI_seconds"})
    m["Season"] = season
    m["Situation"] = label
    if "Player_canon" in m.columns:
        m = m.rename(columns={"Player": "Player_raw"})
        m["Player"] = m["Player_raw"].apply(lambda s: strip_accents(repair_mojibake(s)))
        m = m.drop(columns=["Player_canon"])
        
    rename_map = {
        "CF60": "CF_per60", "CA60": "CA_per60",
        "xGF60": "xGF_per60", "xGA60": "xGA_per60",
        "SF60": "SF_per60", "SA60": "SA_per60",
        "FF60": "FF_per60", "FA60": "FA_per60",
        "SCF60":"SCF_per60","SCA60":"SCA_per60",
        "HDCF60":"HDCF_per60","HDCA60":"HDCA_per60",
        "GF60": "GF_per60", "GA60": "GA_per60",
        "iCF60":"iCF_per60","iFF60":"iFF_per60","iSF60":"iSF_per60","iXG60":"iXG_per60",
        "iHDCF60":"iHDCF_per60","iSCF60":"iSCF_per60","iG60":"iG_per60","iP60":"iP_per60",
        "iA1_60":"iA1_per60","iA2_60":"iA2_per60"
        }
    m = m.rename(columns={c: rename_map[c] for c in m.columns if c in rename_map})
    
    return m

def situation_max(label):
    if label == "Even Strength": return es_max
    if label == "Power Play": return pp_max
    if label == "Penalty Kill": return pk_max
    return es_max

def flag_outliers(df, label):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    th = situation_max(label)
    rows = []
    for r in df.itertuples(index=False):
        t = getattr(r, "TOI_seconds", np.nan)
        v = getattr(r, "xGF_per60", np.nan)
        bad = False
        why = []
        if pd.isna(t) or t <= 0: bad = True
        why.append("zero_or_nan_TOI")
        if pd.notna(t) and float(t) < min_toi_for_rates_seconds and pd.notna(v) and abs(float(v)) > th:
            bad = True 
            why.append("small_TOI_high_rate")
        if pd.notna(v) and float(v) > th*3:
            bad = True 
            why.append("extreme_xGF60")
        if bad:
            rows.append({"Player":getattr(r,"Player",None),"Situation":label,"TOI_seconds":float(t) if pd.notna(t) else np.nan,
                         "xGF_per60":float(v) if pd.notna(v) else np.nan,"Reasons":";".join(why)})
    return pd.DataFrame(rows)

def attach_demographics_from_players_master(contracts_df: pd.DataFrame, players_master_path: Path):
    if contracts_df is None or contracts_df.empty:
        return contracts_df

    if "playerId" not in contracts_df.columns:
        return contracts_df

    if not players_master_path.exists():
        return contracts_df

    players = pd.read_csv(players_master_path, low_memory=False)
    if "playerId" not in players.columns:
        return contracts_df

    keep_cols = ["playerId"]
    for col in ["shootsCatches", "height", "weight"]:
        if col in players.columns:
            keep_cols.append(col)

    demo = players[keep_cols].copy()
    merged = contracts_df.merge(demo, on="playerId", how="left")
    merged["Shot"] = merged["shootsCatches"]
    merged["H(f)"] = merged["height"]
    merged["W(lbs)"] = merged["weight"]

    for c in ["shootsCatches", "height", "weight"]:
        if c in merged.columns:
            merged.drop(columns=c, inplace=True)

    return merged

# Season Driver
def process_season(season):
    paths = nst_paths(season)
    out_dir = Path(paths["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_player_game_stats(paths["PGS"])
    season_map = build_name_to_id_map(paths["PGS"])
    global_map = {}
    if players_master_file.exists():
        api = pd.read_csv(players_master_file, low_memory=False)
    
        if "fullName" in api.columns:
            names = api["fullName"].astype(str)
        elif {"firstName", "lastName"}.issubset(api.columns):
            names = (api["firstName"].fillna("").astype(str) + " " + api["lastName"].fillna("").astype(str)).str.strip()
        elif "playerFullName" in api.columns:
            names = api["playerFullName"].astype(str)
        elif "skaterFullName" in api.columns:
            names = api["skaterFullName"].astype(str)
        elif "name" in api.columns:
            names = api["name"].astype(str)
        else:
            names = pd.Series([], dtype=str)
    
        keys = names.apply(canonical_name)
        ids = pd.to_numeric(api.get("playerId"), errors="coerce")
    
        for key, pid in zip(keys, ids):
            if key and pd.notna(pid) and key not in global_map:
                global_map[key] = int(pid)

    name_to_id = global_map.copy()
    name_to_id.update(season_map)
    es = build_situation_master(paths, season, "EV", "Even Strength", name_to_id)
    pp = build_situation_master(paths, season, "PP", "Power Play", name_to_id)
    pk = build_situation_master(paths, season, "PK", "Penalty Kill", name_to_id)

    if es is not None:
        es.to_csv(out_dir / f"NST_player_master_ES_{season}.csv", index=False)
    if pp is not None: 
        pp.to_csv(out_dir / f"NST_player_master_PP_{season}.csv", index=False)
    if pk is not None: 
        pk.to_csv(out_dir / f"NST_player_master_PK_{season}.csv", index=False)

    outlier_frames = []
    if es is not None:
        o = flag_outliers(es, "Even Strength")
        if o is not None and not o.empty:
            outlier_frames.append(o)
    if pp is not None:
        o = flag_outliers(pp, "Power Play")
        if o is not None and not o.empty:
            outlier_frames.append(o)
    if pk is not None:
        o = flag_outliers(pk, "Penalty Kill")
        if o is not None and not o.empty:
            outlier_frames.append(o)
    
    if outlier_frames:
        outliers = pd.concat(outlier_frames, ignore_index=True)
        missing_dir = out_dir / "missing logs"
        missing_dir.mkdir(parents=True, exist_ok=True)
        outliers.to_csv(missing_dir / f"NST_outliers_{season}.csv", index=False)

def update_contracts_master(contracts_csv=contracts_master_file, api_master_csv=players_master_file):
    contracts_path = Path(contracts_csv).resolve()
    api_path = Path(api_master_csv).resolve()
    api = pd.read_csv(str(api_path), low_memory=False)
    api_name = api["fullName"].astype(str)
    api_keys = api_name.apply(canonical_name)
    api_pids = pd.to_numeric(api["playerId"], errors="coerce")
    api_pos = api["positionCode"].astype(str)
    name_pos_to_id = {}
    name_only_to_id = {}
    for k, p, pid in zip(api_keys, api_pos, api_pids):
        if k and pd.notna(pid):
            up = str(p).strip().upper() if isinstance(p, str) else ""
            if up:
                if (k, up) not in name_pos_to_id:
                    name_pos_to_id[(k, up)] = int(pid)
            if k not in name_only_to_id:
                name_only_to_id[k] = int(pid)

    df = pd.read_csv(str(contracts_path), low_memory=False)
    if "Skaters" not in df.columns:
        return

    df["Skaters_raw"] = df["Skaters"].astype(str)
    df["canon_key"] = (df["Skaters_raw"] .map(lambda s: canonical_name(strip_accents(repair_mojibake(s)))) .fillna(""))
    df["Skaters"] = df["canon_key"].map(title_from_canon).str.strip()
    miss_idx = []
    if "Pos" in df.columns:
        pos_series = df["Pos"].astype(str).str.strip().str.upper()
        pid = []
        for i, (ck, cp) in enumerate(zip(df["canon_key"], pos_series)):
            v = name_pos_to_id.get((ck, cp), name_only_to_id.get(ck))
            if v is None:
                pid.append(np.nan)
                miss_idx.append(i)
            else:
                pid.append(int(v))
        df["playerId"] = pd.Series(pid, dtype="Int64")
    else:
        df["playerId"] = df["canon_key"].map(name_only_to_id).astype("Int64")
        miss_idx = [i for i in df.index if pd.isna(df["playerId"].iloc[i])]

    if miss_idx:
        unresolved_path = contracts_path.parent / "contracts_unresolved_playerId.csv"
        cols_to_log = ["Skaters_raw", "Skaters", "canon_key"]
        if "Pos" in df.columns:
            cols_to_log.insert(2, "Pos")
        df.loc[miss_idx, cols_to_log].to_csv(str(unresolved_path), index=False)

    keep_cols = [c for c in df.columns if c not in ["Skaters_raw", "canon_key"]]
    tmp_path = contracts_path.with_suffix(".csv.tmp")
    df[keep_cols].to_csv(tmp_path, index=False, encoding="utf-8")
    tmp_path.replace(contracts_path)

def main():
    seasons = filter_seasons(find_nst_seasons(), start_year, end_year)
    for s in seasons:
        process_season(s)

    update_contracts_master(contracts_master_file, players_master_file)
    contracts_df = pd.read_csv(contracts_master_file, low_memory=False)
    contracts_df = attach_demographics_from_players_master(contracts_df, players_master_file)
    contracts_df.to_csv(contracts_master_file, index=False, encoding="utf-8-sig")
    print("Finished cleaning run.")

if __name__ == "__main__":
    main()
