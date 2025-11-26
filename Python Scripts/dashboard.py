# -*- coding: utf-8 -*-
"""
NHL Contract Dashboard that packages predictions and GAR calculations nicely into an interactive Dashboard

NOTE: I had AI's help in learning Dash over the past 6 months, and had it help with some of the debugging, especially when it came
to plotting the multiple parts together on the same visual.
"""
# Imports
from pathlib import Path
import webbrowser, threading, time
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, dash_table, Input, Output
import plotly.graph_objs as go

# Config
project_root = Path(__file__).resolve().parents[1]
data_root = project_root / "Data" / "Clean Data"
data_files = {
    "contracts": data_root / "2025-2026" / "contract_results_2025-2026.csv",
    "garwar": data_root / "2024-2025" / "Skater_GAR_WAR_2024-2025.csv",
    "nst_team": data_root / "2024-2025" / "NST_player_master_ES_2024-2025.csv",
    "value_2324": data_root / "2023-2024" / "player_value_2023-2024.csv",
    "value_2425": data_root / "2024-2025" / "player_value_2024-2025.csv",
    "value_2526": data_root / "2025-2026" / "player_value_2025-2026.csv",
    }

cap_table = {
    "2015-2016": 71400000, "2016-2017": 73000000, "2017-2018": 75000000,
    "2018-2019": 79500000, "2019-2020": 81500000, "2020-2021": 81500000,
    "2021-2022": 81500000, "2022-2023": 82500000, "2023-2024": 83500000,
    "2024-2025": 88000000, "2025-2026": 95500000, "2026-2027": 104000000,
    "2027-2028": 113500000
    }
cap_growth_limit = 0.05

# Fill in cap table through 2035 using natural growth rate
last_cap_start_year = int(list(cap_table.keys())[-1].split("-")[0])
for year in range(last_cap_start_year + 1, 2035):
    prev_season_key = f"{year-1}-{year}"
    season_key = f"{year}-{year+1}"
    cap_table[season_key] = round(cap_table[prev_season_key] * (1 + cap_growth_limit))
 
terms_list = list(range(1, 9))

gar_to_spar_ratio = 2.9245                  # 2024-2025 GAR -> SPAR conversion
spar_dollar_rate_2025 = 1571475.16220797    # $ per SPAR for 2025-26
league_min_aav = 750000.0

# [historical line, band fill, projected line]
team_colors = {
    "ANA": ["#F47A38", "#F9C8A8", "#006272"],
    "ARI": ["#8C2633", "#C88B96", "#E2D6B5"],
    "BOS": ["#FFB81C", "#FFE08A", "#000000"],
    "BUF": ["#003087", "#8AA5DA", "#FFC72C"],
    "CGY": ["#C8102E", "#F6A4AE", "#F1BE48"],
    "CAR": ["#CC0000", "#F3A3A3", "#000000"],
    "CHI": ["#CF0A2C", "#F2A4AF", "#000000"],
    "COL": ["#6F263D", "#C09BAC", "#236192"],
    "CBJ": ["#002654", "#89A3CC", "#CE1126"],
    "DAL": ["#006847", "#87C9B0", "#8F8F8C"],
    "DET": ["#CE1126", "#F3A3AE", "#000000"],
    "EDM": ["#041E42", "#8EA0BF", "#FF4C00"],
    "FLA": ["#041E42", "#8EA0BF", "#C8102E"],
    "LAK": ["#111111", "#BDBDBD", "#A2AAAD"],
    "MIN": ["#154734", "#86BAA7", "#A6192E"],
    "MTL": ["#AF1E2D", "#EEA5AE", "#192168"],
    "NSH": ["#FFB81C", "#FFE08A", "#041E42"],
    "NJD": ["#CE1126", "#F3A3AE", "#000000"],
    "NYI": ["#0038A8", "#8FB2EB", "#F47D30"],
    "NYR": ["#0038A8", "#8FB2EB", "#CE1126"],
    "OTT": ["#D10A0A", "#F2A2A2", "#000000"],
    "PHI": ["#F74902", "#FFC2A6", "#000000"],
    "PIT": ["#000000", "#D8D2C4", "#FFB81C"],
    "SJS": ["#006D75", "#94CBCD", "#000000"],
    "SEA": ["#001628", "#94B9C7", "#99D9D9"],
    "STL": ["#002F87", "#8FA8DA", "#FFB81C"],
    "TBL": ["#002868", "#8FA6D9", "#FFFFFF"],
    "TOR": ["#00205B", "#8EA2C8", "#FFFFFF"],
    "VAN": ["#001F5B", "#8FA6CF", "#00843D"],
    "VGK": ["#B4975A", "#E0D2AF", "#333F48"],
    "WSH": ["#041E42", "#8EA0BF", "#C8102E"],
    "WPG": ["#041E42", "#8EA0BF", "#7B303E"],
    "UTA": ["#0C2340", "#A3B1C6", "#2EB36C"],
    "XXX": ["#444444", "#CCCCCC", "#1f77b4"],
    }

column_min_height = "910px"
graph_height_px = 740

# Helpers
def require_cols(df, needed, src):
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise RuntimeError

def to_int_safe(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def percent_to_fraction(x):
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v / 100.0

def season_start_int(s):
    try:
        return int(str(s).split("-")[0])
    except Exception:
        return 0

def shift_season(season_str, delta_years):
    try:
        a = int(season_str.split("-")[0])
        return f"{a+delta_years}-{a+delta_years+1}"
    except Exception:
        return season_str

def projection_season_keys_inclusive(start_season, n_years):
    keys = list(cap_table.keys())
    if start_season in keys:
        si = keys.index(start_season)
        return [keys[min(si+i, len(keys)-1)] for i in range(0, n_years)]
    return []

def percentile_to_color(p):
    if pd.isna(p): return "rgb(200,200,200)"
    p = max(0.0, min(100.0, float(p)))
    if p <= 50:
        t = p / 50.0
        r = 255 - int(127*t) 
        g = int(128*t)
        b = int(128*t)
    else:
        t = (p-50.0)/50.0
        r = 128 - int(128*t) 
        g = 128 - int(128*t)
        b = 128 + int(127*t)
    return f"rgb({r},{g},{b})"

def map_player_value_columns(df):
    require_cols(df, ["playerId","statsSeason","p$SPAR"], "player_value")
    out = df.rename(columns={"p$SPAR":"SPAR_dollars"})
    out["playerId"] = pd.to_numeric(out["playerId"], errors="coerce")
    out["SPAR_dollars"] = pd.to_numeric(out["SPAR_dollars"], errors="coerce")
    return out

def hex_to_rgba(hex_color, alpha):
    if not isinstance(hex_color, str):
        return f"rgba(0,0,0,{alpha})"
    s = hex_color.lstrip("#")
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except Exception:
        r = g = b = 0
    return f"rgba({r},{g},{b},{alpha})"

def load_all_data():
    contracts_df = pd.read_csv(data_files["contracts"], low_memory=False)
    gar_war_df = pd.read_csv(data_files["garwar"], low_memory=False)
    nst_es_df = pd.read_csv(data_files["nst_team"], low_memory=False)

    pv_2324 = map_player_value_columns(pd.read_csv(data_files["value_2324"], low_memory=False))
    pv_2425 = map_player_value_columns(pd.read_csv(data_files["value_2425"], low_memory=False))
    pv_2526 = map_player_value_columns(pd.read_csv(data_files["value_2526"], low_memory=False))
    player_value_df = pd.concat([pv_2324, pv_2425, pv_2526], ignore_index=True)
    player_value_df["curr_season"] = player_value_df["statsSeason"].apply(season_start_int)
    player_value_df = player_value_df.sort_values(["playerId", "curr_season"]).drop(columns="curr_season")

    for frame in [contracts_df, gar_war_df, nst_es_df]:
        if "playerId" in frame.columns:
            frame["playerId"] = pd.to_numeric(frame["playerId"], errors="coerce")

    # GAR/WAR + percentiles
    require_cols(gar_war_df, ["PlayerID", "GAR_total"], "Skater_GAR_WAR_2024-2025.csv")
    gar_war_df = gar_war_df.rename(columns={"PlayerID": "playerId"})
    pct_cols = [c for c in gar_war_df.columns if str(c).startswith("pct_")]

    if pct_cols:
        gar_pct = gar_war_df[["playerId"] + pct_cols].copy()
    else:
        gar_pct = gar_war_df[["playerId"]].copy()

    # Trim contracts to what the dashboard actually needs
    len_cols = [c for c in contracts_df.columns if str(c).startswith("capPct_len")]
    if not len_cols:
        raise RuntimeError("Contracts file must contain capPct_len1..capPct_len8.")

    keep_cols = [
        "playerId", "PlayerName", "Pos", "Shot", "Weight_lb", "Height_in", "Signing_Age",
        "Start_Year", "Length", "Start_Yr_Cap_Pct", "RoleBucket"
        ]
    keep_cols = [c for c in keep_cols if c in contracts_df.columns]
    comp_cols = [c for c in contracts_df.columns if c.startswith("comp")]
    proj_cols = [
        c for c in contracts_df.columns
        if c.startswith("proj_GAR_total_year")
        or c.startswith("proj_GAR_total_low_year")
        or c.startswith("proj_GAR_total_high_year")
        ]

    contracts_trim = contracts_df[keep_cols + len_cols + comp_cols + proj_cols].copy()
    require_cols(nst_es_df, ["playerId", "TeamStd_Primary"], "NST_player_master_ES_2024-2025.csv")
    nst_trim = (nst_es_df[["playerId", "TeamStd_Primary"]].drop_duplicates("playerId").rename(columns={"TeamStd_Primary": "Team"})) 
    master = (gar_war_df.merge(nst_trim, on="playerId", how="left").merge(contracts_trim, on="playerId", how="left"))
    if "RoleBucket_x" in master.columns or "RoleBucket_y" in master.columns:
        role = None
        if "RoleBucket_x" in master.columns:
            role = master["RoleBucket_x"]
        elif "RoleBucket_y" in master.columns:
            role = master["RoleBucket_y"]

        if role is not None:
            master["RoleBucket"] = role

        drop_cols = [c for c in ["RoleBucket_x", "RoleBucket_y"] if c in master.columns]
        master = master.drop(columns=drop_cols)

    player_list = contracts_trim[["playerId", "PlayerName", "Pos"]].copy()

    player_list["playerId"] = pd.to_numeric(player_list["playerId"], errors="coerce")
    player_list = player_list.dropna(subset=["playerId"])

    player_list["PlayerName"] = (
        player_list["PlayerName"]
        .astype(str)
        .str.strip()
        )
    player_list = player_list[player_list["PlayerName"] != ""]

    # Deduplicate by playerId
    player_list = player_list.drop_duplicates(subset=["playerId"]).copy()
    player_list = player_list.sort_values("PlayerName")

    options = []
    for _, r in player_list.iterrows():
        pid = int(r["playerId"])
        nm = r["PlayerName"]
        pos = r.get("Pos", "")
        label = f"{nm} ({pos})" if isinstance(pos, str) and pos else nm
        options.append({"label": label, "value": pid})

    return master, options, pct_cols, player_value_df, gar_pct

app = Dash(__name__)
player_master_df, dropdown_options, pct_columns, player_value, GAR_percentile = load_all_data()
default_pid = dropdown_options[0]["value"] if dropdown_options else None
tile_base_style = {
    "minWidth": "175px", "height": "110px",
    "display": "flex", "flexDirection": "column",
    "justifyContent": "center", "alignItems": "center",
    "borderRadius": "16px", "boxShadow": "0 4px 16px rgba(0,0,0,0.15)",
    "color": "white", "textAlign": "center", "padding": "10px"
    }

app.layout = html.Div([
    html.H2("NHL Contract Dashboard", style={"textAlign":"center","marginBottom":"6px"}),

    html.Div([
        html.Div([
            html.Label("Select Player"),
            dcc.Dropdown(id="player-dropdown", options=dropdown_options,value=default_pid, searchable=True, clearable=False)], style={"minWidth":"260px","flex":"1"}),

        html.Div([
            html.Label("Units"),
            dcc.Dropdown(id="units-dropdown", options=[{"label":"Cap %", "value":"cap"}, {"label":"AAV ($)", "value":"usd"}], value="cap", clearable=False)], style={"minWidth":"160px","marginLeft":"10px"}),

        html.Div([
            html.Label("Term"),
            dcc.Slider(id="term-slider", min=1, max=8, step=1, value=1, marks={i:str(i) for i in terms_list})], style={"minWidth":"320px","marginLeft":"10px","flex":"1"})], 
                style={"display":"flex","flexWrap":"wrap","gap":"8px","alignItems":"center","margin":"0 10px"}),

    html.Div([
        html.Div([
            html.Div(id="bio-card", style={"marginTop":"10px"}),
            html.Div(id="pct-tiles-grid", style={"display":"flex","flexWrap":"wrap","gap":"12px", "justifyContent":"center","marginTop":"8px"}),
            html.Div(id="contract-summary", style={"marginTop":"6px","fontWeight":"600"}),
            dash_table.DataTable(id="comps-table", style_table={"overflowX":"auto"}, style_cell={"textAlign":"left","padding":"4px","fontSize":"12px"})], style={"width":"30%","minWidth":"360px","display":"inline-block","verticalAlign":"top",
                  "padding":"10px","minHeight":column_min_height,"boxSizing":"border-box"}),

        html.Div([
            dcc.Graph(id="value-plot", style={"height": f"{graph_height_px}px"}),
            html.Div([
                dcc.Checklist(
                    id="overlay-check",
                    options=[
                        {"label":" Show projection uncertainty band", "value":"projband"},
                        {"label":" Show contract band", "value":"contract"},
                        ],
                    value=["projband","contract"],
                    inputStyle={"marginRight":"6px"},
                    style={"marginTop":"6px"}),
                html.Div(id="surplus-box",style={"marginTop":"8px","padding":"8px","borderRadius":"10px", "background":"#f5f5f5","fontWeight":"600"})], style={"marginTop":"4px"})], 
                    style={"width":"68%","minWidth":"560px","display":"inline-block","verticalAlign":"top", "padding":"10px","minHeight":column_min_height,"boxSizing":"border-box"})
        ], style={"display":"flex","gap":"10px","alignItems":"stretch"})
], style={"paddingBottom":"12px"})

@app.callback(
    Output("term-slider", "value"),
    Input("player-dropdown", "value"),
    prevent_initial_call=False
    )

def default_term(player_id):
    player_rows = player_master_df[player_master_df["playerId"] == player_id]
    if player_rows.empty: 
        return 1
    row = player_rows.iloc[0]

    if "Length" in row.index and pd.notna(row["Length"]) and to_int_safe(row["Length"], 0) in terms_list:
        return to_int_safe(row["Length"], 1)
    return 1

@app.callback(
    [
        Output("bio-card","children"),
        Output("pct-tiles-grid","children"),
        Output("contract-summary","children"),
        Output("comps-table","data"), Output("comps-table","columns"),
        Output("value-plot","figure"),
        Output("surplus-box","children"),
    ],
    [
        Input("player-dropdown","value"),
        Input("units-dropdown","value"),
        Input("term-slider","value"),
        Input("overlay-check","value")
    ]
)

def update(player_id, units, active_term, overlays):
    player_rows = player_master_df[player_master_df["playerId"] == player_id]
    if player_rows.empty: 
        player_rows = player_master_df.iloc[[0]]
    row = player_rows.iloc[0]
    term = to_int_safe(active_term, 0)
    role_col = next((c for c in row.index if str(c).lower()=="rolebucket"), None)

    pos = row.get("Pos", "NA")
    shot = row.get("Shot", "NA")
    if isinstance(shot, str) and shot: 
        shot = shot.strip().upper()[:1]
    team = row.get("Team", "XXX")
    name = row.get("PlayerName", f"playerId {int(player_id) if pd.notna(player_id) else 'NA'}")
    age = row.get("Signing_Age", np.nan)
    try: 
        age = float(age)
    except Exception: 
        age = np.nan
    h_in = row.get("Height_in", "NA")
    w_lb = row.get("Weight_lb", "NA")
    role = row.get(role_col, "NA") if role_col else "NA"
    bio = html.Div([
        html.H3(name),
        html.P(f"Pos: {pos} | Shot: {shot} | Team: {team}"),
        html.P(f"Age at signing: {age if pd.notna(age) else 'NA'} | H: {h_in} in | W: {w_lb} lb"),
        html.P(f"Role: {role}")
        ])

    # Percentiles
    tiles = []
    gar_row = GAR_percentile[ GAR_percentile["playerId"] == player_id ]
    if gar_row.empty:
        gar_source = row
    else:
        gar_source = gar_row.iloc[0]

    for col in pct_columns:
        val = gar_source.get(col, np.nan)
        color = percentile_to_color(val)
        label = col.replace("pct_", "").replace("_", " ").title()
        txt = "N/A" if pd.isna(val) else f"{float(val):.1f}th"
        tiles.append(html.Div([
            html.Div(label, style={"fontSize":"13px","opacity":0.95}),
            html.Div(txt, style={"fontSize":"28px","fontWeight":"800","marginTop":"4px"})
            ], style=dict(tile_base_style, **{"backgroundColor": color})))
        
    player_value_row = player_value[player_value["playerId"] == player_id].copy()
    if "statsSeason" in player_value_row.columns:
        player_value_row["stat_season_temp"] = player_value_row["statsSeason"].apply(season_start_int)
        player_value_row = player_value_row.sort_values("stat_season_temp").drop(columns="stat_season_temp")
    hist_color, shade_color, proj_color = team_colors.get(team, ["#000000","#CCCCCC","#1f77b4"])

    traces = []
    hist_last_usd = None
    hist_S_cap_pct = np.nan
    start_season = str(row.get("Start_Year", "2025-2026")) if "Start_Year" in row else "2025-2026"
    if "-" not in start_season: start_season = "2025-2026"
    season_S = shift_season(start_season, -1)

    if not player_value_row.empty and "SPAR_dollars" in player_value_row.columns:
        if units == "usd":
            y_vals = player_value_row["SPAR_dollars"]
            hist_last_usd = player_value_row["SPAR_dollars"].iloc[-1]
        else:
            y_vals = player_value_row.apply(
                lambda r: (r["SPAR_dollars"] / cap_table.get(str(r["statsSeason"]), np.nan)) * 100.0
                if pd.notna(r["SPAR_dollars"]) and str(r["statsSeason"]) in cap_table else np.nan,axis=1)
        traces.append(go.Scatter(
            x=player_value_row["statsSeason"].astype(str).tolist(), y=y_vals.tolist(), mode="lines+markers", name="Historical",
            line=dict(color=hist_color, width=3), marker=dict(color=hist_color)
            ))
        rS = player_value_row[player_value_row["statsSeason"] == season_S]
        if not rS.empty:
            sparS = float(rS.iloc[0]["SPAR_dollars"])
            capS = float(cap_table.get(season_S, np.nan))
            if capS > 0: hist_S_cap_pct = (sparS / capS) * 100.0
            if units == "usd": hist_last_usd = sparS

    proj_rows = []
    for k in range(1, 10):
        gar_mid = row.get(f"proj_GAR_total_year{k}", np.nan)
        gar_lo = row.get(f"proj_GAR_total_low_year{k}", np.nan)
        gar_hi = row.get(f"proj_GAR_total_high_year{k}", np.nan)
        if pd.isna(gar_mid):
            continue

        season_k = shift_season(start_season, k - 1)
        cap_k = cap_table.get(season_k, np.nan)
        spar_mid = (gar_mid / gar_to_spar_ratio)
        spar_lo = (gar_lo / gar_to_spar_ratio) if not pd.isna(gar_lo) else np.nan
        spar_hi = (gar_hi / gar_to_spar_ratio) if not pd.isna(gar_hi) else np.nan
        usd_mid = spar_mid * spar_dollar_rate_2025
        usd_lo = spar_lo * spar_dollar_rate_2025 if not pd.isna(spar_lo) else np.nan
        usd_hi = spar_hi * spar_dollar_rate_2025 if not pd.isna(spar_hi) else np.nan
        
        if not pd.isna(usd_mid) and usd_mid < league_min_aav:
            usd_mid = league_min_aav
        if not pd.isna(usd_lo) and usd_lo < league_min_aav:
            usd_lo = league_min_aav
        if not pd.isna(usd_hi) and usd_hi < league_min_aav:
            usd_hi = league_min_aav

        # Dollars -> cap % (equivalent cap fraction at that season's cap)
        if pd.notna(cap_k) and cap_k > 0:
            cap_pct_mid = (usd_mid / cap_k) * 100.0
            cap_pct_lo = (usd_lo / cap_k) * 100.0 if not pd.isna(usd_lo) else np.nan
            cap_pct_hi = (usd_hi / cap_k) * 100.0 if not pd.isna(usd_hi) else np.nan
        else:
            cap_pct_mid = np.nan
            cap_pct_lo = np.nan
            cap_pct_hi = np.nan

        proj_rows.append({
            "statsSeason": season_k,
            "value_usd": usd_mid,
            "value_usd_low": usd_lo,
            "value_usd_high": usd_hi,
            "value_cap_pct": cap_pct_mid,
            "value_cap_pct_low": cap_pct_lo,
            "value_cap_pct_high": cap_pct_hi,
            })

    proj_df = pd.DataFrame(proj_rows)

    if not proj_df.empty:
        if units == "usd":
            proj_x_list = proj_df["statsSeason"].astype(str).tolist()
            proj_y_list = proj_df["value_usd"].astype(float).tolist()
            proj_lo_list = proj_df["value_usd_low"].astype(float).fillna(proj_df["value_usd"]).tolist()
            proj_hi_list = proj_df["value_usd_high"].astype(float).fillna(proj_df["value_usd"]).tolist()
        else:
            proj_x_list = proj_df["statsSeason"].astype(str).tolist()
            proj_y_list = proj_df["value_cap_pct"].astype(float).tolist()
            proj_lo_list = proj_df["value_cap_pct_low"].astype(float).fillna(proj_df["value_cap_pct"]).tolist()
            proj_hi_list = proj_df["value_cap_pct_high"].astype(float).fillna(proj_df["value_cap_pct"]).tolist()
    else:
        proj_x_list = []
        proj_y_list = []
        proj_lo_list = []
        proj_hi_list = []

    # dotted projected curve
    traces.append(go.Scatter(x=proj_x_list, y=proj_y_list, mode="lines+markers", name="Projected", line=dict(color=proj_color, width=3, dash="dot")))
    if pd.notna(hist_S_cap_pct) and len(proj_x_list) > 0 and units == "cap":
        traces.append(go.Scatter(
            x=[str(season_S), proj_x_list[0]],
            y=[float(hist_S_cap_pct), float(proj_y_list[0])],
            mode="lines", name="S→S+1",
            line=dict(color=proj_color, width=2, dash="dot"), showlegend=False
            ))
    if hist_last_usd is not None and pd.notna(hist_last_usd) and len(proj_x_list) > 0 and units == "usd":
        traces.append(go.Scatter(
            x=[str(season_S), proj_x_list[0]],
            y=[float(hist_last_usd), float(proj_y_list[0])],
            mode="lines", name="S→S+1",
            line=dict(color=proj_color, width=2, dash="dot"), showlegend=False
            ))

    # Uncertainty band
    if "projband" in (overlays or []):
        traces.append(go.Scatter(x=proj_x_list, y=proj_lo_list, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False))
        traces.append(go.Scatter(x=proj_x_list, y=proj_hi_list, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)", width=0),
                                 fill="tonexty", fillcolor="rgba(0,0,0,0.12)", name="Projection range"))

    fig = go.Figure(data=traces)
    s_mark_x, s_mark_y, s_lbl = [], [], []
    if not player_value_row.empty and "SPAR_dollars" in player_value_row.columns:
        for delta, tag in [(-2,"S-2"), (-1,"S-1"), (0,"S")]:
            tgt = shift_season(start_season, delta)
            r = player_value_row[player_value_row["statsSeason"] == tgt]
            if not r.empty:
                if units == "usd":
                    yv = float(r.iloc[0]["SPAR_dollars"])
                else:
                    cap = cap_table.get(tgt, np.nan)
                    yv = float(r.iloc[0]["SPAR_dollars"]) / float(cap) * 100.0 if pd.notna(cap) else np.nan
                if pd.notna(yv):
                    s_mark_x.append(str(tgt))
                    s_mark_y.append(yv)
                    s_lbl.append(tag)
    if s_mark_x:
        fig.add_trace(go.Scatter(
            x=s_mark_x, y=s_mark_y, mode="markers+text", text=s_lbl, textposition="top center",
            name="S-2/S-1/S", marker=dict(size=9, color="rgba(0,0,0,0.55)")
            ))

    # Contract AAV band (cap % varies by season, AAV fixed in USD)
    cap_frac_selected = percent_to_fraction(row.get(f"capPct_len{active_term}"))
    term_contract = to_int_safe(active_term, 0)

    if "contract" in (overlays or []) and pd.notna(cap_frac_selected) and term_contract > 0:
        # Anchor AAV off 2025-2026 cap
        anchor_cap = cap_table.get("2025-2026", cap_table.get(start_season, np.nan))
        if pd.notna(anchor_cap):
            aav_usd_contract = float(cap_frac_selected) * float(anchor_cap)
            seasons_in_contract = projection_season_keys_inclusive(start_season, term_contract + 1)
            x_line, y_line = [], []
            for i, s in enumerate(seasons_in_contract):
                is_last_boundary = (i == len(seasons_in_contract) - 1)

                if is_last_boundary and y_line:
                    x_line.append(s)
                    y_line.append(y_line[-1])
                    continue

                cap_s = cap_table.get(s, np.nan)
                if pd.isna(cap_s) or cap_s <= 0:
                    continue

                if units == "cap":
                    y_val = (aav_usd_contract / cap_s) * 100.0
                else:
                    y_val = aav_usd_contract

                x_line.append(s)
                y_line.append(y_val)

            if len(x_line) == 1:
                s0 = x_line[0]
                s1 = shift_season(s0, 1)
                x_line.append(s1)
                y_line.append(y_line[0])

            if x_line:
                band_rgb = team_colors.get(team, ["#000000", "#CCCCCC", "#1f77b4"])[1]
                band_fill = hex_to_rgba(band_rgb, 0.2)

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    fill="tozeroy",
                    fillcolor=band_fill
                    ))
                
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines+markers",
                    name="Contract AAV",
                    line=dict(color=band_rgb, width=2),
                    marker=dict(size=5)
                    ))

    cols = list(row.index)
    idxs = []
    for i in range(1, 10+1):
        need = [f"comp{i}_PlayerName", f"comp{i}_Start_Year", f"comp{i}_Length", f"comp{i}_CapPct"]
        if all(c in cols for c in need): idxs.append(i)
    for i in idxs:
        nm = row.get(f"comp{i}_PlayerName")
        sy = str(row.get(f"comp{i}_Start_Year"))
        trm = to_int_safe(row.get(f"comp{i}_Length"), 0)
        capf = percent_to_fraction(row.get(f"comp{i}_CapPct"))
        if "-" in sy and trm > 0 and pd.notna(capf):
            keys = list(cap_table.keys())
            if sy in keys:
                si = keys.index(sy)
                x0 = keys[si]
                x1 = keys[min(si + trm, len(keys)-1)]
                y = float(capf) * 100.0 if units == "cap" else float(capf) * float(cap_table[sy])
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y, y], mode="lines+markers",
                    line=dict(color="rgba(0,0,0,0.35)", width=2, dash="dash"),
                    marker=dict(size=5, opacity=0.85), name=f"Comp: {nm} ({trm}y)"
                    ))

    all_y = [0.0]
    for t in fig.data:
        ys = t["y"]
        if ys is not None:
            for yy in ys:
                try:
                    if yy is not None and pd.notna(yy):
                        all_y.append(float(yy))
                except Exception:
                    pass
    ymin, ymax = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)
    pad = 0.05 * (ymax - ymin if ymax != ymin else 1.0)
    xcats = list(dict.fromkeys((player_value_row["statsSeason"].astype(str).tolist() if not player_value_row.empty else []) + proj_x_list))

    fig.update_layout(
        title="Value vs Contract (slider selects term)",
        xaxis=dict(type="category", categoryorder="array", categoryarray=xcats, title="Season"),
        yaxis=dict(title=("Value (USD)" if units=="usd" else "Value (Cap %)"),
                   tickformat=("$,.0f" if units=="usd" else ".1f"),
                   autorange=False, range=[ymin - pad, ymax + pad]),
        template="plotly_white", height=graph_height_px)

    # Surplus box
    cap_frac = cap_frac_selected
    anchor_cap_for_aav = cap_table.get("2025-2026", cap_table.get(start_season, np.nan))
    if units == "usd":
        if pd.notna(cap_frac) and pd.notna(anchor_cap_for_aav):
            aav_usd = float(cap_frac) * float(anchor_cap_for_aav)
        else:
            aav_usd = None

        fut = proj_df.head(term)
        if aav_usd is not None and not fut.empty:
            surplus = float((fut["value_usd"] - aav_usd).sum())
            surplus_txt = f"Excess Value over {term}y @ {float(row.get(f'capPct_len{active_term}',0)):.2f}% cap: ${surplus/1e6:.2f}M"
        else:
            surplus_txt = "Excess Value: N/A"
    else:
        fut = proj_df.head(term).copy()
        if not fut.empty and pd.notna(cap_frac) and pd.notna(anchor_cap_for_aav):
            fut["cap"] = fut["statsSeason"].map(cap_table)
            fut["val_usd"] = (fut["value_cap_pct"] / 100.0) * fut["cap"]
            aav_usd = float(cap_frac) * float(anchor_cap_for_aav)
            surplus = float((fut["val_usd"] - aav_usd).sum())
            surplus_txt = f"Excess Value over {term} years @ {float(row.get(f'capPct_len{active_term}',0)):.2f}% cap: ${surplus/1e6:.2f}M"
        else:
            surplus_txt = "Excess Value: N/A"

    # Summary + comps table
    if units == "usd" and pd.notna(cap_frac) and pd.notna(anchor_cap_for_aav):
        aav_display = float(cap_frac) * float(anchor_cap_for_aav)
        summary = f"Selected term: {term} - Cap%: {float(row.get(f'capPct_len{active_term}',0)):.2f} - AAV (anchored to 2025-26 cap): ${aav_display:,.0f}"
    else:
        summary = f"Selected term: {term} - Cap%: {float(row.get(f'capPct_len{active_term}',0)):.2f}"

    comp_cols = [{"name": c, "id": c} for c in ["Name","Pos","Year","Term","Cap%","Similarity"]]
    comp_rows = []

    for i in range(1, 11):
        nm  = row.get(f"comp{i}_PlayerName")
        posc = row.get(f"comp{i}_Pos")
        yr  = row.get(f"comp{i}_Start_Year")
        trm = row.get(f"comp{i}_Length")
        cpc = row.get(f"comp{i}_CapPct")
        simp = row.get(f"comp{i}_sim_pct")

        if isinstance(nm, str) and nm.strip() != "":
            if pd.notna(simp):
                try:
                    sim_str = f"{float(simp):.0f}"
                except Exception:
                    sim_str = ""
            else:
                sim_str = ""
            comp_rows.append({
                "Name": nm,
                "Pos": posc,
                "Year": yr,
                "Term": trm,
                "Cap%": cpc,
                "Similarity": sim_str
                })

    return bio, tiles, summary, comp_rows, comp_cols, fig, surplus_txt

# Main
def main():
    host, port = "127.0.0.1", 8050
    url = f"http://{host}:{port}/"
    def open_browser():
        time.sleep(1.0)
        try:
            webbrowser.open_new(url)
        except Exception:
            pass
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=True, host=host, port=port, use_reloader=False)

if __name__ == "__main__":
    main()