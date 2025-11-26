# -*- coding: utf-8 -*-
"""
WAR & SPAR to Dollar Valuation

Converts WAR and SPAR to dollar valuations via linear regression, then applies market rate to skaters
"""
# Imports
from pathlib import Path
import pandas as pd
import numpy as np

# Config
script_dir = Path(__file__).resolve().parent
data_root = script_dir.parent / "Data" / "Clean Data"
start_year = 2016
end_year = 2025

contracts_file = "NHL_CONTRACTS_MASTER.csv"
GAR_file_pattern = "Skater_GAR_WAR_{season}.csv"

min_bucket_size = 15
conf_sample_k = 50
random_seed = 17
value_center_war = 2.0
value_pos_gamma = 0.6
value_neg_gamma = 0.7
priors_weight_list = [0.55, 0.35, 0.10]

cap_table = {
    "2015-2016": 71400000,
    "2016-2017": 73000000,
    "2017-2018": 75000000,
    "2018-2019": 79500000,
    "2019-2020": 81500000,
    "2020-2021": 81500000,
    "2021-2022": 81500000,
    "2022-2023": 82500000,
    "2023-2024": 83500000,
    "2024-2025": 88000000,
    "2025-2026": 95500000,
    "2026-2027": 104000000,
    "2027-2028": 113500000
    }
cap_growth_rate = 0.05

min_salary_table = {
    "2015-2016": 575000,
    "2016-2017": 575000,
    "2017-2018": 650000,
    "2018-2019": 650000,
    "2019-2020": 700000,
    "2020-2021": 700000,
    "2021-2022": 750000,
    "2022-2023": 750000,
    "2023-2024": 775000,
    "2024-2025": 775000
    }

# Helpers
def season_label(year):
    return f"{year}-{year+1}"

def prev_season_str(contract_season):
    a = int(contract_season.split("-")[0])
    return f"{a-1}-{a}"

def start_year_from_season_str(season):
    return int(season.split("-")[0])

def back_n_seasons(season_str, n):
    y = start_year_from_season_str(season_str)
    return season_label(y - n)

def ensure_folder(folder_path):
    folder_path.mkdir(parents=True, exist_ok=True)

def stats_path_for_season(stats_season):
    return data_root / stats_season / (GAR_file_pattern.format(season=stats_season))

def safe_get_cap(contract_season):
    known = sorted(cap_table.keys(), key=start_year_from_season_str)
    target_y = start_year_from_season_str(contract_season)
    last_known = None
    for ks in known:
        if start_year_from_season_str(ks) <= target_y:
            last_known = ks
    if last_known is None:
        return cap_table[known[0]]
    val = cap_table[last_known]
    y = start_year_from_season_str(last_known)
    while y < target_y:
        val = val * (1.0 + cap_growth_rate)
        y += 1
    return int(round(val))

def safe_get_min_salary(contract_season):
    if contract_season in min_salary_table:
        return min_salary_table[contract_season]
    known = sorted(min_salary_table.keys(), key=start_year_from_season_str)
    target_y = start_year_from_season_str(contract_season)
    last_known = None
    for ks in known:
        if start_year_from_season_str(ks) <= target_y:
            last_known = ks
    if last_known is None:
        last_known = known[-1]
    return min_salary_table[last_known]

def coerce_cap_hit(series):
    if series.dtype.kind in ("i", "f"):
        return series.astype(float)
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def median_ape_and_iqr(actual, pred):
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs(actual_arr - pred_arr) / np.where(actual_arr != 0.0, actual_arr, np.nan)
    ape = ape[~np.isnan(ape)]
    if ape.size == 0:
        return np.nan, np.nan
    median_ape = float(np.median(ape))
    q1 = float(np.percentile(ape, 25))
    q3 = float(np.percentile(ape, 75))
    return median_ape, (q3 - q1)

def compress_value_signal(x, center, pos_gamma, neg_gamma):
    arr = np.asarray(x, dtype=float)
    out = arr.copy()
    mask_pos = arr > center
    out[mask_pos] = center + (arr[mask_pos] - center) * pos_gamma
    mask_neg = arr < -center
    out[mask_neg] = -center + (arr[mask_neg] + center) * neg_gamma

    return out

def build_market_and_values_for_season(contract_season, contracts_df, season_col):
    stats_season = prev_season_str(contract_season)
    out_folder = data_root / contract_season
    ensure_folder(out_folder)

    league_cap = float(safe_get_cap(contract_season))
    league_min = float(safe_get_min_salary(contract_season))
    stats_path = stats_path_for_season(stats_season)

    stats_df = pd.read_csv(stats_path)
    req_stats = ["Player", "PlayerID", "PosBucket", "WAR", "SPAR"]
    stats_df = stats_df[req_stats].copy()
    stats_df = stats_df.dropna(subset=["PlayerID"]).copy()
    stats_df = stats_df.rename(columns={"PlayerID": "playerId"})
    stats_df["playerId"] = stats_df["playerId"].astype(int)
    stats_df["PosBucket"] = stats_df["PosBucket"].astype(str).str.upper()
    stats_df.loc[~stats_df["PosBucket"].isin(["F", "D"]), "PosBucket"] = "ALL"
    stats_df["statsSeason"] = stats_season
    stats_df["contractSeason"] = contract_season

    def load_prior(prior_season):
        pth = stats_path_for_season(prior_season)
        if not pth.exists():
            return None
        prior_df = pd.read_csv(pth)
        required_cols = ["PlayerID", "WAR", "SPAR"]
        if not all(col in prior_df.columns for col in required_cols):
            return None
        prior_df = prior_df[["PlayerID", "WAR", "SPAR"]].rename(columns={"PlayerID": "playerId"})
        prior_df["playerId"] = pd.to_numeric(prior_df["playerId"], errors="coerce").astype("Int64")
        return prior_df

    prior_frames = [stats_df[["playerId","WAR","SPAR"]].copy()]
    for lag in (1,2):
        prior_season = back_n_seasons(stats_season, lag)
        prior_df = load_prior(prior_season) if prior_season else None
        if prior_df is None:
            prior_frames.append(None)
        else:
            prior_frames.append(prior_df)

    stats_small = stats_df[["playerId","WAR","SPAR","PosBucket","statsSeason","contractSeason"]].copy()
    if prior_frames[1] is not None:
        stats_small = stats_small.merge(prior_frames[1].rename(columns={"WAR":"WAR_p1","SPAR":"SPAR_p1"}), on="playerId", how="left")
    if prior_frames[2] is not None:
        stats_small = stats_small.merge(prior_frames[2].rename(columns={"WAR":"WAR_p2","SPAR":"SPAR_p2"}), on="playerId", how="left")

    prior_weights_arr = np.array(priors_weight_list + [0]*(3-len(priors_weight_list)), dtype=float)
    
    def col_or_nan(df, col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)
    
    parts_war  = [
        pd.to_numeric(stats_small["WAR"], errors="coerce"),
        col_or_nan(stats_small, "WAR_p1"),
        col_or_nan(stats_small, "WAR_p2"),
        ]
    parts_spar = [
        pd.to_numeric(stats_small["SPAR"], errors="coerce"),
        col_or_nan(stats_small, "SPAR_p1"),
        col_or_nan(stats_small, "SPAR_p2"),
        ]
    
    def blend_triplet(values_tuple):
        vals = np.array(values_tuple, dtype=float)
        mask = np.isfinite(vals)
        if not mask.any():
            return np.nan
        ww = prior_weights_arr[mask]
        ww = ww / ww.sum()
        return float(np.dot(ww, vals[mask]))
    
    stats_small["WAR_blend"] = [blend_triplet(v) for v in zip(parts_war[0],  parts_war[1],  parts_war[2])]
    stats_small["SPAR_blend"] = [blend_triplet(v) for v in zip(parts_spar[0], parts_spar[1], parts_spar[2])]

    contracts_season_df = contracts_df.copy()
    contracts_season_df = contracts_season_df[contracts_season_df[season_col] == contract_season].copy()
    contracts_season_df = contracts_season_df.dropna(subset=["playerId"]).copy()
    contracts_season_df["playerId"] = contracts_season_df["playerId"].astype(int)
    contracts_season_df["Cap Hit"] = coerce_cap_hit(contracts_season_df["Cap Hit"])
    contracts_season_df = contracts_season_df[contracts_season_df["Cap Hit"] > 0].copy()
    contracts_season_df["Level"] = contracts_season_df["Level"].astype(str).str.strip().str.upper()
    contracts_season_df = contracts_season_df[contracts_season_df["Level"] != "ELC"].copy()
    id_to_bucket = dict(zip(stats_df["playerId"], stats_df["PosBucket"]))
    contracts_season_df["Bucket"] = contracts_season_df["playerId"].map(id_to_bucket)
    contracts_season_df.loc[contracts_season_df["Bucket"].isna(), "Bucket"] = "ALL"

    joined = pd.merge(
        contracts_season_df[["playerId", "Bucket", "Cap Hit"]],
        stats_small[["playerId","WAR_blend","SPAR_blend"]],
        on="playerId",
        how="left")

    joined["cap_pct"] = joined["Cap Hit"] / float(league_cap)
    fit_base = joined.copy()
    fit_base["WAR_fit"]  = pd.to_numeric(fit_base["WAR_blend"], errors="coerce").fillna(0.0).clip(lower=0.0)
    fit_base["SPAR_fit"] = pd.to_numeric(fit_base["SPAR_blend"], errors="coerce").fillna(0.0).clip(lower=0.0)

    def compute_rates(contracts_stats_df, metric_col):
        rows = []
        for bucket in ["ALL", "F", "D"]:
            bucket_df = contracts_stats_df.copy() if bucket == "ALL" else contracts_stats_df[contracts_stats_df["Bucket"] == bucket].copy()
            fit_pos = bucket_df[bucket_df[metric_col] > 0].copy()
            fit_pos["ratio"] = fit_pos["cap_pct"] / fit_pos[metric_col]
            fit_pos = fit_pos[np.isfinite(fit_pos["ratio"])]
    
            n_used = int(fit_pos.shape[0])
            if n_used == 0:
                med_ratio = np.nan
                mdape = np.nan
                iqr = np.nan
            else:
                if n_used >= 20:
                    qlo, qhi = np.percentile(fit_pos["ratio"], [20, 80])
                    fit_mid = fit_pos[(fit_pos["ratio"] >= qlo) & (fit_pos["ratio"] <= qhi)].copy()
                else:
                    fit_mid = fit_pos
    
                med_ratio = float(np.median(fit_mid["ratio"]))
                sample_size = min(conf_sample_k, fit_pos.shape[0])
                conf_sample_df = fit_pos.sample(n=sample_size, random_state=random_seed) if sample_size < fit_pos.shape[0] else fit_pos
                pred_cap_hit = med_ratio * float(league_cap) * conf_sample_df[metric_col]
                mdape, iqr = median_ape_and_iqr(conf_sample_df["Cap Hit"].values, pred_cap_hit.values)
    
            rows.append({
                "Bucket": bucket,
                "n_used": n_used,
                "med_ratio": med_ratio,
                "mdape": mdape,
                "iqr": iqr
                })
        return pd.DataFrame(rows)

    rate_spar_war = compute_rates(fit_base, "WAR_fit")
    rate_spar_spar = compute_rates(fit_base, "SPAR_fit")

    market_rows = []
    for bucket in ["ALL", "F", "D"]:
        rate_spar_row_war = rate_spar_war[rate_spar_war["Bucket"] == bucket].iloc[0]
        rate_spar_row_spar = rate_spar_spar[rate_spar_spar["Bucket"] == bucket].iloc[0]
        cap_pct_per_WAR = rate_spar_row_war["med_ratio"] if not pd.isna(rate_spar_row_war["med_ratio"]) else np.nan
        cap_pct_per_SPAR = rate_spar_row_spar["med_ratio"] if not pd.isna(rate_spar_row_spar["med_ratio"]) else np.nan
        usd_per_WAR = cap_pct_per_WAR * league_cap if not pd.isna(cap_pct_per_WAR) else np.nan
        usd_per_SPAR = cap_pct_per_SPAR * league_cap if not pd.isna(cap_pct_per_SPAR) else np.nan

        market_rows.append({
            "ContractSeason": contract_season,
            "StatsSeason": stats_season,
            "LeagueCap": int(league_cap),
            "LeagueMinSalary": int(league_min),
            "Bucket": bucket,
            "rate_$perWAR": float(usd_per_WAR) if not pd.isna(usd_per_WAR) else np.nan,
            "rate_capPct_perWAR": float(cap_pct_per_WAR) if not pd.isna(cap_pct_per_WAR) else np.nan,
            "N_used_WAR": int(rate_spar_row_war["n_used"]),
            "IQR_APE_WAR": float(rate_spar_row_war["iqr"]) if not pd.isna(rate_spar_row_war["iqr"]) else np.nan,
            "rate_$perSPAR": float(usd_per_SPAR) if not pd.isna(usd_per_SPAR) else np.nan,
            "rate_capPct_perSPAR": float(cap_pct_per_SPAR) if not pd.isna(cap_pct_per_SPAR) else np.nan,
            "N_used_SPAR": int(rate_spar_row_spar["n_used"]),
            "IQR_APE_SPAR": float(rate_spar_row_spar["iqr"]) if not pd.isna(rate_spar_row_spar["iqr"]) else np.nan,
            })

    market_df = pd.DataFrame(market_rows)

    def bucket_rate(rate_spar_df, bucket, min_n, rate_col):
        row = rate_spar_df[rate_spar_df["Bucket"] == bucket].iloc[0]
        if int(row["n_used"]) >= min_n and np.isfinite(row[rate_col]):
            return float(row[rate_col])

        row_all = rate_spar_df[rate_spar_df["Bucket"] == "ALL"].iloc[0]
        return float(row_all[rate_col]) if np.isfinite(row_all[rate_col]) else np.nan

    rate_war = {
        "ALL": bucket_rate(rate_spar_war, "ALL", min_bucket_size, "med_ratio"),
        "F": bucket_rate(rate_spar_war, "F", min_bucket_size, "med_ratio"),
        "D": bucket_rate(rate_spar_war, "D", min_bucket_size, "med_ratio"),
        }
    rate_spar = {
        "ALL": bucket_rate(rate_spar_spar, "ALL", min_bucket_size, "med_ratio"),
        "F": bucket_rate(rate_spar_spar, "F", min_bucket_size, "med_ratio"),
        "D": bucket_rate(rate_spar_spar, "D", min_bucket_size, "med_ratio"),
        }

    stats_df = stats_df.merge(stats_small[["playerId","WAR_blend","SPAR_blend"]], on="playerId", how="left")
    stats_df["PriceBucket"] = stats_df["PosBucket"].where(stats_df["PosBucket"].isin(["F","D"]), "ALL")

    def price_row(row):
        b = row["PriceBucket"]
        war_raw  = float(row["WAR_blend"])  if np.isfinite(row["WAR_blend"])  else 0.0
        spar_raw = float(row["SPAR_blend"]) if np.isfinite(row["SPAR_blend"]) else 0.0

        war_eff = compress_value_signal(
            [war_raw],
            center=value_center_war,
            pos_gamma=value_pos_gamma,
            neg_gamma=value_neg_gamma
            )[0]

        spar_eff = compress_value_signal(
            [spar_raw],
            center=value_center_war,
            pos_gamma=value_pos_gamma,
            neg_gamma=value_neg_gamma
            )[0]

        war_clamped  = max(war_eff, 0.0)
        spar_clamped = max(spar_eff, 0.0)
        rate_war_bucket = rate_war.get(b, rate_war["ALL"])
        rate_spar_bucket = rate_spar.get(b, rate_spar["ALL"])
        cap_pct_war = rate_war_bucket * war_clamped if np.isfinite(rate_war_bucket) else np.nan
        cap_pct_spar = rate_spar_bucket * spar_clamped if np.isfinite(rate_spar_bucket) else np.nan
        price_war = cap_pct_war * league_cap if np.isfinite(cap_pct_war) else np.nan
        price_spar = cap_pct_spar * league_cap if np.isfinite(cap_pct_spar) else np.nan
        price_war = np.nan if not np.isfinite(price_war)  else max(price_war, league_min)
        price_spar = np.nan if not np.isfinite(price_spar) else max(price_spar, league_min)
        price_pct_war = (price_war / league_cap * 100.0) if np.isfinite(price_war) else np.nan
        price_pct_spar = (price_spar / league_cap * 100.0) if np.isfinite(price_spar) else np.nan

        return pd.Series({
            "p$WAR": price_war, "p$SPAR": price_spar,
            "p%WAR": price_pct_war, "p%SPAR": price_pct_spar
            })

    priced = stats_df.copy()
    priced = pd.concat([priced, priced.apply(price_row, axis=1)], axis=1)
    player_value_cols = [
        "playerId","Player","PosBucket","statsSeason","contractSeason","WAR","SPAR",
        "p$WAR","p$SPAR","p%WAR","p%SPAR"
        ]
    player_value_df = priced[player_value_cols].copy()
    market_out = out_folder / f"market_rates_{contract_season}.csv"
    player_out = out_folder / f"player_value_{contract_season}.csv"
    market_df.to_csv(market_out, index=False)
    player_value_df.to_csv(player_out, index=False)

    return market_df, player_value_df

# Main
def main():
    contracts_path = data_root / contracts_file
    contracts_df = pd.read_csv(contracts_path)
    season_col = "Start Year"
    seasons = []
    for season_str in sorted(contracts_df[season_col].dropna().unique()):
        try:
            year = start_year_from_season_str(season_str)
            if start_year <= year <= end_year:
                seasons.append(season_str)
        except Exception:
            continue

    for contract_season in seasons:
        build_market_and_values_for_season(contract_season, contracts_df, season_col)

    print("Market Calculations Complete.")

if __name__ == "__main__":
    main()
