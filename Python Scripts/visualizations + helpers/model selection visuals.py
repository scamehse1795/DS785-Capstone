# -*- coding: utf-8 -*-
"""
Presentation 4 Visuals - histograms + contracts table
"""
# Imports
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
project_root = Path(__file__).resolve().parents[2]
clean_dir = project_root / "Data" / "Clean Data"
season_rapm = "2024-2025"
season_contracts = "2025-2026"
target_year = 2025

rapm_es_csv = clean_dir / season_rapm / f"RAPM_xGF_Even_Strength_{season_rapm}.csv"
diag_csv = clean_dir / season_contracts / f"model_diagnostics_{season_contracts}.csv"
contracts_master = clean_dir / "NHL_contracts_master.csv"
contract_results = clean_dir / season_contracts / f"contract_results_{season_contracts}.csv"
out_dir = Path(__file__).resolve().parent / "figs"

# Helpers
def parse_start_year_series(s):
    s = s.astype(str)
    yrs = s.str.extract(r'^\s*(\d{4})', expand=False)
    return pd.to_numeric(yrs, errors='coerce').astype('Int64')

# RAPM Histogram
def fig_rapm_histograms():
    df = pd.read_csv(rapm_es_csv, low_memory=False)
    off = pd.to_numeric(df["off_xGF60_raw"], errors="coerce")
    ddf = pd.to_numeric(df["def_xGF60_raw"], errors="coerce")
    m_off = float(off.mean(skipna=True))
    m_def = float(ddf.mean(skipna=True))

    plt.figure(figsize=(8, 4.5))
    plt.hist(off.dropna().values, bins=40, edgecolor="black", alpha=0.85)
    plt.axvline(m_off, linestyle="--", linewidth=1.6)
    plt.title("RAPM (Even Strength) - OFF")
    plt.xlabel("OFF RAPM") 
    plt.ylabel("Count")
    plt.tight_layout()
    out1 = out_dir / "rapm_hist_off_2024_25.png"
    plt.savefig(out1, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.hist(ddf.dropna().values, bins=40, edgecolor="black", alpha=0.85)
    plt.axvline(m_def, linestyle="--", linewidth=1.6)
    plt.title("RAPM (Even Strength) - DEF (more negative = better)")
    plt.xlabel("DEF RAPM (negative is better)")
    plt.ylabel("Count")
    plt.tight_layout()
    out2 = out_dir / "rapm_hist_def_2024_25.png"
    plt.savefig(out2, dpi=180)
    plt.close()

# Contract MAE vs Baseline + Table
def fig_term_mae_vs_baseline():
    diag = pd.read_csv(diag_csv, low_memory=False)
    diag["term"] = pd.to_numeric(diag["term"], errors="coerce").astype("Int64")
    c = pd.read_csv(contracts_master, low_memory=False)
    c["Start_Year"] = parse_start_year_series(c["Start Year"])
    c["Length"] = pd.to_numeric(c["Length"], errors="coerce").astype("Int64")
    c["Start_Yr_Cap_Pct"] = pd.to_numeric(c["Start Yr Cap %"], errors="coerce")
    c["playerId"] = pd.to_numeric(c["playerId"], errors="coerce").astype("Int64")
    c["Level"] = c["Level"].astype(str).str.upper().str.strip()

    hist = c[(c["Start_Year"].notna()) &
             (c["Start_Year"] < target_year) &
             (c["playerId"].notna()) &
             (c["Level"] != "ELC")].copy()
    hist = hist.dropna(subset=["Length", "Start_Yr_Cap_Pct"])
    base_medians = (hist.groupby("Length", dropna=False)["Start_Yr_Cap_Pct"]
                         .median().rename("MAE_baseline_pred").reset_index())
    
    exp = c[(c["Start_Year"] == target_year) & (c["playerId"].notna())].copy()
    exp = exp.dropna(subset=["Length", "Start_Yr_Cap_Pct"])
    term_to_baseline = dict(zip(base_medians["Length"].astype(int), base_medians["MAE_baseline_pred"]))
    exp["pred_capPct_baseline"] = exp["Length"].astype(int).map(term_to_baseline)
    exp = exp.dropna(subset=["pred_capPct_baseline"]).copy()
    exp["abs_err_baseline"] = np.abs(exp["pred_capPct_baseline"] - exp["Start_Yr_Cap_Pct"])
    base_mae = (exp.groupby("Length", dropna=False)["abs_err_baseline"]
                  .mean().rename("MAE_baseline").reset_index().rename(columns={"Length":"term"}))
    
    all_terms = pd.DataFrame({"term": list(range(1, 9))})
    comb = (all_terms.merge(diag[["term","MAE_cap_pct"]], on="term", how="left").merge(base_mae, on="term", how="left"))
    out_table = out_dir / "term_mae_vs_baseline_2025_26_values.csv"
    comb.to_csv(out_table, index=False)
    x = np.arange(1, 9)
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, comb["MAE_cap_pct"].values, width, label="Model MAE (cap % pts)")
    plt.bar(x + width/2, comb["MAE_baseline"].values, width, label="Baseline MAE (cap % pts)")
    ymax = np.nanmax([comb["MAE_cap_pct"].values, comb["MAE_baseline"].values])
    if not np.isfinite(ymax): 
        ymax = 1.0
    plt.ylim(0, ymax * 1.2)
    plt.xticks(x, [str(t) for t in x])
    plt.xlabel("Term (years)")
    plt.ylabel("MAE (cap % points)")
    plt.title("Term MAE - Model vs Baseline (2025–26)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    outp = out_dir / "term_mae_vs_baseline_2025_26.png"
    plt.tight_layout()
    plt.savefig(outp, dpi=180)
    plt.close()
    
# Cap % Baseline test
def simple_cap_pct_baseline():
    df = pd.read_csv(contract_results, low_memory=False)
    df["Start_Year"] = pd.to_numeric(df["Start_Year"], errors="coerce").astype("Int64")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce").astype("Int64")
    df["Start_Yr_Cap_Pct"] = pd.to_numeric(df["Start_Yr_Cap_Pct"], errors="coerce")
    df = df[(df["Start_Year"] == target_year)].copy()
    df = df.dropna(subset=["Length", "Start_Yr_Cap_Pct"])
    df["Length"] = df["Length"].astype(int)
    df = df[(df["Length"] >= 1) & (df["Length"] <= 8)]
    if "level_clean" in df.columns:
        lvl = df["level_clean"].astype(str).str.upper().str.strip()
        df = df[lvl != "ELC"].copy()
    errs = []
    terms_used = []
    for _, row in df.iterrows():
        L = int(row["Length"])
        col = f"capPct_len{L}"
        pred = row[col]
        actual = row["Start_Yr_Cap_Pct"]
        if not np.isfinite(pred) or not np.isfinite(actual):
            continue
        errs.append(abs(pred - actual))
        terms_used.append(L)

    errs = np.asarray(errs, dtype=float)
    terms_used = np.asarray(terms_used, dtype=int)
    n = errs.size
    mae = float(np.mean(errs))
    thr_1x = mae
    thr_2x = 2.0 * mae
    pass_1x_mask = errs <= thr_1x
    pass_2x_mask = errs <= thr_2x
    pass_1x_cnt = int(pass_1x_mask.sum())
    pass_2x_cnt = int(pass_2x_mask.sum())
    pass_1x_pct = 100.0 * pass_1x_cnt / n
    pass_2x_pct = 100.0 * pass_2x_cnt / n
    print(f"Cap% Baseline: contracts used: {n}")
    print(f"Cap% Baseline: MAE_cap_pct: {mae:.3f}")
    print(f"Cap% Baseline: pass_err<=1xMAE: {pass_1x_cnt} / {n} = {pass_1x_pct:.1f}%")
    print(f"Cap% Baseline: pass_err<=2xMAE: {pass_2x_cnt} / {n} = {pass_2x_pct:.1f}%")
    print("Cap% Baseline: Per-term breakdown:")
    header = f"{'Term':>4}  {'n':>5}  {'pass<=1xMAE':>12}  {'%<=1xMAE':>10}  {'pass<=2xMAE':>12}  {'%<=2xMAE':>10}"
    print(header)
    print("-" * len(header))
    for L in range(1, 9):
        mask_L = terms_used == L
        if not mask_L.any():
            continue

        errs_L = errs[mask_L]
        n_L = int(errs_L.size)
        pass_1x_L = int((errs_L <= thr_1x).sum())
        pass_2x_L = int((errs_L <= thr_2x).sum())
        pass_1x_L_pct = 100.0 * pass_1x_L / n_L
        pass_2x_L_pct = 100.0 * pass_2x_L / n_L
        print(f"{L:>4}  {n_L:>5}  {pass_1x_L:>12}  {pass_1x_L_pct:>9.1f}%  {pass_2x_L:>12}  {pass_2x_L_pct:>9.1f}%")

    def print_player_case(df_in, player_name, thr1, thr2):
        mask = df_in["PlayerName"].astype(str).str.upper().str.strip() == player_name.upper()
        sub = df_in[mask].copy()
        print(f"Player case: {player_name}")
        for _, row in sub.iterrows():
            L = int(row["Length"])
            col = f"capPct_len{L}"
            pred = row[col]
            actual = row["Start_Yr_Cap_Pct"]
            err = abs(pred - actual)
            pass1 = err <= thr1
            pass2 = err <= thr2
            status1 = "PASS" if pass1 else "FAIL"
            status2 = "PASS" if pass2 else "FAIL"
            print(f"Term={L} yrs, actual={actual:.3f}%, model={pred:.3f}%, err={err:.3f}%, 1xMAE: {status1}, 2xMAE: {status2}")

    # Print Boeser and Faber
    print_player_case(df, "Brock Boeser", thr_1x, thr_2x)
    print_player_case(df, "Brock Faber", thr_1x, thr_2x)

if __name__ == "__main__":
    fig_rapm_histograms()
    fig_term_mae_vs_baseline()
    simple_cap_pct_baseline()
    print("All figures written.")