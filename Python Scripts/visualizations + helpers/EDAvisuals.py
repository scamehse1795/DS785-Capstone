"""
EDA Presentation Visuals
"""
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Config
data_root = Path(__file__).resolve().parents[2]
season = "2024-2025"
contracts_path = data_root / "Data" / "Clean Data" / "NHL_CONTRACTS_MASTER.csv"
out_dir = Path(__file__).resolve().parent / "figs"
out_dir.mkdir(parents=True, exist_ok=True)
nst_es_path = data_root / "Data" / "Clean Data" / season / f"NST_player_master_ES_{season}.csv"
stints_path = data_root / "Data" / "Clean Data" / season / f"stints_{season}.csv"

title_cap_history = "Figure 1: Distribution of Average Annual Contract Value (AAV) in terms of Cap Hit (%)"
title_cap_history_pos = "Contract Cap Hit (%) by Position (Goalies excluded)"
title_term_vs_cap = "Figure 2: Term vs Cap Hit (%): Mean + Median with Min–Max Band"
title_stint_distribution = "Distribution of Stint Lengths (1 Second Bins)"
title_percent_xg_stint = "Percentage Share of Stints with xGF"
title_top_correlated_pairs = "Figure 3: Top Correlated Feature Pairs in NST Skater Data"

# Helpers
def savefig(path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def parse_money_to_float(x):
    s = str(x).replace("$", "").replace(",", "").strip()
    try:
        return float(s) if s else np.nan
    except:
        return np.nan

def fd_bucket(pos_text):
    if pd.isna(pos_text):
        return np.nan
    s = str(pos_text).strip().upper()
    return "D" if s == "D" else "F"

def r_to_color(rabs):
    if rabs >= 0.85: return "#d7301f" # red
    if rabs >= 0.70: return "#fc8d59" # orange
    if rabs >= 0.60: return "#fdae6b" # yellow-orange
    if rabs >= 0.50: return "#fee08b" # yellow
    return "#cccccc"

# Cap % histogram
def plot_cap_hist(contracts):
    df = contracts.copy()
    if "Level" in df.columns:
        df = df[df["Level"].astype(str).str.upper() != "ELC"] # exclude ELCs
    pct = pd.to_numeric(df["Start Yr Cap %"], errors="coerce").dropna().clip(0, 20)
    mean_pct = pct.mean()
    mean_dol = df["Cap Hit"].apply(parse_money_to_float).mean()
    bins = np.arange(0, 21, 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(pct, bins=bins, edgecolor="black")
    ax.set_xlabel("Cap Hit (% of League Cap)")
    ax.set_ylabel("Count")
    ax.set_title(title_cap_history)
    ax.axvline(mean_pct, linestyle="--", linewidth=1.2)
    ymax = ax.get_ylim()[1]
    ax.text(mean_pct + 0.15, ymax*0.92,
            f"mean {mean_pct:.2f}%\nmean $ {mean_dol:,.0f}",
            rotation=90, va="top", ha="left", fontsize=9)
    savefig(f"{out_dir}/caphitpct_hist_{season}.png")

# Cap % by position
def plot_cap_by_position(contracts):
    df = contracts.copy()
    if "Level" in df.columns:
        df = df[df["Level"].astype(str).str.upper() != "ELC"] # exclude ELCs
    if "Pos" in df.columns:
        df["FD"] = df["Pos"].apply(fd_bucket)
    else:
        df["FD"] = np.nan
    df["pct"] = pd.to_numeric(df["Start Yr Cap %"], errors="coerce").clip(0, 20)
    df["aav"] = df["Cap Hit"].apply(parse_money_to_float)
    vF = df.loc[df["FD"] == "F", "pct"].dropna()
    vD = df.loc[df["FD"] == "D", "pct"].dropna()
    
    bins = np.arange(0, 21, 1)
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(vF, bins=bins, edgecolor="black")
    ax1.set_title("Forwards")
    ax1.set_xlabel("% Cap")
    ax1.set_ylabel("Count")
    if len(vF) > 0:
        m_pct = vF.mean()
        m_dol = df.loc[df["FD"]=="F","aav"].mean()
        ax1.axvline(m_pct, linestyle="--", linewidth=1.2)
        ax1.text(m_pct + 0.15, ax1.get_ylim()[1]*0.92,
                 f"mean {m_pct:.2f}%\n${m_dol:,.0f}",
                 rotation=90, va="top", ha="left", fontsize=9)

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(vD, bins=bins, edgecolor="black")
    ax2.set_title("Defense") 
    ax2.set_xlabel("% Cap")
    if len(vD) > 0:
        m_pct = vD.mean() 
        m_dol = df.loc[df["FD"]=="D","aav"].mean()
        ax2.axvline(m_pct, linestyle="--", linewidth=1.2)
        ax2.text(m_pct + 0.15, ax2.get_ylim()[1]*0.92, f"mean {m_pct:.2f}%\n${m_dol:,.0f}", rotation=90, va="top", ha="left", fontsize=9)

    plt.suptitle(title_cap_history_pos, y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(f"{out_dir}/caphitpct_by_position_{season}.png")
    plt.close()

# Term vs Cap%
def plot_term_vs_cap(contracts):
    df = contracts.copy()
    if "Level" in df.columns:
        df = df[df["Level"].astype(str).str.upper() != "ELC"]
    df["term"] = pd.to_numeric(df["Length"], errors="coerce")
    df["pct"] = pd.to_numeric(df["Start Yr Cap %"], errors="coerce").clip(0, 20)
    df = df.dropna(subset=["term","pct"])
    df = df[(df["term"] >= 1) & (df["term"] <= 8)]
    if df.empty:
        return

    g = df.groupby("term")["pct"]
    stats = g.agg(mean="mean", median="median", min="min", max="max").reset_index().sort_values("term")
    x, mu, med, mn, mx = stats["term"].values, stats["mean"].values, stats["median"].values, stats["min"].values, stats["max"].values
    plt.figure(figsize=(8.6, 4.8))
    plt.fill_between(x, mn, mx, alpha=0.18, linewidth=0, label="Min–Max")
    plt.plot(x, mu, marker="o", label="Mean")
    plt.plot(x, med, marker="o", linestyle="--", label="Median")
    plt.xlabel("Term (years)")
    plt.ylabel("Cap Hit (% of Cap)")
    plt.title(title_term_vs_cap)
    plt.legend()
    plt.tight_layout()
    savefig(f"{out_dir}/term_vs_cap_mean_minmax_{season}.png")

# Stint lengths + xG %
def plot_stints(stints):
    dur = (pd.to_numeric(stints["endSec"], errors="coerce") - pd.to_numeric(stints["startSec"], errors="coerce")).clip(lower=0).dropna()
    dur = dur[dur <= 80]
    bins = np.arange(0, 81, 1)
    plt.figure(figsize=(12, 4.8))
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(dur, bins=bins, edgecolor="black")
    ax1.set_xlabel("Stint Duration (sec)")
    ax1.set_ylabel("Count")
    ax1.set_title(title_stint_distribution)
    med = dur.median()
    p95 = dur.quantile(0.95)
    ax1.axvline(med, linestyle="--", linewidth=1.2)
    ax1.axvline(p95, linestyle=":", linewidth=1.2)
    ax1.text(med + 0.3, ax1.get_ylim()[1]*0.92, "median", rotation=90, va="top", ha="right", fontsize=9)
    ax1.text(p95 + 0.3, ax1.get_ylim()[1]*0.92, "95th %", rotation=90, va="top", ha="right", fontsize=9)
    ax2 = plt.subplot(1, 2, 2)
    df = stints[["Situation", "xGF"]].copy()
    keep = ["Even Strength", "Power Play", "Penalty Kill"]
    df = df[df["Situation"].isin(keep)]
    df["xg_flag"] = pd.to_numeric(df["xGF"], errors="coerce").fillna(0) > 0

    share = (df.groupby("Situation")["xg_flag"].mean() * 100.0)
    order = ["Even Strength", "Power Play", "Penalty Kill"]
    vals = [share.get(k, 0.0) for k in order]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax2.bar(order, [100, 100, 100], color="#e6e6e6", edgecolor="black")
    ax2.bar(order, vals, color=colors, edgecolor="black")
    for i, v in enumerate(vals):
        ax2.text(i, min(v, 98), f"{v:.1f}%", ha="center", va="bottom", fontsize=9) # keep label inside axis

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% of stints with any xG")
    ax2.set_title(title_percent_xg_stint)
    plt.tight_layout()
    savefig(f"{out_dir}/stints_duration_and_pct_xg_{season}.png")


# Correlation pairs
def plot_corr_pairs(nst_es):
    cols = [c for c in [
        "xGF_per60","xGA_per60","CF_per60","CA_per60","GF_per60","GA_per60",
        "SCF_per60","SCA_per60","HDCF_per60","HDCA_per60","iXG_per60","iCF_per60",
        "TOI_seconds"
        ] if c in nst_es.columns]
    if len(cols) < 3:
        print("[WARN] not enough NST ES columns for correlations.")
        return

    df = nst_es[cols].apply(pd.to_numeric, errors="coerce")
    corr = df.corr()
    pairs = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            pairs.append((cols[i], cols[j], abs(r), r))
    pairs.sort(key=lambda x: -x[2])
    top = pairs[:12]
    labels = [f"{a} ~ {b}" for a, b, _, _ in top]
    rabs = [p[2] for p in top]
    colors = [r_to_color(v) for v in rabs]

    plt.figure(figsize=(12, 5.2))
    y = np.arange(len(labels))
    plt.barh(y, rabs, color=colors, edgecolor="black")
    plt.yticks(y, labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("|Pearson r|")
    plt.title(title_top_correlated_pairs)
    for c, txt in [("#fee08b","≥0.50"),("#fdae6b","≥0.60"),("#fc8d59","≥0.70"),("#d7301f","≥0.85")]:
        plt.plot([], [], color=c, linewidth=10, label=txt)
    plt.legend(title="Color by |r|", loc="lower right", frameon=False)
    plt.tight_layout()
    savefig(f"{out_dir}/top_corr_pairs_{season}.png")

if __name__ == "__main__":
    contracts = pd.read_csv(contracts_path)
    nst_es = pd.read_csv(nst_es_path)
    stints = pd.read_csv(stints_path)
    plot_cap_hist(contracts)
    plot_cap_by_position(contracts)
    plot_term_vs_cap(contracts)
    plot_stints(stints)
    plot_corr_pairs(nst_es)
    print("[DONE] Saved figures to:", out_dir)
