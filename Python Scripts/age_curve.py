# -*- coding: utf-8 -*-
"""
Synthetic Age Curves

This age curve appraoch is due to the volatility in my GAR data making it very difficult to fit a solid age curve in
this V1 of my pipeline. As such, I looked to industry leaders (Dom Luszczyszyn of the Athletic, Eric Tulsky, current GM
of the Carolina Hurricanes, and EvolvingHockey) to craft three separate artificial curves, creating a min/max band to
simulate uncertainty, and fit the center of that band.

Curves are normalized so peak = 1.0.
Shapes are set to visually/qualitatively match findings:
 - Dom Luszczyszyn: early peaks, sharper post-30 decline
 - Eric Tulsky: symmetric hump around mid-20s
 - EvolvingWild: broad plateau 22–28, decline after ~29 (F) / ~31 (D)
"""
# Imports
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Config
script_dir = Path(__file__).resolve().parent
out_dir = script_dir.parent / "Data" / "Clean Data"

output_csv = "AGE_CURVE_GAR_BY_AGE.csv"
offense_curve_plot_name = "age_curve_offense.png"
defense_curve_plot_name = "age_curve_defense.png"
age_list = list(range(18, 39))

curve_parameters = {
    "Dom": {
        "F": dict(
                start_age=18, start_val=0.10,
                rise_end_age=24, rise_end_val=1.00,
                plateau_end_age=26, plateau_val=0.97,
                decline_end_age=34, decline_val=0.20,
                tail_age=38, tail_val=0.05
            ),
        "D": dict(
                start_age=18, start_val=0.12,
                rise_end_age=26, rise_end_val=1.00,
                plateau_end_age=29, plateau_val=0.98,
                decline_end_age=36, decline_val=0.20,
                tail_age=38, tail_val=0.08
            ),
        },
    "Tulsky": {
        "F": dict(
                start_age=18, start_val=0.20,
                rise_end_age=25, rise_end_val=1.00,
                plateau_end_age=27, plateau_val=0.95,
                decline_end_age=35, decline_val=0.40,
                tail_age=38, tail_val=0.30
            ),
        "D": dict(
                start_age=18, start_val=0.22,
                rise_end_age=26, rise_end_val=1.00,
                plateau_end_age=29, plateau_val=0.97,
                decline_end_age=36, decline_val=0.45,
                tail_age=38, tail_val=0.35
            ),
        },
    "EvolvingWild": {
        "F": dict(
                start_age=18, start_val=0.18,
                rise_end_age=25, rise_end_val=1.00,
                plateau_end_age=28, plateau_val=0.98,
                decline_end_age=35, decline_val=0.25,
                tail_age=38, tail_val=0.10
            ),
        "D": dict(
                start_age=18, start_val=0.20,
                rise_end_age=26, rise_end_val=1.00,
                plateau_end_age=30, plateau_val=0.98,
                decline_end_age=37, decline_val=0.30,
                tail_age=38, tail_val=0.15
            ),
        }
    }

# Helpers
def interpolate_between_points(x0, y0, x1, y1, x):
    if x1 == x0:
        return y0
    t = (x - x0) / float(x1 - x0)
    return y0 + t * (y1 - y0)

def piecewise_curve(age, p):
    start_age, start_val = p["start_age"], p["start_val"]
    rise_end_age, rise_end_val = p["rise_end_age"], p["rise_end_val"]
    plateau_end_age, plateau_val = p["plateau_end_age"], p["plateau_val"]
    decline_end_age, decline_val = p["decline_end_age"], p["decline_val"]
    tail_age, tail_val = p["tail_age"], p["tail_val"]

    if age <= start_age: 
        y = start_val
    elif age <= rise_end_age: 
        y = interpolate_between_points(start_age, start_val, rise_end_age, rise_end_val, age)
    elif age <= plateau_end_age: 
        y = interpolate_between_points(rise_end_age, rise_end_val, plateau_end_age, plateau_val, age)
    elif age <= decline_end_age: 
        y = interpolate_between_points(plateau_end_age, plateau_val, decline_end_age, decline_val, age)
    elif age <= tail_age: 
        y = interpolate_between_points(decline_end_age, decline_val, tail_age, tail_val, age)
    else: y = tail_val

    return max(0.0, min(1.05, float(y)))

def build_source_curves(age_list, params_for_source):
    out = {}
    for pos in ["F", "D"]:
        spec = params_for_source[pos]
        ys = [piecewise_curve(a, spec) for a in age_list]
        peak = max(ys) if ys else 1.0
        ys = [y / peak if peak > 0 else 0.0 for y in ys]
        out[pos] = ys
    return out

def build_all_curves(ages, params):
    curves = {}
    for src in ["Dom", "Tulsky", "EvolvingWild"]:
        curves[src] = build_source_curves(ages, params[src])
    return curves

def build_age_curve_table(ages, curves):
    rows = []
    for pos in ["F", "D"]:
        for i, age in enumerate(ages):
            dom = curves["Dom"][pos][i]
            tul = curves["Tulsky"][pos][i]
            ew = curves["EvolvingWild"][pos][i]
            vals = [dom, tul, ew]
            avg = sum(vals) / 3.0
            lo = min(vals)
            hi = max(vals)
            rows.append({
                "age": age,
                "pos": pos,
                "Dom": round(dom, 3),
                "Tulsky": round(tul, 3),
                "EvolvingWild": round(ew, 3),
                "avg": round(avg, 3),
                "lower": round(lo, 3),
                "upper": round(hi, 3)
            })
    return pd.DataFrame(rows, columns=["age","pos","Dom","Tulsky","EvolvingWild","avg","lower","upper"])

def plot_position(df_pos, title, outfile_path):
    plt.figure(figsize=(8.5, 5.0))
    plt.fill_between(df_pos["age"], df_pos["lower"], df_pos["upper"], alpha=0.18, label="Uncertainty band")
    plt.plot(df_pos["age"], df_pos["avg"], linewidth=3, label="Average")
    plt.plot(df_pos["age"], df_pos["Dom"], linestyle="--", linewidth=1.4, label="Dom")
    plt.plot(df_pos["age"], df_pos["Tulsky"], linestyle="--", linewidth=1.4, label="Tulsky")
    plt.plot(df_pos["age"], df_pos["EvolvingWild"], linestyle="--", linewidth=1.4, label="EvolvingWild")
    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Relative value (peak = 1.0)")
    plt.xlim(min(age_list), max(age_list))
    plt.ylim(0.0, 1.1)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(str(outfile_path), dpi=160)
    plt.close()

# Main
def main():
    curves = build_all_curves(age_list, curve_parameters)
    df = build_age_curve_table(age_list, curves)
    csv_path = out_dir / output_csv
    df.to_csv(csv_path, index=False)
    df_f = df[df["pos"] == "F"].copy()
    df_d = df[df["pos"] == "D"].copy()

    plot_position(df_f, "Age Curve - Offense (Forwards, blended)", out_dir / offense_curve_plot_name)
    plot_position(df_d, "Age Curve - Defense (Defensemen, blended)", out_dir / defense_curve_plot_name)

if __name__ == "__main__":
    main()