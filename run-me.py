"""
Simple run-me file that executes all my scripts in the proper order and times the full end-to-end run

To toggle certain scripts on/off in a run, you just have to change the "enable" tag to True/False
"""
import sys
import time
from pathlib import Path
from datetime import datetime

import nhl_api_scraper
import capwages_scraper
import map_nst_api_gameids
import clean_data
import rapm_calculator
import gar_war_spar_calculator
import war_value_conversion
import age_curve
import contract_prediction
import dashboard

scripts_dir_name = "Python Scripts"

script_main_lookup = {
    "NHL_API_Scraper.py":           nhl_api_scraper.main,
    "CapWages_Scraper.py":          capwages_scraper.main,
    "Map_NST_API_GameIDs.py":       map_nst_api_gameids.main,
    "Clean_Data.py":                clean_data.main,
    "RAPM_Calculator.py":           rapm_calculator.main,
    "GAR_WAR_SPAR_Calculator.py":   gar_war_spar_calculator.main,
    "WAR_value_conversion.py":      war_value_conversion.main,
    "age_curve.py":                 age_curve.main,
    "contract_prediction.py":       contract_prediction.main,
    "dashboard.py":                 dashboard.main,
}

def python_bin():
    return sys.executable

def seconds_fmt(s):
    return f"{s:,.2f}s"

def main():
    steps = [
        {"file": "NHL_API_Scraper.py",              "enabled": True},
       #{"file": "puckpediaScrape_master.py",       "enabled": False}, This is the version that included the old PuckPedia scraping and is currently non-funcitonal. 
       #                                                               The CapWages portions were stripped out and included in the replacement file below
        {"file": "CapWages_Scraper.py",             "enabled": True},
        {"file": "Map_NST_API_GameIDs.py",          "enabled": True},
        {"file": "Clean_Data.py",                   "enabled": True},
        {"file": "RAPM_Calculator.py",              "enabled": True},
        {"file": "GAR_WAR_SPAR_Calculator.py",      "enabled": True},
        {"file": "WAR$Conversion.py",               "enabled": True},
        {"file": "age_curve.py",                    "enabled": True},
        {"file": "contract_prediction.py",          "enabled": True},
        {"file": "dashboard.py",                    "enabled": True},
        ]
    
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"PIPELINE START {start_ts}")

    t0_all = time.perf_counter()

    for step in steps:
        if not step.get("enabled", True):
            continue

        fname = step["file"]
        print(f"PIPELINE RUN: {fname}")

        t0 = time.perf_counter()
        if fname not in script_main_lookup:
            raise SystemExit(f"No module mapping found for script: {fname}")

        script_fn = script_main_lookup[fname]
        script_fn()

        dt = time.perf_counter() - t0
        print(f"PIPELINE COMPLETE {fname} in {seconds_fmt(dt)}")

    total_dt = time.perf_counter() - t0_all
    print(f"PIPELINE TOTAL RUNTIME: {seconds_fmt(total_dt)}")


if __name__ == "__main__":
    main()