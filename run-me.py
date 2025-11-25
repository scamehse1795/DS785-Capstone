"""
Simple run-me file that executes all my scripts in the proper order and times the full end-to-end run

To toggle certain scripts on/off in a run, you just have to change the "enable" tag to True/False
"""
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

scripts_dir_name = "Python Scripts"

def python_bin():
    return sys.executable

def seconds_fmt(s):
    return f"{s:,.2f}s"

def main():
    root = Path(__file__).resolve().parent
    scripts_dir = root / scripts_dir_name

    steps = [
        {"file": "NHL API Scraper.py",              "enabled": True},
       #{"file": "puckpediaScrape_master.py",       "enabled": False}, This is the version that included the old PuckPedia scraping and is currently non-funcitonal. 
       #                                                               The CapWages portions were stripped out and included in the replacement file below
        {"file": "CapWages Scraper.py",              "enabled": True},
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
        script_path = scripts_dir / fname
        print(f"PIPELINE RUN: {fname}")
        t0 = time.perf_counter()
        proc = subprocess.run([python_bin(), "-u", str(script_path)], cwd=str(scripts_dir),
            stderr=subprocess.STDOUT, text=True)
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            raise SystemExit(f"PIPELINE FAILED {fname}: (return code {proc.returncode})")
        print(f"PIPELINE COMPLETE {fname} in {seconds_fmt(dt)}")
    total_dt = time.perf_counter() - t0_all
    print(f"PIPELINE TOTAL RUNTIME: {seconds_fmt(total_dt)}")

if __name__ == "__main__":
    main()
