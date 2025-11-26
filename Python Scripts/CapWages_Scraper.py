# -*- coding: utf-8 -*-
"""
Puckpedia and CapWages Scraper

Uses Puckpedia columns, and converts CapWages data to match where possible.

I had AI assist me with setting up the Chrome folder appraoch, as I was running into issues after
updating my computer to Windows 11 getting Chromedriver to work the way I had it functioning
before. This appraoch should also allow it to run standalone. I also had AI help me with cleaning the script up,
especially after the Windows 11 update and after PuckPedia updated their HTML formatting for the 4th time this semester.

EDIT: I had to drop PuckPedia entirely, as the HTML background change hid several columns I need that I can parse from CapWages but no longer
can access on PuckPedia. As such, this script ONLY scrapes CapWages and still performs the necessary parsing to PuckPedia headers,
but I need more time than I have to fix the PuckPedia scraping itself.

NOTE: DO NOT CLOSE THE BROWSER POP-UP, as that will kill the scraping.
"""
# Imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

# Config
script_dir  = Path(__file__).resolve().parent
project_root = script_dir.parent
chrome_binary = project_root / "Chrome" / "chrome-win64" / "chrome.exe"
chromedriver_path = project_root / "Chrome" / "chromedriver-win64" / "chromedriver.exe"
master_out = project_root / "Data" / "Clean Data" / "NHL_CONTRACTS_MASTER.csv"

start_year = 2015
end_year = 2025

target_columns = [
    "Skaters","Pos","Shot","W(lbs)","H(f)","Length","Level","Cap Hit","Start Yr Cap %",
    "Structure","Clauses","Start Year","Signing Age","Signing Status","Expiry Year",
    "Expiry Status","Signing Agent","Signing GM","Signing Season"
    ]

# For calcualting Start Yr Cap % for capwages data
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

# Helpers
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    
def season_label(year):
    return f"{year}-{year+1}"

def clean_spaces(s):
    return " ".join(str(s).split()) if s is not None else ""

def extract_numeric_value(s):
    return "".join(ch for ch in str(s) if ch.isdigit() or ch == ".")

def extract_first_int(s):
    s = str(s)
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif num:
            break
    return int(num) if num else None

def contains_year(s):
    return any(token.isdigit() and len(token) == 4 for token in str(s).split())

def parse_first_year(s):
    s = str(s).replace("/", "-").strip()
    first = s.split("-")[0]
    return int(first) if first.isdigit() and len(first) == 4 else None

def fix_headers(headers):
    fixed = []
    for h in headers:
        if not isinstance(h, str):
            fixed.append(h)
            continue
        txt = " ".join(h.split())

        low = txt.lower().replace(" ", "")
        if low.endswith("skaters") or "skater" in low:
            fixed.append("Skaters")
        elif txt.upper().startswith("DATE"):
            fixed.append("DATE")
        else:
            fixed.append(txt)
    return fixed

def normalize_player_name(name):
    if not isinstance(name, str):
        return name
    txt = clean_spaces(name)
    if "," not in txt:
        return txt
    last_part, first_part = txt.split(",", 1)
    last_part = clean_spaces(last_part)
    first_part = clean_spaces(first_part)
    suffixes = {"Jr","Jr.","Sr","Sr.","II","III","IV","V"}
    last_tokens = last_part.split()
    trail = ""
    if len(last_tokens) > 1 and last_tokens[-1] in suffixes:
        trail = last_tokens[-1]
        last_part = " ".join(last_tokens[:-1])
    parts = [first_part, last_part]
    if trail:
        parts.append(trail)
    return " ".join(parts)

def fix_player_names(df):
    if "Skaters" in df.columns:
        df["Skaters"] = df["Skaters"].astype(str).apply(normalize_player_name)
    return df

def standardize_start_year(s):
    y1 = parse_first_year(s)
    return f"{y1}-{y1+1}" if y1 else s

def add_years_to_start(year_str, delta):
    y1 = parse_first_year(year_str)
    if y1 is None:
        return year_str
    y = y1 + delta
    return f"{y}-{y+1}"

def expiry_year_from_start_and_length(start_year, length):
    L = int(length)
    return add_years_to_start(start_year, L - 1)

def compute_start_cap_pct(cap_hit_val, start_year):
    cap = cap_table.get(start_year)
    if not cap:
        return ""
    num = extract_numeric_value(cap_hit_val)
    numf = float(num) if num != "" else 0.0
    return round(100.0 * numf / float(cap), 1)

def parse_position_letter(pos_text):
    if not isinstance(pos_text, str):
        return ""
    p = pos_text.upper()

    if "G" in p:
        return "G"
    if "D" in p:
        return "D"
    if "LW" in p:
        return "L"
    if "RW" in p:
        return "R"
    if "C" in p:
        return "C"

    if p == "L":
        return "L"
    if p == "R":
        return "R"

    return ""

def expand_season_label(label):
    y1 = parse_first_year(label)
    return f"{y1}-{y1+1}" if y1 else label

def make_contract_key(row):
    name = str(row.get("Skaters","")).strip().lower()
    pos = row.get("Pos","")
    if pos == "D":
        bucket = "D"
    elif pos in ("L", "R", "C"):
        bucket = "F"
    elif pos == "G":
        bucket = "G"
    else:
        bucket = ""

    start = str(row.get("Start Year","")).strip()
    length = row.get("Length","")
    length = int(length)
    caphit = extract_numeric_value(row.get("Cap Hit",""))
    return (name, bucket, start, length, caphit)

def today_scrape_season():
    today = datetime.today()
    thisYear = today.year
    return f"{thisYear}-{thisYear+1}" if today.month >= 7 else f"{thisYear-1}-{thisYear}"

def standardize_status(s):
    if not s:
        return ""
    t = " ".join(str(s).split()).upper().replace(" ", "")
    t = t.replace("+=", "+")
    if t == "ARB":
        return "UFA+ARB"
    return t

# Webdriver
def make_driver():
    opts = webdriver.ChromeOptions()
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--window-size=1400,1000")
    if chrome_binary.exists() and chromedriver_path.exists():
        opts.binary_location = str(chrome_binary)
        service = Service(str(chromedriver_path))
        driver = webdriver.Chrome(service=service, options=opts)
    else:
        driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(90)
    driver.set_script_timeout(90)
    driver.implicitly_wait(0)
    return driver

def js_click(driver, el):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    try:
        driver.execute_script("arguments[0].click();", el)
    except:
        el.click()

# CapWages Scraping
capwages_base_url = "https://capwages.com/signings"
cw_thead_th = "//main//table//thead//th | //main//table//thead//tr//th"
cw_tbody_rows = "//main//table//tbody/tr"

def capwages_open(driver):
    time.sleep(1.25)
    driver.get(capwages_base_url)
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//main")))
    WebDriverWait(driver, 20).until(lambda d:
        len(d.find_elements(By.XPATH, cw_tbody_rows)) > 0 or len(d.find_elements(By.XPATH, "//main//*[self::select or @role='combobox']")) > 0)

def capwages_open_menu(driver):
    selects = driver.find_elements(By.XPATH, "//main//select")
    if selects:
        return
    for xp in [
        "//main//*[@role='combobox']",
        "//main//button",
        "//main//*[contains(@class,'select') or contains(@class,'dropdown')][self::div or self::span]"
        ]:
        els = driver.find_elements(By.XPATH, xp)
        if els:
            js_click(driver, els[0])
            time.sleep(0.2)
            return

def capwages_select_option(driver, label_text):
    selects = driver.find_elements(By.XPATH, "//main//select")
    if selects:
        try:
            Select(selects[0]).select_by_visible_text(label_text)
            time.sleep(0.3)
            return True
        except Exception:
            sel = Select(selects[0])
            for opt in sel.options:
                if label_text.strip() in (opt.text or "").strip():
                    opt.click()
                    time.sleep(0.3)
                    return True
            return False
    for xp in [
        "//*[@role='listbox']//*[self::li or @role='option']",
        "//ul//li",
        "//button",
        "//div[@role='option' or @role='menuitem']"
       ]:
        for el in driver.find_elements(By.XPATH, xp):
            txt = (el.get_attribute("textContent") or el.text or "").strip()
            if not txt:
                continue
            if txt == label_text or label_text in txt:
                js_click(driver, el)
                time.sleep(0.3)
                return True
    return False

def scrape_capwages_table(driver):
    container = None
    prev = -1
    stable = 0
    max_rounds = 10

    for _ in range(max_rounds):
        # current rendered row count
        rows_now = driver.find_elements(By.XPATH, cw_tbody_rows)
        count = len(rows_now)
        if count == prev:
            stable += 1
        else:
            stable = 0

        if stable >= 2:
            break

        # scroll inner container to load all lines
        if container is not None:
            time.sleep(1.0) 
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", container)
        
        time.sleep(1.5) 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.5)
        prev = count

    # return to top
    if container is not None:
        time.sleep(1.0)
        driver.execute_script("arguments[0].scrollTop = 0", container)
    time.sleep(1.5)
    driver.execute_script("window.scrollTo(0, 0)")


    # read headers and rows
    ths = driver.find_elements(By.XPATH, cw_thead_th)
    raw_headers = [(t.get_attribute('textContent') or t.text or "").strip() for t in ths]
    headers = fix_headers(raw_headers)

    rows = driver.find_elements(By.XPATH, cw_tbody_rows)
    data = []
    max_cells = 0
    for r in rows:
        try:
            tds = r.find_elements(By.TAG_NAME, "td")
            cells = []
            for td in tds:
                try:
                    cells.append(td.text.strip())
                except StaleElementReferenceException:
                    # Refind td
                    fresh_tds = r.find_elements(By.TAG_NAME, "td")
                    idx = tds.index(td)
                    if idx < len(fresh_tds):
                        cells.append(fresh_tds[idx].text.strip())
                    else:
                        cells.append("")
            if any(cells):
                data.append(cells)
                max_cells = max(max_cells, len(cells))
    
        except StaleElementReferenceException:
            # Refetch the row and retry once
            try:
                fresh_rows = driver.find_elements(By.XPATH, cw_tbody_rows)
                idx = rows.index(r)
                fresh_r = fresh_rows[idx]
                fresh_tds = fresh_r.find_elements(By.TAG_NAME, "td")
                cells = [td.text.strip() for td in fresh_tds]
                if any(cells):
                    data.append(cells)
                    max_cells = max(max_cells, len(cells))
            except Exception:
                continue

    if not headers and max_cells > 0:
        headers = [f"COL{i+1}" for i in range(max_cells)]

    if headers and max_cells > 0:
        if len(headers) > max_cells:
            headers = headers[:max_cells]
        elif len(headers) < max_cells:
            headers = headers + [f"COL{i+1}" for i in range(len(headers), max_cells)]

        fixed = []
        for row in data:
            if len(row) > len(headers):
                fixed.append(row[:len(headers)])
            elif len(row) < len(headers):
                fixed.append(row + [""] * (len(headers) - len(row)))
            else:
                fixed.append(row)
        data = fixed

        return pd.DataFrame(data, columns=headers)
    else:
        return pd.DataFrame(data)

def harmonize_capwages_df(source_df, season_label_str):
    temp_df = source_df.copy()

    rename = {}
    for c in temp_df.columns:
        cu = clean_spaces(c).upper()
        if cu in ("PLAYER", "SKATER", "SKATERS"):
            rename[c] = "Skaters"
        elif cu in ("YRS","TERM","LENGTH"):
            rename[c] = "YRS"
        elif cu == "CAP HIT":
            rename[c] = "CAP HIT"
        elif cu == "TYPE":
            rename[c] = "TYPE"
        elif cu == "DATE":
            rename[c] = "DATE"
        elif cu == "POS":
            rename[c] = "POS"
    if rename:
        temp_df = temp_df.rename(columns=rename)

    if "Skaters" in temp_df.columns:
        temp_df["Skaters"] = temp_df["Skaters"].astype(str).apply(normalize_player_name)

    if "DATE" in temp_df.columns:
        def ensure_year(s, lab):
            txt = clean_spaces(s)
            if txt == "":
                return txt
            if contains_year(txt):
                return txt
            y1 = parse_first_year(lab)
            return f"{txt}-{y1}" if y1 is not None else txt
        temp_df["DATE"] = temp_df["DATE"].apply(lambda x: ensure_year(x, season_label_str))

    temp_df["Signing Season"] = season_label_str

    def parse_type(t):
        txt = str(t).strip().lower()
        structure = "2-way" if ("2-way" in txt or "(2-way" in txt) else "1-way"
        base = txt.split("(")[0].strip().upper() if "(" in txt else txt.upper()
        base = base if base else "STD"
        is_ext = "ext" in txt
        return structure, base, is_ext

    if "TYPE" in temp_df.columns:
        parsed_values = temp_df["TYPE"].map(parse_type)
        temp_df["Structure"] = [v[0] for v in parsed_values]
        temp_df["Level"] = [v[1] for v in parsed_values]
        temp_df["is_extension_flag"] = [v[2] for v in parsed_values]
    else:
        temp_df["Structure"] = "1-way"
        temp_df["Level"] = "STD"
        temp_df["is_extension_flag"] = False

    temp_df["Start Year"] = standardize_start_year(season_label_str)
    if "is_extension_flag" in temp_df.columns:
        mask_ext = temp_df["is_extension_flag"] == True
        if mask_ext.any():
            temp_df.loc[mask_ext, "Start Year"] = temp_df.loc[mask_ext, "Start Year"].apply(lambda s: add_years_to_start(s, 1))

    if "POS" in temp_df.columns:
        temp_df["Pos"] = temp_df["POS"].apply(parse_position_letter)
    else:
        temp_df["Pos"] = ""
    temp_df = temp_df[temp_df["Pos"] != "G"].copy()

    # approximate signing age (data has current age, so use delta from current season to signing season. Eventually erplace with birth date calculation)
    scrape_season = today_scrape_season()
    try:
        scrape_y1 = int(scrape_season.split("-")[0])
        sign_y1 = int(season_label_str.split("-")[0])
        delta = max(0, scrape_y1 - sign_y1)
    except Exception:
        delta = 0

    def age_to_signing(a):
        s = str(a)
        digits = "".join([c for c in s if c.isdigit()])
        return max(0, int(digits) - delta) if digits != "" else ""

    if "AGE" in temp_df.columns:
        temp_df["Signing Age"] = temp_df["AGE"].apply(age_to_signing)
    else:
        temp_df["Signing Age"] = ""

    if "YRS" in temp_df.columns:
        temp_df["Length"] = temp_df["YRS"].apply(lambda x: "".join([c for c in str(x) if c.isdigit()]) if pd.notnull(x) else "")
        temp_df["Length"] = temp_df["Length"].replace("", pd.NA).astype("Int64")
    else:
        temp_df["Length"] = pd.NA

    if "Length" in temp_df.columns:
        eight_mask = temp_df["Length"] == 8
        if "is_extension_flag" in temp_df.columns:
            eight_mask = eight_mask & (temp_df["is_extension_flag"] == False)
        if eight_mask.any():
            temp_df.loc[eight_mask, "Start Year"] = temp_df.loc[eight_mask, "Start Year"].apply(lambda s: add_years_to_start(s, 1))

    temp_df = temp_df.drop(columns=["is_extension_flag"], errors="ignore")
    temp_df["Cap Hit"] = temp_df["CAP HIT"].astype(str) if "CAP HIT" in temp_df.columns else ""
    temp_df["Start Year"] = temp_df["Start Year"].apply(standardize_start_year)
    temp_df["Expiry Year"] = temp_df.apply(lambda r: expiry_year_from_start_and_length(r.get("Start Year",""), r.get("Length","")), axis=1)
    temp_df["Start Yr Cap %"] = temp_df.apply(lambda r: compute_start_cap_pct(r.get("Cap Hit",""), r.get("Start Year","")), axis=1)

    for c in ["Shot","W(lbs)","H(f)","Clauses","Signing Status","Expiry Status","Signing Agent","Signing GM"]:
        temp_df[c] = ""

    for col in ["Signing Status", "Expiry Status"]:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].apply(standardize_status)

    keep = ["Skaters","Pos","Length","Level","Cap Hit","Start Yr Cap %","Structure","Start Year",
            "Signing Age","Expiry Year","Signing Season","Shot","W(lbs)","H(f)","Clauses",
            "Signing Status","Expiry Status","Signing Agent","Signing GM"]
    for c in keep:
        if c not in temp_df.columns:
            temp_df[c] = ""
    return temp_df[keep].copy()

def scrape_capwages(start_year, end_year):
    time.sleep(1.0)
    driver = make_driver()
    out = {}
    capwages_open(driver)
    first_year = max(2015, start_year)
    for year in range(first_year, end_year + 1):
        label = f"{year}-{str(year + 1)[-2:]}"
        capwages_open_menu(driver)
        if not capwages_select_option(driver, label):
            continue

        time.sleep(1.25)
        df = scrape_capwages_table(driver)
        if not df.empty:
            out[label] = df

    driver.quit()
    return out

# Main driver
def main():
    ensure_dir(master_out)
    print("Scraping CapWages")
    cw_raw = scrape_capwages(start_year, end_year)
    master = pd.DataFrame(columns=target_columns)
    for label, source_df in cw_raw.items():
        full_label = expand_season_label(label)
        cw_df = harmonize_capwages_df(source_df, full_label)
        if len(master) > 0:
            master["dedup_key"] = master.apply(make_contract_key, axis=1)
        
        cw_df["dedup_key"] = cw_df.apply(make_contract_key, axis=1)
        if len(master) > 0:
            mask = ~cw_df["dedup_key"].isin(set(master["dedup_key"]))
            keep_new = cw_df[mask].drop(columns=["dedup_key"], errors="ignore")
            master = master.drop(columns=["dedup_key"], errors="ignore")
            master = pd.concat([master, keep_new], ignore_index=True)
        else:
            master = cw_df.drop(columns=["dedup_key"], errors="ignore").copy()

    for c in target_columns:
        if c not in master.columns:
            master[c] = ""
    master = master[target_columns].copy()
    if len(master):
        master["dedup_key"] = master.apply(make_contract_key, axis=1)
        master = master.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"])

    master.to_csv(master_out, index=False, encoding="utf-8-sig")
    print(f"Contract Master saved with {len(master)} rows")

if __name__ == "__main__":
    main()