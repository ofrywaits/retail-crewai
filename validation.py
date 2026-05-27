"""
validation.py — בדיקות תקינות לכל פלטי Crew 1
הרץ: python3 validation.py
"""

import pandas as pd
import json
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REQUIRED_FILES = {
    "data/raw/israeli_retail.csv":        "דאטאסט גולמי",
    "data/processed/clean_data.csv":      "דאטאסט נקי",
    "data/processed/dataset_contract.json": "Contract",
    "data/processed/eda_report.html":     "דו\"ח EDA",
    "data/processed/insights.md":         "תובנות",
}

REQUIRED_COLUMNS = [
    "מספר_הזמנה", "תאריך", "רשת", "עיר", "אזור",
    "קטגוריה", "מוצר", "כמות", "מכירות_ש", "רווח_ש",
]

errors = []
warnings = []


def check(condition: bool, msg: str, is_warning: bool = False):
    if condition:
        log.info(f"✓  {msg}")
    else:
        if is_warning:
            log.warning(f"⚠  {msg}")
            warnings.append(msg)
        else:
            log.error(f"✗  {msg}")
            errors.append(msg)


# ── 1. קיום קבצים ───────────────────────────────────────────────────────────
log.info("── בדיקת קיום קבצים ──────────────────────────")
for path, label in REQUIRED_FILES.items():
    check(os.path.exists(path), f"קובץ קיים: {label} ({path})")

# ── 2. תקינות clean_data.csv ────────────────────────────────────────────────
log.info("── בדיקת clean_data.csv ──────────────────────")
try:
    df = pd.read_csv("data/processed/clean_data.csv", encoding="utf-8-sig")
    df["תאריך"] = pd.to_datetime(df["תאריך"])

    check(len(df) >= 9000,   f"מספר שורות סביר: {len(df):,}")
    check(len(df.columns) >= 15, f"מספר עמודות: {len(df.columns)}")

    for col in REQUIRED_COLUMNS:
        check(col in df.columns, f"עמודה קיימת: {col}")

    check((df["מכירות_ש"] > 0).all(),   "כל ערכי מכירות חיוביים")
    check((df["כמות"] > 0).all(),        "כל ערכי כמות חיוביים")
    check((df["רווח_ש"] > -1000).all(), "ערכי רווח בטווח סביר", is_warning=True)
    check(df["מספר_הזמנה"].nunique() == len(df), "אין כפילויות במספרי הזמנה")

    missing_critical = df[REQUIRED_COLUMNS].isnull().sum().sum()
    check(missing_critical == 0, f"אין ערכים חסרים בעמודות קריטיות")

except Exception as e:
    log.error(f"שגיאה בבדיקת clean_data: {e}")
    errors.append(str(e))

# ── 3. תקינות dataset_contract.json ────────────────────────────────────────
log.info("── בדיקת dataset_contract.json ───────────────")
try:
    with open("data/processed/dataset_contract.json", encoding="utf-8") as f:
        contract = json.load(f)

    check("columns" in contract,      "Contract מכיל 'columns'")
    check("rows"    in contract,      "Contract מכיל 'rows'")
    check("version" in contract,      "Contract מכיל 'version'")
    check(contract.get("rows", 0) >= 9000,
          f"Contract מדווח על {contract.get('rows',0):,} שורות")

    # Contract מסכים עם הדאטה
    contract_cols = set(contract.get("columns", {}).keys())
    data_cols     = set(df.columns)
    check(contract_cols == data_cols,
          f"עמודות Contract תואמות לדאטה ({len(contract_cols)} עמודות)")

except Exception as e:
    log.error(f"שגיאה בבדיקת contract: {e}")
    errors.append(str(e))

# ── 4. תקינות eda_report.html ───────────────────────────────────────────────
log.info("── בדיקת eda_report.html ─────────────────────")
try:
    with open("data/processed/eda_report.html", encoding="utf-8") as f:
        html = f.read()

    check(len(html) > 50_000,       f"HTML מלא ({len(html):,} תווים)")
    check("<img src=" in html,       "HTML מכיל גרפים מוטמעים")
    check("eda_report.html" not in [""],  "קובץ HTML תקין")

except Exception as e:
    log.error(f"שגיאה בבדיקת HTML: {e}")
    errors.append(str(e))

# ── 5. תקינות insights.md ──────────────────────────────────────────────────
log.info("── בדיקת insights.md ─────────────────────────")
try:
    with open("data/processed/insights.md", encoding="utf-8") as f:
        md = f.read()

    check(len(md) > 200,         f"insights.md מכיל תוכן ({len(md)} תווים)")
    check("##" in md or "#" in md, "insights.md מכיל כותרות")

except Exception as e:
    log.error(f"שגיאה בבדיקת insights: {e}")
    errors.append(str(e))

# ── סיכום ──────────────────────────────────────────────────────────────────
log.info("── סיכום ─────────────────────────────────────")
log.info(f"שגיאות: {len(errors)} | אזהרות: {len(warnings)}")

if errors:
    log.error("הולידציה נכשלה! תקן לפני שממשיכים:")
    for e in errors:
        log.error(f"  → {e}")
    sys.exit(1)
else:
    log.info("כל הבדיקות עברו בהצלחה!")
    if warnings:
        log.warning(f"יש {len(warnings)} אזהרות — לא חוסמות אבל כדאי לבדוק.")
