"""
בדיקות — איכות נתונים
"""
import pytest
import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).parent.parent
CLEAN_PATH = ROOT / "data/processed/clean_data.csv"
RAW_PATH   = ROOT / "data/raw/israeli_retail.csv"

REQUIRED_COLS = [
    "מספר_הזמנה", "תאריך", "יום_בשבוע", "עונה",
    "רשת", "עיר", "אזור", "קטגוריה", "כמות",
    "מחיר_יחידה", "הנחה_אחוז", "מכירות_ש", "רווח_ש", "אחוז_רווח",
]


@pytest.fixture
def clean_df():
    return pd.read_csv(CLEAN_PATH, encoding="utf-8-sig")


def test_clean_file_exists():
    assert CLEAN_PATH.exists(), "clean_data.csv לא נמצא — הרץ run_flow.py"


def test_clean_has_required_columns(clean_df):
    missing = [c for c in REQUIRED_COLS if c not in clean_df.columns]
    assert not missing, f"עמודות חסרות: {missing}"


def test_clean_has_enough_rows(clean_df):
    assert len(clean_df) >= 9000, f"יותר מדי שורות הוסרו: {len(clean_df)}"


def test_no_negative_sales(clean_df):
    assert (clean_df["מכירות_ש"] > 0).all(), "יש ערכי מכירות שליליים"


def test_no_negative_quantity(clean_df):
    assert (clean_df["כמות"] > 0).all(), "יש ערכי כמות שליליים"


def test_no_duplicate_orders(clean_df):
    dups = clean_df.duplicated(subset=["מספר_הזמנה"]).sum()
    assert dups == 0, f"יש {dups} כפילויות במספרי הזמנה"


def test_profit_percentage_range(clean_df):
    assert (clean_df["אחוז_רווח"] >= 0).all(), "יש אחוזי רווח שליליים"
    assert (clean_df["אחוז_רווח"] <= 100).all(), "יש אחוזי רווח מעל 100%"


def test_date_column_parseable(clean_df):
    dates = pd.to_datetime(clean_df["תאריך"], errors="coerce")
    assert dates.notna().all(), "יש תאריכים לא תקינים"


def test_date_range(clean_df):
    dates = pd.to_datetime(clean_df["תאריך"])
    assert dates.min().year >= 2020, "תאריכים עתיקים מדי"
    assert dates.max().year <= 2026, "תאריכים עתידיים"
