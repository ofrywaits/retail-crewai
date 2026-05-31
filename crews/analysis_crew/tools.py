"""
כלים (Tools) ל-Crew 1
כל כלי עושה פעולה אמיתית על הקבצים — הסוכנים קוראים לכלים האלה
"""

from crewai.tools import tool
import pandas as pd
import json
import os

RAW_PATH   = "data/raw/israeli_retail.csv"
CLEAN_PATH = "data/processed/clean_data.csv"
CONTRACT_PATH = "data/processed/dataset_contract.json"


@tool("load_and_inspect_data")
def load_and_inspect_data(action: str) -> str:
    """
    טוען את הדאטאסט הגולמי ומחזיר דו"ח מפורט.
    Call this tool with action="run" to execute. No other value is valid.
    """
    try:
        df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")

        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].to_dict()

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                "min":  round(float(df[col].min()), 2),
                "max":  round(float(df[col].max()), 2),
                "mean": round(float(df[col].mean()), 2),
            }

        neg_sales = int((df["מכירות_ש"] <= 0).sum())
        neg_qty   = int((df["כמות"] <= 0).sum())
        dups      = int(df.duplicated(subset=["מספר_הזמנה"]).sum())

        # קיצור הפלט לחיסכון בטוקנים
        key_stats = {c: stats[c] for c in ["מכירות_ש", "כמות", "מחיר_יחידה", "אחוז_רווח"] if c in stats}
        report = {
            "rows": len(df),
            "columns_count": len(df.columns),
            "columns": list(df.columns),
            "missing_values": missing_cols,
            "duplicate_order_ids": dups,
            "negative_sales_rows": neg_sales,
            "key_numeric_stats": key_stats,
            "categories": df["קטגוריה"].unique().tolist(),
            "date_range": {"from": str(df["תאריך"].min()), "to": str(df["תאריך"].max())},
        }

        return json.dumps(report, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        return f"שגיאה בטעינת הנתונים: {e}"


@tool("clean_and_save_data")
def clean_and_save_data(action: str) -> str:
    """
    מנקה את הדאטאסט הגולמי.
    Call this tool with action="run" to execute. No other value is valid.
    - מסיר כפילויות
    - מסיר שורות עם מכירות/כמות שליליות
    - מתקן סוגי נתונים
    - מסיר חריגים קיצוניים (top 1%)
    שומר את התוצאה ב-data/processed/clean_data.csv
    """
    try:
        df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
        original = len(df)
        log = []

        # המרת תאריך
        df["תאריך"] = pd.to_datetime(df["תאריך"])
        log.append("תאריך הומר ל-datetime")

        # הסרת כפילויות
        before = len(df)
        df = df.drop_duplicates(subset=["מספר_הזמנה"])
        log.append(f"הוסרו {before - len(df)} כפילויות")

        # הסרת ערכים שליליים
        before = len(df)
        df = df[(df["מכירות_ש"] > 0) & (df["כמות"] > 0) & (df["מחיר_יחידה"] > 0)]
        log.append(f"הוסרו {before - len(df)} שורות עם ערכים שליליים")

        # מילוי חג ריק
        df["חג"] = df["חג"].fillna("")
        log.append("ערכי חג חסרים מולאו במחרוזת ריקה")

        # הסרת חריגים (מעל percentile 99 במכירות)
        before = len(df)
        q99 = df["מכירות_ש"].quantile(0.99)
        df = df[df["מכירות_ש"] <= q99]
        log.append(f"הוסרו {before - len(df)} חריגים מעל ₪{q99:,.0f} (99th percentile)")

        # שמירה
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")

        summary = {
            "status": "success",
            "original_rows": original,
            "clean_rows": len(df),
            "removed_rows": original - len(df),
            "cleaning_steps": log,
            "saved_to": CLEAN_PATH,
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"שגיאה בניקוי הנתונים: {e}"


@tool("generate_dataset_contract")
def generate_dataset_contract(action: str) -> str:
    """
    קורא את clean_data.csv ומייצר dataset_contract.json.
    Call this tool with action="run" to execute. No other value is valid.
    סכמה מלאה של הדאטאסט — שמות עמודות, סוגים, טווחים, ערכים אפשריים.
    """
    try:
        df = pd.read_csv(CLEAN_PATH, encoding="utf-8-sig")

        columns_schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            col_info = {
                "type": dtype,
                "nullable": bool(df[col].isnull().any()),
                "unique_count": int(df[col].nunique()),
            }
            if df[col].dtype in ["float64", "int64"]:
                col_info["min"]  = round(float(df[col].min()), 2)
                col_info["max"]  = round(float(df[col].max()), 2)
                col_info["mean"] = round(float(df[col].mean()), 2)
            else:
                col_info["allowed_values"] = df[col].dropna().unique().tolist()[:10]

            columns_schema[col] = col_info

        contract = {
            "dataset_name": "Israeli Retail Sales Dataset",
            "version": "1.0",
            "created": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "source": "Synthetic — generated by data/generate_dataset.py",
            "description": "נתוני מכירות קמעונאיות ישראליות סינתטיים לשנים 2022–2024",
            "rows": len(df),
            "columns_count": len(df.columns),
            "date_range": {
                "from": str(df["תאריך"].min()),
                "to":   str(df["תאריך"].max()),
            },
            "columns": columns_schema,
        }

        with open(CONTRACT_PATH, "w", encoding="utf-8") as f:
            json.dump(contract, f, ensure_ascii=False, indent=2, default=str)

        return json.dumps({
            "status": "success",
            "saved_to": CONTRACT_PATH,
            "columns_documented": len(columns_schema),
            "rows_in_dataset": len(df),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"שגיאה ביצירת החוזה: {e}"
