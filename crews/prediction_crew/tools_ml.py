"""
כלים ל-Crew 2 — Feature Engineering + ML
"""

from crewai.tools import tool
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
import joblib, json, os
from datetime import datetime

CLEAN_PATH      = "data/processed/clean_data.csv"
FEATURES_PATH   = "data/processed/features.csv"
MODEL_RF_PATH   = "models/model_rf.pkl"
MODEL_LR_PATH   = "models/model_lr.pkl"
MODEL_BEST_PATH = "models/model.pkl"
META_PATH       = "models/model_meta.json"
EVAL_PATH       = "data/processed/evaluation_report.md"
CARD_PATH       = "data/processed/model_card.md"


@tool("run_feature_engineering")
def create_features(action: str) -> str:
    """
    טוען clean_data.csv, מבצע Feature Engineering ושומר features.csv.
    Call this tool with action="run" to execute. No other value is valid.
    """
    try:
        df = pd.read_csv(CLEAN_PATH, encoding="utf-8-sig")
        df["תאריך"] = pd.to_datetime(df["תאריך"])

        # ── משתנה מטרה: רווח גבוה = מעל החציון ─────────────────────────
        median_profit = df["אחוז_רווח"].median()
        df["is_high_profit"] = (df["אחוז_רווח"] > median_profit).astype(int)

        # ── פיצ'רים מתאריך ───────────────────────────────────────────────
        df["month"]      = df["תאריך"].dt.month
        df["day_of_week_num"] = df["תאריך"].dt.dayofweek
        df["is_holiday"] = (df["חג"].notna() & (df["חג"] != "")).astype(int)
        df["is_weekend"]  = (df["יום_בשבוע"].isin(["שישי", "שבת"])).astype(int)

        # ── קידוד קטגוריאלי (Label Encoding) ─────────────────────────────
        cat_cols = ["קטגוריה", "רשת", "אזור", "עונה", "יום_בשבוע"]
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le.classes_.tolist()

        # ── בחירת פיצ'רים סופיים ──────────────────────────────────────────
        feature_cols = [
            "כמות", "מחיר_יחידה", "הנחה_אחוז", "month",
            "day_of_week_num", "is_holiday", "is_weekend",
            "קטגוריה_enc", "רשת_enc", "אזור_enc", "עונה_enc",
        ]

        features_df = df[feature_cols + ["is_high_profit"]].copy()

        os.makedirs("data/processed", exist_ok=True)
        features_df.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")

        # שמירת מטאדאטה של ה-encoding לשימוש עתידי
        os.makedirs("models", exist_ok=True)
        with open("models/encoders.json", "w", encoding="utf-8") as f:
            json.dump({"encoders": encoders,
                       "median_profit_threshold": median_profit,
                       "feature_cols": feature_cols}, f, ensure_ascii=False, indent=2)

        class_dist = df["is_high_profit"].value_counts().to_dict()

        return json.dumps({
            "status": "success",
            "saved_to": FEATURES_PATH,
            "total_rows": len(features_df),
            "features_count": len(feature_cols),
            "feature_names": feature_cols,
            "target": "is_high_profit",
            "target_distribution": {
                "high_profit (1)": class_dist.get(1, 0),
                "low_profit (0)":  class_dist.get(0, 0),
            },
            "profit_threshold": round(median_profit, 2),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"שגיאה ב-Feature Engineering: {e}"


@tool("run_ml_training")
def train_models(action: str) -> str:
    """
    טוען features.csv, מאמן Random Forest ו-Logistic Regression,
    שומר את שני המודלים ואת המודל הטוב יותר כ-model.pkl.
    Call this tool with action="run" to execute. No other value is valid.
    """
    try:
        df = pd.read_csv(FEATURES_PATH, encoding="utf-8-sig")

        X = df.drop(columns=["is_high_profit"])
        y = df["is_high_profit"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # נרמול לרגרסיה לוגיסטית
        scaler  = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ── Random Forest ────────────────────────────────────────────────
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        rf_f1  = f1_score(y_test, rf.predict(X_test), average="weighted")
        rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

        # ── Logistic Regression ──────────────────────────────────────────
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
        lr_f1  = f1_score(y_test, lr.predict(X_test_scaled), average="weighted")
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

        # שמירת מודלים
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf, MODEL_RF_PATH)
        joblib.dump({"model": lr, "scaler": scaler}, MODEL_LR_PATH)

        # המודל הטוב יותר = זה עם F1 גבוה יותר
        best_name = "Random Forest" if rf_f1 >= lr_f1 else "Logistic Regression"
        if rf_f1 >= lr_f1:
            joblib.dump(rf, MODEL_BEST_PATH)
        else:
            joblib.dump({"model": lr, "scaler": scaler}, MODEL_BEST_PATH)

        # שמירת מטאדאטה
        meta = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "train_size": len(X_train),
            "test_size":  len(X_test),
            "best_model": best_name,
            "random_forest":        {"accuracy": round(rf_acc,4), "f1": round(rf_f1,4), "auc": round(rf_auc,4)},
            "logistic_regression":  {"accuracy": round(lr_acc,4), "f1": round(lr_f1,4), "auc": round(lr_auc,4)},
            "feature_importances":  dict(sorted(
                zip(X.columns, rf.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:5]),
        }
        # המרה ל-float רגיל לשמירה ב-JSON
        for k in meta["feature_importances"]:
            meta["feature_importances"][k] = round(float(meta["feature_importances"][k]), 4)

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return json.dumps(meta, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"שגיאה באימון מודלים: {e}"


@tool("run_ml_evaluation")
def evaluate_models(action: str) -> str:
    """
    מעריך את שני המודלים ושומר evaluation_report.md מפורט.
    Call this tool with action="run" to execute. No other value is valid.
    """
    try:
        df   = pd.read_csv(FEATURES_PATH, encoding="utf-8-sig")
        X    = df.drop(columns=["is_high_profit"])
        y    = df["is_high_profit"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler  = StandardScaler()
        scaler.fit(X)
        X_test_scaled = scaler.transform(X_test)

        rf = joblib.load(MODEL_RF_PATH)
        lr_bundle = joblib.load(MODEL_LR_PATH)
        lr, sc = lr_bundle["model"], lr_bundle["scaler"]

        rf_pred = rf.predict(X_test)
        lr_pred = lr.predict(sc.transform(X_test))

        def make_report(name, y_true, y_pred, model=None, X=None):
            cr = classification_report(y_true, y_pred,
                                       target_names=["Low Profit","High Profit"])
            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="weighted")
            section = f"""
## {name}

| מדד | ערך |
|-----|-----|
| Accuracy | {acc:.4f} |
| F1 Score (weighted) | {f1:.4f} |

### Classification Report
```
{cr}
```

### Confusion Matrix
|  | Predicted Low | Predicted High |
|--|--------------|----------------|
| **Actual Low**  | {cm[0][0]} | {cm[0][1]} |
| **Actual High** | {cm[1][0]} | {cm[1][1]} |
"""
            if model and hasattr(model, "feature_importances_") and X is not None:
                top5 = sorted(zip(X.columns, model.feature_importances_),
                              key=lambda x: x[1], reverse=True)[:5]
                fi_rows = "\n".join(f"| {n} | {v:.4f} |" for n,v in top5)
                section += f"""
### Feature Importance (Top 5)
| Feature | Importance |
|---------|-----------|
{fi_rows}
"""
            return section

        rf_section = make_report("Random Forest", y_test, rf_pred, rf, X_test)
        lr_section = make_report("Logistic Regression", y_test, lr_pred)

        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

        best = meta.get("best_model", "Random Forest")
        rf_f1 = meta["random_forest"]["f1"]
        lr_f1 = meta["logistic_regression"]["f1"]

        report = f"""# Evaluation Report — Israeli Retail Profit Classifier

**תאריך:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**מטרה:** חיזוי האם עסקה תניב רווח גבוה (מעל החציון)
**גודל Test Set:** {len(X_test):,} עסקאות

## השוואת מודלים

| מודל | Accuracy | F1 Score | AUC |
|------|---------|---------|-----|
| Random Forest | {meta['random_forest']['accuracy']} | {meta['random_forest']['f1']} | {meta['random_forest']['auc']} |
| Logistic Regression | {meta['logistic_regression']['accuracy']} | {meta['logistic_regression']['f1']} | {meta['logistic_regression']['auc']} |

**מודל מומלץ: {best}**
{'Random Forest עדיף ב-F1: ' + str(round(rf_f1-lr_f1,4)) if rf_f1>lr_f1 else 'Logistic Regression עדיף ב-F1: ' + str(round(lr_f1-rf_f1,4))}

{rf_section}
{lr_section}

---
*נוצר אוטומטית על ידי Evaluation Agent — CrewAI Israeli Retail Project*
"""

        with open(EVAL_PATH, "w", encoding="utf-8") as f:
            f.write(report)

        return json.dumps({
            "status": "success",
            "saved_to": EVAL_PATH,
            "best_model": best,
            "rf_f1":  rf_f1,
            "lr_f1":  lr_f1,
            "rf_accuracy":  meta["random_forest"]["accuracy"],
            "lr_accuracy":  meta["logistic_regression"]["accuracy"],
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"שגיאה בהערכת מודלים: {e}"


@tool("create_model_card")
def create_model_card(card_text: str) -> str:
    """
    מקבל טקסט Model Card ושומר כ-model_card.md.
    """
    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

        content = f"""# Model Card — Israeli Retail Profit Classifier

**תאריך:** {datetime.now().strftime('%Y-%m-%d')}

## פרטי המודל

| שדה | ערך |
|-----|-----|
| שם המודל | {meta.get('best_model','Random Forest')} |
| גרסה | 1.0 |
| משימה | Binary Classification |
| מטרה | חיזוי רווח גבוה/נמוך בעסקה קמעונאית |

## ביצועים

| מדד | Random Forest | Logistic Regression |
|-----|--------------|---------------------|
| Accuracy | {meta['random_forest']['accuracy']} | {meta['logistic_regression']['accuracy']} |
| F1 Score | {meta['random_forest']['f1']} | {meta['logistic_regression']['f1']} |
| AUC | {meta['random_forest']['auc']} | {meta['logistic_regression']['auc']} |

## נתוני אימון

- **מקור:** Israeli Retail Dataset (סינתטי)
- **גודל Train:** {meta.get('train_size',0):,} עסקאות
- **גודל Test:**  {meta.get('test_size',0):,} עסקאות
- **תקופה:** 2022–2024

## פיצ'רים

{chr(10).join(f"- `{f}`" for f in ["כמות", "מחיר_יחידה", "הנחה_אחוז", "month", "day_of_week_num", "is_holiday", "is_weekend", "קטגוריה_enc", "רשת_enc", "אזור_enc", "עונה_enc"])}

## מגבלות

- הדאטאסט סינתטי — יש לאמת מול נתונים אמיתיים לפני שימוש בייצור
- המודל לא לוקח בחשבון מחזוריות ארוכת-טווח
- ביצועים עשויים להשתנות בסגמנטים קטנים (ערים קטנות, מוצרים נדירים)

## שימוש מיועד

זיהוי עסקאות עם פוטנציאל רווח גבוה לצורך:
- תמחור דינמי
- ניהול מלאי חכם
- מיקוד מאמצי שיווק

---

{card_text}

---
*נוצר אוטומטית על ידי Model Card Agent — CrewAI Israeli Retail Project*
"""
        with open(CARD_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        return f"model_card.md נשמר בהצלחה ב-{CARD_PATH}"

    except Exception as e:
        return f"שגיאה ביצירת model card: {e}"
