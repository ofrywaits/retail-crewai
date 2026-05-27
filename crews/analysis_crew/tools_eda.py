"""
כלים ל-Day 3: EDA ותובנות
"""

from crewai.tools import tool
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ללא חלון גרפי
import matplotlib.pyplot as plt
import seaborn as sns
import base64, io, json, os
from datetime import datetime

CLEAN_PATH    = "data/processed/clean_data.csv"
EDA_HTML_PATH = "data/processed/eda_report.html"
INSIGHTS_PATH = "data/processed/insights.md"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# ─── עזר: המרת figure ל-base64 ──────────────────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded


# ─── עזר: צבעים ──────────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]


@tool("generate_eda_report")
def generate_eda_report(dummy: str = "run") -> str:
    """
    קורא את clean_data.csv, מייצר 6 גרפים ובונה דו"ח HTML מלא.
    שומר ב-data/processed/eda_report.html.
    מחזיר סיכום של הממצאים המרכזיים.
    """
    try:
        df = pd.read_csv(CLEAN_PATH, encoding="utf-8-sig")
        df["תאריך"] = pd.to_datetime(df["תאריך"])
        charts = {}

        # ── גרף 1: מכירות לפי קטגוריה ─────────────────────────────────────
        cat = df.groupby("קטגוריה")["מכירות_ש"].sum().sort_values()
        fig, ax = plt.subplots()
        cat.plot(kind="barh", ax=ax, color=COLORS[:len(cat)])
        ax.set_title("מכירות לפי קטגוריה (₪)", fontsize=13)
        ax.set_xlabel("סה\"כ מכירות (₪)")
        for bar in ax.patches:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"₪{bar.get_width():,.0f}", va="center", fontsize=8)
        charts["קטגוריות"] = _fig_to_b64(fig)

        # ── גרף 2: טרנד חודשי ──────────────────────────────────────────────
        monthly = df.groupby(df["תאריך"].dt.to_period("M"))["מכירות_ש"].sum()
        fig, ax = plt.subplots(figsize=(12, 4))
        monthly.plot(ax=ax, color="#2196F3", linewidth=2, marker="o", markersize=3)
        ax.set_title("טרנד מכירות חודשי 2022–2024", fontsize=13)
        ax.set_ylabel("מכירות (₪)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=7)
        charts["טרנד_חודשי"] = _fig_to_b64(fig)

        # ── גרף 3: מכירות לפי רשת ──────────────────────────────────────────
        chain = df.groupby("רשת")["מכירות_ש"].sum().sort_values()
        fig, ax = plt.subplots()
        chain.plot(kind="barh", ax=ax, color="#FF9800")
        ax.set_title("מכירות לפי רשת (₪)", fontsize=13)
        charts["רשתות"] = _fig_to_b64(fig)

        # ── גרף 4: ימי שבוע ────────────────────────────────────────────────
        day_order = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
        day = df.groupby("יום_בשבוע")["מכירות_ש"].sum().reindex(day_order, fill_value=0)
        colors_day = ["#E91E63" if d == "שבת" else "#4CAF50" for d in day_order]
        fig, ax = plt.subplots()
        day.plot(kind="bar", ax=ax, color=colors_day)
        ax.set_title("מכירות לפי יום בשבוע (אדום = שבת)", fontsize=13)
        ax.set_ylabel("מכירות (₪)")
        plt.xticks(rotation=0)
        charts["ימי_שבוע"] = _fig_to_b64(fig)

        # ── גרף 5: אזורים ──────────────────────────────────────────────────
        region = df.groupby("אזור")["מכירות_ש"].sum().sort_values()
        fig, ax = plt.subplots()
        region.plot(kind="pie", ax=ax, autopct="%1.1f%%",
                    colors=COLORS[:len(region)], startangle=90)
        ax.set_title("חלוקת מכירות לפי אזור", fontsize=13)
        ax.set_ylabel("")
        charts["אזורים"] = _fig_to_b64(fig)

        # ── גרף 6: רווחיות לפי קטגוריה ────────────────────────────────────
        prof = df.groupby("קטגוריה").agg(
            מכירות=("מכירות_ש", "sum"),
            רווח=("רווח_ש", "sum")
        )
        prof["אחוז_רווח"] = (prof["רווח"] / prof["מכירות"] * 100).round(1)
        fig, ax = plt.subplots()
        prof["אחוז_רווח"].sort_values().plot(kind="barh", ax=ax, color="#9C27B0")
        ax.set_title("אחוז רווח לפי קטגוריה (%)", fontsize=13)
        ax.set_xlabel("אחוז רווח (%)")
        charts["רווחיות"] = _fig_to_b64(fig)

        # ── סטטיסטיקות לדו"ח ───────────────────────────────────────────────
        top_cat    = df.groupby("קטגוריה")["מכירות_ש"].sum().idxmax()
        top_chain  = df.groupby("רשת")["מכירות_ש"].sum().idxmax()
        top_city   = df.groupby("עיר")["מכירות_ש"].sum().idxmax()
        top_month  = df.groupby(df["תאריך"].dt.month)["מכירות_ש"].sum().idxmax()
        months_heb = {1:"ינואר",2:"פברואר",3:"מרץ",4:"אפריל",5:"מאי",6:"יוני",
                      7:"יולי",8:"אוגוסט",9:"ספטמבר",10:"אוקטובר",11:"נובמבר",12:"דצמבר"}

        stats = {
            "סה\"כ_מכירות": f"₪{df['מכירות_ש'].sum():,.0f}",
            "סה\"כ_רווח":    f"₪{df['רווח_ש'].sum():,.0f}",
            "אחוז_רווח_כללי": f"{df['רווח_ש'].sum()/df['מכירות_ש'].sum()*100:.1f}%",
            "מספר_עסקאות":   f"{len(df):,}",
            "ממוצע_עסקה":    f"₪{df['מכירות_ש'].mean():,.1f}",
            "קטגוריה_מובילה": top_cat,
            "רשת_מובילה":     top_chain,
            "עיר_מובילה":     top_city,
            "חודש_שיא":       months_heb.get(top_month, str(top_month)),
        }

        # ── בניית HTML ──────────────────────────────────────────────────────
        stats_rows = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in stats.items()
        )
        chart_tags = "".join(
            f'<div class="chart-box"><h3>{title}</h3>'
            f'<img src="data:image/png;base64,{b64}" alt="{title}"></div>'
            for title, b64 in charts.items()
        )

        html = f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
  <meta charset="UTF-8">
  <title>EDA Report — Israeli Retail Dataset</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; direction: rtl; }}
    h1   {{ color: #1565C0; text-align: center; }}
    h2   {{ color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 5px; }}
    .kpi-table {{ width: 60%; margin: 0 auto 30px; border-collapse: collapse; }}
    .kpi-table td {{ padding: 8px 14px; border: 1px solid #ddd; font-size: 15px; }}
    .kpi-table tr:nth-child(even) {{ background: #E3F2FD; }}
    .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .chart-box {{ background: white; border-radius: 8px; padding: 15px;
                  box-shadow: 0 2px 6px rgba(0,0,0,.1); }}
    .chart-box h3 {{ margin: 0 0 10px; color: #555; font-size: 14px; }}
    .chart-box img {{ width: 100%; border-radius: 4px; }}
    .footer {{ text-align: center; color: #999; margin-top: 30px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>דו"ח EDA — דאטאסט קמעונאי ישראלי</h1>
  <p style="text-align:center;color:#666">נוצר: {datetime.now().strftime('%Y-%m-%d %H:%M')} | שורות: {len(df):,} | תקופה: 2022–2024</p>

  <h2>מדדים מרכזיים</h2>
  <table class="kpi-table">{stats_rows}</table>

  <h2>גרפים</h2>
  <div class="charts-grid">{chart_tags}</div>

  <div class="footer">נוצר אוטומטית על ידי EDA Agent — CrewAI Israeli Retail Project</div>
</body>
</html>"""

        os.makedirs("data/processed", exist_ok=True)
        with open(EDA_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html)

        return json.dumps({
            "status": "success",
            "saved_to": EDA_HTML_PATH,
            "charts_generated": list(charts.keys()),
            "key_findings": stats,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"שגיאה ביצירת דו\"ח EDA: {e}"


@tool("generate_insights_md")
def generate_insights_md(insights_text: str) -> str:
    """
    מקבל טקסט תובנות ושומר כ-insights.md.
    השתמש בכלי זה לאחר שניתחת את הנתונים וגיבשת תובנות עסקיות.
    """
    try:
        os.makedirs("data/processed", exist_ok=True)

        # מוסיפים כותרת ותאריך
        content = f"""# תובנות עסקיות — דאטאסט קמעונאי ישראלי

**תאריך:** {datetime.now().strftime('%Y-%m-%d')}
**מקור נתונים:** clean_data.csv (9,900 עסקאות, 2022–2024)

---

{insights_text}

---
*נוצר אוטומטית על ידי Insights Agent — CrewAI Israeli Retail Project*
"""
        with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        return f"insights.md נשמר בהצלחה ב-{INSIGHTS_PATH} ({len(content)} תווים)"

    except Exception as e:
        return f"שגיאה בשמירת insights.md: {e}"


@tool("get_data_statistics")
def get_data_statistics(dummy: str = "run") -> str:
    """
    מחשב סטטיסטיקות מפורטות מ-clean_data.csv לצורך כתיבת תובנות.
    """
    try:
        df = pd.read_csv(CLEAN_PATH, encoding="utf-8-sig")
        df["תאריך"] = pd.to_datetime(df["תאריך"])

        months_heb = {1:"ינואר",2:"פברואר",3:"מרץ",4:"אפריל",5:"מאי",6:"יוני",
                      7:"יולי",8:"אוגוסט",9:"ספטמבר",10:"אוקטובר",11:"נובמבר",12:"דצמבר"}

        stats = {
            "כללי": {
                "סה\"כ_מכירות_ש\"ח": round(df["מכירות_ש"].sum(), 0),
                "סה\"כ_רווח_ש\"ח":    round(df["רווח_ש"].sum(), 0),
                "אחוז_רווח":          round(df["רווח_ש"].sum() / df["מכירות_ש"].sum() * 100, 1),
                "מספר_עסקאות":        len(df),
            },
            "לפי_קטגוריה": df.groupby("קטגוריה").agg(
                מכירות=("מכירות_ש", "sum"),
                רווח=("רווח_ש", "sum"),
                עסקאות=("מספר_הזמנה", "count")
            ).round(0).to_dict(),
            "לפי_רשת": df.groupby("רשת")["מכירות_ש"].sum().round(0).to_dict(),
            "לפי_אזור": df.groupby("אזור")["מכירות_ש"].sum().round(0).to_dict(),
            "לפי_יום_שבוע": df.groupby("יום_בשבוע")["מכירות_ש"].sum().round(0).to_dict(),
            "לפי_עונה": df.groupby("עונה")["מכירות_ש"].sum().round(0).to_dict(),
            "חגים": df[df["חג"].notna() & (df["חג"] != "")].groupby("חג")["מכירות_ש"].sum().round(0).to_dict(),
            "חודש_שיא": months_heb.get(
                int(df.groupby(df["תאריך"].dt.month)["מכירות_ש"].sum().idxmax()), "?"
            ),
            "יום_חזק_ביותר": df.groupby("יום_בשבוע")["מכירות_ש"].sum().idxmax(),
            "קטגוריה_רווחית_ביותר": df.groupby("קטגוריה")["אחוז_רווח"].mean().idxmax(),
        }

        return json.dumps(stats, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        return f"שגיאה: {e}"
