"""
Streamlit Dashboard — Israeli Retail CrewAI Project
יום 6: ממשק משתמש
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import numpy as np
import subprocess
import sys
import datetime
import os
from pathlib import Path
from bidi.algorithm import get_display

def heb(text):
    return get_display(str(text))

def heb_list(lst):
    return [heb(x) for x in lst]

def style_chart(fig, ax):
    """Glassmorphism dark style for all matplotlib charts."""
    fig.patch.set_facecolor('#12122a')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d2d4e')

# ── נתיבים ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
CLEAN_PATH    = ROOT / "data/processed/clean_data.csv"
RAW_PATH      = ROOT / "data/raw/israeli_retail.csv"
INSIGHTS_PATH = ROOT / "data/processed/insights.md"
EVAL_PATH     = ROOT / "data/processed/evaluation_report.md"
META_PATH     = ROOT / "models/model_meta.json"
MODEL_PATH    = ROOT / "models/model.pkl"
ENCODERS_PATH = ROOT / "models/encoders.json"

REQUIRED_COLS = [
    "מספר_הזמנה", "תאריך", "יום_בשבוע", "עונה", "חג",
    "רשת", "עיר", "אזור", "קטגוריה", "כמות",
    "מחיר_יחידה", "הנחה_אחוז", "מכירות_ש", "רווח_ש", "אחוז_רווח",
]

COLORS = ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]

sns.set_theme(style="dark")
matplotlib.rcParams.update({
    "text.color": "#e2e8f0",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "figure.facecolor": "#12122a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#2d2d4e",
    "grid.color": "#2d2d4e",
    "grid.alpha": 0.5,
})

# ── הגדרות עמוד ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Israeli Retail AI Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ═══════════════════════════════════════════
   GLASSMORPHISM THEME — Israeli Retail AI
═══════════════════════════════════════════ */

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    background-attachment: fixed !important;
}
.main .block-container {
    background: transparent !important;
    padding-top: 1.5rem !important;
}

/* Metric cards — glass */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    padding: 18px 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(99,102,241,0.25) !important;
}
[data-testid="stMetricValue"] > div {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] > div {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.85) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: rgba(255,255,255,0.85) !important;
}

/* Titles — gradient text */
h1 {
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}
h2 {
    color: #c4b5fd !important;
    font-weight: 700 !important;
}
h3, h4 {
    color: #93c5fd !important;
    font-weight: 600 !important;
}
p, .stMarkdown {
    color: rgba(255,255,255,0.8) !important;
}

/* Buttons — gradient */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.6) !important;
}

/* Alerts */
.stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 12px !important; color: #6ee7b7 !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; border-radius: 12px !important; color: #fcd34d !important; }
.stError   { background: rgba(239,68,68,0.1)  !important; border: 1px solid rgba(239,68,68,0.3)  !important; border-radius: 12px !important; color: #fca5a5 !important; }
.stInfo    { background: rgba(99,102,241,0.1)  !important; border: 1px solid rgba(99,102,241,0.3) !important; border-radius: 12px !important; color: #a5b4fc !important; }

/* Inputs */
.stSelectbox > div > div,
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
/* Slider */
.stSlider [data-baseweb="slider"] { filter: hue-rotate(220deg); }

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Code */
code, pre {
    background: rgba(99,102,241,0.12) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.45); border-radius: 3px; }

/* DataFrames */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Caption */
.stCaption, small { color: rgba(255,255,255,0.45) !important; }

/* ═══════════════════════════════════════════
   RTL — עברית מימין לשמאל
═══════════════════════════════════════════ */
.stMarkdown,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol,
.stCaption,
.element-container p {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: plaintext !important;
}

/* כותרות עמוד */
h1, h2, h3, h4 {
    direction: rtl !important;
    text-align: right !important;
}

/* סיידבר — RTL */
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    direction: rtl !important;
    text-align: right !important;
}

/* שמור על קוד LTR */
code, pre, .stCode {
    direction: ltr !important;
    text-align: left !important;
}

/* metric labels — center */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    text-align: center !important;
    direction: rtl !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cache ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_PATH, encoding="utf-8-sig")
    df["תאריך"] = pd.to_datetime(df["תאריך"])
    return df

@st.cache_data
def load_meta():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_encoders():
    with open(ENCODERS_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🛒 Israeli Retail AI")
st.sidebar.markdown("**פרויקט CrewAI | ימים 1–5**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "בחר עמוד:",
    ["🏠 דשבורד ראשי", "📊 ניתוח נתונים", "💡 תובנות עסקיות",
     "🤖 מודל ML", "🔮 ניבוי", "📤 העלה נתונים", "📋 מוניטורינג"],
)

st.sidebar.markdown("---")
st.sidebar.caption("מונע על ידי CrewAI + Groq LLM")
st.sidebar.caption("מאגר: Israeli Retail 2022–2024")
st.sidebar.caption("~9,900 עסקאות | 19 עמודות")

# ── פידבק בצד (שלב 18: Iteration) ───────────────────────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("💬 פידבק"):
    feedback = st.text_area("כתוב פה הערות / רעיונות לשיפור:", key="feedback_text", height=80)
    if st.button("שלח פידבק", key="send_feedback"):
        if feedback.strip():
            feedback_path = ROOT / "logs/feedback.txt"
            feedback_path.parent.mkdir(exist_ok=True)
            with open(feedback_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M}] {feedback}\n")
            st.success("תודה! הפידבק נשמר ✅")
        else:
            st.warning("נא לכתוב משהו לפני השליחה")


# ════════════════════════════════════════════════════════════════════════════
# עמוד 1: דשבורד ראשי
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 דשבורד ראשי":
    st.title("🏠 דשבורד ראשי — Israeli Retail Sales")
    st.markdown("**ניתוח מכירות קמעונאיות ישראליות 2022–2024 | מונע על ידי CrewAI**")
    st.markdown("---")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ קובץ הנתונים לא נמצא. הרץ תחילה: `python3 run_flow.py`")
        st.info("לחלופין, עבור לעמוד **📤 העלה נתונים** כדי להעלות CSV משלך.")
        st.stop()
    except Exception as e:
        st.error(f"שגיאה בטעינת נתונים: {e}")
        st.stop()

    months_heb = {1:"ינואר",2:"פברואר",3:"מרץ",4:"אפריל",5:"מאי",6:"יוני",
                  7:"יולי",8:"אוגוסט",9:"ספטמבר",10:"אוקטובר",11:"נובמבר",12:"דצמבר"}

    # שורת KPI ראשונה
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 סה\"כ מכירות",   f"₪{df['מכירות_ש'].sum():,.0f}")
    c2.metric("📈 סה\"כ רווח",     f"₪{df['רווח_ש'].sum():,.0f}")
    c3.metric("📊 אחוז רווח",      f"{df['רווח_ש'].sum()/df['מכירות_ש'].sum()*100:.1f}%")
    c4.metric("🧾 עסקאות",         f"{len(df):,}")

    st.markdown("")

    # שורת KPI שנייה
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🏆 קטגוריה מובילה", df.groupby("קטגוריה")["מכירות_ש"].sum().idxmax())
    c6.metric("🏪 רשת מובילה",     df.groupby("רשת")["מכירות_ש"].sum().idxmax())
    c7.metric("🏙️ עיר מובילה",    df.groupby("עיר")["מכירות_ש"].sum().idxmax())
    c8.metric("📅 חודש שיא",       months_heb[int(df.groupby(df["תאריך"].dt.month)["מכירות_ש"].sum().idxmax())])

    st.markdown("---")

    # גרפים
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("מכירות לפי קטגוריה")
        cat = df.groupby("קטגוריה")["מכירות_ש"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        ax.barh(heb_list(cat.index), cat.values, color=COLORS[:len(cat)])
        ax.set_xlabel(heb("מכירות (₪)"))
        for bar in ax.patches:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"₪{bar.get_width()/1e6:.1f}M", va="center", fontsize=8, color="#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("טרנד מכירות חודשי")
        monthly = df.groupby(df["תאריך"].dt.to_period("M"))["מכירות_ש"].sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        monthly.plot(ax=ax, color="#6366f1", linewidth=2, marker="o", markersize=3)
        ax.set_ylabel(heb("מכירות (₪)"))
        ax.grid(True, color="#2d2d4e", alpha=0.5)
        plt.xticks(rotation=45, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# עמוד 2: EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 ניתוח נתונים":
    st.title("📊 ניתוח נתונים (EDA)")
    st.markdown("**6 גרפים של הדאטאסט — ניתן לסנן לפי קטגוריה ושנה**")
    st.markdown("---")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ נתונים לא נמצאו. הרץ `python3 run_flow.py` או העלה CSV.")
        st.stop()
    except Exception as e:
        st.error(f"שגיאה בטעינת נתונים: {e}")
        st.stop()

    with st.expander("🔍 סינון נתונים", expanded=False):
        fc1, fc2 = st.columns(2)
        cats  = fc1.multiselect("קטגוריות",
                                df["קטגוריה"].unique().tolist(),
                                default=df["קטגוריה"].unique().tolist())
        years = fc2.multiselect("שנים",
                                sorted(df["תאריך"].dt.year.unique()),
                                default=sorted(df["תאריך"].dt.year.unique()))
        if cats:
            df = df[df["קטגוריה"].isin(cats)]
        if years:
            df = df[df["תאריך"].dt.year.isin(years)]
        st.caption(f"מציג {len(df):,} עסקאות")

    # שורה 1
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("מכירות לפי קטגוריה")
        cat = df.groupby("קטגוריה")["מכירות_ש"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        ax.barh(heb_list(cat.index), cat.values, color=COLORS[:len(cat)])
        ax.set_xlabel(heb("מכירות (₪)"))
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with r1c2:
        st.subheader("חלוקת מכירות לפי אזור")
        region = df.groupby("אזור")["מכירות_ש"].sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        wedges, texts, autotexts = ax.pie(
            region.values, labels=heb_list(region.index), autopct="%1.1f%%",
            colors=COLORS[:len(region)], startangle=90)
        for t in texts + autotexts:
            t.set_color("#e2e8f0")
        ax.set_ylabel("")
        st.pyplot(fig); plt.close()

    # שורה 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("מכירות לפי יום בשבוע")
        day_order  = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
        day        = df.groupby("יום_בשבוע")["מכירות_ש"].sum().reindex(day_order, fill_value=0)
        colors_day = ["#ef4444" if d == "שבת" else "#6366f1" for d in day_order]
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        ax.bar(heb_list(day.index), day.values, color=colors_day)
        ax.set_ylabel(heb("מכירות (₪)"))
        plt.xticks(rotation=0, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with r2c2:
        st.subheader("אחוז רווח לפי קטגוריה")
        prof = df.groupby("קטגוריה")["אחוז_רווח"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        ax.barh(heb_list(prof.index), prof.values, color="#8b5cf6")
        ax.set_xlabel(heb("אחוז רווח ממוצע (%)"))
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    # שורה 3: טרנד + עונות
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.subheader("טרנד מכירות חודשי")
        monthly = df.groupby(df["תאריך"].dt.to_period("M"))["מכירות_ש"].sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        monthly.plot(ax=ax, color="#6366f1", linewidth=2, marker="o", markersize=3)
        ax.set_ylabel(heb("מכירות (₪)"))
        ax.grid(True, color="#2d2d4e", alpha=0.5)
        plt.xticks(rotation=45, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with r3c2:
        st.subheader("מכירות לפי עונה")
        season = df.groupby("עונה")["מכירות_ש"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        ax.bar(heb_list(season.index), season.values,
               color=["#f59e0b", "#6366f1", "#10b981", "#ef4444"][:len(season)])
        ax.set_ylabel(heb("מכירות (₪)"))
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# עמוד 3: תובנות עסקיות
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 תובנות עסקיות":
    st.title("💡 תובנות עסקיות")
    st.markdown("**ניתוח עסקי שנוצר אוטומטית על ידי Insights Agent (CrewAI + Groq)**")
    st.markdown("---")

    try:
        with open(INSIGHTS_PATH, encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    except FileNotFoundError:
        st.warning("⚠️ קובץ insights.md לא נמצא. הרץ תחילה: `python3 run_flow.py`")


# ════════════════════════════════════════════════════════════════════════════
# עמוד 4: מודל ML
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 מודל ML":
    st.title("🤖 ביצועי מודל Machine Learning")
    st.markdown("**Random Forest vs Logistic Regression | מטרה: חיזוי רווח גבוה/נמוך**")
    st.markdown("---")

    try:
        meta = load_meta()
    except FileNotFoundError:
        st.warning("⚠️ קובץ model_meta.json לא נמצא. הרץ תחילה: `python3 run_flow.py`")
        st.stop()

    # ביצועים
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌲 Random Forest")
        rf = meta["random_forest"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{rf['accuracy']:.4f}")
        m2.metric("F1 Score", f"{rf['f1']:.4f}")
        m3.metric("AUC",      f"{rf['auc']:.4f}")

    with col2:
        st.subheader("📉 Logistic Regression")
        lr = meta["logistic_regression"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{lr['accuracy']:.4f}")
        m2.metric("F1 Score", f"{lr['f1']:.4f}")
        m3.metric("AUC",      f"{lr['auc']:.4f}")

    best = meta.get("best_model", "Random Forest")
    st.success(f"🏆 **מודל מנצח: {best}**  |  אומן על {meta.get('train_size',0):,} עסקאות")

    st.markdown("---")

    # גרפים זה לצד זה
    gc1, gc2 = st.columns(2)

    with gc1:
        st.subheader("השוואת מודלים")
        metrics_labels = ["Accuracy", "F1 Score", "AUC"]
        rf_vals = [meta["random_forest"]["accuracy"],
                   meta["random_forest"]["f1"],
                   meta["random_forest"]["auc"]]
        lr_vals = [meta["logistic_regression"]["accuracy"],
                   meta["logistic_regression"]["f1"],
                   meta["logistic_regression"]["auc"]]
        x = np.arange(len(metrics_labels))
        fig, ax = plt.subplots(figsize=(7, 4))
        style_chart(fig, ax)
        ax.bar(x - 0.2, rf_vals, 0.4, label="Random Forest",       color="#6366f1")
        ax.bar(x + 0.2, lr_vals, 0.4, label="Logistic Regression", color="#06b6d4")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels, color="#e2e8f0")
        ax.set_ylim(0, 1.1)
        legend = ax.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e")
        for text in legend.get_texts():
            text.set_color("#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with gc2:
        if "feature_importances" in meta:
            st.subheader("חשיבות פיצ'רים (Top 5)")
            fi    = meta["feature_importances"]
            fi_df = pd.DataFrame(list(fi.items()),
                                 columns=["Feature", "Importance"]).sort_values("Importance")
            fig, ax = plt.subplots(figsize=(7, 4))
            style_chart(fig, ax)
            ax.barh(fi_df["Feature"], fi_df["Importance"], color="#10b981")
            ax.set_xlabel("Importance")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig); plt.close()

    # דו"ח מלא
    with st.expander("📋 דו\"ח הערכה מלא (Markdown)"):
        try:
            with open(EVAL_PATH, encoding="utf-8") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.warning("קובץ evaluation_report.md לא נמצא.")


# ════════════════════════════════════════════════════════════════════════════
# עמוד 5: ניבוי
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 ניבוי":
    st.title("🔮 ניבוי רווחיות בזמן אמת")
    st.markdown("**הזן פרטי עסקה — המודל יחזה אם הרווח יהיה גבוה או נמוך**")
    st.markdown("---")

    try:
        enc_data      = load_encoders()
        encoders      = enc_data["encoders"]
        threshold     = enc_data.get("median_profit_threshold", 25.0)
        model         = load_model()
    except FileNotFoundError:
        st.warning("⚠️ קבצי מודל לא נמצאו. הרץ תחילה: `python3 run_flow.py`")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ פרטי המוצר")
        category = st.selectbox("קטגוריה",     sorted(encoders["קטגוריה"]))
        chain    = st.selectbox("רשת",          sorted(encoders["רשת"]))
        region   = st.selectbox("אזור",         sorted(encoders["אזור"]))
        season   = st.selectbox("עונה",         sorted(encoders["עונה"]))

    with col2:
        st.subheader("💵 פרטי העסקה")
        qty        = st.number_input("כמות יחידות",     min_value=1,   max_value=100,    value=3)
        price      = st.number_input("מחיר יחידה (₪)", min_value=1.0, max_value=2000.0, value=120.0, step=10.0)
        discount   = st.slider("הנחה (%)",   min_value=0,  max_value=50,  value=5)
        month_val  = st.slider("חודש",       min_value=1,  max_value=12,  value=6)
        is_weekend = st.checkbox("סוף שבוע (שישי/שבת)")
        is_holiday = st.checkbox("יום חג / אירוע מיוחד")

    st.markdown("---")

    if st.button("🔮 חשב ניבוי", use_container_width=True, type="primary"):
        cat_enc    = sorted(encoders["קטגוריה"]).index(category)
        chain_enc  = sorted(encoders["רשת"]).index(chain)
        region_enc = sorted(encoders["אזור"]).index(region)
        season_enc = sorted(encoders["עונה"]).index(season)
        dow_num    = 4 if is_weekend else 2

        features = np.array([[
            qty, price, discount, month_val,
            dow_num, int(is_holiday), int(is_weekend),
            cat_enc, chain_enc, region_enc, season_enc,
        ]], dtype=float)

        pred  = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        high_prob = float(proba[1])

        rc1, rc2 = st.columns([2, 1])
        with rc1:
            if pred == 1:
                st.success(f"### ✅ רווח גבוה צפוי!")
                st.markdown(f"סיכוי לרווח גבוה: **{high_prob*100:.1f}%**")
            else:
                st.warning(f"### ⚠️ רווח נמוך צפוי")
                st.markdown(f"סיכוי לרווח גבוה: **{high_prob*100:.1f}%**")
                st.markdown("💡 שקול: להוריד הנחה, לשנות קטגוריה, או להגדיל כמות")

            st.progress(high_prob)
            st.caption(f"סף ה'רווח הגבוה': {threshold:.1f}% רווח על העסקה")

        with rc2:
            fig, ax = plt.subplots(figsize=(3, 3))
            style_chart(fig, ax)
            wedges, texts, autotexts = ax.pie(
                [high_prob, 1 - high_prob],
                labels=[heb("רווח גבוה"), heb("רווח נמוך")],
                colors=["#10b981", "#ef4444"],
                autopct="%1.1f%%", startangle=90)
            for t in texts + autotexts:
                t.set_color("#e2e8f0")
            st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# עמוד 6: העלה נתונים
# ════════════════════════════════════════════════════════════════════════════
elif page == "📤 העלה נתונים":
    st.title("📤 העלה נתונים שלך")
    st.markdown("**העלה קובץ CSV עם נתוני מכירות — המערכת תנתח אותם אוטומטית עם CrewAI**")
    st.markdown("---")

    # הורדת תבנית
    st.subheader("שלב 1: הורד תבנית")
    st.markdown("הקובץ חייב להכיל את העמודות הבאות:")
    st.code(", ".join(REQUIRED_COLS), language=None)

    try:
        df_template = pd.read_csv(RAW_PATH, encoding="utf-8-sig").head(10)
        csv_bytes = df_template.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ הורד קובץ דוגמה (10 שורות)",
            data=csv_bytes,
            file_name="retail_template.csv",
            mime="text/csv",
        )
    except FileNotFoundError:
        st.info("קובץ הדוגמה לא נמצא.")

    st.markdown("---")

    # העלאת קובץ
    st.subheader("שלב 2: העלה את הקובץ שלך")
    uploaded_file = st.file_uploader("בחר קובץ CSV", type=["csv"])

    if uploaded_file:
        try:
            # ניסיון קריאה עם utf-8-sig, אחר כך utf-8
            try:
                df_new = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            except Exception:
                uploaded_file.seek(0)
                df_new = pd.read_csv(uploaded_file, encoding="utf-8")

            # ולידציה
            missing_cols = [c for c in REQUIRED_COLS if c not in df_new.columns]

            if missing_cols:
                st.error(f"❌ עמודות חסרות בקובץ: {', '.join(missing_cols)}")
                st.info("הורד את קובץ הדוגמה למעלה ובדוק שהעמודות תואמות.")
            else:
                st.success(f"✅ הקובץ תקין — {len(df_new):,} שורות, {len(df_new.columns)} עמודות")

                # תצוגה מקדימה
                with st.expander("👀 תצוגה מקדימה (5 שורות ראשונות)"):
                    st.dataframe(df_new.head())

                st.markdown("---")
                st.subheader("שלב 3: הרץ ניתוח")
                st.markdown("לחץ על הכפתור — **7 סוכני AI** ינתחו את הנתונים שלך (~2 דקות)")

                if st.button("🚀 הרץ ניתוח מלא", use_container_width=True, type="primary"):

                    # שמירת הקובץ כ-Raw Data
                    df_new.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
                    st.info(f"✅ הקובץ נשמר ({len(df_new):,} שורות)")

                    # הרצת ה-Flow
                    with st.spinner("🤖 סוכני CrewAI עובדים על הנתונים שלך... (אנא המתן)"):
                        result = subprocess.run(
                            [sys.executable, "run_flow.py"],
                            capture_output=True,
                            text=True,
                            cwd=str(ROOT),
                            timeout=600,
                        )

                    if result.returncode == 0:
                        st.success("### ✅ הניתוח הושלם בהצלחה!")
                        st.markdown("כל הקבצים עודכנו עם הנתונים שלך:")
                        st.markdown("""
- `clean_data.csv` — נתונים נקיים
- `eda_report.html` — דו"ח גרפים
- `insights.md` — תובנות עסקיות
- `model.pkl` — מודל ML מאומן
- `evaluation_report.md` — ביצועי המודל
""")
                        st.info("🔄 עבור לעמוד **דשבורד ראשי** כדי לראות את התוצאות על הנתונים שלך")
                        st.cache_data.clear()
                    else:
                        st.error("❌ אירעה שגיאה בהרצת הניתוח")
                        with st.expander("פרטי השגיאה"):
                            st.code(result.stderr[-1000:] if result.stderr else "אין פרטים")

        except Exception as e:
            st.error(f"שגיאה בקריאת הקובץ: {e}")
            st.info("וודא שהקובץ הוא CSV תקין עם encoding UTF-8")


# ════════════════════════════════════════════════════════════════════════════
# עמוד 7: מוניטורינג (שלב 17)
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 מוניטורינג":
    st.title("📋 מוניטורינג ולוגים")
    st.markdown("**סטטוס מערכת, לוגים, וביצועים — שלב 17 מתוך 18**")
    st.markdown("---")

    # ── סטטוס קבצים ──────────────────────────────────────────────────────────
    st.subheader("🟢 סטטוס מערכת")
    files_to_check = {
        "📄 clean_data.csv":       ROOT / "data/processed/clean_data.csv",
        "🤖 model.pkl":            ROOT / "models/model.pkl",
        "📊 model_meta.json":      ROOT / "models/model_meta.json",
        "🔑 encoders.json":        ROOT / "models/encoders.json",
        "💡 insights.md":          ROOT / "data/processed/insights.md",
        "📝 evaluation_report.md": ROOT / "data/processed/evaluation_report.md",
        "📋 model_card.md":        ROOT / "data/processed/model_card.md",
    }

    cols = st.columns(4)
    for i, (label, path) in enumerate(files_to_check.items()):
        col = cols[i % 4]
        if path.exists():
            size_kb = path.stat().st_size / 1024
            mtime   = datetime.datetime.fromtimestamp(path.stat().st_mtime)
            col.success(f"✅ {label}")
            col.caption(f"{size_kb:.1f} KB | {mtime:%d/%m %H:%M}")
        else:
            col.error(f"❌ {label}")
            col.caption("לא נמצא")

    st.markdown("---")

    # ── הרצה אחרונה ──────────────────────────────────────────────────────────
    st.subheader("⏱️ הרצה אחרונה")
    logs_dir = ROOT / "logs"
    summary_files = sorted(logs_dir.glob("summary_*.json"), reverse=True) if logs_dir.exists() else []

    if summary_files:
        latest = summary_files[0]
        try:
            with open(latest, encoding="utf-8") as f:
                summary = json.load(f)
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("🕒 Run ID",   summary.get("run_id", "—"))
            sc2.metric("✅ סטטוס",    summary.get("status", "—"))
            sc3.metric("⏱️ זמן",     f"{summary.get('duration_sec', 0):.0f}s")
            sc4.metric("🤖 מודל",    summary.get("best_model", "—"))

            with st.expander("📄 סיכום מלא"):
                st.json(summary)
        except Exception as e:
            st.warning(f"לא ניתן לקרוא summary: {e}")
    else:
        st.info("אין נתוני הרצה עדיין — הרץ run_flow.py תחילה")

    st.markdown("---")

    # ── ביצועי מודל ──────────────────────────────────────────────────────────
    st.subheader("📈 ביצועי מודל אחרונים")
    try:
        meta = load_meta()
        pc1, pc2, pc3 = st.columns(3)
        rf = meta["random_forest"]
        pc1.metric("🌲 RF Accuracy", f"{rf['accuracy']:.4f}", delta=f"{rf['accuracy']-0.75:.2f} vs baseline")
        pc2.metric("🌲 RF F1 Score", f"{rf['f1']:.4f}")
        pc3.metric("🌲 RF AUC",      f"{rf['auc']:.4f}")

        fig, ax = plt.subplots(figsize=(8, 3))
        style_chart(fig, ax)
        metrics_vals = [rf["accuracy"], rf["f1"], rf["auc"]]
        metrics_labels = ["Accuracy", "F1 Score", "AUC"]
        bars = ax.bar(metrics_labels, metrics_vals,
                      color=["#6366f1", "#10b981", "#f59e0b"], width=0.4)
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, metrics_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=11, fontweight="bold", color="#e2e8f0")
        ax.axhline(0.7, color="#ef4444", linestyle="--", alpha=0.6, label="סף מינימלי (70%)")
        legend = ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#2d2d4e")
        for text in legend.get_texts():
            text.set_color("#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
    except FileNotFoundError:
        st.warning("⚠️ קובץ model_meta.json לא נמצא")

    st.markdown("---")

    # ── לוגים ────────────────────────────────────────────────────────────────
    st.subheader("📜 לוגים אחרונים")
    log_file = ROOT / "logs/run.log"
    if log_file.exists():
        with open(log_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        last_lines = lines[-60:] if len(lines) > 60 else lines

        log_text = "".join(last_lines)
        st.code(log_text, language=None)
        st.caption(f"מציג {len(last_lines)} שורות אחרונות מתוך {len(lines)} | {log_file}")
    else:
        st.info("קובץ run.log לא נמצא — יווצר בהרצה הבאה")

    # ── פידבק שהתקבל ─────────────────────────────────────────────────────────
    feedback_file = ROOT / "logs/feedback.txt"
    if feedback_file.exists():
        st.markdown("---")
        st.subheader("💬 פידבק שהתקבל")
        with open(feedback_file, encoding="utf-8") as f:
            fb_content = f.read().strip()
        if fb_content:
            st.text_area("פידבק מהמשתמשים:", value=fb_content, height=120, disabled=True)
        else:
            st.info("אין פידבק עדיין")
