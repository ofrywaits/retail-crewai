# Israeli Retail AI Analysis System

**מערכת ניתוח מכירות קמעונאיות ישראליות מבוססת AI**
פרויקט גמר — AI Developer Program | Jonathan Zouari, PhD.

> **Live Demo:** https://retail-crewai-2mtb4rmrhdewfzqy2rwyyu.streamlit.app

---

## הדאטאסט — נתוני דמו סינתטיים

> **שקיפות מלאה:** הדאטאסט המשמש בפרויקט זה הוא **נתוני דמו סינתטיים** שנוצרו לצורך הדגמת המערכת.

| פרט | ערך |
|-----|-----|
| שם | Israeli Retail Dataset (Demo) |
| גודל | 9,900 עסקאות |
| תקופה | ינואר 2022 — דצמבר 2024 |
| מבנה | מבוסס על מבנה ריאלי של קמעונאות ישראלית |
| רשתות | ויקטורי, שופרסל, רמי לוי, מגה |
| ערים | תל אביב, ירושלים, חיפה, חולון, באר שבע |
| מקור | נוצר סינתטית לצורך הדגמת ה-AI Pipeline |

**למה נתונים סינתטיים?**
הפרויקט מדגים **pipeline של AI** — CrewAI Flow עם 7 סוכנים, מודל ML, ודשבורד אינטראקטיבי. הדאטאסט הסינתטי מאפשר הדגמה מלאה של כל הפונקציונליות ללא תלות בנתוני ייצור רגישים. המערכת תעבוד באותה צורה עם כל CSV קמעונאי אמיתי שיוזן דרך עמוד ה"העלה נתונים".

**להחלפה בנתונים אמיתיים:**
ניתן למצוא דאטאסטים של קמעונאות בציבור ב-[Kaggle](https://www.kaggle.com/datasets?search=retail) ולהעלות אותם ישירות דרך ממשק האפליקציה.

---

## 18-Step Development Roadmap

פרויקט זה נבנה לפי מתודולוגיית ה-18 שלבים — מהגדרת הבעיה ועד לפריסה ומוניטורינג.

---

## Phase 1 — Discovery

### שלב 1: הגדרת הבעיה

**מה הבעיה?**
מנהלי מכירות קמעונאיים בישראל מקבלים החלטות עסקיות (תמחור, מלאי, הנחות) על בסיס אינטואיציה בלבד. אין להם כלי נגיש שמנתח נתוני מכירות היסטוריים ומציג תובנות בזמן אמת.

**מי חווה את הכאב?**
מנהלי חנויות ורשתות קמעונאיות ישראליות (ויקטורי, שופרסל, רמי לוי) ללא צוות BI ייעודי.

**למה כדאי לפתור?**
החלטות תמחור שגויות עולות לרשת קמעונאית ממוצעת 5–15% מהרווח השנתי. כלי ניתוח AI זמין יכול להפוך נתוני CSV רגילים לתובנות עסקיות תוך שניות.

**איך נראה פתרון טוב?**
מנהל חנות מעלה קובץ Excel/CSV → המערכת מנתחת אוטומטית → מקבל דשבורד עם גרפים + תחזית רווחיות לכל עסקה.

---

### שלב 2: הגדרת המשתמש

**פרסונת משתמש ראשית:**

| שדה | פרטים |
|-----|--------|
| **שם** | דוד כהן |
| **תפקיד** | מנהל רשת קמעונאית, 3 סניפים |
| **גיל** | 42 |
| **רמת טכנולוגיה** | בינונית — Excel כן, Python לא |
| **אתגר יומי** | "אני יודע שיש בנתונים שלי תובנות זהב, אבל אין לי זמן לחפש" |
| **מה הוא רוצה** | תשובה מהירה: איזו קטגוריה הכי רווחית? מתי לתת הנחות? |

**User Story:**
> "As a retail manager, I want to upload my sales CSV and get instant AI-powered insights, so that I can make data-driven pricing and inventory decisions without needing a data scientist."

**מה חוסם אותו היום?**
- אין זמן ללמוד כלי BI מורכבים (Tableau, Power BI)
- אין תקציב לצוות דאטה
- Excel לא מספיק חכם לתחזיות

---

### שלב 3: MVP — Minimum Viable Product

| Must Have ✅ | Nice to Have 🔶 | Out of Scope ❌ |
|-------------|----------------|----------------|
| דשבורד עם KPIs | השוואה בין תקופות | ממשק מובייל |
| 6 גרפים אינטראקטיביים | ייצוא PDF | תשלומים |
| ניבוי רווחיות בזמן אמת | התראות email | מולטי-טננט |
| העלאת CSV משלך | לוח זמנים אוטומטי | ERP integration |
| תובנות עסקיות ב-AI | מפת חום גאוגרפית | CRM חיבור |
| מוניטורינג ולוגים | A/B testing | |

---

## Phase 2 — Design

### שלבים 4–5: UX / UI

**ארכיטקטורת מסכים (6 עמודים):**

```
Sidebar Navigation
├── 🏠 דשבורד ראשי      → KPIs + 2 גרפי סיכום
├── 📊 ניתוח נתונים     → 6 גרפים + סינון דינמי
├── 💡 תובנות עסקיות   → AI-generated insights
├── 🤖 מודל ML          → השוואת מודלים + feature importance
├── 🔮 ניבוי            → טופס + תחזית real-time
├── 📤 העלה נתונים     → drag & drop CSV + הרצת Flow
└── 📊 מוניטורינג       → לוגים + סטטוס מערכת
```

**עקרונות UI:**
- **RTL First** — כל הטקסט העברי נתמך נכון עם `python-bidi`
- **Mobile-friendly** — Streamlit layout responsive
- **Feedback loops** — כל פעולה מחזירה spinner + הודעת success/error
- **Progressive disclosure** — Expandable sections לפרטים מתקדמים

---

## Phase 3 — Architecture

### שלב 6: ארכיטקטורת המערכת

```
┌─────────────────────────────────────────────────────┐
│                   User (Browser)                    │
└─────────────────┬───────────────────────────────────┘
                  │ HTTP
┌─────────────────▼───────────────────────────────────┐
│            Streamlit App (app.py)                   │
│   Dashboard | EDA | Insights | ML | Predict | Upload│
└─────────────────┬───────────────────────────────────┘
                  │ subprocess
┌─────────────────▼───────────────────────────────────┐
│              CrewAI Flow (flow.py)                  │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │  Crew 1:     │  │  Crew 2:     │                 │
│  │  Data + EDA  │  │  ML Pipeline │                 │
│  │  (3 agents)  │  │  (4 agents)  │                 │
│  └──────┬───────┘  └──────┬───────┘                 │
└─────────┼─────────────────┼───────────────────────── ┘
          │                 │
┌─────────▼─────────────────▼─────────────────────────┐
│                    Groq API                         │
│          llama-3.3-70b-versatile (Free)             │
└─────────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────────── ┐
│                  File System                        │
│  data/raw/       → CSV input                        │
│  data/processed/ → clean_data, insights, reports    │
│  models/         → model.pkl, encoders, meta        │
│  logs/           → run.log, summary_*.json          │
└─────────────────────────────────────────────────────┘
```

### שלב 7: Tech Stack

| טכנולוגיה | גרסה | שימוש |
|-----------|------|--------|
| Python | 3.11 | שפה ראשית |
| CrewAI | 1.9.3+ | מסגרת ריבוי סוכנים |
| Groq LLM | Llama 3.3 70B | מנוע שפה (חינמי) |
| Pandas | 2.0+ | עיבוד נתונים |
| Scikit-learn | 1.3+ | Random Forest, Logistic Regression |
| Streamlit | 1.28+ | ממשק משתמש |
| Matplotlib / Seaborn | 3.7+ / 0.12+ | ויזואליזציה |
| python-bidi | 0.4.2+ | תמיכת RTL בגרפים |
| Joblib | 1.3+ | שמירת מודלים |
| python-dotenv | 1.0+ | ניהול סודות |

---

## Phase 4 — Setup

### שלב 8: Project Setup

```bash
# Clone
git clone https://github.com/ofrywaits/retail-crewai.git
cd retail-crewai

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# Environment
echo "GROQ_API_KEY=your_key_here" > .env
# מפתח חינמי: https://console.groq.com
```

**מבנה הפרויקט:**
```
projectai_1-5_26/
├── app/
│   └── app.py                  # Streamlit dashboard (7 עמודים)
├── crews/
│   ├── analysis_crew/          # Crew 1: ניקוי + EDA
│   │   ├── agents.py           # Data Loader, Cleaner, Contract
│   │   ├── agents_eda.py       # EDA Analyst, Insights Analyst
│   │   ├── tasks.py / tasks_eda.py
│   │   └── tools.py / tools_eda.py
│   └── prediction_crew/        # Crew 2: ML Pipeline
│       ├── agents_ml.py        # Feature Eng, Trainer, Evaluator, Card
│       ├── tasks_ml.py
│       └── tools_ml.py
├── flows/
│   └── flow.py                 # CrewAI Flow ראשי (5 שלבים)
├── data/
│   ├── raw/                    # CSV input (לא ב-git)
│   └── processed/              # פלט: clean_data, insights, reports
├── models/                     # model.pkl, encoders.json, model_meta.json
├── logs/                       # run.log, summary_*.json
├── tests/
│   ├── test_data.py            # 9 בדיקות איכות נתונים
│   └── test_model.py           # 9 בדיקות מודל ML
├── run_flow.py                 # נקודת כניסה
├── requirements.txt
└── .env                        # סודות (לא ב-git)
```

### שלב 9: Git / GitHub

```bash
git init
git remote add origin https://github.com/ofrywaits/retail-crewai.git

# .gitignore חשוב:
# .env, data/raw/, logs/, __pycache__, venv/

git add .
git commit -m "feat: complete AI retail analysis system"
git push origin main
```

**Repo:** https://github.com/ofrywaits/retail-crewai (public ✅)

---

## Phase 5 — AI Integration

### שלב 10: AI Workflow

```
CSV Input
    │
    ▼
[Data Loader Agent]
    Uses: load_and_validate_data tool
    Output: validated dataset summary
    │
    ▼
[Data Cleaner Agent]
    Uses: clean_data tool
    Output: clean_data.csv (9,900 rows)
    │
    ▼
[Contract Agent]
    Uses: generate_dataset_contract tool
    Output: dataset_contract.json
    │
    ▼
[EDA Analyst Agent]
    Uses: run_eda_analysis tool
    Output: 5 chart PNGs + eda_report.html
    │
    ▼
[Insights Analyst Agent]
    Uses: generate_insights tool
    Output: insights.md (6 business insights in Hebrew)
    │
    ▼
[Feature Engineer Agent]
    Uses: run_feature_engineering tool
    Output: features.csv (11 features)
    │
    ▼
[Model Trainer Agent]
    Uses: run_ml_training tool
    Output: model.pkl + encoders.json
    │
    ▼
[Model Evaluator Agent]
    Uses: run_ml_evaluation tool
    Output: model_meta.json + evaluation_report.md
    │
    ▼
[Model Card Agent]
    Output: model_card.md (business value doc)
```

**זמן ריצה:** ~25 שניות | **עלות:** $0 (Groq free tier)

---

### שלב 11: MCP — Model Context Protocol

MCP (Model Context Protocol) הוא פרוטוקול סטנדרטי שמאפשר ל-LLM לקרוא לכלים חיצוניים בצורה מובנית.

**כיצד פרויקט זה ממש את עקרונות ה-MCP:**

כל סוכן CrewAI מקבל **tools** — פונקציות Python שמדמות בדיוק את מודל ה-MCP:

```python
# דוגמה — tools_ml.py
@tool("run_ml_training")
def run_ml_training(action: str) -> str:
    """
    MCP-style tool: trains Random Forest + Logistic Regression models.
    Input:  action="run"
    Output: JSON with accuracy, F1, AUC metrics
    """
    ...
```

| מאפיין MCP | מימוש בפרויקט |
|-----------|---------------|
| Tool definition | `@tool("name")` decorator + docstring |
| Input schema | `action: str` parameter |
| Structured output | JSON string return value |
| Tool discovery | CrewAI agent receives tool list automatically |
| Error handling | try/except → error message returned to agent |

**LLM ↔ Tool flow:**
```
Agent (LLM) → decides to call tool
    → CrewAI router → Python function
    → result returned to LLM
    → LLM incorporates result in Final Answer
```

---

### שלב 12: CrewAI Agents

**7 סוכנים | 2 Crews | 1 Flow**

**Crew 1 — Data & EDA (5 סוכנים):**

| סוכן | תפקיד | כלים |
|------|--------|------|
| Data Loader | טוען ומאמת CSV | `load_and_validate_data` |
| Data Cleaner | מנקה + שומר clean_data.csv | `clean_data` |
| Contract Agent | מתעד סכמת הדאטאסט | `generate_dataset_contract` |
| EDA Analyst | יוצר 5 גרפים + HTML report | `run_eda_analysis` |
| Insights Analyst | מייצר 6 תובנות עסקיות בעברית | `generate_insights` |

**Crew 2 — ML Pipeline (4 סוכנים):**

| סוכן | תפקיד | כלים |
|------|--------|------|
| Feature Engineer | יוצר 11 features | `run_feature_engineering` |
| Model Trainer | מאמן RF + LR | `run_ml_training` |
| Model Evaluator | מחשב Accuracy/F1/AUC | `run_ml_evaluation` |
| Model Card Agent | כותב תיעוד ערך עסקי | (LLM only) |

---

## Phase 6 — Development

### שלב 13: Build

**Streamlit Dashboard — 7 עמודים:**

1. **🏠 דשבורד ראשי** — 8 KPIs + 2 גרפי סיכום
2. **📊 ניתוח נתונים** — 6 גרפי EDA + סינון דינמי לפי קטגוריה/שנה
3. **💡 תובנות עסקיות** — 6 תובנות AI + evaluation report
4. **🤖 מודל ML** — השוואת RF vs LR + feature importance
5. **🔮 ניבוי** — טופס real-time + gauge chart + המלצות
6. **📤 העלה נתונים** — CSV upload + ולידציה + הרצת Flow מלא
7. **📊 מוניטורינג** — לוגים, סטטוס מערכת, זמני ריצה

**טכניקות מפתח:**
- `@st.cache_data` — מניעת reload מיותר
- `python-bidi` — תיקון RTL בגרפי matplotlib
- `subprocess.run()` — הרצת Flow מ-Streamlit
- `st.cache_data.clear()` — רענון אחרי upload

### שלב 14: Testing

**18 בדיקות pytest — 2 קבצים:**

```bash
pytest tests/ -v
```

**test_data.py** (9 בדיקות):
- קיום קובץ clean_data.csv
- עמודות נדרשות (14 עמודות)
- מינימום 9,000 שורות
- אין מכירות שליליות
- אין כמויות שליליות
- אין כפילויות בהזמנות
- אחוז רווח בטווח 0–100%
- תאריכים תקינים
- טווח תאריכים 2020–2026

**test_model.py** (9 בדיקות):
- קיום model.pkl
- קיום model_meta.json
- מפתחות נדרשים ב-meta
- Random Forest accuracy > 70%
- Random Forest F1 > 70%
- ניבוי מחזיר 0 או 1
- predict_proba מסתכמת ל-1
- מפתחות נדרשים ב-encoders
- מספר features = 11

### שלב 15: Bug Fixing & Stability

**בעיות שנפתרו:**

| בעיה | פתרון |
|------|--------|
| עברית הפוכה בגרפים | `python-bidi` + `get_display()` wrapper |
| Tool call XML error | הוסף `action: str` parameter לכל tool |
| Rate limit Groq | Retry mechanism עם 70s wait, 5 ניסיונות |
| Encoding UTF-8/UTF-8-sig | try/except עם fallback encoding |
| Model.pkl לא ב-git | הסרה מ-.gitignore |

**Error Handling בקוד:**
```python
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ נתונים לא נמצאו — הרץ run_flow.py תחילה")
    st.stop()
except Exception as e:
    st.error(f"שגיאה בטעינת נתונים: {e}")
    st.stop()
```

---

## Phase 7 — Deployment

### שלב 16: Deploy

**Streamlit Cloud (ייצור):**

1. Push ל-GitHub (public repo)
2. https://share.streamlit.io → Connect repo
3. Main file: `app/app.py`
4. Secrets: `GROQ_API_KEY = "..."`
5. Deploy!

**Live URL:** https://retail-crewai-2mtb4rmrhdewfzqy2rwyyu.streamlit.app

**סביבת פיתוח:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 run_flow.py          # הרצת ניתוח
streamlit run app/app.py     # הפעלת ממשק
```

### שלב 17: Monitoring & Logs

**לוגים בזמן אמת:**
- `logs/run.log` — כל אירוע בכל הרצה
- `logs/summary_YYYYMMDD_HHMMSS.json` — סיכום כל run עם metrics

**עמוד מוניטורינג בממשק:**
- סטטוס כל קובץ מערכת (✅/❌)
- זמן הרצה אחרון
- 50 שורות לוג אחרונות
- גרף ביצועים (F1, Accuracy, AUC)

**Streamlit Cloud Logs:**
```
App → ⋮ → View app logs
```

### שלב 18: Iteration & Roadmap

**V1 (נוכחי) — הושלם ✅**
- ניתוח CSV עם 7 סוכני AI
- מודל ML עם 89% accuracy
- דשבורד 7 עמודים
- פריסה ב-Streamlit Cloud

**V2 (מתוכנן):**
- [ ] אימות משתמשים (Supabase Auth)
- [ ] שמירת היסטוריית runs בענן
- [ ] השוואה בין תקופות זמן
- [ ] ייצוא דוח PDF אוטומטי
- [ ] תמיכה ב-Excel (.xlsx)

**V3 (עתידי):**
- [ ] Multi-tenant — כל לקוח עם data נפרד
- [ ] Real-time dashboard עם WebSocket
- [ ] Mobile app (React Native)

---

## ביצועי המודל

| מדד | Random Forest | Logistic Regression |
|-----|--------------|---------------------|
| Accuracy | **0.8939** | 0.7520 |
| F1 Score | **0.8933** | 0.7517 |
| AUC | **0.9569** | 0.7916 |
| Train Size | 7,920 | 7,920 |
| Features | 11 | 11 |

**מודל מנצח: Random Forest** — עדיף ב-14% על Logistic Regression.

**11 פיצ'רים:**
`כמות`, `מחיר_יחידה`, `הנחה_אחוז`, `חודש`, `יום_בשבוע_מספר`, `חג`, `סוף_שבוע`, `קטגוריה_קוד`, `רשת_קוד`, `אזור_קוד`, `עונה_קוד`

---

## הרצה מהירה

```bash
# 1. Clone + Setup
git clone https://github.com/ofrywaits/retail-crewai.git
cd retail-crewai
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env

# 2. הרץ ניתוח (25 שניות)
python3 run_flow.py

# 3. הפעל ממשק
streamlit run app/app.py
# → http://localhost:8501

# 4. הרץ בדיקות
pytest tests/ -v   # צפוי: 18/18 passed
```

---

## משתני סביבה

| משתנה | תיאור | קבל מ |
|-------|--------|--------|
| `GROQ_API_KEY` | מפתח Groq LLM | https://console.groq.com |

---

*פרויקט גמר | AI Developer Program | Jonathan Zouari, PhD.*
*בניה: 18 שלבים מהגדרת הבעיה עד לפריסה בייצור*
