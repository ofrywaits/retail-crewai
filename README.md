# Israeli Retail AI Analysis System

**מערכת ניתוח מכירות קמעונאיות ישראליות מבוססת AI**  
פרויקט גמר — AI Developer Program | Jonathan Zouari, PhD.

---

## מה המערכת עושה

המשתמש מעלה קובץ CSV עם נתוני מכירות → 7 סוכני AI מנתחים את הנתונים באופן מלא אוטומטי → ממשק אינטראקטיבי מציג תוצאות + ניבוי רווחיות בזמן אמת.

```
CSV שלך → CrewAI Flow → דשבורד + תובנות + מודל ML
```

---

## ממשק המשתמש — 6 עמודים

| עמוד | תוכן |
|------|------|
| 🏠 דשבורד ראשי | KPIs + גרפים מרכזיים |
| 📊 ניתוח נתונים | 6 גרפים אינטראקטיביים עם סינון |
| 💡 תובנות עסקיות | תובנות שנוצרו אוטומטית ע"י AI |
| 🤖 מודל ML | השוואת Random Forest vs Logistic Regression |
| 🔮 ניבוי | הזן עסקה → קבל חיזוי רווח בזמן אמת |
| 📤 העלה נתונים | העלה CSV משלך → הרץ ניתוח מלא |

---

## ארכיטקטורת ה-AI

```
CrewAI Flow
├── Crew 1: Data Analysis (3 סוכנים)
│   ├── Data Loader      → בדיקת איכות נתונים
│   ├── Data Cleaner     → ניקוי ושמירת clean_data.csv
│   └── Contract Agent   → תיעוד סכמת הדאטאסט
│
├── Crew 1 EDA (2 סוכנים)
│   ├── EDA Analyst      → 6 גרפים + דו"ח HTML
│   └── Insights Analyst → 6 תובנות עסקיות בעברית
│
└── Crew 2: Machine Learning (4 סוכנים)
    ├── Feature Engineer → Feature Engineering
    ├── Model Trainer    → Random Forest + Logistic Regression
    ├── Model Evaluator  → Accuracy / F1 / AUC + Confusion Matrix
    └── Model Card Agent → תיעוד ערך עסקי
```

**LLM:** Groq — `llama-3.3-70b-versatile` (חינמי)  
**זמן ריצה:** ~25 שניות לניתוח מלא

---

## התקנה והרצה

```bash
# 1. שכפל את הפרויקט
git clone https://github.com/ofrywaits/retail-crewai.git
cd retail-crewai

# 2. צור סביבה וירטואלית
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. התקן תלויות
pip install -r requirements.txt

# 4. הוסף מפתח API
echo "GROQ_API_KEY=your_key_here" > .env
# מפתח חינמי ב: https://console.groq.com

# 5. (אופציונלי) הרץ ניתוח על הדאטאסט הדוגמה
python3 run_flow.py

# 6. הפעל את הממשק
streamlit run app/app.py
```

פתח בדפדפן: **http://localhost:8501**

---

## מבנה הפרויקט

```
projectai_1-5_26/
├── app/
│   └── app.py                  # Streamlit dashboard
├── crews/
│   ├── analysis_crew/          # Crew 1: ניקוי + EDA
│   │   ├── agents.py
│   │   ├── agents_eda.py
│   │   ├── tasks.py
│   │   ├── tasks_eda.py
│   │   ├── tools.py
│   │   └── tools_eda.py
│   └── prediction_crew/        # Crew 2: ML
│       ├── agents_ml.py
│       ├── tasks_ml.py
│       └── tools_ml.py
├── flows/
│   └── flow.py                 # CrewAI Flow ראשי
├── data/
│   └── processed/              # קבצי פלט
├── models/                     # מודלי ML שמורים
├── tests/                      # בדיקות
├── run_flow.py                 # נקודת כניסה
├── requirements.txt
└── .env                        # סודות (לא ב-git)
```

---

## משתני סביבה

```env
GROQ_API_KEY=your_groq_api_key_here
```

קבל מפתח חינמי: https://console.groq.com

---

## טכנולוגיות

| טכנולוגיה | שימוש |
|-----------|-------|
| Python 3.11 | שפה ראשית |
| CrewAI 1.9+ | מסגרת ריבוי סוכנים |
| Groq LLM | מנוע שפה (Llama 3.3 70B) |
| Pandas | עיבוד נתונים |
| Scikit-learn | Machine Learning |
| Streamlit | ממשק משתמש |
| Matplotlib / Seaborn | גרפים |
| python-bidi | תמיכת RTL בגרפים |

---

## ביצועי המודל

| מדד | Random Forest | Logistic Regression |
|-----|--------------|---------------------|
| Accuracy | 0.8939 | 0.7520 |
| F1 Score | 0.8933 | 0.7517 |
| AUC | 0.9569 | 0.7916 |

**מודל מנצח: Random Forest** ✅

---

*פרויקט גמר | AI Developer Program*
