"""
RetailAnalysisFlow — מחבר את Crew 1 ו-Crew 2 לצינור אחד
@start → Crew1 Cleaning → Crew1 EDA → Validate → Crew2 ML → Summary
"""

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel
import logging
import os
import json
import time
from datetime import datetime


# ── Retry helper: מנסה שוב אחרי Rate Limit של Groq ───────────────────────────
def _kickoff(crew, label: str, log):
    """מריץ crew.kickoff() — מנסה עד 5 פעמים אם Groq מחזיר Rate Limit."""
    for attempt in range(1, 6):
        try:
            return crew.kickoff()
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg.lower() or "429" in msg or "RateLimitError" in msg:
                wait = 70
                log.warning(f"  [{label}] Rate Limit — ממתין {wait}s (ניסיון {attempt}/5)...")
                time.sleep(wait)
            else:
                raise
    return crew.kickoff()


# ── הגדרת Logging ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("logs/run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("RetailFlow")


# ── State: מה ה-Flow זוכר בין השלבים ────────────────────────────────────────
class RetailFlowState(BaseModel):
    run_id:             str  = ""
    started_at:         str  = ""
    crew1_clean_done:   bool = False
    crew1_eda_done:     bool = False
    validation_passed:  bool = False
    crew2_done:         bool = False
    errors:             list = []
    timings:            dict = {}
    outputs:            dict = {}


# ── ה-Flow ───────────────────────────────────────────────────────────────────
class RetailAnalysisFlow(Flow[RetailFlowState]):

    # ─────────────────────────────────────────────────────────────────────────
    @start()
    def initialize(self):
        self.state.run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state.started_at = datetime.now().isoformat()
        log.info("=" * 55)
        log.info(f"  RetailAnalysisFlow started | run_id={self.state.run_id}")
        log.info("=" * 55)
        return "initialized"

    # ─────────────────────────────────────────────────────────────────────────
    @listen(initialize)
    def run_crew1_cleaning(self, _):
        log.info("── שלב 1: Crew 1 — ניקוי נתונים ──────────────────────")
        t0 = time.time()
        try:
            from crewai import Crew, Process
            from crews.analysis_crew.agents import (
                data_loader_agent, data_cleaner_agent, contract_agent
            )
            from crews.analysis_crew.tasks import (
                ingest_task, clean_task, contract_task
            )

            crew = Crew(
                agents=[data_loader_agent, data_cleaner_agent, contract_agent],
                tasks=[ingest_task, clean_task, contract_task],
                process=Process.sequential,
                verbose=False,
            )
            result = _kickoff(crew, "Crew1-Cleaning", log)

            elapsed = round(time.time() - t0, 1)
            self.state.crew1_clean_done = True
            self.state.timings["crew1_cleaning"] = elapsed
            self.state.outputs["crew1_cleaning"] = str(result)[:300]

            log.info(f"  Crew 1 Cleaning הסתיים ✓ ({elapsed}s)")
            log.info(f"  קבצים: clean_data.csv, dataset_contract.json")
            return "crew1_clean_done"

        except Exception as e:
            self.state.errors.append(f"Crew1 Cleaning: {e}")
            log.error(f"  שגיאה ב-Crew 1 Cleaning: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    @listen(run_crew1_cleaning)
    def run_crew1_eda(self, _):
        log.info("── שלב 2: Crew 1 — EDA ותובנות ────────────────────────")
        t0 = time.time()
        try:
            from crewai import Crew, Process
            from crews.analysis_crew.agents_eda import eda_agent, insights_agent
            from crews.analysis_crew.tasks_eda  import eda_task, insights_task

            crew = Crew(
                agents=[eda_agent, insights_agent],
                tasks=[eda_task, insights_task],
                process=Process.sequential,
                verbose=False,
            )
            result = _kickoff(crew, "Crew1-EDA", log)

            # שמירת insights.md ישירות מ-Python (בלי tool call)
            insights_content = str(result)
            os.makedirs("data/processed", exist_ok=True)
            with open("data/processed/insights.md", "w", encoding="utf-8") as f:
                f.write(
                    f"# תובנות עסקיות — דאטאסט קמעונאי ישראלי\n\n"
                    f"**תאריך:** {datetime.now().strftime('%Y-%m-%d')}\n"
                    f"**מקור נתונים:** clean_data.csv (9,900 עסקאות, 2022–2024)\n\n"
                    f"---\n\n{insights_content}\n\n---\n"
                    f"*נוצר אוטומטית על ידי Insights Agent — CrewAI Israeli Retail Project*\n"
                )
            log.info("  insights.md נשמר ✓")

            elapsed = round(time.time() - t0, 1)
            self.state.crew1_eda_done = True
            self.state.timings["crew1_eda"] = elapsed
            self.state.outputs["crew1_eda"] = str(result)[:300]

            log.info(f"  Crew 1 EDA הסתיים ✓ ({elapsed}s)")
            log.info(f"  קבצים: eda_report.html, insights.md")
            return "crew1_eda_done"

        except Exception as e:
            self.state.errors.append(f"Crew1 EDA: {e}")
            log.error(f"  שגיאה ב-Crew 1 EDA: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    @listen(run_crew1_eda)
    def validate_outputs(self, _):
        log.info("── שלב 3: ולידציה — בודק קבצי Crew 1 ─────────────────")
        required = {
            "data/processed/clean_data.csv":        "clean_data.csv",
            "data/processed/dataset_contract.json": "dataset_contract.json",
            "data/processed/eda_report.html":       "eda_report.html",
            "data/processed/insights.md":           "insights.md",
        }
        missing = []
        for path, label in required.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                log.info(f"  ✓ {label} ({size:,} bytes)")
            else:
                log.error(f"  ✗ {label} — חסר!")
                missing.append(label)

        # בדיקת תקינות clean_data
        try:
            import pandas as pd
            df = pd.read_csv("data/processed/clean_data.csv", encoding="utf-8-sig")
            assert len(df) >= 9000,   f"יותר מדי שורות הוסרו: {len(df)}"
            assert (df["מכירות_ש"] > 0).all(), "יש ערכי מכירות שליליים"
            log.info(f"  ✓ clean_data תקין ({len(df):,} שורות)")
        except Exception as e:
            missing.append(f"clean_data validation: {e}")
            log.error(f"  ✗ בדיקת clean_data נכשלה: {e}")

        if missing:
            self.state.errors.extend(missing)
            raise ValueError(f"ולידציה נכשלה: {missing}")

        self.state.validation_passed = True
        log.info("  כל הבדיקות עברו ✓ — ממשיכים ל-Crew 2")
        return "validation_passed"

    # ─────────────────────────────────────────────────────────────────────────
    @listen(validate_outputs)
    def run_crew2(self, _):
        log.info("── שלב 4: Crew 2 — Feature Engineering + ML ────────────")
        t0 = time.time()
        try:
            from crewai import Crew, Process
            from crews.prediction_crew.agents_ml import (
                feature_engineer, model_trainer, model_evaluator, model_card_agent
            )
            from crews.prediction_crew.tasks_ml import (
                features_task, training_task, evaluation_task, model_card_task
            )

            crew = Crew(
                agents=[feature_engineer, model_trainer, model_evaluator, model_card_agent],
                tasks=[features_task, training_task, evaluation_task, model_card_task],
                process=Process.sequential,
                verbose=False,
            )
            result = _kickoff(crew, "Crew2-ML", log)

            # שמירת model_card.md ישירות מ-Python (בלי tool call)
            business_text = str(result)
            try:
                with open("models/model_meta.json", encoding="utf-8") as f:
                    meta = json.load(f)
                card_content = (
                    f"# Model Card — Israeli Retail Profit Classifier\n\n"
                    f"**תאריך:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
                    f"## פרטי המודל\n\n"
                    f"| שדה | ערך |\n|-----|-----|\n"
                    f"| שם המודל | {meta.get('best_model','Random Forest')} |\n"
                    f"| גרסה | 1.0 |\n| משימה | Binary Classification |\n"
                    f"| מטרה | חיזוי רווח גבוה/נמוך בעסקה קמעונאית |\n\n"
                    f"## ביצועים\n\n"
                    f"| מדד | Random Forest | Logistic Regression |\n"
                    f"|-----|--------------|---------------------|\n"
                    f"| Accuracy | {meta['random_forest']['accuracy']} | {meta['logistic_regression']['accuracy']} |\n"
                    f"| F1 Score | {meta['random_forest']['f1']} | {meta['logistic_regression']['f1']} |\n"
                    f"| AUC | {meta['random_forest']['auc']} | {meta['logistic_regression']['auc']} |\n\n"
                    f"## ערך עסקי\n\n{business_text}\n\n"
                    f"---\n*נוצר אוטומטית על ידי Model Card Agent — CrewAI Israeli Retail Project*\n"
                )
                os.makedirs("data/processed", exist_ok=True)
                with open("data/processed/model_card.md", "w", encoding="utf-8") as f:
                    f.write(card_content)
                log.info("  model_card.md נשמר ✓")
            except Exception as me:
                log.warning(f"  model_card.md שגיאה: {me}")

            elapsed = round(time.time() - t0, 1)
            self.state.crew2_done = True
            self.state.timings["crew2"] = elapsed
            self.state.outputs["crew2"] = str(result)[:300]

            log.info(f"  Crew 2 הסתיים ✓ ({elapsed}s)")
            log.info(f"  קבצים: features.csv, model.pkl, evaluation_report.md, model_card.md")
            return "crew2_done"

        except Exception as e:
            self.state.errors.append(f"Crew2: {e}")
            log.error(f"  שגיאה ב-Crew 2: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    @listen(run_crew2)
    def generate_summary(self, _):
        log.info("── שלב 5: סיכום ─────────────────────────────────────────")
        total = sum(self.state.timings.values())

        # קריאת מטאדאטה של המודל
        model_info = {}
        try:
            with open("models/model_meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            model_info = {
                "best_model": meta.get("best_model"),
                "rf_f1":  meta.get("random_forest", {}).get("f1"),
                "lr_f1":  meta.get("logistic_regression", {}).get("f1"),
            }
        except Exception:
            pass

        summary = {
            "run_id":            self.state.run_id,
            "started_at":        self.state.started_at,
            "completed_at":      datetime.now().isoformat(),
            "total_time_sec":    round(total, 1),
            "steps_completed":   4,
            "errors":            self.state.errors,
            "timings":           self.state.timings,
            "model_performance": model_info,
            "outputs_generated": [
                "data/processed/clean_data.csv",
                "data/processed/dataset_contract.json",
                "data/processed/eda_report.html",
                "data/processed/insights.md",
                "data/processed/features.csv",
                "models/model.pkl",
                "data/processed/evaluation_report.md",
                "data/processed/model_card.md",
            ],
        }

        # שמירת סיכום
        os.makedirs("logs", exist_ok=True)
        summary_path = f"logs/summary_{self.state.run_id}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        log.info("=" * 55)
        log.info("  Flow הסתיים בהצלחה!")
        log.info(f"  זמן כולל:  {round(total/60, 1)} דקות")
        log.info(f"  מודל טוב:  {model_info.get('best_model','?')} (F1={model_info.get('rf_f1','?')})")
        log.info(f"  שגיאות:    {len(self.state.errors)}")
        log.info(f"  סיכום:     {summary_path}")
        log.info("=" * 55)

        return summary
