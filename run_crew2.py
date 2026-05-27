"""
הרצת Crew 2 — Feature Engineering + ML Models
הרץ: python3 run_crew2.py
"""

from crewai import Crew, Process
from crews.prediction_crew.agents_ml import (
    feature_engineer, model_trainer, model_evaluator, model_card_agent
)
from crews.prediction_crew.tasks_ml import (
    features_task, training_task, evaluation_task, model_card_task
)
import os, json

if __name__ == "__main__":
    print("=" * 55)
    print("  Crew 2 — Feature Engineering & ML Models")
    print("=" * 55)

    crew = Crew(
        agents=[feature_engineer, model_trainer, model_evaluator, model_card_agent],
        tasks=[features_task, training_task, evaluation_task, model_card_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 55)
    print("  תוצאה סופית:")
    print("=" * 55)
    print(result)

    print("\n" + "=" * 55)
    print("  בדיקת קבצי פלט:")
    print("=" * 55)
    outputs = [
        ("data/processed/features.csv",         "features.csv"),
        ("models/model.pkl",                     "model.pkl (best)"),
        ("models/model_rf.pkl",                  "model_rf.pkl"),
        ("models/model_lr.pkl",                  "model_lr.pkl"),
        ("data/processed/evaluation_report.md",  "evaluation_report.md"),
        ("data/processed/model_card.md",         "model_card.md"),
    ]
    for path, label in outputs:
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else 0
        status = f"✓ ({size:,} bytes)" if exists else "✗ חסר!"
        print(f"  {label:30s} {status}")

    if os.path.exists("models/model_meta.json"):
        with open("models/model_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"\n  המודל הטוב ביותר: {meta['best_model']}")
        print(f"  RF  → F1: {meta['random_forest']['f1']}  | AUC: {meta['random_forest']['auc']}")
        print(f"  LR  → F1: {meta['logistic_regression']['f1']}  | AUC: {meta['logistic_regression']['auc']}")
