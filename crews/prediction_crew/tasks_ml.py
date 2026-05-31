"""
4 משימות Crew 2
"""

from crewai import Task
from crews.prediction_crew.agents_ml import (
    feature_engineer, model_trainer, model_evaluator, model_card_agent
)

features_task = Task(
    description=(
        "Use run_feature_engineering with action='run' to build the ML dataset. "
        "Report: how many features were created, what the target distribution looks like, "
        "and what threshold was used for 'high profit'."
    ),
    expected_output=(
        "Summary of features.csv: number of features, target distribution "
        "(high vs low profit counts), and the profit threshold value."
    ),
    agent=feature_engineer,
)

training_task = Task(
    description=(
        "Use run_ml_training with action='run' to train both models. "
        "Report the accuracy, F1, and AUC for each model, and name the winner."
    ),
    expected_output=(
        "Training results: accuracy/F1/AUC for Random Forest and Logistic Regression, "
        "and which model won overall."
    ),
    agent=model_trainer,
    context=[features_task],
)

evaluation_task = Task(
    description=(
        "Use run_ml_evaluation with action='run' to create the full evaluation report. "
        "Then summarize: which model is better and why, based on F1 and the confusion matrix."
    ),
    expected_output=(
        "Confirmation evaluation_report.md was saved, plus a 3-sentence summary of "
        "which model is better and the key metric difference."
    ),
    agent=model_evaluator,
    context=[training_task],
)

model_card_task = Task(
    description=(
        "Based on the evaluation results, write a model card in Hebrew with exactly these 5 sections:\n"
        "1. מטרת המודל — one sentence describing the model's goal.\n"
        "2. סיכום נתוני אימון — brief summary: dataset size, features, time period.\n"
        "3. מדדים — Accuracy, F1, AUC for both models (one line each).\n"
        "4. מגבלות — 3 bullet points: data scope, time period, missing external factors.\n"
        "5. שיקולים אתיים — 3 bullet points: geographic bias, transparency, fair use.\n"
        "Return ONLY this structured text as your Final Answer — do NOT call any tool."
    ),
    expected_output=(
        "A model card in Hebrew with 5 sections: מטרת המודל, סיכום נתוני אימון, מדדים, מגבלות, שיקולים אתיים."
    ),
    agent=model_card_agent,
    context=[evaluation_task],
)
