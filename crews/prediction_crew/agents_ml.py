"""
4 סוכני Crew 2 — Feature Engineering + ML
"""

from crewai import Agent, LLM
from dotenv import load_dotenv
import os
from crews.prediction_crew.tools_ml import (
    create_features,
    train_models,
    evaluate_models,
    create_model_card,
)

load_dotenv()

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

feature_engineer = Agent(
    role="Feature Engineer",
    goal=(
        "Transform the clean retail dataset into ML-ready features. "
        "Encode categorical variables, extract date features, and create the target variable."
    ),
    backstory=(
        "You are an ML engineer who knows that good features are 80% of model success. "
        "You carefully encode categories, extract time patterns, and document every decision."
    ),
    tools=[create_features],
    llm=llm,
    verbose=True,
)

model_trainer = Agent(
    role="ML Model Trainer",
    goal=(
        "Train Random Forest and Logistic Regression models on the features dataset. "
        "Save both models and identify which performs better."
    ),
    backstory=(
        "You are a machine learning practitioner who always compares at least two models. "
        "You believe in reproducibility — you always set random_state and document hyperparameters."
    ),
    tools=[train_models],
    llm=llm,
    verbose=True,
)

model_evaluator = Agent(
    role="Model Evaluator",
    goal=(
        "Evaluate both trained models using accuracy, F1 score, AUC, and confusion matrix. "
        "Save a detailed evaluation_report.md."
    ),
    backstory=(
        "You are a rigorous ML evaluator who never trusts a model without checking it from "
        "multiple angles. You always look at the confusion matrix, not just accuracy."
    ),
    tools=[evaluate_models],
    llm=llm,
    verbose=True,
)

model_card_agent = Agent(
    role="Model Documentation Specialist",
    goal=(
        "Write 2-3 sentences about the business value of the trained model: "
        "how retailers can use it to improve profit margins."
    ),
    backstory=(
        "You believe AI models must be transparent. You write model cards that any "
        "stakeholder — technical or not — can understand."
    ),
    tools=[],
    llm=llm,
    verbose=True,
)
