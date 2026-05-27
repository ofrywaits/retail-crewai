"""
3 סוכנים של Crew 1 — ניקוי נתונים
"""

from crewai import Agent, LLM
from dotenv import load_dotenv
import os

from crews.analysis_crew.tools import (
    load_and_inspect_data,
    clean_and_save_data,
    generate_dataset_contract,
)

load_dotenv()

# מודל Groq — חינמי ומהיר
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)


# ========================================
# סוכן 1 — Data Loader
# תפקיד: טוען את הדאטא ומדווח מה יש בו
# ========================================
data_loader_agent = Agent(
    role="Data Loader",
    goal=(
        "Load the Israeli retail dataset and produce a clear inspection report. "
        "Identify missing values, duplicates, negative values, and data quality issues."
    ),
    backstory=(
        "You are a meticulous data engineer who always validates data before "
        "passing it to the next stage. You never assume data is clean — you check everything."
    ),
    tools=[load_and_inspect_data],
    llm=llm,
    verbose=True,
)


# ========================================
# סוכן 2 — Data Cleaner
# תפקיד: מנקה את הנתונים ושומר clean_data.csv
# ========================================
data_cleaner_agent = Agent(
    role="Data Cleaner",
    goal=(
        "Clean the Israeli retail dataset by removing duplicates, fixing data types, "
        "handling missing values, and removing outliers. Save the result as clean_data.csv."
    ),
    backstory=(
        "You are a data cleaning specialist with years of experience in retail data. "
        "You know exactly which rows to keep and which to remove, and you document every step."
    ),
    tools=[clean_and_save_data],
    llm=llm,
    verbose=True,
)


# ========================================
# סוכן 3 — Contract Generator
# תפקיד: מייצר JSON שמתאר את הסכמה של הדאטאסט
# ========================================
contract_agent = Agent(
    role="Dataset Contract Generator",
    goal=(
        "Generate a comprehensive dataset_contract.json that documents every column "
        "in the cleaned dataset — its type, range, allowed values, and nullable status."
    ),
    backstory=(
        "You are a data governance expert. You believe every dataset needs a contract "
        "so that downstream models and agents know exactly what to expect."
    ),
    tools=[generate_dataset_contract],
    llm=llm,
    verbose=True,
)
