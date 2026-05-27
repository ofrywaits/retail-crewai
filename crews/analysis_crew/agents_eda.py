"""
2 סוכני Day 3: EDA + Insights
"""

from crewai import Agent, LLM
from dotenv import load_dotenv
import os
from crews.analysis_crew.tools_eda import (
    generate_eda_report,
    get_data_statistics,
    generate_insights_md,
)

load_dotenv()

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

# ========================================
# סוכן 4 — EDA Agent
# תפקיד: יוצר גרפים ודו"ח HTML
# ========================================
eda_agent = Agent(
    role="EDA Analyst",
    goal=(
        "Generate a comprehensive HTML EDA report with 6 charts for the Israeli retail dataset. "
        "Save it as eda_report.html. Then summarize the most important visual findings."
    ),
    backstory=(
        "You are a senior data analyst who believes every dataset tells a story. "
        "You create beautiful, informative reports that help business stakeholders "
        "understand the data at a glance."
    ),
    tools=[generate_eda_report],
    llm=llm,
    verbose=True,
)

# ========================================
# סוכן 5 — Insights Agent
# תפקיד: מנתח נתונים ושומר תובנות ל-insights.md
# ========================================
insights_agent = Agent(
    role="Business Insights Analyst",
    goal=(
        "Analyze the Israeli retail data statistics and write 6 clear, actionable "
        "business insights in Hebrew. Save them to insights.md."
    ),
    backstory=(
        "You are a retail business consultant who worked with Israeli chains like "
        "Shufersal and Rami Levy. You understand Israeli consumer behavior, holiday "
        "effects, and regional differences. You write insights that drive real decisions."
    ),
    tools=[get_data_statistics, generate_insights_md],
    llm=llm,
    verbose=True,
)
