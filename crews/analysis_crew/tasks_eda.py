"""
2 משימות Day 3: EDA + Insights
"""

from crewai import Task
from crews.analysis_crew.agents_eda import eda_agent, insights_agent

eda_task = Task(
    description=(
        "Use the generate_eda_report tool to create the full HTML report. "
        "After it's saved, list the 3 most important findings you see in the charts."
    ),
    expected_output=(
        "Confirmation that eda_report.html was saved, followed by 3 key visual findings "
        "from the charts (e.g., which category dominates, which month peaks, etc.)."
    ),
    agent=eda_agent,
)

insights_task = Task(
    description=(
        "First use get_data_statistics to get the numbers. "
        "Then write 6 business insights in Hebrew, each with: "
        "a title, the finding with specific numbers, and a practical recommendation. "
        "Finally use generate_insights_md to save them."
    ),
    expected_output=(
        "6 Hebrew business insights saved to insights.md, each containing "
        "a title, data-backed finding, and actionable recommendation."
    ),
    agent=insights_agent,
    context=[eda_task],
)
