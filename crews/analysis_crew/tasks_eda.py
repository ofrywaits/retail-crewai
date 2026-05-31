"""
2 משימות Day 3: EDA + Insights
"""

from crewai import Task
from crews.analysis_crew.agents_eda import eda_agent, insights_agent

eda_task = Task(
    description=(
        "Use the generate_eda_report tool with action='run' to create the full HTML report. "
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
        "First use get_data_statistics with action='run' to get the numbers. "
        "Then write 6 business insights in Hebrew, each with: "
        "a title (## כותרת), the finding with specific numbers, and a practical recommendation. "
        "Return ALL 6 insights as your complete Final Answer — do NOT call any saving tool."
    ),
    expected_output=(
        "6 Hebrew business insights, each containing a title (## heading), "
        "a data-backed finding with specific numbers, and an actionable recommendation."
    ),
    agent=insights_agent,
    context=[eda_task],
)
