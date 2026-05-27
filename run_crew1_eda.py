"""
הרצת Day 3 — EDA + Insights
הרץ: python3 run_crew1_eda.py
"""

from crewai import Crew, Process
from crews.analysis_crew.agents_eda import eda_agent, insights_agent
from crews.analysis_crew.tasks_eda  import eda_task, insights_task
import os

if __name__ == "__main__":
    print("=" * 55)
    print("  Day 3 — EDA Report + Business Insights")
    print("=" * 55)

    crew = Crew(
        agents=[eda_agent, insights_agent],
        tasks=[eda_task, insights_task],
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
    for path, label in [
        ("data/processed/eda_report.html", "eda_report.html"),
        ("data/processed/insights.md",     "insights.md"),
    ]:
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else 0
        status = f"✓ קיים ({size:,} bytes)" if exists else "✗ חסר!"
        print(f"  {label:30s} {status}")
