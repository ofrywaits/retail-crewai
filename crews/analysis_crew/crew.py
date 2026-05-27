"""
Crew 1 — Analysis Crew
מחבר את 3 הסוכנים ו-3 המשימות לצוות אחד
"""

from crewai import Crew, Process
from crews.analysis_crew.agents import data_loader_agent, data_cleaner_agent, contract_agent
from crews.analysis_crew.tasks  import ingest_task, clean_task, contract_task


def build_analysis_crew() -> Crew:
    return Crew(
        agents=[data_loader_agent, data_cleaner_agent, contract_agent],
        tasks=[ingest_task, clean_task, contract_task],
        process=Process.sequential,  # כל סוכן מחכה לסוכן הקודם
        verbose=True,
    )
