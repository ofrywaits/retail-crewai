"""
3 משימות של Crew 1
"""

from crewai import Task
from crews.analysis_crew.agents import (
    data_loader_agent,
    data_cleaner_agent,
    contract_agent,
)


# ========================================
# משימה 1 — טעינה ובדיקה
# ========================================
ingest_task = Task(
    description=(
        "Use the load_and_inspect_data tool with action='run' to load the Israeli retail dataset. "
        "Report: total rows, columns list, missing values per column, duplicate orders, "
        "negative values, numeric statistics, and date range. "
        "Be specific with numbers."
    ),
    expected_output=(
        "A structured inspection report with: row count, column names, "
        "missing value counts, duplicate count, negative value count, "
        "min/max/mean for numeric columns, and date range."
    ),
    agent=data_loader_agent,
)


# ========================================
# משימה 2 — ניקוי ושמירה
# ========================================
clean_task = Task(
    description=(
        "Use the clean_and_save_data tool with action='run' to clean the dataset. "
        "After cleaning, summarize: how many rows were removed, what was removed, "
        "and confirm clean_data.csv was saved successfully."
    ),
    expected_output=(
        "A cleaning summary with: original row count, final row count, "
        "number of rows removed per step, and confirmation that clean_data.csv was saved."
    ),
    agent=data_cleaner_agent,
    context=[ingest_task],
)


# ========================================
# משימה 3 — Contract
# ========================================
contract_task = Task(
    description=(
        "Use the generate_dataset_contract tool with action='run' to create dataset_contract.json. "
        "Then summarize the contract: how many columns were documented, "
        "what types exist, and confirm the file was saved."
    ),
    expected_output=(
        "Confirmation that dataset_contract.json was saved, with a summary of "
        "columns documented, data types found, and key ranges."
    ),
    agent=contract_agent,
    context=[clean_task],
)
