"""
הרצת ה-Flow המלא — מ-raw data ועד מודל מאומן
הרץ: python3 run_flow.py
"""

from flows.flow import RetailAnalysisFlow

if __name__ == "__main__":
    flow   = RetailAnalysisFlow()
    result = flow.kickoff()

    print("\nתוצאה סופית:")
    print(result)
