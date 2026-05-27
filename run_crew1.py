"""
הרצת Crew 1 — ניקוי נתונים
הרץ: python3 run_crew1.py
"""

from crews.analysis_crew.crew import build_analysis_crew
import os, json

if __name__ == "__main__":
    print("=" * 55)
    print("  Crew 1 — Data Cleaning & Contract Generation")
    print("=" * 55)

    crew   = build_analysis_crew()
    result = crew.kickoff()

    print("\n" + "=" * 55)
    print("  תוצאה סופית:")
    print("=" * 55)
    print(result)

    # בדיקה שהפלטים נוצרו
    print("\n" + "=" * 55)
    print("  בדיקת קבצי פלט:")
    print("=" * 55)
    for path, label in [
        ("data/processed/clean_data.csv",        "clean_data.csv"),
        ("data/processed/dataset_contract.json", "dataset_contract.json"),
    ]:
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else 0
        status = f"✓ קיים ({size:,} bytes)" if exists else "✗ חסר!"
        print(f"  {label:30s} {status}")
