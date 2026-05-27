"""
בדיקה ראשונה - Crew עם 2 סוכנים
מטרה: לוודא שהכל עובד לפני שנבנה את המערכת הגדולה
"""

from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

# טוען את המפתחות מקובץ .env
load_dotenv()

# ========================================
# הגדרת המודל - Groq (חינמי וחזק)
# ========================================
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

# ========================================
# סוכן 1: אנליסט נתונים
# תפקיד: מנתח נתונים קמעונאיים ומוצא תובנות
# ========================================
data_analyst = Agent(
    role="Data Analyst",
    goal="Analyze retail sales data and identify key patterns and trends",
    backstory=(
        "You are an expert data analyst specializing in retail. "
        "You love finding hidden patterns in sales data."
    ),
    llm=groq_llm,
    verbose=True,
)

# ========================================
# סוכן 2: יועץ עסקי
# תפקיד: הופך את הניתוח להמלצות עסקיות
# ========================================
business_advisor = Agent(
    role="Business Advisor",
    goal="Turn data insights into clear business recommendations",
    backstory=(
        "You are a senior retail business consultant. "
        "You translate complex data analysis into simple, actionable advice."
    ),
    llm=groq_llm,
    verbose=True,
)

# ========================================
# משימה 1: ניתוח נתונים
# ========================================
analysis_task = Task(
    description=(
        "Analyze the following retail data summary and identify the top 3 insights:\n"
        "- Total sales: $500,000\n"
        "- Best selling category: Electronics (40% of sales)\n"
        "- Worst performing day: Monday\n"
        "- Customer return rate: 15%\n"
        "- Top product: iPhone case (1,200 units sold)"
    ),
    expected_output="A list of 3 key insights from the retail data, each explained in 2 sentences.",
    agent=data_analyst,
)

# ========================================
# משימה 2: המלצות עסקיות
# ========================================
recommendation_task = Task(
    description=(
        "Based on the data analysis, create 3 specific business recommendations "
        "to improve sales performance. Be practical and specific."
    ),
    expected_output="3 clear business recommendations, each with a title and 2-sentence explanation.",
    agent=business_advisor,
    context=[analysis_task],  # מקבל את תוצאת המשימה הקודמת
)

# ========================================
# יצירת ה-Crew (הצוות)
# ========================================
crew = Crew(
    agents=[data_analyst, business_advisor],
    tasks=[analysis_task, recommendation_task],
    verbose=True,
)

# ========================================
# הרצת הצוות
# ========================================
if __name__ == "__main__":
    print("=" * 50)
    print("מריץ את ה-Crew הראשון...")
    print("=" * 50)

    result = crew.kickoff()

    print("\n" + "=" * 50)
    print("תוצאה סופית:")
    print("=" * 50)
    print(result)
