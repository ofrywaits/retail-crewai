# Retail AI Analysis System

A multi-agent AI system built with CrewAI Flow for analyzing retail data and generating business insights.

## What It Does

- **Team 1 (Analysis Crew):** Analyzes retail sales data to find patterns and trends
- **Team 2 (Prediction Crew):** Generates business recommendations and predictions
- **CrewAI Flow:** Orchestrates both teams in a coordinated pipeline
- **Streamlit Dashboard:** Interactive UI to visualize results

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Main language |
| CrewAI | Multi-agent orchestration |
| Groq (Llama 3.3) | LLM for agents |
| Pandas | Data processing |
| Scikit-learn | Machine learning |
| Streamlit | Web dashboard |
| Matplotlib / Seaborn | Charts and graphs |

## Project Structure

```
retail-crewai/
├── data/
│   ├── raw/              # Original Kaggle dataset
│   └── processed/        # Cleaned data
├── crews/
│   ├── analysis_crew/    # Team 1: Data analysis agents
│   └── prediction_crew/  # Team 2: Prediction agents
├── flows/                # CrewAI Flow connecting both teams
├── models/               # Saved ML models
├── app/                  # Streamlit dashboard
├── tests/                # Test files
└── test_crew.py          # Quick test to verify setup
```

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd retail-crewai

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Test the setup
python3 test_crew.py
```

## Dataset

Using the [Online Retail Dataset](https://www.kaggle.com/) from Kaggle.

## Course

Final project for the AI Developer Program.
