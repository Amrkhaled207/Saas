# Universal Data Cleaner & Q&A Dashboard

A reusable Streamlit app that:
- Accepts **any CSV/Excel dataset**.
- Performs **automatic cleaning** (duplicates, whitespace, types, missing values).
- Applies **preprocessing** (encoders, scaling) with a **configurable pipeline**.
- Generates on-demand **EDA** and **interactive dashboards**.
- Supports **question answering** about your data with simple natural-language patterns and SQL (DuckDB).
- Exposes a minimal **AutoML** starter (optional) for quick modeling.

> Built for portability: plain Python + Streamlit + DuckDB. No heavy external services by default.
> Optional LLM-powered Q&A hooks (OpenAI / Gemini) are included but **disabled by default**.

---

## 🧩 Tech Stack
- **Python 3.10+**
- **pandas**, **numpy**
- **duckdb** for fast SQL over DataFrames
- **scikit-learn**, **feature-engine**, **category-encoders**
- **Streamlit** for UI
- **plotly** and **altair** for charts
- **ydata-profiling** (optional) for quick HTML profiles
- **pydantic** & **yaml** for config validation

---

## 🚀 Quickstart

```bash
# 1) Create venv
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run app
streamlit run app.py
```

Open http://localhost:8501

---

## 📂 Structure
```
universal_data_cleaner_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── config.yaml
├── core/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── cleaning.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── viz.py
│   ├── qa.py
│   └── utils.py
└── examples/
    └── sample.csv
```

---

## ⚙️ Config
Edit `config.yaml` to control cleaning and preprocessing behaviors.

---

## 🔌 Optional: LLM for Q&A
If you want natural-language → SQL/analysis with an LLM, add API keys to `.env` and set:
```yaml
qa:
  llm_enabled: true
  provider: "openai"   # or "gemini"
```
You’ll also need to implement the provider in `core/qa.py` (stubs included).

---

## 🧪 Roadmap
- Richer intent parser for NL questions
- Save/load pipelines as YAML
- Model selection & experiment tracking (MLflow)
