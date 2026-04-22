# 📈 Stock Sentiment & Price Movement Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?style=flat-square&logo=streamlit)
![Kafka](https://img.shields.io/badge/Kafka-streaming-black?style=flat-square&logo=apachekafka)

> Real-time pipeline ingesting Reddit + Yahoo Finance headlines → FinBERT sentiment scoring → XGBoost price movement prediction → live Streamlit dashboard.

## Stack
- **Ingestion:** Reddit (PRAW), Yahoo Finance, Kafka producer
- **NLP:** FinBERT sentiment scoring (rule-based fallback)
- **ML:** XGBoost with TimeSeriesSplit cross-validation
- **API:** FastAPI REST endpoint
- **Dashboard:** Streamlit + Plotly (4 pages)
- **Infra:** Docker + PostgreSQL

## Quick Start
```bash
pip install -r requirements.txt
python run_all.py
streamlit run dashboard/app.py   # http://localhost:8501
uvicorn api.main:app --reload    # http://localhost:8000/docs
```

## Resume Bullets
- Built real-time sentiment pipeline ingesting 2,000+ Reddit/finance headlines using FinBERT NLP scoring
- Engineered 18 features combining technical indicators (RSI, MA, volatility) with sentiment signals
- Trained XGBoost price movement classifier with TimeSeriesSplit cross-validation for temporal integrity
- Served predictions via FastAPI; visualized 4-page interactive Streamlit dashboard with live gauge charts
- Architected Kafka producer for real-time headline streaming with PostgreSQL persistence layer
