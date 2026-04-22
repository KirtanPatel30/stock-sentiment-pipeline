"""
dashboard/app.py
Streamlit dashboard for Stock Sentiment Pipeline.
Pages: Overview | Ticker Deep Dive | Predict Movement | Data Explorer
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR     = Path(__file__).parent.parent / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"

st.set_page_config(
    page_title="Stock Sentiment Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        return (
            joblib.load(MODEL_DIR / "sentiment_model.pkl"),
            joblib.load(MODEL_DIR / "scaler.pkl"),
            joblib.load(MODEL_DIR / "feature_names.pkl"),
        )
    except:
        return None, None, None

@st.cache_data
def load_features():
    p = PROCESSED_DIR / "features.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_headlines():
    p = PROCESSED_DIR / "headlines_scored.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_daily_sentiment():
    p = PROCESSED_DIR / "daily_sentiment.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_metrics():
    p = MODEL_DIR / "metrics.json"
    return json.load(open(p)) if p.exists() else None

@st.cache_data
def load_importance():
    p = MODEL_DIR / "feature_importance.csv"
    return pd.read_csv(p) if p.exists() else None

# ── Load all data ──
model, scaler, feature_names = load_model()
df_feat = load_features()
df_head = load_headlines()
df_sent = load_daily_sentiment()
metrics = load_metrics()
df_imp  = load_importance()

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD"]

# ── Sidebar ──
st.sidebar.title("📈 Stock Sentiment")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🔍 Ticker Deep Dive", "🤖 Predict Movement", "🗃️ Data Explorer"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** {'✅ Loaded' if model else '❌ Not found'}")
if metrics:
    st.sidebar.markdown("### Performance")
    st.sidebar.metric("AUC-ROC",  f"{metrics.get('auc_roc', 'N/A'):.4f}")
    st.sidebar.metric("F1 Score", f"{metrics.get('f1_score', 'N/A'):.4f}")
    st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 'N/A'):.4f}")

# ── PAGE: OVERVIEW ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📈 Stock Sentiment & Price Movement Pipeline")
    st.markdown("**Real-time sentiment analysis** | FinBERT NLP | XGBoost price prediction | Kafka streaming")
    st.divider()

    if df_head is not None and df_feat is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Headlines Analyzed", f"{len(df_head):,}")
        c2.metric("Tickers Tracked", str(df_head["ticker"].nunique()))
        c3.metric("Bullish Headlines", f"{(df_head['sentiment_label']=='bullish').sum():,}")
        c4.metric("Bearish Headlines", f"{(df_head['sentiment_label']=='bearish').sum():,}")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution by Ticker")
            sent_by_ticker = df_head.groupby(["ticker","sentiment_label"]).size().reset_index(name="count")
            fig = px.bar(sent_by_ticker, x="ticker", y="count", color="sentiment_label",
                         color_discrete_map={"bullish":"#2ecc71","bearish":"#e74c3c","neutral":"#95a5a6"},
                         barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Sentiment Score Distribution")
            fig2 = px.histogram(df_head, x="sentiment_score", color="sentiment_label",
                                nbins=40, opacity=0.7,
                                color_discrete_map={"bullish":"#2ecc71","bearish":"#e74c3c","neutral":"#95a5a6"})
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Daily Sentiment Ratio Over Time")
        if df_sent is not None:
            df_sent["date"] = pd.to_datetime(df_sent["date"])
            avg_daily = df_sent.groupby("date")["sentiment_ratio"].mean().reset_index()
            fig3 = px.line(avg_daily, x="date", y="sentiment_ratio",
                           title="Avg Daily Sentiment Ratio (Bullish - Bearish) / Total")
            fig3.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Run `python run_all.py` first to generate data.")

# ── PAGE: TICKER DEEP DIVE ──────────────────────────────────────────────────
elif page == "🔍 Ticker Deep Dive":
    st.title("🔍 Ticker Deep Dive")
    ticker = st.selectbox("Select Ticker", TICKERS)

    if df_feat is not None and df_head is not None:
        tf = df_feat[df_feat["ticker"] == ticker].copy()
        th = df_head[df_head["ticker"] == ticker].copy()

        tf["date"] = pd.to_datetime(tf["date"])
        th["timestamp"] = pd.to_datetime(th["timestamp"])

        # Price + sentiment overlay
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Price", "Sentiment Score"],
                            row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(x=tf["date"], y=tf["close"], name="Close Price",
                                  line=dict(color="#3498db")), row=1, col=1)
        fig.add_trace(go.Scatter(x=tf["date"], y=tf["ma_20"], name="MA20",
                                  line=dict(color="#e67e22", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=tf["date"], y=tf["avg_sentiment_score"],
                                  name="Sentiment Score", line=dict(color="#2ecc71"),
                                  fill="tozeroy"), row=2, col=1)
        fig.update_layout(height=500, title=f"{ticker} — Price vs Sentiment")
        st.plotly_chart(fig, use_container_width=True)

        # Recent headlines
        st.subheader(f"Recent Headlines — {ticker}")
        cols = ["timestamp", "text", "sentiment_label", "sentiment_score", "source"]
        available = [c for c in cols if c in th.columns]
        st.dataframe(
            th[available].sort_values("timestamp", ascending=False).head(20),
            use_container_width=True
        )
    else:
        st.warning("Run the pipeline first.")

# ── PAGE: PREDICT MOVEMENT ──────────────────────────────────────────────────
elif page == "🤖 Predict Movement":
    st.title("🤖 Predict Next-Day Price Movement")
    st.markdown("Enter current market signals to predict whether the stock will move UP or DOWN tomorrow.")

    if model is None:
        st.error("Model not loaded. Run `python run_all.py` first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker     = st.selectbox("Ticker", TICKERS)
            daily_ret  = st.number_input("Daily Return", -0.2, 0.2, 0.01, 0.001, format="%.3f")
            price_rng  = st.number_input("Price Range (High-Low/Close)", 0.0, 0.2, 0.02, 0.001)
        with col2:
            rsi        = st.slider("RSI", 0.0, 100.0, 55.0, 1.0)
            vol_chg    = st.number_input("Volume Change", -0.5, 2.0, 0.05, 0.01)
            vs_ma5     = st.number_input("Price vs MA5", -0.1, 0.1, 0.01, 0.001, format="%.3f")
        with col3:
            sentiment  = st.slider("Avg Sentiment Score", 0.0, 1.0, 0.6, 0.01)
            sent_ratio = st.slider("Sentiment Ratio (Bull-Bear)", -1.0, 1.0, 0.2, 0.01)
            posts      = st.number_input("Total Posts Today", 0, 1000, 50)

        if st.button("🔮 Predict Tomorrow's Movement", type="primary"):
            row = {f: 0.0 for f in feature_names}
            row.update({
                "daily_return": daily_ret,
                "log_return": np.log1p(daily_ret),
                "price_range": price_rng,
                "rsi": rsi,
                "avg_sentiment_score": sentiment,
                "sentiment_ratio": sent_ratio,
                "total_posts": posts,
                "volume_change": vol_chg,
                "price_vs_ma5": vs_ma5,
            })
            X    = pd.DataFrame([row])[feature_names]
            X_sc = pd.DataFrame(scaler.transform(X), columns=feature_names)
            prob = float(model.predict_proba(X_sc)[0][1])
            pred = "📈 UP" if prob >= 0.5 else "📉 DOWN"
            conf = "HIGH" if abs(prob-0.5)>0.2 else "MEDIUM" if abs(prob-0.5)>0.1 else "LOW"

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Prediction",   pred)
            r2.metric("Probability",  f"{prob:.2%}")
            r3.metric("Confidence",   conf)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                title={"text": "Probability of UP Movement (%)"},
                gauge={
                    "axis": {"range": [0,100]},
                    "bar":  {"color": "#2ecc71" if prob>=0.5 else "#e74c3c"},
                    "steps": [
                        {"range": [0,40],   "color": "#fadbd8"},
                        {"range": [40,60],  "color": "#fdebd0"},
                        {"range": [60,100], "color": "#d5f5e3"},
                    ],
                    "threshold": {"line": {"color":"gray","width":4}, "value":50}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# ── PAGE: DATA EXPLORER ─────────────────────────────────────────────────────
elif page == "🗃️ Data Explorer":
    st.title("🗃️ Data Explorer")
    tab1, tab2 = st.tabs(["📰 Headlines", "📊 Features"])

    with tab1:
        if df_head is not None:
            t_filter = st.multiselect("Ticker", TICKERS, default=TICKERS[:3])
            s_filter = st.multiselect("Sentiment", ["bullish","bearish","neutral"], default=["bullish","bearish","neutral"])
            filtered = df_head[(df_head["ticker"].isin(t_filter)) & (df_head["sentiment_label"].isin(s_filter))]
            st.write(f"{len(filtered):,} headlines")
            cols = ["timestamp","ticker","text","sentiment_label","sentiment_score","source"]
            available = [c for c in cols if c in filtered.columns]
            st.dataframe(filtered[available].head(300), use_container_width=True)

    with tab2:
        if df_feat is not None:
            st.dataframe(df_feat.head(300), use_container_width=True)
        else:
            st.warning("Run pipeline first.")
