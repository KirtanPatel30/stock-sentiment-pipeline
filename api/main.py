"""
api/main.py
FastAPI REST API for stock sentiment + price movement prediction.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(
    title="Stock Sentiment & Price Movement API",
    description="Predicts next-day price movement using sentiment + technical signals",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = scaler = feature_names = None


@app.on_event("startup")
def load_model():
    global model, scaler, feature_names
    if (MODEL_DIR / "sentiment_model.pkl").exists():
        model         = joblib.load(MODEL_DIR / "sentiment_model.pkl")
        scaler        = joblib.load(MODEL_DIR / "scaler.pkl")
        feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        print("Model loaded.")
    else:
        print("WARNING: Model not found. Run models/train.py first.")


class PredictionInput(BaseModel):
    ticker: str = Field(..., example="AAPL")
    daily_return: float = Field(..., example=0.012)
    price_range: float = Field(..., example=0.025)
    rsi: float = Field(..., example=58.0)
    avg_sentiment_score: float = Field(..., example=0.72)
    sentiment_ratio: float = Field(..., example=0.35)
    total_posts: int = Field(..., example=45)
    volume_change: float = Field(..., example=0.08)
    price_vs_ma5: float = Field(..., example=0.012)
    price_vs_ma20: float = Field(..., example=-0.005)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    p = MODEL_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "Metrics not found. Train model first.")
    return json.load(open(p))


@app.post("/predict")
def predict(inp: PredictionInput):
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    row = {f: 0.0 for f in feature_names}
    row.update({
        "daily_return": inp.daily_return,
        "log_return": np.log1p(inp.daily_return),
        "price_range": inp.price_range,
        "rsi": inp.rsi,
        "avg_sentiment_score": inp.avg_sentiment_score,
        "sentiment_ratio": inp.sentiment_ratio,
        "total_posts": inp.total_posts,
        "volume_change": inp.volume_change,
        "price_vs_ma5": inp.price_vs_ma5,
        "price_vs_ma20": inp.price_vs_ma20,
    })

    X = pd.DataFrame([row])[feature_names]
    X_sc = pd.DataFrame(scaler.transform(X), columns=feature_names)
    prob = float(model.predict_proba(X_sc)[0][1])
    pred = "UP" if prob >= 0.5 else "DOWN"
    conf = "HIGH" if abs(prob - 0.5) > 0.2 else "MEDIUM" if abs(prob - 0.5) > 0.1 else "LOW"

    sentiment_str = (
        "BULLISH" if inp.avg_sentiment_score > 0.6
        else "BEARISH" if inp.avg_sentiment_score < 0.4
        else "NEUTRAL"
    )

    return {
        "ticker": inp.ticker,
        "predicted_movement": pred,
        "probability_up": round(prob, 4),
        "confidence": conf,
        "sentiment_signal": sentiment_str,
        "rsi_signal": "OVERBOUGHT" if inp.rsi > 70 else "OVERSOLD" if inp.rsi < 30 else "NEUTRAL",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
