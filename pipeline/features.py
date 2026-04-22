"""
pipeline/features.py
Joins sentiment signals with price data and engineers ML features.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def engineer_features():
    print("[FEATURES] Engineering features...")

    prices_path    = Path(__file__).parent.parent / "data" / "raw" / "prices.csv"
    sentiment_path = PROCESSED_DIR / "daily_sentiment.csv"

    prices    = pd.read_csv(prices_path)
    sentiment = pd.read_csv(sentiment_path)

    prices["date"]    = pd.to_datetime(prices["date"]).dt.date
    sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date

    df = prices.merge(sentiment, on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Fill missing sentiment with neutral
    df["avg_sentiment_score"] = df["avg_sentiment_score"].fillna(0.5)
    df["sentiment_ratio"]     = df["sentiment_ratio"].fillna(0.0)
    df["total_posts"]         = df["total_posts"].fillna(0)

    # Price features
    df["daily_return"]     = df.groupby("ticker")["close"].pct_change()
    df["log_return"]       = np.log1p(df["daily_return"])
    df["price_range"]      = (df["high"] - df["low"]) / df["close"]
    df["volume_change"]    = df.groupby("ticker")["volume"].pct_change()

    # Rolling price features
    for w in [5, 10, 20]:
        df[f"ma_{w}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"volatility_{w}"] = df.groupby("ticker")["daily_return"].transform(
            lambda x: x.rolling(w, min_periods=1).std()
        )

    df["price_vs_ma5"]  = df["close"] / (df["ma_5"]  + 1e-6) - 1
    df["price_vs_ma20"] = df["close"] / (df["ma_20"] + 1e-6) - 1

    # RSI
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs    = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))

    df["rsi"] = df.groupby("ticker")["close"].transform(compute_rsi)

    # Rolling sentiment features
    for w in [3, 7]:
        df[f"sentiment_ma_{w}"] = df.groupby("ticker")["avg_sentiment_score"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )

    # Target: next-day price movement (1 = up, 0 = down)
    df["next_return"] = df.groupby("ticker")["daily_return"].shift(-1)
    df["target"]      = (df["next_return"] > 0).astype(int)

    # Drop rows without target
    df = df.dropna(subset=["next_return", "daily_return"])

    out = PROCESSED_DIR / "features.csv"
    df.to_csv(out, index=False)
    print(f"[FEATURES] Saved {len(df):,} rows with {len(df.columns)} features to {out}")
    return df


if __name__ == "__main__":
    engineer_features()
