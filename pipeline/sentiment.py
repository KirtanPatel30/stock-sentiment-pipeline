"""
pipeline/sentiment.py
Runs FinBERT-style sentiment analysis on headlines.
Uses a lightweight rule-based model as fallback if transformers/torch is unavailable.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BULLISH_WORDS = {
    "moon", "mooning", "bull", "bullish", "buy", "bought", "calls", "long",
    "rally", "soar", "surge", "beat", "upgrade", "undervalued", "breakout",
    "growth", "loaded", "strong", "partnership", "exceeded", "upside",
}
BEARISH_WORDS = {
    "crash", "bear", "bearish", "sell", "sold", "puts", "short", "drop",
    "dump", "dumping", "miss", "downgrade", "overvalued", "fall", "avoid",
    "risk", "cut", "headwinds", "regulatory", "disaster",
}


def rule_based_sentiment(text: str) -> tuple:
    """Fast rule-based sentiment scorer."""
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    bull_score = len(tokens & BULLISH_WORDS)
    bear_score = len(tokens & BEARISH_WORDS)
    total = bull_score + bear_score + 1e-6
    if bull_score > bear_score:
        label = "bullish"
        score = bull_score / total
    elif bear_score > bull_score:
        label = "bearish"
        score = bear_score / total
    else:
        label = "neutral"
        score = 0.5
    return label, round(score, 4)


def load_finbert():
    """Try to load FinBERT. Returns None if unavailable."""
    try:
        from transformers import pipeline
        print("[SENTIMENT] Loading FinBERT model...")
        nlp = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=False,
            truncation=True,
            max_length=512,
        )
        print("[SENTIMENT] FinBERT loaded successfully.")
        return nlp
    except Exception as e:
        print(f"[SENTIMENT] FinBERT unavailable ({e}). Using rule-based scorer.")
        return None


def score_batch(texts: list, nlp=None, batch_size: int = 64) -> list:
    """Score a list of texts. Uses FinBERT if available, else rule-based."""
    results = []
    if nlp is not None:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            preds = nlp(batch)
            for pred in preds:
                label = pred["label"].lower()
                score = round(pred["score"], 4)
                results.append((label, score))
            if i % 200 == 0:
                print(f"[SENTIMENT] Scored {min(i+batch_size, len(texts))}/{len(texts)}")
    else:
        results = [rule_based_sentiment(t) for t in texts]
    return results


def run_sentiment_pipeline(headlines_path=None):
    print("[SENTIMENT] Running sentiment analysis...")
    if headlines_path is None:
        headlines_path = Path(__file__).parent.parent / "data" / "raw" / "headlines.csv"

    df = pd.read_csv(headlines_path)
    print(f"[SENTIMENT] Loaded {len(df):,} headlines")

    nlp = load_finbert()
    results = score_batch(df["text"].tolist(), nlp)

    df["sentiment_label"] = [r[0] for r in results]
    df["sentiment_score"] = [r[1] for r in results]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Aggregate daily sentiment per ticker
    daily = df.groupby(["ticker", "date"]).agg(
        avg_sentiment_score=("sentiment_score", "mean"),
        bullish_count=("sentiment_label", lambda x: (x == "bullish").sum()),
        bearish_count=("sentiment_label", lambda x: (x == "bearish").sum()),
        neutral_count=("sentiment_label", lambda x: (x == "neutral").sum()),
        total_posts=("sentiment_label", "count"),
        avg_post_score=("score", "mean"),
    ).reset_index()

    daily["sentiment_ratio"] = (
        (daily["bullish_count"] - daily["bearish_count"]) /
        (daily["total_posts"] + 1e-6)
    )

    out_headlines = PROCESSED_DIR / "headlines_scored.csv"
    out_daily = PROCESSED_DIR / "daily_sentiment.csv"
    df.to_csv(out_headlines, index=False)
    daily.to_csv(out_daily, index=False)

    print(f"[SENTIMENT] Saved scored headlines to {out_headlines}")
    print(f"[SENTIMENT] Saved daily aggregates to {out_daily}")
    print(f"[SENTIMENT] Label distribution:\n{df['sentiment_label'].value_counts()}")
    return df, daily


if __name__ == "__main__":
    run_sentiment_pipeline()
