"""
data/fetch_news.py
Fetches headlines from Reddit (WSB, stocks, investing) and Yahoo Finance news.
Falls back to realistic mock data if API keys are not configured.
"""

import os
import random
import hashlib
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD"]

BULLISH_TEMPLATES = [
    "{t} is absolutely mooning right now, bought more calls",
    "Just loaded up on {t} — earnings beat expectations massively",
    "{t} breaking out of resistance, this is going to $500 easy",
    "Analysts upgrading {t} to strong buy — massive upside target",
    "{t} partnership announcement is a game changer for the stock",
    "Bought {t} dip, this company is undervalued at current prices",
    "{t} revenue growth is insane, holding long term no question",
    "All in on {t} calls, product launch exceeded all expectations",
]

BEARISH_TEMPLATES = [
    "{t} is overvalued and about to crash hard, puts loaded",
    "Sold all my {t} — earnings miss was a disaster, getting out",
    "{t} CEO dumping shares is a massive red flag for investors",
    "Short {t} here, chart pattern screaming distribution top",
    "{t} competition is eating their lunch, avoid this stock",
    "Bought {t} puts — macro headwinds are going to crush margins",
    "{t} guidance cut is just the beginning of bad news ahead",
    "Regulatory risk on {t} is being completely underpriced right now",
]

NEUTRAL_TEMPLATES = [
    "What does everyone think about {t} going into earnings season?",
    "Anyone else watching {t} closely this week for a breakout?",
    "{t} held support today but volume was pretty low overall",
    "Interesting price action on {t} — waiting for more clarity",
    "{t} quarterly report comes out next week, setting up nicely",
    "Is {t} a buy at these levels or wait for a better entry?",
]


def generate_mock_headlines(n=2000, days_back=30):
    """Generate realistic financial headlines with sentiment labels."""
    np.random.seed(42)
    records = []
    base_time = datetime.now()

    for i in range(n):
        ticker = random.choice(TICKERS)
        sentiment_roll = random.random()

        if sentiment_roll < 0.35:
            text = random.choice(BULLISH_TEMPLATES).format(t=ticker)
            true_sentiment = "bullish"
        elif sentiment_roll < 0.65:
            text = random.choice(BEARISH_TEMPLATES).format(t=ticker)
            true_sentiment = "bearish"
        else:
            text = random.choice(NEUTRAL_TEMPLATES).format(t=ticker)
            true_sentiment = "neutral"

        timestamp = base_time - timedelta(
            minutes=random.randint(0, days_back * 24 * 60)
        )
        score = random.randint(0, 5000)
        source = random.choice(["reddit_wsb", "reddit_stocks", "reddit_investing", "yahoo_finance"])

        records.append({
            "id": hashlib.md5(f"{text}{i}".encode()).hexdigest()[:12],
            "ticker": ticker,
            "text": text,
            "source": source,
            "score": score,
            "timestamp": timestamp.isoformat(),
            "true_sentiment": true_sentiment,
        })

    df = pd.DataFrame(records)
    return df


def fetch_yahoo_news(ticker: str) -> list:
    """Fetch real news from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news or []
        results = []
        for item in news[:10]:
            results.append({
                "id": item.get("uuid", ""),
                "ticker": ticker,
                "text": item.get("title", ""),
                "source": "yahoo_finance",
                "score": 100,
                "timestamp": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).isoformat(),
                "true_sentiment": None,
            })
        return results
    except Exception:
        return []


def fetch_reddit_posts(ticker: str, limit: int = 25) -> list:
    """Fetch Reddit posts using PRAW if credentials are set."""
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if not client_id or client_id == "your_client_id":
        return []
    try:
        import praw
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=os.getenv("REDDIT_USER_AGENT", "StockSentimentBot/1.0"),
        )
        results = []
        for sub in ["wallstreetbets", "stocks", "investing"]:
            for post in reddit.subreddit(sub).search(ticker, limit=limit // 3, sort="new"):
                results.append({
                    "id": post.id,
                    "ticker": ticker,
                    "text": post.title,
                    "source": f"reddit_{sub}",
                    "score": post.score,
                    "timestamp": datetime.fromtimestamp(post.created_utc).isoformat(),
                    "true_sentiment": None,
                })
        return results
    except Exception:
        return []


def fetch_all(use_mock=True):
    print("[FETCH] Fetching headlines...")
    all_records = []

    if use_mock:
        df = generate_mock_headlines(n=2000)
        print(f"[FETCH] Generated {len(df):,} mock headlines")
        out = RAW_DIR / "headlines.csv"
        df.to_csv(out, index=False)
        print(f"[FETCH] Saved to {out}")
        return df

    for ticker in TICKERS:
        all_records.extend(fetch_yahoo_news(ticker))
        all_records.extend(fetch_reddit_posts(ticker))

    df = pd.DataFrame(all_records).drop_duplicates(subset=["id"])
    print(f"[FETCH] Fetched {len(df):,} real headlines")
    out = RAW_DIR / "headlines.csv"
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    df = fetch_all(use_mock=True)
    print(df.head())
    print(df["ticker"].value_counts())
