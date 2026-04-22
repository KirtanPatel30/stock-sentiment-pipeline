"""
data/fetch_prices.py
Fetches historical stock prices via yfinance.
Falls back to realistic synthetic price data if yfinance fails.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD"]

BASE_PRICES = {
    "AAPL": 185.0, "TSLA": 245.0, "NVDA": 620.0,
    "MSFT": 415.0, "GOOGL": 165.0, "AMZN": 178.0,
    "META": 490.0, "AMD": 170.0,
}


def fetch_real_prices(ticker: str, days: int = 90) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{days}d", progress=False)
        if df.empty:
            return None
        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["ticker"] = ticker
        df.columns = ["date", "open", "high", "low", "close", "volume", "ticker"]
        return df
    except Exception:
        return None


def generate_synthetic_prices(ticker: str, days: int = 90) -> pd.DataFrame:
    """Generate realistic OHLCV price data using geometric Brownian motion."""
    np.random.seed(hash(ticker) % 2**31)
    base = BASE_PRICES.get(ticker, 100.0)
    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
    returns = np.random.normal(0.0005, 0.02, days)
    closes = [base]
    for r in returns[1:]:
        closes.append(closes[-1] * (1 + r))
    closes = np.array(closes)
    opens  = closes * np.random.uniform(0.99, 1.01, days)
    highs  = np.maximum(opens, closes) * np.random.uniform(1.0, 1.02, days)
    lows   = np.minimum(opens, closes) * np.random.uniform(0.98, 1.0, days)
    vols   = np.random.randint(20_000_000, 100_000_000, days)

    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": opens.round(2),
        "high": highs.round(2),
        "low": lows.round(2),
        "close": closes.round(2),
        "volume": vols,
        "ticker": ticker,
    })


def fetch_all_prices(days: int = 90):
    print("[PRICES] Fetching stock prices...")
    frames = []
    for ticker in TICKERS:
        df = fetch_real_prices(ticker, days)
        if df is None or len(df) < 10:
            print(f"[PRICES] {ticker}: using synthetic data")
            df = generate_synthetic_prices(ticker, days)
        else:
            print(f"[PRICES] {ticker}: fetched {len(df)} real rows")
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    out = RAW_DIR / "prices.csv"
    result.to_csv(out, index=False)
    print(f"[PRICES] Saved {len(result):,} rows to {out}")
    return result


if __name__ == "__main__":
    df = fetch_all_prices()
    print(df.tail())
