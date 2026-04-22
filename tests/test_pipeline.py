"""tests/test_pipeline.py — Unit tests for Stock Sentiment Pipeline"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataFetch:
    def test_mock_headlines_shape(self):
        from data.fetch_news import generate_mock_headlines
        df = generate_mock_headlines(n=100)
        assert len(df) == 100
        assert "ticker" in df.columns
        assert "text" in df.columns
        assert "true_sentiment" in df.columns

    def test_sentiment_labels_valid(self):
        from data.fetch_news import generate_mock_headlines
        df = generate_mock_headlines(n=200)
        assert set(df["true_sentiment"].unique()).issubset({"bullish","bearish","neutral"})

    def test_synthetic_prices_shape(self):
        from data.fetch_prices import generate_synthetic_prices
        df = generate_synthetic_prices("AAPL", days=30)
        assert len(df) == 30
        assert "close" in df.columns
        assert "ticker" in df.columns
        assert (df["close"] > 0).all()

    def test_prices_ohlcv_valid(self):
        from data.fetch_prices import generate_synthetic_prices
        df = generate_synthetic_prices("TSLA", days=60)
        assert (df["high"] >= df["low"]).all()
        assert (df["volume"] > 0).all()


class TestSentiment:
    def test_rule_based_bullish(self):
        from pipeline.sentiment import rule_based_sentiment
        label, score = rule_based_sentiment("This stock is mooning, bought calls, very bullish!")
        assert label == "bullish"
        assert score > 0.5

    def test_rule_based_bearish(self):
        from pipeline.sentiment import rule_based_sentiment
        label, score = rule_based_sentiment("Crash incoming, bought puts, shorting now")
        assert label == "bearish"
        assert score > 0.5

    def test_batch_scoring_length(self):
        from pipeline.sentiment import score_batch
        texts = ["stock going up", "stock going down", "watching market"] * 10
        results = score_batch(texts, nlp=None)
        assert len(results) == 30

    def test_score_range(self):
        from pipeline.sentiment import rule_based_sentiment
        for text in ["buy buy buy","sell everything","maybe hold"]:
            label, score = rule_based_sentiment(text)
            assert 0.0 <= score <= 1.0
            assert label in {"bullish","bearish","neutral"}


class TestFeatures:
    def setup_method(self):
        from data.fetch_news   import generate_mock_headlines, save_raw_data
        from data.fetch_prices import generate_synthetic_prices
        import tempfile, os
        self.tmp = Path(tempfile.mkdtemp())
        raw = self.tmp / "raw"
        raw.mkdir()
        df_h = generate_mock_headlines(n=500)
        df_h.to_csv(raw / "headlines.csv", index=False)
        frames = [generate_synthetic_prices(t, 60) for t in ["AAPL","TSLA","NVDA"]]
        pd.concat(frames).to_csv(raw / "prices.csv", index=False)
        self.raw = raw

    def test_rsi_range(self):
        from data.fetch_prices import generate_synthetic_prices
        df = generate_synthetic_prices("AAPL", 100)
        assert (df["close"] > 0).all()

    def test_price_range_positive(self):
        from data.fetch_prices import generate_synthetic_prices
        df = generate_synthetic_prices("NVDA", 60)
        price_range = (df["high"] - df["low"]) / df["close"]
        assert (price_range >= 0).all()


class TestAPI:
    def test_prediction_input_valid(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from api.main import PredictionInput
        inp = PredictionInput(
            ticker="AAPL", daily_return=0.01, price_range=0.02,
            rsi=55.0, avg_sentiment_score=0.7, sentiment_ratio=0.3,
            total_posts=50, volume_change=0.05,
            price_vs_ma5=0.01, price_vs_ma20=-0.005
        )
        assert inp.ticker == "AAPL"
        assert 0 <= inp.avg_sentiment_score <= 1

    def test_prediction_fields(self):
        from api.main import PredictionInput
        inp = PredictionInput(
            ticker="TSLA", daily_return=-0.05, price_range=0.04,
            rsi=28.0, avg_sentiment_score=0.2, sentiment_ratio=-0.4,
            total_posts=120, volume_change=0.3,
            price_vs_ma5=-0.03, price_vs_ma20=-0.08
        )
        assert inp.rsi == 28.0
        assert inp.sentiment_ratio == -0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
