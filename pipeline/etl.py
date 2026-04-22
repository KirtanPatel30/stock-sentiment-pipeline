"""
pipeline/etl.py
Full ETL orchestrator: fetch → sentiment → features → load DB
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sentiment_user:sentiment_pass@localhost:5432/stock_sentiment")


def run_etl():
    print("=" * 60)
    print("STOCK SENTIMENT PIPELINE — ETL")
    print("=" * 60)

    from data.fetch_news   import fetch_all
    from data.fetch_prices import fetch_all_prices
    from pipeline.sentiment import run_sentiment_pipeline
    from pipeline.features  import engineer_features

    print("\n[1/4] Fetching headlines...")
    fetch_all(use_mock=True)

    print("\n[2/4] Fetching stock prices...")
    fetch_all_prices(days=90)

    print("\n[3/4] Running sentiment analysis...")
    run_sentiment_pipeline()

    print("\n[4/4] Engineering features...")
    df = engineer_features()

    # Load to PostgreSQL if available
    try:
        engine = create_engine(DATABASE_URL)
        df.to_sql("sentiment_features", engine, schema="sentiment",
                  if_exists="replace", index=False, chunksize=5000)
        print("[ETL] Loaded to PostgreSQL")
    except Exception as e:
        print(f"[ETL] PostgreSQL unavailable ({e}). CSV saved.")

    print("\n[ETL] Complete!")
    return df


if __name__ == "__main__":
    run_etl()
