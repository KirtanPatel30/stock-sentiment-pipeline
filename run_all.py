"""
run_all.py — Run the entire Stock Sentiment Pipeline end-to-end.
Usage: python run_all.py
"""

import subprocess
import sys
from pathlib import Path


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"ERROR: step failed with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("\n📈 STOCK SENTIMENT PIPELINE — FULL RUN")
    print("=" * 60)

    run("python pipeline/etl.py",    "STEP 1/3: ETL — fetch data + sentiment + features")
    run("python models/train.py",    "STEP 2/3: Train XGBoost price movement model")
    run("python -m pytest tests/ -v","STEP 3/3: Run unit tests")

    print("\n" + "="*60)
    print("  ✅ ALL STEPS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  Dashboard:  streamlit run dashboard/app.py  → http://localhost:8501")
    print("  API:        uvicorn api.main:app --reload   → http://localhost:8000/docs")
    print("=" * 60)
