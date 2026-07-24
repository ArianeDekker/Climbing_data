"""Build the compact processed dataset from the raw Kaggle CSV.

Usage (from the project root):
    python scripts/build_data.py data/raw/mp_routes.csv
"""
import sys
from pathlib import Path

# Make the project root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_compact_dataset, DEFAULT_PROCESSED

if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/mp_routes.csv"
    df = build_compact_dataset(raw, DEFAULT_PROCESSED)
    print(f"Wrote {len(df)} rows -> {DEFAULT_PROCESSED}")
