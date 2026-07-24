"""Data loading, caching, and compact-dataset preparation.

The full Kaggle export (~116k rows) is never loaded at query time. Instead a
compact Parquet with only the columns the recommender touches is built once
from the raw CSV and cached; the app loads that.

Because the raw Kaggle dataset is not redistributable and requires a Kaggle
login, this module also ships a small **synthetic** sample so the app and
tests run end-to-end offline. The synthetic rows are clearly labelled and must
be replaced with the real export for real recommendations.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Columns the recommender actually reads.
KEEP_COLUMNS = ["Route", "Location", "URL", "Avg Stars", "Route Type", "Rating", "desc", "num_votes", "Pitches"]

DEFAULT_RAW = "data/raw/mp_routes.csv"
DEFAULT_PROCESSED = "data/processed/routes_compact.parquet"


def build_compact_dataset(raw_path: str = DEFAULT_RAW, out_path: str = DEFAULT_PROCESSED) -> pd.DataFrame:
    """Read the raw Kaggle CSV, keep only needed columns, write Parquet.

    Row selection is deliberately conservative -- it drops rows the original
    scoring would drop anyway (missing description or grade) but does not
    otherwise filter, so results are unaffected.
    """
    df = pd.read_csv(raw_path, index_col=0)
    df.columns = df.columns.str.strip()
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols].copy()
    df = df.dropna(subset=[c for c in ["desc", "Rating"] if c in df.columns])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


@functools.lru_cache(maxsize=1)
def load_routes(processed_path: str = DEFAULT_PROCESSED) -> pd.DataFrame:
    """Load the compact dataset, falling back to the synthetic sample.

    Searches a few sensible locations so the file is found regardless of
    where Streamlit was launched from:
      1. ``processed_path`` as given (relative to the current directory).
      2. Same, resolved against the project root (parent of ``src/``).
    The DataFrame carries a ``.attrs['source']`` marker so the UI can show a
    prominent warning if it fell back to the demo sample.
    """
    project_root = Path(__file__).resolve().parents[1]
    candidates = [Path(processed_path), project_root / processed_path]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            df.attrs["source"] = f"real:{p}"
            return df
    df = make_synthetic_sample()
    df.attrs["source"] = "synthetic"
    return df


def make_synthetic_sample(seed: int = 0) -> pd.DataFrame:
    """Return a small, clearly-fake dataset for offline demos and tests."""
    rng = np.random.default_rng(seed)
    templates = [
        ("Slab Dancer", "Sport", "5.10a", "Delicate technical slab with tiny crimps and precise footwork.",
         "Siurana > Catalonia > Spain > Europe > International"),
        ("Pocket Rocket", "Sport", "5.11c", "Pumpy overhanging limestone with pockets and a mono crux.",
         "Margalef > Catalonia > Spain > Europe > International"),
        ("Roof Warrior", "Sport", "5.12a", "Steep roof pull on big jugs, powerful and athletic.",
         "Ceuse > Southern Alps > France > Europe > International"),
        ("Vertical Limit", "Sport", "5.10d", "Dead vertical face climbing on small edges, sustained.",
         "Volx > Provence > France > Europe > International"),
        ("Crimp City", "Sport", "5.11a", "Sharp crimps on a vertical wall, very technical and balancy.",
         "Montsant > Catalonia > Spain > Europe > International"),
        ("Jug Haul", "Sport", "5.9", "Juggy overhang with huge buckets, endurance fest.",
         "Ceuse > Southern Alps > France > Europe > International"),
        ("Boulder Problem A", "Boulder", "V4", "Crimpy technical slab, delicate and precise.",
         "Fontainebleau > Ile-de-France > France > Europe > International"),
        ("Boulder Problem B", "Boulder", "V2", "Juggy overhanging roof, powerful and dynamic.",
         "Fontainebleau > Ile-de-France > France > Europe > International"),
        ("Chicago Choss", "Sport", "5.8", "Vertical face with edges near the city, technical.",
         "Devils Lake > Chicago Area > Illinois"),
        ("Pumpfest", "Sport", "5.11d", "Sustained pumpy overhang with pockets, power endurance.",
         "Siurana > Catalonia > Spain > Europe > International"),
    ]
    rows = []
    for i, (name, rtype, grade, desc, loc) in enumerate(templates):
        rows.append({
            "Route": f"[SAMPLE] {name}",
            "Location": loc,
            "URL": f"https://www.mountainproject.com/route/{100000 + i}/sample",
            "Avg Stars": round(float(rng.uniform(2.5, 4.0)), 1),
            "Route Type": rtype,
            "Rating": grade,
            "desc": desc,
            "num_votes": int(rng.integers(5, 300)),
            "Pitches": 1,
        })
    return pd.DataFrame(rows)


def clear_cache() -> None:
    """Clear the cached dataset (useful after rebuilding the Parquet)."""
    load_routes.cache_clear()
    location_gazetteer.cache_clear()


@functools.lru_cache(maxsize=1)
def location_gazetteer(min_count: int = 3) -> frozenset:
    """Set of place phrases that actually occur in the data's ``Location`` field.

    Every ">"-delimited level of every ``Location`` breadcrumb is a candidate
    place phrase (country, state, region, crag, ...). Phrases are kept only if
    they appear in at least ``min_count`` distinct locations, which drops
    one-off crag-name oddities while keeping real regions like "spain" or
    "red rocks". This is a pure lookup over existing data -- no external model.
    """
    df = load_routes()
    counts: dict[str, int] = {}
    for loc in df["Location"].dropna().astype(str).unique():
        for part in loc.split(">"):
            token = part.strip().lower()
            if len(token) >= 3:
                counts[token] = counts.get(token, 0) + 1
    return frozenset(t for t, c in counts.items() if c >= min_count)
