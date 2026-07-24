"""Regression tests: refactored functions must match the originals exactly.

The original ``Route_recommender.py`` is imported from ``reference/`` (a
byte-for-byte copy of the repo file) and used as the ground truth. Its pure
functions have no side effects at import, so they can be called directly.

``Climbing_rate_regression.py`` executes a CSV read at import time, so its pure
parts (the ``yds_to_num`` function and the synonym dictionaries) are compared
against vendored copies whose source lines are identical.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import features, grades, scoring, vocabulary

REF = Path(__file__).resolve().parents[1] / "reference" / "Route_recommender.py"


def _load_original():
    spec = importlib.util.spec_from_file_location("orig_rr", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    ORIG = _load_original()
    HAVE_TORCH = True
except Exception:  # torch not installed in this environment
    ORIG = None
    HAVE_TORCH = False

needs_orig = pytest.mark.skipif(not HAVE_TORCH, reason="original module needs torch to import")

GRADES = ["5.10a", "5.10a+", "5.10a-", "5.11d", "5.9", "V0", "V5", "V17",
          "5.15d", "5.12c/d", "3rd Class", "4th class", "", "nonsense"]


@needs_orig
@pytest.mark.parametrize("g", GRADES)
def test_yds_to_ordinal_matches_original(g):
    a = grades.yds_to_ordinal(g)
    b = ORIG.yds_to_ordinal(g)
    assert (np.isnan(a) and np.isnan(b)) or a == b


@needs_orig
@pytest.mark.parametrize("desc", [
    "steep pumpy pocketed roof", "delicate technical slab with crimps",
    "juggy overhang", "", None, "vertical face edges",
])
def test_build_multihot_styles_matches_original(desc):
    assert features.build_multihot_styles(desc) == ORIG.build_multihot_styles(desc)


@needs_orig
@pytest.mark.parametrize("rt", ["Sport", "Trad, TR", "Boulder", "Sport, Trad", None, "Ice"])
def test_build_multihot_route_types_matches_original(rt):
    assert features.build_multihot_route_types(rt) == ORIG.build_multihot_route_types(rt)


@needs_orig
def test_votes_multiplier_matches_original():
    v = [0.0, 0.1, 0.5, 0.9, 1.0]
    np.testing.assert_allclose(
        scoring.votes_confidence_multiplier(v), ORIG._votes_confidence_multiplier(v)
    )


@needs_orig
def test_style_keywords_identical():
    assert vocabulary.STYLE_KEYWORDS == ORIG.STYLE_KEYWORDS
    assert vocabulary.ROUTE_TYPES == ORIG.ROUTE_TYPES


@needs_orig
def test_preprocess_normalisation_matches_original():
    df = pd.DataFrame({
        "Route": [f"r{i}" for i in range(6)],
        "Rating": ["5.10a", "5.11b", "5.9", "V3", "5.12a", "5.10d"],
        "Avg Stars": [3.1, 2.5, 4.0, 3.8, 1.9, 3.3],
        "num_votes": [10, 5, 100, 7, 3, 50],
        "desc": ["slab"] * 6,
        "Route Type": ["Sport"] * 6,
        "Location": ["A"] * 6,
        "URL": ["u"] * 6,
    })
    ours = scoring.preprocess_routes_only(df)
    theirs = ORIG.preprocess_routes_only(df)
    for col in ["grade_ord", "grade_norm", "stars_norm", "votes_norm", "avg_stars", "num_votes"]:
        np.testing.assert_allclose(
            ours[col].to_numpy(dtype=float), theirs[col].to_numpy(dtype=float), rtol=1e-9, atol=1e-9
        )


@needs_orig
def test_area_aggregation_matches_original():
    """Our aggregate_areas must equal the original given the same pred_prob."""
    from src.recommender import aggregate_areas
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "Location": (["A"] * 5 + ["B"] * 4 + ["C"] * 3 + ["D"] * 2),
        "Route": [f"r{i}" for i in range(14)],
        "Rating": ["5.10a"] * 14,
        "pred_prob": rng.uniform(0, 1, 14),
    })
    ours = aggregate_areas(df.copy(), like_threshold=0.4, min_routes=3)
    theirs = ORIG._aggregate_areas_by_liked_count(df.copy(), like_threshold=0.4, min_routes=3)
    pd.testing.assert_frame_equal(
        ours.reset_index(drop=True), theirs.reset_index(drop=True), check_like=True
    )
