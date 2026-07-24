"""Route scoring.

**Important honesty note.** The source repository scores routes with a
*personalised* dual-input neural network (``Route_recommender.py``): it takes a
specific user's tick history and predicts the probability that *that user*
likes each unseen route. It contains **no** mechanism that scores a route
against a free-text query. The requested "type a request, get matches"
interface is therefore not something the existing model can do directly.

Rather than swap in a generic recommender (which the brief forbids), this
module keeps every reusable piece of the original scoring and adds the single,
clearly-flagged bridge needed to turn a parsed query into a per-route score:

Preserved verbatim from ``Route_recommender.py``:
  * :func:`preprocess_routes_only` -- grade/stars/votes normalisation.
  * :func:`votes_confidence_multiplier` -- ``0.5 + 0.5 * votes_norm`` clip.

Documented bridge (flagged, minimal, reuses only existing feature defs):
  * A route's match score is ``match_ratio * votes_confidence_multiplier``,
    where ``match_ratio`` is the fraction of the query's requested style flags
    that the route's description contains, detected with the **existing**
    Project 1 synonym flags. Community rating (``Avg Stars``) enters exactly
    where the original used it -- as the tie-break/quality signal alongside the
    same vote-confidence multiplier. No embedding or new similarity model is
    introduced.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .features import ALL_STYLE_FLAGS, add_style_flag_columns
from .grades import yds_to_ordinal
from .query_parser import ParsedQuery

# ---------------------------------------------------------------------------
# Verbatim from Route_recommender.py
# ---------------------------------------------------------------------------


def votes_confidence_multiplier(votes_norm, floor: float = 0.5) -> np.ndarray:
    """``0.5 + 0.5 * votes_norm`` clipped to [0, 1] (verbatim)."""
    return np.clip(floor + (1.0 - floor) * np.asarray(votes_norm, dtype=np.float64), 0.0, 1.0)


def preprocess_routes_only(routes_subset: pd.DataFrame) -> pd.DataFrame:
    """Normalise grade/stars/votes exactly as the original does.

    Mirrors ``preprocess_routes_only`` in ``Route_recommender.py`` (minus the
    torch-only ``style_vec``/``type_vec`` columns, which the query matcher does
    not need). ``grade_ord``/``avg_stars``/``num_votes``/``grade_norm``/
    ``stars_norm``/``votes_norm`` are computed with identical formulas.
    """
    r = routes_subset.copy()
    r["grade_ord"] = r["Rating"].apply(yds_to_ordinal)
    r = r.dropna(subset=["grade_ord", "desc"])
    r["avg_stars"] = pd.to_numeric(r["Avg Stars"], errors="coerce").fillna(0)
    r["num_votes"] = pd.to_numeric(r["num_votes"], errors="coerce").fillna(0).clip(lower=0)
    g = r["grade_ord"]
    r["grade_norm"] = (g - g.mean()) / (g.std() + 1e-8)
    s = r["avg_stars"]
    r["stars_norm"] = (s - s.mean()) / (s.std() + 1e-8)
    v = r["num_votes"].replace(0, 1)
    r["votes_norm"] = np.log1p(r["num_votes"].clip(0)) / (np.log1p(v).max() + 1e-8)
    return r


# ---------------------------------------------------------------------------
# Documented query-match bridge
# ---------------------------------------------------------------------------


def score_routes(unseen: pd.DataFrame, query: ParsedQuery) -> pd.DataFrame:
    """Attach ``match_count``, ``match_ratio`` and ``pred_prob`` to routes.

    ``pred_prob`` is named to match the downstream area-aggregation code so
    that logic runs unchanged. It equals ``match_ratio * vote_multiplier``.

    Routes containing any *excluded* style flag are dropped. When the query
    requests no styles at all, ``match_ratio`` is 1.0 for every route (the
    request is purely discipline/grade/geography), so ranking falls back to the
    community-quality signal exactly like the original's "no likes" branch.
    """
    df = add_style_flag_columns(unseen, desc_col="desc")

    requested = [f for f in query.style_flags if f in ALL_STYLE_FLAGS]
    excluded = [f for f in query.excluded_style_flags if f in ALL_STYLE_FLAGS]

    if excluded:
        keep = ~df[excluded].any(axis=1)
        df = df[keep].copy()

    if requested:
        df["match_count"] = df[requested].sum(axis=1).astype(int)
        df["match_ratio"] = df["match_count"] / float(len(requested))
    else:
        df["match_count"] = 0
        df["match_ratio"] = 1.0

    mult = votes_confidence_multiplier(df["votes_norm"].values)
    df["pred_prob"] = df["match_ratio"].values * mult
    return df


def matched_flag_summary(row: pd.Series, requested: List[str]) -> str:
    """Short per-route explanation from the existing matched style flags."""
    if not requested:
        return "Matched on discipline / grade / location; no specific style requested."
    hits = [f for f in requested if int(row.get(f, 0)) == 1]
    stars = row.get("avg_stars", 0)
    if not hits:
        return f"Matched filters only (0 of {len(requested)} styles); community rating {stars:.1f}\u2605."
    return (
        f"Matches {len(hits)} of {len(requested)} styles ({', '.join(hits)}); "
        f"community rating {stars:.1f}\u2605."
    )
