"""End-to-end query -> route/area recommendation.

Filtering and area aggregation are lifted directly from
``Route_recommender.py``:

* Grade / route-type / location filters reproduce the filter block inside
  ``generate_recommendations`` line-for-line.
* :func:`aggregate_areas` reproduces ``_aggregate_areas_by_liked_count``
  verbatim -- including ``score = liked_count * liked_ratio``, the
  ``total_routes >= min_routes`` gate, and the ``mean_prob`` fallback branch.

The only substitution is *what counts as a "liked" route*: the original uses
``pred_prob >= like_threshold`` from the neural net; here ``pred_prob`` is the
query-match score from :mod:`src.scoring`, so an area's ``liked_count`` is its
number of strong query matches. The aggregation arithmetic is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .query_parser import ParsedQuery
from .scoring import matched_flag_summary, preprocess_routes_only, score_routes
from .vocabulary import ROUTE_TYPES


# ---------------------------------------------------------------------------
# Verbatim helpers from Route_recommender.py
# ---------------------------------------------------------------------------


def route_matches_type_preference(route_type_str, preferred_types) -> bool:
    """Verbatim ``_route_matches_type_preference``."""
    if not preferred_types:
        return True
    if pd.isna(route_type_str):
        return False
    route_types = {p.strip() for p in str(route_type_str).split(",") if p.strip()}
    return bool(route_types & preferred_types)


def aggregate_areas(unseen: pd.DataFrame, like_threshold: float = 0.4, min_routes: int = 3) -> pd.DataFrame:
    """Verbatim ``_aggregate_areas_by_liked_count`` (score = liked_count * liked_ratio)."""
    unseen = unseen.copy()
    unseen["predicted_like"] = unseen["pred_prob"] >= like_threshold

    liked_count = unseen.groupby("Location")["predicted_like"].sum().astype(int)
    total_routes = unseen.groupby("Location").size()
    mean_prob = unseen.groupby("Location")["pred_prob"].mean()
    grouped = pd.DataFrame(
        {"liked_count": liked_count, "total_routes": total_routes, "mean_prob": mean_prob}
    ).reset_index()
    grouped["liked_ratio"] = grouped["liked_count"] / grouped["total_routes"].replace(0, np.nan)

    grouped = grouped[grouped["total_routes"] >= min_routes]
    if len(grouped) == 0:
        return grouped

    has_any_liked = grouped["liked_count"].max() >= 1
    if has_any_liked:
        grouped = grouped[grouped["liked_count"] >= 1]
        grouped["score"] = grouped["liked_count"] * grouped["liked_ratio"]
        sort_col = "score"
    else:
        grouped["score"] = grouped["mean_prob"]
        sort_col = "mean_prob"

    best_routes = (
        unseen.sort_values("pred_prob", ascending=False)
        .groupby("Location")
        .first()
        .reset_index()[["Location", "Route", "Rating", "pred_prob"]]
    )
    grouped = grouped.merge(best_routes, on="Location", how="left")
    return grouped.sort_values(sort_col, ascending=False)


# ---------------------------------------------------------------------------
# Location display helper (presentation only, not scoring)
# ---------------------------------------------------------------------------


def split_location(location: object) -> Dict[str, Optional[str]]:
    """Split a Mountain Project ``Location`` (">"-delimited) for display.

    Mountain Project stores geography as a single hierarchical string; there
    are no separate country/region columns in the dataset. This helper only
    parses that string for the UI and does not affect any score.
    """
    if pd.isna(location):
        return {"country": None, "region": None, "subregion": None, "name": None, "full": None}
    parts = [p.strip() for p in str(location).split(">") if p.strip()]
    # Mountain Project stores geography as a single broadest-LAST breadcrumb:
    #   international: "<crag> > ... > <country> > <continent> > International"
    #   US:           "<crag> > ... > <state>"
    # There are no separate political columns, so these fields are best-effort
    # and the full breadcrumb is always shown alongside them.
    name = parts[0] if parts else None
    rev = parts[::-1]
    if rev and rev[0].lower() == "international":
        # rev = [International, continent, country, region, subregion, ...]
        country = rev[2] if len(rev) > 2 else (rev[1] if len(rev) > 1 else None)
        region = rev[3] if len(rev) > 3 else None
        subregion = rev[4] if len(rev) > 4 else None
    else:
        country = "USA"
        region = rev[0] if rev else None            # US state
        subregion = rev[1] if len(rev) > 1 else None
    return {"country": country, "region": region, "subregion": subregion, "name": name, "full": str(location)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class Recommendations:
    parsed: ParsedQuery
    routes: List[dict]
    areas: List[dict]
    n_candidates: int
    message: Optional[str] = None


def recommend(
    routes_df: pd.DataFrame,
    query: ParsedQuery,
    grade_min: Optional[float] = None,
    grade_max: Optional[float] = None,
    pitch_min: Optional[int] = None,
    pitch_max: Optional[int] = None,
    top_routes: int = 5,
    top_areas: int = 5,
    like_threshold: float = 1.0,
    min_routes: int = 3,
) -> Recommendations:
    """Run the full pipeline for a parsed query.

    ``like_threshold`` defaults to 1.0 so that, for a style query, an area's
    ``liked_count`` counts routes matching *all* requested style flags -- the
    strictest reading of "matches the request". Lower it to relax.

    ``pitch_min``/``pitch_max`` filter on the ``Pitches`` column using the
    exact ``pd.to_numeric(..., errors='coerce')`` comparison used by the
    original ``generate_recommendations``.
    """
    unseen = preprocess_routes_only(routes_df)
    if len(unseen) == 0:
        return Recommendations(query, [], [], 0, "No routes with usable grade/description.")

    # --- filters: identical to generate_recommendations -------------------
    if grade_min is not None:
        unseen = unseen[unseen["grade_ord"] >= grade_min]
    if grade_max is not None:
        unseen = unseen[unseen["grade_ord"] <= grade_max]

    disc = query.discipline
    if disc:
        preferred = {disc}
        mask = unseen["Route Type"].apply(lambda x: route_matches_type_preference(x, preferred))
        unseen = unseen[mask]

    if pitch_min is not None:
        pts = pd.to_numeric(unseen["Pitches"], errors="coerce")
        unseen = unseen[pts >= pitch_min]
    if pitch_max is not None:
        pts = pd.to_numeric(unseen["Pitches"], errors="coerce")
        unseen = unseen[pts <= pitch_max]

    if query.geography:
        unseen = unseen[unseen["Location"].astype(str).str.contains(query.geography, case=False, na=False)]

    if len(unseen) == 0:
        return Recommendations(query, [], [], 0, "No routes matched the filters (discipline / grade / location).")

    scored = score_routes(unseen, query)
    if len(scored) == 0:
        return Recommendations(query, [], [], 0, "All candidate routes were excluded by your negative terms.")

    requested = [f for f in query.style_flags]

    # --- routes -----------------------------------------------------------
    route_rows = scored.sort_values(["pred_prob", "avg_stars"], ascending=False).head(top_routes)
    routes = []
    for _, row in route_rows.iterrows():
        loc = split_location(row.get("Location"))
        routes.append({
            "name": row.get("Route"),
            "discipline": row.get("Route Type"),
            "grade": row.get("Rating"),
            "pitches": row.get("Pitches"),
            "location": loc["full"],
            "country": loc["country"],
            "region": loc["region"],
            "avg_stars": float(row.get("avg_stars", 0)),
            "num_votes": int(row.get("num_votes", 0)),
            "match_score": round(float(row.get("pred_prob", 0)), 4),
            "match_count": int(row.get("match_count", 0)),
            "explanation": matched_flag_summary(row, requested),
            "url": row.get("URL"),
        })

    # --- areas: aggregation formula unchanged -----------------------------
    # The verbatim aggregator reads a "pred_prob" column and counts routes with
    # pred_prob >= like_threshold as "liked". We feed it match_ratio there so an
    # area's liked_count is its number of routes matching the requested styles
    # (threshold 1.0 == matches all requested flags). The route *list* above is
    # still ranked by the vote-adjusted score, exactly as the original ranked by
    # the vote-adjusted model probability.
    agg_input = scored.copy()
    agg_input["pred_prob"] = scored["match_ratio"].values
    grouped = aggregate_areas(agg_input, like_threshold=like_threshold, min_routes=min_routes)
    areas = []
    for _, row in grouped.head(top_areas).iterrows():
        loc = split_location(row.get("Location"))
        areas.append({
            "name": loc["name"] or row.get("Location"),
            "country": loc["country"],
            "region": loc["region"],
            "subregion": loc["subregion"],
            "location": loc["full"],
            "score": round(float(row.get("score", 0)), 4),
            "matching_routes": int(row.get("liked_count", 0)),
            "total_routes": int(row.get("total_routes", 0)),
            "mean_score": round(float(row.get("mean_prob", 0)), 4),
            "top_route": row.get("Route"),
            "top_route_grade": row.get("Rating"),
            "explanation": _area_explanation(row, requested),
        })

    msg = None
    if not areas:
        msg = f"No area had at least {min_routes} candidate routes after filtering."
    return Recommendations(query, routes, areas, len(scored), msg)


def _area_explanation(row: pd.Series, requested: List[str]) -> str:
    lc = int(row.get("liked_count", 0))
    tr = int(row.get("total_routes", 0))
    if requested:
        return f"{lc} of {tr} candidate routes match all of: {', '.join(requested)}."
    return f"{tr} candidate routes in range; ranked by mean community-quality score."
