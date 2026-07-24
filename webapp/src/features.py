"""Feature detection on route descriptions.

Both detectors that exist in the repository are preserved verbatim:

* :func:`build_multihot_styles` / :func:`build_multihot_route_types` from
  ``Route_recommender.py`` (substring matching against the flat
  ``STYLE_KEYWORDS`` / ``ROUTE_TYPES`` lists).
* :func:`style_flags_regex` reproduces the Project 1 flagging expression
  ``desc.str.contains(r"\\bword\\b")`` over the ``THEMES`` synonym
  dictionaries.

The query-to-route matcher in :mod:`src.scoring` uses the Project 1 regex
flags, because those are the only detectors in the repo that resolve synonyms
(e.g. "steep" -> overhang). No new feature is engineered here.
"""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from .vocabulary import ROUTE_TYPES, STYLE_KEYWORDS, THEMES

# ---------------------------------------------------------------------------
# Project 2 multi-hot detectors (Route_recommender.py) -- verbatim
# ---------------------------------------------------------------------------


def build_multihot_styles(desc: object) -> List[int]:
    """Substring multi-hot over ``STYLE_KEYWORDS`` (Project 2, verbatim)."""
    if pd.isna(desc):
        return [0] * len(STYLE_KEYWORDS)
    text = str(desc).lower()
    return [1 if kw in text else 0 for kw in STYLE_KEYWORDS]


def build_multihot_route_types(route_type_str: object) -> List[int]:
    """Multi-hot over ``ROUTE_TYPES`` (Project 2, verbatim)."""
    if pd.isna(route_type_str):
        return [0] * len(ROUTE_TYPES)
    parts = [p.strip() for p in str(route_type_str).split(",")]
    out = [0] * len(ROUTE_TYPES)
    for p in parts:
        for i, rt in enumerate(ROUTE_TYPES):
            if rt.lower() in p.lower() or p.lower() in rt.lower():
                out[i] = 1
                break
    return out


# ---------------------------------------------------------------------------
# Project 1 synonym flags (Climbing_rate_regression.py) -- verbatim logic
# ---------------------------------------------------------------------------

#: Flat map: canonical style flag -> list of synonym surface words. Built from
#: THEMES exactly as Project 1 flattens ``themes`` before regex flagging.
STYLE_TO_WORDS: Dict[str, List[str]] = {
    style: words
    for theme in THEMES.values()
    for style, words in theme.items()
}

ALL_STYLE_FLAGS: List[str] = list(STYLE_TO_WORDS.keys())


def _word_pattern(words: List[str]) -> str:
    """Reproduce Project 1's ``"|".join(rf"\\b{w}\\b" ...)`` join."""
    return "|".join(rf"\b{w}\b" for w in words)


def style_flags_regex(desc: object) -> Dict[str, int]:
    """Binary style flags for one description (Project 1 regex, verbatim).

    Returns a dict ``{style_flag: 0/1}`` using the same
    word-boundary ``str.contains`` test Project 1 applied per style.
    """
    text = "" if pd.isna(desc) else str(desc).lower()
    flags: Dict[str, int] = {}
    for style, words in STYLE_TO_WORDS.items():
        flags[style] = int(bool(re.search(_word_pattern(words), text)))
    return flags


def add_style_flag_columns(df: pd.DataFrame, desc_col: str = "desc") -> pd.DataFrame:
    """Vectorised equivalent of the Project 1 per-style flagging loop.

    Adds one 0/1 column per style flag, using ``Series.str.contains`` with the
    same joined ``\\bword\\b`` pattern, matching Project 1 line-for-line.
    """
    out = df.copy()
    desc = out[desc_col].astype("string").str.lower()
    for style, words in STYLE_TO_WORDS.items():
        out[style] = desc.str.contains(_word_pattern(words), regex=True).fillna(False).astype(int)
    return out
