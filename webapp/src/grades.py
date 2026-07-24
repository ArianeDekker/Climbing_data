"""Grade parsing and conversion.

This module preserves, without mathematical change, the two grade-conversion
functions that already exist in the source repository:

* ``yds_to_num`` -- from ``Climbing_rate_regression.py`` (Project 1). YDS only,
  ``+``/``-`` modifiers weighted +/-0.2, no bouldering support.
* ``yds_to_ordinal`` -- from ``Route_recommender.py`` (Project 2). Handles V
  grades (``V<n>`` -> ``20 + n``) and YDS, ``+``/``-`` modifiers weighted
  +/-0.15.

The two functions are intentionally *not* reconciled here: they disagree
(different modifier weights, different bouldering support) and the regression
tests pin each to its original behaviour. The web app uses ``yds_to_ordinal``
because it is the only one of the two that supports every discipline the
interface exposes; this choice is documented in the README.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Project 1 grade conversion (Climbing_rate_regression.py) -- verbatim logic
# --------------------------------------------------------------------------


def yds_to_num(r: object) -> float:
    """Convert a YDS rating string to a number (Project 1 behaviour).

    Copied verbatim from ``Climbing_rate_regression.py``. Sport-only project,
    so there is no bouldering support and ``+``/``-`` are weighted +/-0.2.
    Returns ``np.nan`` when no YDS token is found.
    """
    m = re.search(r"5\.(\d+)([abcd]?)([+-]?)", str(r))
    if not m:
        return np.nan
    base = int(m.group(1))
    letter = {"a": 0, "b": 0.25, "c": 0.5, "d": 0.75}.get(m.group(2), 0)
    modifier = {"+": 0.2, "-": -0.2}.get(m.group(3), 0)
    return base + letter + modifier


# --------------------------------------------------------------------------
# Project 2 grade conversion (Route_recommender.py) -- verbatim logic
# --------------------------------------------------------------------------


def yds_to_ordinal(rating: object) -> float:
    """Convert a YDS or V grade to a continuous ordinal (Project 2 behaviour).

    Copied verbatim from ``Route_recommender.py``. ``V<n> -> 20 + n``; YDS
    ``5.<base><letter><mod>`` with letters a/b/c/d -> 0/.25/.5/.75 and
    ``+``/``-`` -> +/-0.15. ``3rd``/``4th`` class and other non-matching
    strings return ``np.nan`` (a documented limitation of the source code).
    """
    if pd.isna(rating):
        return np.nan
    s = str(rating).strip()
    m = re.search(r"V(\d+)", s, re.I)
    if m:
        return 20 + int(m.group(1))
    m = re.search(r"5\.(\d+)([abcd]?)([+-]?)(?:\s|$|,|/)", s)
    if not m:
        m = re.search(r"5\.(\d+)([abcd]?)([+-]?)", s)
    if not m:
        return np.nan
    base = int(m.group(1))
    letter = {"a": 0, "b": 0.25, "c": 0.5, "d": 0.75}.get((m.group(2) or "").lower(), 0)
    modifier = {"+": 0.15, "-": -0.15}.get(m.group(3) or "", 0)
    return base + letter + modifier


# --------------------------------------------------------------------------
# UI helpers -- new, non-methodological. These only build the dropdown
# choices and convert them to the ordinals produced by yds_to_ordinal above.
# They do not change any scoring behaviour.
# --------------------------------------------------------------------------

#: Ordered YDS labels for roped-climbing selectors. 3rd/4th class are listed
#: because the interface spec asks for them, but note yds_to_ordinal returns
#: NaN for them -- the parser flags this rather than silently coercing.
ROPED_GRADE_LABELS = (
    ["3rd Class", "4th Class"]
    + [f"5.{n}" for n in range(0, 10)]
    + [f"5.{n}{L}" for n in range(10, 16) for L in ("a", "b", "c", "d")]
)

#: Ordered V-scale labels for bouldering selectors.
BOULDER_GRADE_LABELS = [f"V{n}" for n in range(0, 18)]

DISCIPLINE_GRADE_SYSTEM = {
    "Sport": "roped",
    "Trad": "roped",
    "Top rope": "roped",
    "TR": "roped",
    "Aid": "roped",
    "Mixed": "roped",
    "Ice": "roped",
    "Alpine": "roped",
    "Boulder": "boulder",
}


def grade_labels_for_discipline(discipline: str) -> list[str]:
    """Return the ordered dropdown labels appropriate to a discipline."""
    system = DISCIPLINE_GRADE_SYSTEM.get(discipline, "roped")
    return BOULDER_GRADE_LABELS if system == "boulder" else ROPED_GRADE_LABELS


def label_to_ordinal(label: str) -> Optional[float]:
    """Convert a dropdown label to an ordinal via the original conversion.

    Returns ``None`` when the label is not representable by the source
    ``yds_to_ordinal`` (e.g. 3rd/4th class), so callers can surface the
    limitation instead of inventing a value.
    """
    val = yds_to_ordinal(label)
    return None if pd.isna(val) else float(val)
