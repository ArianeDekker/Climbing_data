"""Climbing vocabulary, copied verbatim from the source repository.

Two independent vocabularies exist in the repo and they do **not** agree.
Both are preserved here unchanged so nothing is silently invented:

1. ``THEMES`` -- the rich synonym dictionaries from
   ``Climbing_rate_regression.py`` (Project 1), grouped into angle / feature /
   hold / movement. These map many surface words onto a smaller set of
   canonical style flags and are what the query parser uses, because they are
   the only place in the repo that defines climbing *synonyms*.

2. ``STYLE_KEYWORDS`` -- the flat 27-word list from ``Route_recommender.py``
   (Project 2) used to build the model's multi-hot style vector via substring
   matching.

Known inconsistency (documented, not fixed): Project 1 treats "steep" as a
synonym for *overhang*, while Project 2's flat list has no "steep" entry, so
the model never sees it. The regression tests pin both behaviours.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Project 1 dictionaries (Climbing_rate_regression.py) -- verbatim
# ---------------------------------------------------------------------------

ANGLE_DICT = {
    "slab": [
        "slab", "slabby", "slabbing", "low angle", "low angled",
        "slab face", "vertical slab", "technical slab", "slab crux",
    ],
    "vertical": [
        "vertical", "vertical face", "vertical section",
        "vertical terrain", "dead vertical",
    ],
    "overhang": [
        "overhang", "overhanging", "overhung",
        "steep", "steeper", "steep face",
        "steep section", "steep roof", "overhanging face",
        "overhanging section",
    ],
    "roof": [
        "roof", "roofs", "roof crux", "roof pull",
        "pull roof", "roof section", "lip", "ceiling",
    ],
}

FEATURE_DICT = {
    "crack": [
        "crack", "cracks", "hand crack", "finger crack",
        "wide crack", "offwidth", "crack corner",
    ],
    "arete": ["arete", "aretes", "arete crux", "rounded arete"],
    "dihedral": [
        "corner", "corners", "dihedral", "dihedrals",
        "shallow dihedral", "facing dihedral",
    ],
    "flake": ["flake", "flakes", "large flake", "detached flake"],
    "chimney": ["chimney"],
}

HOLD_DICT = {
    "crimpy": ["crimp", "crimps", "crimpy", "small crimps", "tiny crimps", "sharp crimps"],
    "juggy": ["jug", "jugs", "juggy", "big jugs", "huge jugs", "bucket", "buckets"],
    "pockets": ["pocket", "pockets", "pocketed", "finger pocket", "mono"],
    "slopers": ["sloper", "slopers", "slopey", "sloping", "sloping holds"],
    "edges": ["edge", "edges", "small edges", "positive edges"],
    "undercling": ["undercling", "underclings"],
}

MOVEMENT_DICT = {
    "technical": ["technical", "techy", "delicate", "precise", "balance", "balancy", "footwork"],
    "powerful": ["powerful", "burly", "bouldery", "dynamic", "athletic"],
    "pumpy": ["pumpy", "endurance", "sustained", "power endurance", "fight pump"],
    "reachy": ["reachy", "long reach", "big reach", "long reaches"],
    "runout": ["runout", "runouts", "spaced bolts"],
}

#: Mirrors the ``themes`` dict in Project 1.
THEMES = {
    "angle": ANGLE_DICT,
    "feature": FEATURE_DICT,
    "hold": HOLD_DICT,
    "movement": MOVEMENT_DICT,
}

# ---------------------------------------------------------------------------
# Project 2 flat list (Route_recommender.py) -- verbatim
# ---------------------------------------------------------------------------

STYLE_KEYWORDS = [
    # angle / wall shape
    "slab", "vertical", "overhang", "roof", "face",
    # features
    "crack", "arete", "corner", "dihedral", "chimney", "flake",
    # holds
    "crimp", "jug", "pocket", "mono", "sloper", "undercling",
    # movement / feel
    "technical", "delicate", "balanc", "precise", "pump", "sustained",
    "boulder", "powerful", "reach", "runout",
]

#: Canonical route-type categories, verbatim from Project 2.
ROUTE_TYPES = ["Sport", "Trad", "Boulder", "Aid", "Mixed", "Ice", "Alpine", "TR"]
