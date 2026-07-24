"""Transparent natural-language query parser.

The parser maps free text **only** onto concepts that already exist in the
repository: the Project 1 synonym dictionaries (styles), the Project 2
``ROUTE_TYPES`` (discipline), the original grade conversion (grades), and the
Location substring filter (geography). There is no embedding, no external
model, and no learned semantic matching -- every mapping is a lookup you can
read off in this file.

Anything the parser cannot map to an existing concept is returned in
``unmatched`` so the interface can tell the user exactly which words were
ignored, per the spec's transparency requirement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .features import STYLE_TO_WORDS, _word_pattern
from .grades import yds_to_ordinal
from .vocabulary import ROUTE_TYPES

# Discipline surface words -> canonical ROUTE_TYPES entry. Only disciplines the
# repo already recognises are listed.
_DISCIPLINE_WORDS: Dict[str, str] = {
    "sport": "Sport",
    "trad": "Trad",
    "traditional": "Trad",
    "boulder": "Boulder",
    "bouldering": "Boulder",
    "top rope": "TR",
    "toprope": "TR",
    "top-rope": "TR",
    "tr": "TR",
    "aid": "Aid",
    "mixed": "Mixed",
    "ice": "Ice",
    "alpine": "Alpine",
}

# Words that introduce a place; the following phrase is passed to the existing
# Location substring filter verbatim.
_GEO_TRIGGERS = ["near", "around", "close to", "in", "at"]

_EXCLUSION_TRIGGERS = ["without", "no ", "not ", "avoid", "except"]

_STOPWORDS = {
    "a", "an", "the", "with", "and", "or", "some", "climbing", "climb", "route",
    "routes", "area", "areas", "of", "for", "me", "i", "want", "looking", "find",
    "good", "nice", "please", "on", "that", "has", "have",
}


@dataclass
class ParsedQuery:
    """Structured interpretation of a user query."""

    raw: str
    discipline: Optional[str] = None
    style_flags: List[str] = field(default_factory=list)
    excluded_style_flags: List[str] = field(default_factory=list)
    geography: Optional[str] = None
    grade_min: Optional[float] = None
    grade_max: Optional[float] = None
    grade_labels: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_display(self) -> Dict[str, object]:
        """Human-readable summary for the "parsed interpretation" panel."""
        return {
            "Discipline": self.discipline or "(any / from selector)",
            "Styles": ", ".join(self.style_flags) or "(none detected)",
            "Excluded": ", ".join(self.excluded_style_flags) or "(none)",
            "Geography": self.geography or "(anywhere)",
            "Grades": self._grade_display(),
            "Ignored terms": ", ".join(self.unmatched) or "(none)",
            "Notes": " ".join(self.notes) or "",
        }

    def _grade_display(self) -> str:
        if self.grade_labels:
            return "–".join(self.grade_labels)
        if self.grade_min is not None or self.grade_max is not None:
            return f"{self.grade_min}–{self.grade_max} (ordinal)"
        return "(from selector)"


def _extract_grades(text: str) -> Tuple[Optional[float], Optional[float], List[str]]:
    """Pull explicit grade tokens (5.x / V x) and convert with yds_to_ordinal."""
    labels = re.findall(r"5\.\d+[abcd]?[+-]?|[vV]\d+", text)
    ordinals = []
    kept = []
    for lab in labels:
        o = yds_to_ordinal(lab)
        if o == o:  # not NaN
            ordinals.append(o)
            kept.append(lab)
    if not ordinals:
        return None, None, []
    return min(ordinals), max(ordinals), kept


def _extract_geography(text: str) -> Optional[str]:
    """Return the place phrase following a geo trigger, or None."""
    for trig in _GEO_TRIGGERS:
        m = re.search(rf"\b{re.escape(trig.strip())}\b\s+([a-zA-Z][a-zA-Z\s\-]*)", text)
        if m:
            phrase = m.group(1).strip()
            # keep first 1-3 words, stop at another trigger/style word
            words = phrase.split()
            cleaned = []
            for w in words[:3]:
                if w in _DISCIPLINE_WORDS or w in _STOPWORDS:
                    break
                cleaned.append(w)
            if cleaned:
                return " ".join(cleaned)
    return None


def _gazetteer_geography(text: str, geo_terms, consumed: set) -> Optional[str]:
    """Find a place phrase anywhere in the query using the data gazetteer.

    Matches 3-, then 2-, then 1-word windows against ``geo_terms`` (place
    phrases that occur in the dataset's ``Location`` field), returning the
    longest match whose words were not already consumed by a style/discipline
    term. This lets "slabby spain" resolve "spain" without an "in"/"near".
    """
    if not geo_terms:
        return None
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text)
    n = len(tokens)
    for size in (3, 2, 1):
        for i in range(n - size + 1):
            window = tokens[i:i + size]
            if set(window) & consumed:
                continue
            phrase = " ".join(window)
            if phrase in _STOPWORDS:
                continue
            if phrase in geo_terms:
                return phrase
    return None


def parse_query(text: str, geo_terms=None) -> ParsedQuery:
    """Parse a natural-language request into existing-vocabulary concepts.

    The mapping is deterministic and transparent:

    * **styles** -- a style flag is set when any of its Project 1 synonyms
      matches the query with the same ``\\bword\\b`` test used on route
      descriptions.
    * **exclusions** -- synonyms appearing after "without/no/not/avoid/except"
      set an excluded flag instead.
    * **discipline** -- matched against :data:`_DISCIPLINE_WORDS`.
    * **grades** -- explicit ``5.x`` / ``V<n>`` tokens via ``yds_to_ordinal``.
    * **geography** -- a phrase after a geo trigger ("in"/"near"/...) OR, when
      ``geo_terms`` (a gazetteer of place phrases from the data's ``Location``
      field) is supplied, any place name appearing anywhere in the query. This
      means "slabby spain" resolves "spain" without needing "in spain".
    """
    q = ParsedQuery(raw=text)
    lowered = text.lower().strip()

    # --- exclusions: capture spans introduced by an exclusion trigger -------
    excluded_region = ""
    for trig in _EXCLUSION_TRIGGERS:
        for m in re.finditer(re.escape(trig), lowered):
            tail = lowered[m.end(): m.end() + 25]
            excluded_region += " " + tail

    # --- styles (and exclusions) using Project 1 synonym flags --------------
    for style, words in STYLE_TO_WORDS.items():
        pat = _word_pattern(words)
        if re.search(pat, excluded_region):
            q.excluded_style_flags.append(style)
        elif re.search(pat, lowered):
            q.style_flags.append(style)

    # --- discipline ---------------------------------------------------------
    for word, canon in sorted(_DISCIPLINE_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            q.discipline = canon
            break

    # --- grades -------------------------------------------------------------
    q.grade_min, q.grade_max, q.grade_labels = _extract_grades(lowered)

    # words already claimed by a style or discipline, so geography can't reuse
    consumed: set = set()
    for style in q.style_flags + q.excluded_style_flags:
        for w in STYLE_TO_WORDS[style]:
            if re.search(rf"\b{w}\b", lowered):
                consumed.update(w.split())
    if q.discipline:
        consumed.update(w for w in _DISCIPLINE_WORDS if _DISCIPLINE_WORDS[w] == q.discipline)

    # --- geography: trigger phrase first, then gazetteer anywhere -----------
    q.geography = _extract_geography(lowered)
    if not q.geography:
        q.geography = _gazetteer_geography(lowered, geo_terms, consumed)

    # --- qualitative words the repo has no mapping for ----------------------
    for word in ("easy", "hard", "moderate", "beginner", "advanced"):
        if re.search(rf"\b{word}\b", lowered):
            q.notes.append(
                f'"{word}" has no grade mapping in the source code; use the grade selector.'
            )

    # --- unmatched terms ----------------------------------------------------
    matched_words = set(consumed)
    if q.geography:
        matched_words.update(q.geography.lower().split())
    matched_words.update(_GEO_TRIGGERS)
    matched_words.update(_EXCLUSION_TRIGGERS)

    for tok in re.findall(r"[a-zA-Z][a-zA-Z\-]+", lowered):
        if tok in _STOPWORDS or tok in matched_words:
            continue
        if tok in {"easy", "hard", "moderate", "beginner", "advanced"}:
            continue
        if any(tok in trig for trig in _GEO_TRIGGERS + _EXCLUSION_TRIGGERS):
            continue
        q.unmatched.append(tok)
    # de-dupe, preserve order
    q.unmatched = list(dict.fromkeys(q.unmatched))
    return q
