"""Streamlit interface for the Mountain Project climbing recommender.

This is a thin, production interface around the existing analysis code in
``src/``. All scoring, filtering and ranking come from functions refactored out
of the original repository; this file only handles input widgets, validation,
loading state and presentation.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.data import load_routes, location_gazetteer
from src.grades import grade_labels_for_discipline, label_to_ordinal
from src.query_parser import parse_query
from src.recommender import recommend

st.set_page_config(page_title="Climbing Route Finder", page_icon="🧗", layout="wide")

# --- lightweight styling to echo a clean portfolio look ---------------------
# Only the *main* content area gets the cream background + forced-dark text.
# The sidebar keeps Streamlit's default appearance, per user preference.
st.markdown(
    """
    <style>
      /* Main content area: cream bg, dark ink, forced through Streamlit's
         inner containers so nothing inherits the washed-out default grey. */
      section[data-testid="stMain"] { background:#faf9f6; }
      section[data-testid="stMain"], section[data-testid="stMain"] p,
      section[data-testid="stMain"] li, section[data-testid="stMain"] label,
      section[data-testid="stMain"] span,
      section[data-testid="stMain"] [data-testid="stMarkdownContainer"],
      section[data-testid="stMain"] [data-testid="stCaptionContainer"],
      section[data-testid="stMain"] [data-testid="stWidgetLabel"] {
          color:#111827 !important;
      }
      section[data-testid="stMain"] h1,
      section[data-testid="stMain"] h2,
      section[data-testid="stMain"] h3,
      section[data-testid="stMain"] h4 {
          font-family: Georgia, "Times New Roman", serif;
          color:#0f1a2b !important;
      }
      .card { background:#fff; border:1px solid #d6d2c8; border-radius:10px;
              padding:16px 18px; margin-bottom:12px; color:#111827; }
      .card b { color:#0f1a2b; }
      .score { font-weight:700; color:#1f5d3f; }
      .muted { color:#374151; font-size:0.9rem; }
      .pill { display:inline-block; background:#e7ecf3; color:#0f1a2b;
              border-radius:999px; padding:3px 11px; margin:2px 4px 2px 0;
              font-size:0.82rem; }
      section[data-testid="stMain"] a { color:#1d4ed8; }
      /* Sidebar intentionally left as Streamlit default. */
    </style>
    """,
    unsafe_allow_html=True,
)

DISCIPLINES = ["Sport", "Trad", "Top rope", "Boulder", "Aid", "Mixed", "Ice", "Alpine"]

st.title("🧗 Climbing Route Finder")
st.markdown(
    'Describe your ideal climbing route and get recommendations based on '
    '[Mountain Project analysis](https://arianedekker.github.io/blog).'
)

# Loud warning if the real dataset is missing, so the app never quietly
# serves fake demo data (which would look like real bugs: /sample URLs,
# [SAMPLE] prefixed names, only a handful of matches for any query).
_probe_df = load_routes()
if _probe_df.attrs.get("source") == "synthetic":
    st.error(
        "⚠️ **Running on demo data (10 fake routes).** "
        "The real Mountain Project dataset was not found at "
        "`data/processed/routes_compact.parquet`.\n\n"
        "To load the real 116k-route dataset, from the project folder run:\n\n"
        "```\npython scripts/build_data.py data/raw/mp_routes.csv\n```\n"
        "then restart Streamlit. Make sure `mp_routes.csv` is in `data/raw/` first."
    )

with st.sidebar:
    st.header("Filters")
    discipline = st.selectbox("Style", DISCIPLINES, index=0)
    if discipline == "Boulder":
        st.info("The Mountain Project routes dataset is roped-only — it has no standalone boulders, so this discipline returns no results.")
    labels = grade_labels_for_discipline(discipline)
    c1, c2 = st.columns(2)
    with c1:
        g_lo = st.selectbox("Min grade", labels, index=0)
    with c2:
        g_hi = st.selectbox("Max grade", labels, index=len(labels) - 1)
    pitches_choice = st.selectbox(
        "Pitches",
        ["Any", "1", "2", "3", "4", "5", "5+"],
        index=0,
    )
    p_lo = p_hi = None
    if pitches_choice in {"1", "2", "3", "4", "5"}:
        p_lo = p_hi = int(pitches_choice)
    elif pitches_choice == "5+":
        p_lo = 5
    top_n = st.slider("How many routes to show", 1, 25, 5)

query_text = st.text_input(
    "What are you looking for?",
    placeholder="e.g. vertical crimps technical no crack routes in Spain",
)
go = st.button("Search", type="primary")


def _grade_bounds(lo_label: str, hi_label: str):
    lo, hi = label_to_ordinal(lo_label), label_to_ordinal(hi_label)
    warn = None
    if lo is None or hi is None:
        warn = (
            "3rd/4th class grades are not representable by the source grade "
            "conversion, so that bound was ignored."
        )
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi, warn


if go:
    if not query_text.strip():
        st.error("Enter a description of what you want to climb.")
        st.stop()

    lo, hi, gwarn = _grade_bounds(g_lo, g_hi)
    if gwarn:
        st.warning(gwarn)

    with st.spinner("Matching routes and areas…"):
        routes_df = load_routes()
        parsed = parse_query(query_text, geo_terms=location_gazetteer())
        # discipline selector wins over any discipline word in the text
        parsed.discipline = {"Top rope": "TR"}.get(discipline, discipline)
        rec = recommend(
            routes_df, parsed,
            grade_min=lo, grade_max=hi,
            pitch_min=p_lo, pitch_max=p_hi,
            top_routes=top_n,
        )

    # --- parsed interpretation -------------------------------------------
    st.subheader("How your request was read")
    disp = parsed.as_display()
    st.markdown(
        "".join(f"<span class='pill'>{k}: {v}</span>" for k, v in disp.items() if v),
        unsafe_allow_html=True,
    )
    if parsed.unmatched:
        st.info(
            "These terms have no matching field in the source data and were "
            f"ignored: {', '.join(parsed.unmatched)}."
        )

    if rec.message and not rec.routes:
        st.warning(rec.message)
        st.stop()

    st.caption(f"{rec.n_candidates} candidate routes after filtering.")

    st.subheader(f"Top {len(rec.routes)} routes")

    # Helpful heads-up when the filters left us with genuinely few candidates.
    if rec.n_candidates < top_n:
        st.info(
            f"Only {rec.n_candidates} route(s) matched all filters. "
            "Try widening the grade range or removing the location term."
        )

    for i, r in enumerate(rec.routes, 1):
        try:
            p_val = int(float(r.get("pitches"))) if r.get("pitches") is not None else None
        except (ValueError, TypeError):
            p_val = None
        pitch_txt = ""
        if p_val == 1:
            pitch_txt = " · 1 pitch"
        elif p_val and p_val > 1:
            pitch_txt = f" · {p_val} pitches"

        url = r.get("url") or ""
        link_html = (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
            f'Mountain Project ↗</a>'
        ) if url else ""

        # Convert the 0–1 blended score to a percent for readability.
        match_pct = int(round(float(r["match_score"]) * 100))

        st.markdown(
            f"<div class='card'><b>{i}. {r['name']}</b> "
            f"<span class='muted'>{r['grade']} · {r['discipline']}{pitch_txt}</span><br>"
            f"<span class='muted'>{r['location']}</span><br>"
            f"<span class='score' title='style-match ratio × vote-confidence'>"
            f"match {match_pct}%</span> · "
            f"{r['avg_stars']:.1f}★ ({r['num_votes']} votes)<br>"
            f"<span class='muted'>{r['explanation']}</span><br>{link_html}</div>",
            unsafe_allow_html=True,
        )
