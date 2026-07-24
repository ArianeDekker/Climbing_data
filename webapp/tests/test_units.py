"""Unit tests for the app-facing behaviour."""

import numpy as np
import pandas as pd

from src import grades
from src.data import make_synthetic_sample
from src.query_parser import parse_query
from src.recommender import recommend, split_location


# --- grades ----------------------------------------------------------------

def test_grade_labels_by_discipline():
    assert grades.grade_labels_for_discipline("Boulder")[0] == "V0"
    assert grades.grade_labels_for_discipline("Boulder")[-1] == "V17"
    assert "5.15d" in grades.grade_labels_for_discipline("Sport")


def test_label_to_ordinal_flags_unsupported():
    assert grades.label_to_ordinal("3rd Class") is None
    assert grades.label_to_ordinal("5.10a") == 10.0
    assert grades.label_to_ordinal("V5") == 25.0


# --- query parsing ---------------------------------------------------------

def test_parse_styles_and_discipline():
    q = parse_query("pumpy 5.11 sport climbing with pockets")
    assert q.discipline == "Sport"
    assert "pumpy" in q.style_flags
    assert "pockets" in q.style_flags
    assert q.grade_min == q.grade_max == 11.0


def test_parse_steep_maps_to_overhang():
    # Project 1 treats "steep" as an overhang synonym.
    q = parse_query("steep limestone")
    assert "overhang" in q.style_flags


def test_parse_exclusion():
    q = parse_query("sport climbing without runout")
    assert "runout" in q.excluded_style_flags
    assert "runout" not in q.style_flags


def test_parse_geography_and_unmatched():
    q = parse_query("technical vertical limestone near Chicago")
    assert q.geography and "chicago" in q.geography.lower()
    assert "technical" in q.style_flags and "vertical" in q.style_flags
    assert "limestone" in q.unmatched  # no rock-type field in the source data


def test_parse_easy_is_flagged_not_mapped():
    q = parse_query("easy bouldering with crimps in France")
    assert q.discipline == "Boulder"
    assert "crimpy" in q.style_flags
    assert any("easy" in n for n in q.notes)


def test_geography_without_trigger_word():
    # A gazetteer of known places lets "slabby spain" resolve "spain"
    # even though there is no "in"/"near" trigger.
    gaz = frozenset({"spain", "france", "colorado"})
    q1 = parse_query("slabby spain", geo_terms=gaz)
    q2 = parse_query("slabby in spain", geo_terms=gaz)
    assert q1.geography == q2.geography == "spain"
    assert "slab" in q1.style_flags


def test_style_word_not_taken_as_place():
    # "roof" is a style; even if a place gazetteer is passed, roof/jugs must
    # stay styles and not be consumed as geography.
    gaz = frozenset({"roof", "spain"})  # adversarial: 'roof' also a "place"
    q = parse_query("roof climbing with jugs", geo_terms=gaz)
    assert "roof" in q.style_flags
    assert q.geography is None


# --- filtering + scoring ---------------------------------------------------

def test_grade_filter_applied():
    df = make_synthetic_sample()
    q = parse_query("sport climbing with pockets")
    rec = recommend(df, q, grade_min=11.0, grade_max=12.0, top_routes=5)
    for r in rec.routes:
        o = grades.yds_to_ordinal(r["grade"])
        assert 11.0 <= o <= 12.0


def test_discipline_filter_applied():
    df = make_synthetic_sample()
    q = parse_query("bouldering with crimps")
    rec = recommend(df, q, top_routes=5)
    assert all("Boulder" in (r["discipline"] or "") for r in rec.routes)


def test_pitch_filter_single_only():
    df = make_synthetic_sample()
    q = parse_query("sport climbing")
    rec = recommend(df, q, pitch_min=1, pitch_max=1, top_routes=5)
    assert all(int(r["pitches"]) == 1 for r in rec.routes)


def test_pitch_filter_multi_only_empty_on_synthetic():
    # The synthetic sample is all single-pitch, so multi-pitch should yield nothing.
    df = make_synthetic_sample()
    q = parse_query("sport climbing")
    rec = recommend(df, q, pitch_min=2, top_routes=5)
    assert rec.routes == []


def test_no_results_query_is_handled():
    df = make_synthetic_sample()
    q = parse_query("sport climbing in Antarctica")
    rec = recommend(df, q, top_routes=5)
    assert rec.routes == [] and rec.message is not None


def test_style_matches_rank_higher():
    df = make_synthetic_sample()
    q = parse_query("overhang with jugs")
    rec = recommend(df, q, top_routes=5)
    # top route should contain at least one requested flag
    assert rec.routes[0]["match_count"] >= 1


def test_area_aggregation_counts_matches():
    df = make_synthetic_sample()
    q = parse_query("pumpy pockets")
    rec = recommend(df, q, min_routes=1, top_areas=5)
    assert any(a["matching_routes"] >= 1 for a in rec.areas)


# --- location split --------------------------------------------------------

def test_split_location():
    us = split_location("El Capitan > Yosemite National Park > California")
    assert us["name"] == "El Capitan"       # most specific first
    assert us["region"] == "California"     # US state
    assert us["country"] == "USA"
    intl = split_location("Ceuse > Southern Alps > France > Europe > International")
    assert intl["name"] == "Ceuse"
    assert intl["country"] == "France"
    assert intl["region"] == "Southern Alps"
