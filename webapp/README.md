# Climbing Route Finder

A production web interface around the existing Mountain Project analysis in
[`ArianeDekker/Climbing_data`](https://github.com/ArianeDekker/Climbing_data).
You pick a discipline and grade range, type a request in plain English, and get
the top five matching **areas** and top five matching **routes**.

Every scoring, filtering and ranking decision is reused from the original two
scripts. Nothing about the recommendation math was replaced with a generic
recommender. Where the original code could **not** support part of the
requested interface, that gap is called out below rather than papered over.

---

## 1. What was inspected

The repository contains exactly two scripts and a README — no saved models, no
processed data, no notebooks:

| File | What it is |
|------|------------|
| `Climbing_rate_regression.py` | **Explanatory** OLS regression of `Avg Stars` on grade, log-votes and style flags. Sport-only. Produces coefficients, *not* recommendations. |
| `Route_recommender.py` | A **personalised** recommender: a dual-input neural net (LSTM on descriptions + MLP on structured features) that predicts whether *a specific user* will like each unseen route, trained on that user's `ticks.csv`. |

## 2. The central limitation (read this first)

**The repository has no query-driven recommender.** `Route_recommender.py`
scores routes by *predicted personal preference* from a trained network that
needs (a) the full Kaggle dataset and (b) the user's private tick history —
neither is in the repo. It never scores a route against a text request like
"slab and overhang in Europe."

So the requested "type a request → get matches" interface is a **different
task** from what the code does. Rather than substitute a generic/embedding
recommender (explicitly forbidden), this project:

* reuses every reusable piece unchanged (grade conversion, style dictionaries,
  vote handling, area-aggregation formula), and
* adds **one clearly-flagged bridge** to turn a parsed query into a per-route
  score, built only from the existing feature definitions.

The bridge is documented in §4. If you would rather drive the app from the
personal neural network instead, that requires the Kaggle export + your
`ticks.csv` and a training run; the wiring points are noted in §7.

## 3. What is reused, and from where

| Concern | Source | Where it now lives | Changed? |
|--------|--------|--------------------|----------|
| YDS→number (Sport project) | `Climbing_rate_regression.yds_to_num` | `src/grades.py` | verbatim |
| YDS/V→ordinal | `Route_recommender.yds_to_ordinal` | `src/grades.py` | verbatim |
| Style synonym dictionaries (angle/feature/hold/movement) | `Climbing_rate_regression` | `src/vocabulary.py` | verbatim |
| Flat `STYLE_KEYWORDS`, `ROUTE_TYPES` | `Route_recommender` | `src/vocabulary.py` | verbatim |
| Description style flags (`\bword\b`) | `Climbing_rate_regression` | `src/features.py::style_flags_regex` | verbatim logic |
| `build_multihot_styles/_route_types` | `Route_recommender` | `src/features.py` | verbatim |
| Grade/stars/votes normalisation | `Route_recommender.preprocess_routes_only` | `src/scoring.py` | verbatim |
| Vote-confidence multiplier | `Route_recommender._votes_confidence_multiplier` | `src/scoring.py` | verbatim |
| Grade / route-type / location filters | `Route_recommender.generate_recommendations` | `src/recommender.py::recommend` | verbatim |
| Area aggregation | `Route_recommender._aggregate_areas_by_liked_count` | `src/recommender.py::aggregate_areas` | verbatim |

## 4. Exact scoring / ranking logic

**Route type match** (verbatim): a route matches a discipline iff
`{types on the route} ∩ {preferred} ≠ ∅`, splitting `Route Type` on commas.

**Vote confidence** (verbatim):
```
votes_norm      = log1p(num_votes) / max(log1p(num_votes))
vote_multiplier = clip(0.5 + 0.5 * votes_norm, 0, 1)
```

**Query→route match (the one bridge, `src/scoring.py`):**
```
requested_flags = styles parsed from the query (via the Project-1 synonym dicts)
match_count     = number of requested_flags whose synonyms appear in the route
                  description, using the SAME \bword\b test Project 1 used
match_ratio     = match_count / len(requested_flags)     # 1.0 if no styles asked
pred_prob       = match_ratio * vote_multiplier          # used to RANK routes
```
Routes containing any *excluded* flag (from "without/no/not …") are dropped.
Routes are ranked by `(pred_prob, avg_stars)` descending — community rating
(`Avg Stars`) enters as the quality tie-break, alongside the same vote
multiplier the original applied to its model probability.

**Area ranking (verbatim formula, `aggregate_areas`):** per `Location`,
```
liked_count = #routes matching all requested flags   (match_ratio >= like_threshold, default 1.0)
total_routes= #candidate routes
liked_ratio = liked_count / total_routes
keep areas with total_routes >= min_routes (default 3)
if any area has liked_count >= 1:  score = liked_count * liked_ratio
else:                              score = mean(match_ratio)
sort by score descending
```
This is `_aggregate_areas_by_liked_count` unchanged; only the definition of
"liked" moved from *NN probability ≥ threshold* to *query match ≥ threshold*.

## 5. Methodological inconsistencies & leakage found (documented, not silently fixed)

1. **Two disagreeing grade scales.** `yds_to_num` weights `+/-` as ±0.2 and has
   no bouldering; `yds_to_ordinal` uses ±0.15 and adds `V<n> = 20 + n`. The app
   uses `yds_to_ordinal` (only one covering all disciplines). Both preserved.
2. **Two disagreeing style vocabularies.** Project 1's dictionaries treat
   "steep" as *overhang*; Project 2's flat list omits "steep" entirely. The
   parser uses the richer Project 1 dicts. Regression tests pin both.
3. **`3rd`/`4th class` are unsupported** by `yds_to_ordinal` (returns NaN). The
   selectors expose them per the brief, but the app surfaces a warning instead
   of coercing a value.
4. **No structured geography.** The dataset has a single `Location` breadcrumb
   (broadest-first, e.g. `California > Yosemite > El Capitan`); there are no
   country/region columns and US routes have a state, not "USA", at the top.
   Geography filtering is the original `Location.str.contains(...)`, so a
   continent term like "Europe" only matches if it literally appears in the
   path. `split_location` parses the breadcrumb for display only.
5. **Train/serve normalisation skew (in the original).** `preprocess_routes_only`
   recomputes `grade_norm`/`stars_norm` from each *subset's* mean/std, so the
   stats at recommendation time differ from training. Preserved as-is; flagged.
6. **Mild target leakage in the original global pretraining.** The global label
   is `avg_stars >= 3.5` while `stars_norm` (a function of `avg_stars`) is fed
   as an input feature — the label is partly encoded in an input. Not used by
   this app's query path, but noted for the model path.

## 6. Architecture

GitHub Pages serves static files only and **cannot run a Python backend**, so
the recommender cannot live on your existing Pages site. Options considered:

| Option | Verdict |
|--------|---------|
| **Streamlit Community Cloud** | **Recommended.** Free, Python-native, one file, no separate API. |
| Static Pages + separate Python API (e.g. Render/Fly) | More moving parts, two deploys, CORS. Overkill here. |
| Gradio on HF Spaces | Fine alternative; Streamlit fits the tabular UI better. |
| Fully in-browser (Pyodide) | Possible only with a tiny dataset; pandas+parquet in-browser is fragile. |

**Chosen:** keep `arianedekker.github.io/blog` as the portfolio home and add a
link/button to a Streamlit app on Streamlit Community Cloud (see §8).

## 7. Project layout

```
project/
├── app.py                # Streamlit UI (presentation only)
├── src/
│   ├── data.py           # cached loader, compact-dataset builder, synthetic sample
│   ├── grades.py         # both original grade conversions + selector helpers
│   ├── vocabulary.py     # both original style vocabularies + ROUTE_TYPES
│   ├── features.py       # original style detectors (regex + multi-hot)
│   ├── query_parser.py   # text → existing concepts, with unmatched-term reporting
│   ├── scoring.py        # original normalisation/vote logic + the documented bridge
│   └── recommender.py    # original filters + verbatim area aggregation
├── data/
│   ├── raw/              # put mp_routes.csv here (gitignored)
│   └── processed/        # routes_compact.parquet built here (gitignored)
├── scripts/build_data.py # raw CSV → compact parquet
├── tests/                # regression + unit tests
├── reference/            # byte-for-byte copies of the two original scripts (test ground truth)
├── requirements.txt
└── .github/workflows/tests.yml
```

To drive the app from the **personal NN** instead of the query bridge: train
via the original `main()` (needs `data/raw/mp_routes.csv` + `MP_personal/ticks.csv`),
cache `route_recommender_global.pt` + vocab, then swap `score_routes`'s
`pred_prob` for the model's `sigmoid(logit) * vote_multiplier`. The area code
already consumes a `pred_prob` column and needs no change.

## 8. Setup, run, deploy

### Local
```bash
git clone https://github.com/ArianeDekker/Climbing_data.git
cd Climbing_data            # (or wherever this project/ lives)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional but recommended: build the compact dataset from the Kaggle CSV.
# Download mp_routes.csv from the Kaggle dataset first, put it in data/raw/.
python scripts/build_data.py data/raw/mp_routes.csv

streamlit run app.py        # http://localhost:8501
```
Without the Kaggle file the app still runs on a small `[SAMPLE]` dataset so you
can see the full interface offline.

### Tests
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # only for regression tests
python -m pytest -q
```

### Deploy to Streamlit Community Cloud (free)
1. Push this `project/` to a GitHub repo (or a `web/` subfolder of your existing one).
2. Go to https://share.streamlit.io → **New app** → pick the repo/branch and
   `app.py`.
3. Because the raw dataset is large and not redistributable, either commit a
   compact `data/processed/routes_compact.parquet` (small subset) or let the app
   fall back to the sample. **Do not commit `mp_routes.csv` or `ticks.csv`.**
4. Streamlit gives you a URL like `https://<app>.streamlit.app`.

### Link from your GitHub Pages site
Add to your blog (Markdown or HTML):
```html
<a class="button" href="https://YOUR-APP.streamlit.app" target="_blank">
  🧗 Try the Climbing Route Finder →
</a>
```
Replace `YOUR-APP` with the URL from step 4. *(Placeholder — needs your actual
Streamlit URL once deployed.)*

## 9. Validation queries

`tests/test_units.py` exercises the brief's examples: style+discipline parsing,
"steep"→overhang, exclusions, geography + unmatched terms ("limestone" has no
field in the data → reported), "easy" flagged rather than mapped, grade and
discipline filters, a no-results query, style-match ranking, and area counts.
`tests/test_regression.py` pins the refactor to the original functions
(`yds_to_ordinal`, `build_multihot_*`, vote multiplier, normalisation, area
aggregation) within `1e-9`.

## 10. Placeholders needing your values
* Streamlit app URL (§8, step 4) → the Pages link in §8.
* `data/raw/mp_routes.csv` (Kaggle download) for real (non-sample) results.
* `MP_personal/ticks.csv` only if you switch to the personal-NN scoring path.
