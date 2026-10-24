"""
common.py
---------
Shared paths, settings and helpers for every step of the pipeline.

Run each script from the repo root, e.g. `python src/01_merge_data.py`,
or run the whole pipeline with `python run_all.py`.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Never crash on printing an emoji or accented name to a console that can't show it
# (e.g. Windows cp1252); unprintable characters become "?" instead.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")          # tables, reports, metrics (git-ignored)
METRICS_DIR = OUTPUT_DIR / "metrics"
ASSETS_DIR = Path("assets")           # charts shown in the README

BUSINESS_CSV = DATA_DIR / "yelp_academic_dataset_business.csv"
REVIEW_CSV = DATA_DIR / "yelp_academic_dataset_review.csv"
MERGED_CSV = DATA_DIR / "az_reviews.csv"
PREPROCESSED_CSV = DATA_DIR / "reviews_preprocessed.csv"
# Written by the first version of step 02. Its cleaned text is identical to
# what step 02 produces now, so step 02 reuses it when present (no spaCy needed).
FIRST_RUN_PREPROCESSED = DATA_DIR / "vader_preprocessed_output.csv"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
STATE = "AZ"
RANDOM_STATE = 42

# A business name found at this many locations across the full Yelp dataset
# counts as a chain. Luca (HBS) found that Yelp ratings move revenue for
# independent restaurants but not for chains.
CHAIN_MIN_LOCATIONS = 5

# Closure backtest: risk signals use only reviews written before this date.
# The outcome is Yelp's is_open flag at the dataset snapshot (early 2022).
CUTOFF_DATE = "2019-01-01"

# Revenue illustration, independent restaurants only.
REVENUE_UPLIFT_PER_STAR = 0.05    # Luca: +1 star -> +5-9% revenue; low end used
BASELINE_REVENUE_USD = 800_000    # assumed annual revenue, not observed

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
SENTIMENT_COLORS = {"Negative": "#E53935", "Neutral": "#FFC107", "Positive": "#4CAF50"}

# Broad sectors, checked in order: a business gets the first one that matches.
SECTORS = [
    ("Restaurants & Food", {"Restaurants", "Food", "Bars", "Nightlife", "Coffee & Tea"}),
    ("Automotive", {"Automotive"}),
    ("Health & Medical", {"Health & Medical"}),
    ("Home Services", {"Home Services"}),
    ("Beauty & Spas", {"Beauty & Spas"}),
    ("Hotels & Travel", {"Hotels & Travel"}),
    ("Shopping", {"Shopping"}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in (OUTPUT_DIR, METRICS_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def category_set(categories):
    """Yelp stores categories as one comma-separated string."""
    if not isinstance(categories, str):
        return set()
    return {c.strip() for c in categories.split(",") if c.strip()}


def sector_of(categories):
    cats = category_set(categories)
    for sector, keys in SECTORS:
        if cats & keys:
            return sector
    return "Other"


def is_restaurant(categories):
    return "Restaurants" in category_set(categories)


def normalize_name(name):
    """Normalise a business name so chain locations match ("McDonald's" = "Mcdonalds")."""
    if not isinstance(name, str):
        return ""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[4:] if s.startswith("the ") else s


def star_sentiment(stars):
    """The customer's own verdict: 1-2 stars Negative, 3 Neutral, 4-5 Positive."""
    return pd.Series(
        np.select([stars <= 2, stars == 3], ["Negative", "Neutral"], default="Positive"),
        index=stars.index,
    )


def _to_builtin(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def save_metrics(name, metrics):
    ensure_dirs()
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2, default=_to_builtin), encoding="utf-8")
    print(f"Metrics saved to {path}")


def load_metrics(name):
    path = METRICS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
