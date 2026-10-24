"""
01_merge_data.py
----------------
Filters the Yelp Open Dataset to Arizona and joins each review to its business.

The review file is read in chunks, so the full multi-GB file never has to fit
in memory. Before filtering, it also counts how many locations share each
business name across the whole dataset; step 06 uses that to separate chains
from independents.

Usage:
    python src/01_merge_data.py

Input:  data/yelp_academic_dataset_business.csv
        data/yelp_academic_dataset_review.csv
Output: data/az_reviews.csv
"""

import pandas as pd

from common import BUSINESS_CSV, MERGED_CSV, REVIEW_CSV, STATE, normalize_name

REQUIRED_BUSINESS_COLS = ["business_id", "name", "state", "categories"]
OPTIONAL_BUSINESS_COLS = ["city", "stars", "review_count", "is_open"]
REVIEW_COLS = ["review_id", "business_id", "stars", "text", "date"]


def check_columns(path, required):
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in required if c not in header]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    return header


def load_businesses():
    header = check_columns(BUSINESS_CSV, REQUIRED_BUSINESS_COLS)
    cols = REQUIRED_BUSINESS_COLS + [c for c in OPTIONAL_BUSINESS_COLS if c in header]
    if "is_open" not in cols:
        print("WARNING: no is_open column, so step 06 will skip the closure backtest.")

    biz = pd.read_csv(BUSINESS_CSV, usecols=cols)
    # Chain proxy: number of businesses in the full dataset with the same name.
    name_key = biz["name"].map(normalize_name)
    biz["name_locations"] = name_key.map(name_key.value_counts())
    biz = biz.rename(columns={"stars": "business_stars", "review_count": "business_review_count"})

    local = biz[biz["state"] == STATE].copy()
    print(f"Businesses: {len(biz):,} in the dataset, {len(local):,} in {STATE}")
    return local


def load_reviews(business_ids):
    check_columns(REVIEW_CSV, REVIEW_COLS)
    keep, scanned = [], 0
    for i, chunk in enumerate(pd.read_csv(REVIEW_CSV, usecols=REVIEW_COLS, chunksize=250_000)):
        scanned += len(chunk)
        keep.append(chunk[chunk["business_id"].isin(business_ids)])
        if i % 4 == 3:
            print(f"  scanned {scanned:,} reviews")
    reviews = pd.concat(keep, ignore_index=True).rename(columns={"stars": "review_stars"})
    print(f"Reviews: {scanned:,} scanned, {len(reviews):,} for {STATE} businesses")
    return reviews


def main():
    businesses = load_businesses()
    reviews = load_reviews(set(businesses["business_id"]))
    merged = reviews.merge(businesses, on="business_id", how="inner")
    merged.to_csv(MERGED_CSV, index=False)
    print(
        f"Saved {len(merged):,} reviews across {merged['business_id'].nunique():,} "
        f"businesses to {MERGED_CSV}"
    )


if __name__ == "__main__":
    main()
