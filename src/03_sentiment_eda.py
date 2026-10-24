"""
03_sentiment_eda.py
-------------------
Charts for the README:

    assets/star_rating_distribution.png    How customers rate.
    assets/vader_vs_stars.png              VADER's labels against the customer's
                                           own star rating, before vs after the fix.
    assets/negative_share_by_category.png  Share of 1-2 star reviews by category
                                           (the customer's verdict, not VADER's).

Usage:
    python src/03_sentiment_eda.py

Input:  data/reviews_preprocessed.csv
Output: the charts above, outputs/metrics/03_sentiment_eda.json
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from common import (ASSETS_DIR, PREPROCESSED_CSV, SENTIMENT_COLORS, SENTIMENT_LABELS,
                    ensure_dirs, save_metrics)

MIN_CATEGORY_REVIEWS = 500
TOP_CATEGORIES = 12

sns.set_theme(style="whitegrid")


def plot_star_distribution(df):
    counts = df["review_stars"].value_counts().sort_index()
    shares = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#42A5F5", edgecolor="white")
    for bar, share in zip(bars, shares):
        ax.annotate(f"{share:.0%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    ax.set_title("How customers rate: distribution of star ratings", fontsize=14, pad=12)
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Number of reviews")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = ASSETS_DIR / "star_rating_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_vader_vs_stars(df):
    """For each star rating, the share of reviews VADER calls negative/neutral/positive."""
    panels = [
        ("vader_label_old", "Before: VADER on cleaned text"),
        ("vader_label", "After: VADER on the original text"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    shares_by_version = {}
    for ax, (col, title) in zip(axes, panels):
        shares = (pd.crosstab(df["review_stars"], df[col], normalize="index")
                  .reindex(columns=SENTIMENT_LABELS, fill_value=0))
        shares_by_version[col] = shares
        shares.plot(kind="bar", stacked=True, ax=ax, width=0.8, edgecolor="white", legend=False,
                    color=[SENTIMENT_COLORS[label] for label in SENTIMENT_LABELS])
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Customer's star rating")
        ax.set_xticklabels([str(s) for s in shares.index], rotation=0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Share of reviews")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, [f"VADER: {label}" for label in labels], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    path = ASSETS_DIR / "vader_vs_stars.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return shares_by_version


def plot_negative_share_by_category(df):
    cats = df[["categories", "review_stars"]].dropna(subset=["categories"]).copy()
    cats["category"] = cats["categories"].str.split(",")
    cats = cats.explode("category")
    cats["category"] = cats["category"].str.strip()
    cats["negative"] = cats["review_stars"] <= 2

    stats = cats.groupby("category")["negative"].agg(share="mean", reviews="size")
    stats = stats[stats["reviews"] >= MIN_CATEGORY_REVIEWS]
    top = stats.sort_values("share", ascending=False).head(TOP_CATEGORIES)
    overall = float((df["review_stars"] <= 2).mean())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index[::-1], top["share"][::-1], color="#EF5350", edgecolor="white")
    ax.axvline(overall, color="#37474F", linestyle="--", linewidth=1.2,
               label=f"All reviews: {overall:.0%}")
    ax.set_title(f"Categories with the highest share of 1-2 star reviews\n"
                 f"(categories with {MIN_CATEGORY_REVIEWS:,}+ reviews)", fontsize=13, pad=12)
    ax.set_xlabel("Share of reviews rated 1-2 stars")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = ASSETS_DIR / "negative_share_by_category.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return overall, top


def main():
    ensure_dirs()
    print(f"Loading {PREPROCESSED_CSV}...")
    df = pd.read_csv(PREPROCESSED_CSV,
                     usecols=["review_stars", "categories", "vader_label", "vader_label_old"])

    print("Generating charts...")
    plot_star_distribution(df)
    shares = plot_vader_vs_stars(df)
    overall, top = plot_negative_share_by_category(df)

    metrics = {
        "negative_share_overall": overall,
        "vader_negative_share_by_star": {
            version: {int(star): float(row["Negative"]) for star, row in table.iterrows()}
            for version, table in shares.items()
        },
        "most_negative_categories": [
            {"category": cat, "share_1_2_star": float(row["share"]), "reviews": int(row["reviews"])}
            for cat, row in top.iterrows()
        ],
    }
    save_metrics("03_sentiment_eda", metrics)


if __name__ == "__main__":
    main()
