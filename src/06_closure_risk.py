"""
06_closure_risk.py
------------------
Do Yelp reviews warn that a business is going to close? A backtest
against real closures.

The first version combined sentiment, stars and review count into a
weighted score with fixed cut-offs (50 and 70). It flagged half of all
businesses, penalised businesses just for having few reviews, and was
never checked against any outcome. This version tests the idea directly:

  1. Pick a cutoff date (CUTOFF_DATE in common.py, default 2019-01-01).
  2. Using only reviews written before the cutoff, compute each business's
     review signals: rating, recent share of 1-2 star reviews, rating trend,
     review momentum, recent activity, age, and restaurant/chain flags.
  3. Keep businesses that were clearly open at the cutoff (5+ reviews and
     at least one in the last 90 days).
  4. Outcome: Yelp's is_open flag at the dataset snapshot (early 2022).
     The closure window includes the pandemic.
  5. Logistic regression, scored with 5-fold cross-validated AUC, compared
     with a ratings-only model: do ratings alone predict closure?
  6. Risk tiers from the out-of-fold predictions: top 10% High risk,
     next 20% Elevated, then the closure rate actually seen in each tier.

It also sizes an illustrative revenue opportunity for independent
restaurants only. Luca (HBS) estimated that one extra Yelp star lifts an
independent restaurant's revenue 5-9%, found no effect for chains, and
studied restaurants only, so other businesses are left out.

Usage:
    python src/06_closure_risk.py

Input:  data/reviews_preprocessed.csv
Output: assets/closure_rate_by_tier.png
        assets/closure_risk_drivers.png
        assets/closure_backtest_top15.png
        assets/revenue_opportunity_independents.png
        outputs/closure_backtest_businesses.csv
        outputs/revenue_opportunity_independents.csv
        outputs/metrics/06_closure_risk.json
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (ASSETS_DIR, BASELINE_REVENUE_USD, CHAIN_MIN_LOCATIONS, CUTOFF_DATE, OUTPUT_DIR,
                    PREPROCESSED_CSV, RANDOM_STATE, REVENUE_UPLIFT_PER_STAR, ensure_dirs,
                    is_restaurant, save_metrics)

WINDOW = pd.Timedelta(days=365)
ACTIVE = pd.Timedelta(days=90)
MIN_REVIEWS = 5
SHRINK_M = 10           # pseudo-reviews at the market average (Bayesian average)
MIN_CLOSURES = 30       # below this, the backtest is too thin to report

FEATURES = {
    "rating": "Rating (Bayesian average)",
    "neg_share_12m": "Share of 1-2 star reviews, last 12 months",
    "rating_trend": "Rating trend, last 12 vs prior 12 months",
    "review_momentum": "Review volume trend, last 12 vs prior 12 months",
    "log_reviews_12m": "Reviews in the last 12 months (log)",
    "age_years": "Years since first review",
    "is_restaurant": "Restaurant",
    "is_chain": "Chain",
}
RATING_FEATURES = ["rating", "neg_share_12m", "rating_trend"]
TIER_ORDER = ["High risk", "Elevated", "Low"]

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Review signals
# ---------------------------------------------------------------------------
def shrink(mean, n, prior, m=SHRINK_M):
    """Bayesian average: pulls businesses with few reviews toward the market average."""
    return (n * mean.fillna(prior) + m * prior) / (n + m)


def review_signals(reviews, as_of):
    """Per-business review signals, using only reviews written before `as_of`."""
    r = reviews[reviews["date"] < as_of]
    last12 = r["date"] >= as_of - WINDOW
    prev12 = (r["date"] >= as_of - 2 * WINDOW) & ~last12
    stars = r["review_stars"].astype(float)
    negative = (r["review_stars"] <= 2).astype(float)

    tmp = pd.DataFrame({
        "business_id": r["business_id"],
        "date": r["date"],
        "stars": stars,
        "last12": last12.astype(int),
        "prev12": prev12.astype(int),
        "recent90": (r["date"] >= as_of - ACTIVE).astype(int),
        "stars_last12": stars.where(last12),
        "stars_prev12": stars.where(prev12),
        "neg_last12": negative.where(last12),
    })
    g = tmp.groupby("business_id").agg(
        n_reviews=("stars", "size"),
        mean_stars=("stars", "mean"),
        first_review=("date", "min"),
        n_last12=("last12", "sum"),
        n_prev12=("prev12", "sum"),
        n_recent90=("recent90", "sum"),
        mean_last12=("stars_last12", "mean"),
        mean_prev12=("stars_prev12", "mean"),
        neg_last12=("neg_last12", "mean"),
    )
    star_prior, neg_prior = stars.mean(), negative.mean()
    g["rating"] = shrink(g["mean_stars"], g["n_reviews"], star_prior)
    g["rating_trend"] = (shrink(g["mean_last12"], g["n_last12"], star_prior)
                         - shrink(g["mean_prev12"], g["n_prev12"], star_prior))
    g["neg_share_12m"] = shrink(g["neg_last12"], g["n_last12"], neg_prior)
    g["review_momentum"] = np.log((g["n_last12"] + 1) / (g["n_prev12"] + 1))
    g["log_reviews_12m"] = np.log1p(g["n_last12"])
    g["age_years"] = (as_of - g["first_review"]).dt.days / 365.25
    return g


def business_info(df):
    cols = [c for c in ["business_id", "name", "city", "categories", "is_open", "name_locations"]
            if c in df.columns]
    info = df[cols].drop_duplicates("business_id").set_index("business_id")
    info["is_restaurant"] = info["categories"].map(is_restaurant).astype(int)
    if "name_locations" in info.columns:
        info["is_chain"] = (info["name_locations"] >= CHAIN_MIN_LOCATIONS).astype(int)
    else:
        info["is_chain"] = 0
    return info


BROAD_CATEGORIES = {"Restaurants", "Food", "Nightlife", "Bars", "Shopping", "Home Services",
                    "Health & Medical", "Automotive", "Beauty & Spas", "Hotels & Travel",
                    "Local Services", "Event Planning & Services", "Active Life",
                    "Arts & Entertainment"}


def chart_labels(rows):
    """Rank + specific category + city. Public charts don't name businesses;
    the names stay in the git-ignored CSVs in outputs/."""
    labels = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        cats = [c.strip() for c in str(row.get("categories", "")).split(",") if c.strip()]
        specific = next((c for c in cats if c not in BROAD_CATEGORIES), cats[0] if cats else "Business")
        city = row.get("city")
        labels.append(f"{rank}. {specific}" + (f", {city}" if isinstance(city, str) else ""))
    return labels


# ---------------------------------------------------------------------------
# Closure backtest
# ---------------------------------------------------------------------------
def assign_tiers(risk):
    pct = pd.Series(risk).rank(pct=True, method="first").to_numpy()
    return np.where(pct > 0.90, "High risk", np.where(pct > 0.70, "Elevated", "Low"))


def plot_tiers(tiers, overall_rate, cutoff_year):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"High risk": "#E53935", "Elevated": "#FFA726", "Low": "#66BB6A"}
    bars = ax.bar(tiers.index, tiers["closure_rate"], color=[colors[t] for t in tiers.index],
                  edgecolor="white")
    for bar, (_, row) in zip(bars, tiers.iterrows()):
        ax.annotate(f"{row['closure_rate']:.0%}\n({int(row['businesses']):,} businesses)",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center",
                    va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    ax.axhline(overall_rate, color="#37474F", linestyle="--", linewidth=1.2,
               label=f"All businesses: {overall_rate:.0%}")
    ax.set_title(f"Share that closed, by risk tier assigned from reviews before {cutoff_year}",
                 fontsize=13, pad=12)
    ax.set_ylabel("Closed by the dataset snapshot")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(tiers["closure_rate"].max() * 1.3, overall_rate * 1.5))
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = ASSETS_DIR / "closure_rate_by_tier.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_drivers(coefs):
    s = pd.Series(coefs).rename(index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(s.index, s.values, color=["#E53935" if v > 0 else "#42A5F5" for v in s.values])
    ax.axvline(0, color="#37474F", linewidth=1)
    ax.set_title("What moves closure risk\n(standardised logistic regression weights)",
                 fontsize=13, pad=12)
    ax.set_xlabel("Weight (positive = more likely to close)")
    plt.tight_layout()
    path = ASSETS_DIR / "closure_risk_drivers.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_top15(top, cutoff_year):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(top["closed"] == 1, "#E53935", "#90A4AE")
    ax.barh(top["label"][::-1], top["risk"][::-1], color=colors[::-1])
    ax.set_title(f"The 15 businesses flagged as highest risk from reviews before {cutoff_year}",
                 fontsize=13, pad=12)
    ax.set_xlabel("Predicted closure risk")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.legend(handles=[Patch(color="#E53935", label="Closed by the snapshot"),
                       Patch(color="#90A4AE", label="Still open")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    plt.tight_layout()
    path = ASSETS_DIR / "closure_backtest_top15.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def closure_backtest(df, info):
    if "is_open" not in info.columns:
        print("No is_open column: skipping the closure backtest.")
        return None

    cutoff = pd.Timestamp(CUTOFF_DATE)
    signals = review_signals(df, cutoff).join(info, how="left")
    data = signals[(signals["n_reviews"] >= MIN_REVIEWS) & (signals["n_recent90"] >= 1)].copy()
    data = data[data["is_open"].notna()]
    data["closed"] = (data["is_open"] == 0).astype(int)
    n_closed = int(data["closed"].sum())
    print(f"  Open at {CUTOFF_DATE}: {len(data):,} businesses; "
          f"{n_closed:,} closed by the snapshot ({data['closed'].mean():.1%})")
    if n_closed < MIN_CLOSURES or len(data) - n_closed < MIN_CLOSURES:
        print("  Too few closures (or survivors) for a reliable backtest: skipping.")
        return None

    features = list(FEATURES)
    y = data["closed"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000))
    oof_full = cross_val_predict(model, data[features].to_numpy(float), y, cv=cv,
                                 method="predict_proba")[:, 1]
    oof_ratings = cross_val_predict(model, data[RATING_FEATURES].to_numpy(float), y, cv=cv,
                                    method="predict_proba")[:, 1]
    auc_full, auc_ratings = roc_auc_score(y, oof_full), roc_auc_score(y, oof_ratings)
    print(f"  Cross-validated AUC: all signals {auc_full:.3f}, ratings only {auc_ratings:.3f}")

    data["risk"] = oof_full
    data["tier"] = assign_tiers(oof_full)
    overall = float(y.mean())
    tiers = (data.groupby("tier")["closed"]
             .agg(businesses="size", closed="sum", closure_rate="mean")
             .reindex(TIER_ORDER))
    tiers["lift"] = tiers["closure_rate"] / overall
    print(tiers.to_string(float_format=lambda v: f"{v:.3f}"))

    model.fit(data[features].to_numpy(float), y)
    coefs = dict(zip(features, model.named_steps["logisticregression"].coef_[0]))

    top = data.nlargest(15, "risk").copy()
    top["label"] = chart_labels(top)
    cutoff_year = cutoff.year

    plot_tiers(tiers, overall, cutoff_year)
    plot_drivers(coefs)
    plot_top15(top, cutoff_year)

    keep = ["name", "city", "categories", "n_reviews", *features, "risk", "tier", "closed"]
    data[[c for c in keep if c in data.columns]].sort_values("risk", ascending=False).to_csv(
        OUTPUT_DIR / "closure_backtest_businesses.csv")

    return {
        "cutoff_date": CUTOFF_DATE,
        "n_businesses": len(data),
        "n_closed": n_closed,
        "closure_rate": overall,
        "auc_all_signals": float(auc_full),
        "auc_ratings_only": float(auc_ratings),
        "tiers": {t: {"businesses": int(r["businesses"]), "closed": int(r["closed"]),
                      "closure_rate": float(r["closure_rate"]), "lift": float(r["lift"])}
                  for t, r in tiers.iterrows()},
        "standardised_weights": {FEATURES[k]: float(v) for k, v in coefs.items()},
        "top15_closed": int(top["closed"].sum()),
    }


# ---------------------------------------------------------------------------
# Revenue illustration (independent restaurants only)
# ---------------------------------------------------------------------------
def revenue_opportunity(df, info):
    as_of = df["date"].max() + pd.Timedelta(days=1)
    signals = review_signals(df, as_of).join(info, how="left")
    mask = ((signals["is_restaurant"] == 1) & (signals["is_chain"] == 0)
            & (signals["n_reviews"] >= MIN_REVIEWS))
    if "is_open" in signals.columns:
        mask &= signals["is_open"] == 1
    ind = signals[mask].copy()
    if ind.empty:
        print("No open independent restaurants found: skipping the revenue illustration.")
        return None

    market = float(ind["rating"].mean())
    ind["star_gap"] = (market - ind["rating"]).clip(lower=0)
    ind["revenue_opportunity_usd"] = ind["star_gap"] * REVENUE_UPLIFT_PER_STAR * BASELINE_REVENUE_USD
    below = ind[ind["star_gap"] > 0]
    top = ind.nlargest(15, "revenue_opportunity_usd").copy()
    top["label"] = chart_labels(top)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["label"][::-1], top["revenue_opportunity_usd"][::-1], color="#7E57C2")
    ax.set_title("Illustrative revenue upside from reaching the average rating\n"
                 f"(open independent restaurants; {REVENUE_UPLIFT_PER_STAR:.0%} per star, "
                 f"${BASELINE_REVENUE_USD:,.0f} assumed revenue)", fontsize=12, pad=12)
    ax.set_xlabel("Estimated annual revenue upside (USD)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
    plt.tight_layout()
    path = ASSETS_DIR / "revenue_opportunity_independents.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    ind.sort_values("revenue_opportunity_usd", ascending=False).to_csv(
        OUTPUT_DIR / "revenue_opportunity_independents.csv")
    return {
        "as_of": str(as_of.date()),
        "n_independent_restaurants": len(ind),
        "average_rating": market,
        "n_below_average": len(below),
        "median_opportunity_usd": float(below["revenue_opportunity_usd"].median()),
        "uplift_per_star": REVENUE_UPLIFT_PER_STAR,
        "baseline_revenue_usd": BASELINE_REVENUE_USD,
        "chain_min_locations": CHAIN_MIN_LOCATIONS,
    }


def main():
    ensure_dirs()
    print(f"Loading {PREPROCESSED_CSV}...")
    cols = ["business_id", "name", "city", "categories", "is_open", "name_locations",
            "review_stars", "date"]
    header = pd.read_csv(PREPROCESSED_CSV, nrows=0).columns
    df = pd.read_csv(PREPROCESSED_CSV, usecols=[c for c in cols if c in header])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "review_stars"])
    info = business_info(df)
    print(f"  {len(df):,} reviews, {len(info):,} businesses, "
          f"latest review {df['date'].max().date()}")

    print("\nClosure backtest...")
    backtest = closure_backtest(df, info)
    print("\nRevenue illustration (independent restaurants)...")
    revenue = revenue_opportunity(df, info)
    save_metrics("06_closure_risk", {"backtest": backtest, "revenue": revenue})


if __name__ == "__main__":
    main()
