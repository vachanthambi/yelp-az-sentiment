"""
07_summary.py
-------------
Collects the metrics from every step into one readable file,
outputs/RESULTS.md, for writing up the README.

Usage:
    python src/07_summary.py

Input:  outputs/metrics/*.json
Output: outputs/RESULTS.md
"""

from common import OUTPUT_DIR, load_metrics


def pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def main():
    lines = ["# Results summary", ""]

    pre = load_metrics("02_preprocess")
    if pre:
        old, new = pre["vader_old"], pre["vader_fixed"]
        lines += [
            "## Data",
            f"- {pre['n_reviews']:,} reviews across {pre['n_businesses']:,} businesses",
            "- Star ratings: " + ", ".join(f"{s} star {pct(v)}" for s, v in pre["star_share"].items()),
            "",
            "## VADER fix (checked against the customers' own ratings)",
            "| | Before (cleaned text) | After (original text) |",
            "|---|---|---|",
            f"| 1-2 star reviews labelled negative | {pct(old['complaints_labelled_negative'])} "
            f"| {pct(new['complaints_labelled_negative'])} |",
            f"| 1-2 star reviews labelled positive | {pct(old['complaints_labelled_positive'])} "
            f"| {pct(new['complaints_labelled_positive'])} |",
            f"| Agreement with star rating (neg/neutral/pos) | {pct(old['agreement_with_stars'])} "
            f"| {pct(new['agreement_with_stars'])} |",
            f"| Reviews labelled positive overall | {pct(old['label_share']['Positive'])} "
            f"| {pct(new['label_share']['Positive'])} |",
            "",
        ]

    eda = load_metrics("03_sentiment_eda")
    if eda:
        lines += [f"## Most negative categories (share of 1-2 star reviews; all reviews "
                  f"{pct(eda['negative_share_overall'])})"]
        lines += [f"- {c['category']}: {pct(c['share_1_2_star'])} of {c['reviews']:,} reviews"
                  for c in eda["most_negative_categories"][:8]]
        lines.append("")

    stars = load_metrics("05_star_predictor")
    if stars:
        lines += [
            f"## Star rating predictor (test set: {stars['n_test']:,} reviews)",
            "| Model | Accuracy | Within 1 star | Mean error (stars) | Flags 1-2 star: recall "
            "| Flags 1-2 star: precision |",
            "|---|---|---|---|---|---|",
        ]
        for m in stars["models"].values():
            lines.append(f"| {m['name']} | {pct(m['accuracy'])} | {pct(m['within_one_star'])} "
                         f"| {m['mae_stars']:.2f} | {pct(m['flag_1_2_recall'])} "
                         f"| {pct(m['flag_1_2_precision'])} |")
        lines += [
            "",
            "Recall by actual rating (new setup): "
            + ", ".join(f"{s} star {pct(v)}" for s, v in stars["per_class_recall_new"].items()),
            "",
            "Strongest 1-star phrases: " + ", ".join(stars["top_phrases"]["1_star"]),
            "",
            "Strongest 5-star phrases: " + ", ".join(stars["top_phrases"]["5_star"]),
            "",
        ]

    topics = load_metrics("04_topics")
    if topics:
        lines += [f"## Complaint themes ({topics['n_complaints']:,} reviews rated 1-2 stars; "
                  f"{topics['best_k']} topics chosen by NPMI coherence)"]
        lines += [f"- T{t['topic']} ({pct(t['share_of_complaints'])}): {', '.join(t['top_words'])}"
                  for t in topics["topics"]]
        lines += ["", "Largest theme in each sector:"]
        for sector, shares in topics["theme_share_by_sector"].items():
            label, share = max(shares.items(), key=lambda kv: kv[1])
            lines.append(f"- {sector}: {label} ({pct(share)})")
        lines.append("")

    risk = load_metrics("06_closure_risk")
    if risk and risk.get("backtest"):
        b = risk["backtest"]
        lines += [
            f"## Closure backtest (signals from reviews before {b['cutoff_date']})",
            f"- {b['n_businesses']:,} businesses open at the cutoff; {b['n_closed']:,} closed by "
            f"the snapshot ({pct(b['closure_rate'])})",
            f"- Cross-validated AUC: all signals {b['auc_all_signals']:.3f}, "
            f"ratings only {b['auc_ratings_only']:.3f}",
            f"- Of the 15 businesses flagged highest risk, {b['top15_closed']} closed",
            "",
            "| Tier | Businesses | Closed | Closure rate | Lift vs average |",
            "|---|---|---|---|---|",
        ]
        lines += [f"| {t} | {r['businesses']:,} | {r['closed']:,} | {pct(r['closure_rate'])} "
                  f"| {r['lift']:.2f}x |" for t, r in b["tiers"].items()]
        lines += ["", "Standardised weights (positive = more likely to close):"]
        lines += [f"- {k}: {v:+.2f}" for k, v in
                  sorted(b["standardised_weights"].items(), key=lambda kv: -abs(kv[1]))]
        lines.append("")
    if risk and risk.get("revenue"):
        r = risk["revenue"]
        lines += [
            "## Revenue illustration (open independent restaurants)",
            f"- {r['n_independent_restaurants']:,} restaurants; average rating "
            f"{r['average_rating']:.2f}; {r['n_below_average']:,} below average",
            f"- Median upside for those below average: ${r['median_opportunity_usd']:,.0f} a year "
            f"(assumes {pct(r['uplift_per_star'])} per star and ${r['baseline_revenue_usd']:,} revenue)",
            "",
        ]

    path = OUTPUT_DIR / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
