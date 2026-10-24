"""
04_topics.py
------------
Finds complaint themes with LDA topic modelling.

What changed from the first version:
  - Complaints are reviews the customer rated 1-2 stars. The first version
    used reviews VADER labelled negative, and that VADER step missed more
    than half of the 1-2 star reviews (see step 02).
  - The number of topics is chosen by NPMI coherence (do each topic's top
    words actually appear together in reviews?) instead of being fixed at 5.
  - K-Means was dropped. It produced a different number of groups (8) than
    LDA (5) with no way to reconcile them, and LDA's mixed-membership model
    suits reviews that raise several complaints at once.

Usage:
    python src/04_topics.py

Input:  data/reviews_preprocessed.csv
Output: assets/topic_coherence.png
        assets/complaint_themes_by_sector.png
        outputs/lda_topics.txt   (top words and example reviews per topic)
        outputs/metrics/04_topics.json
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from common import (ASSETS_DIR, OUTPUT_DIR, PREPROCESSED_CSV, RANDOM_STATE, ensure_dirs,
                    save_metrics, sector_of)

K_RANGE = range(4, 13)
SELECTION_SAMPLE = 30_000       # reviews used to compare topic counts, for speed
TOP_N_WORDS = 10
EXAMPLES_PER_TOPIC = 3
MIN_EXAMPLE_WORDS = 15          # skip very short reviews when picking examples
MIN_SECTOR_COMPLAINTS = 300

# Generic words that appear in complaints about anything and crowd out the themes.
DOMAIN_STOP_WORDS = {
    "get", "go", "say", "tell", "come", "would", "could", "one", "like", "even",
    "also", "back", "us", "know", "make", "take", "want", "really", "ask", "look",
    "thing", "way", "day", "time", "place", "give", "try", "see", "need", "well",
    "still", "never", "ever", "think", "much", "star", "review",
}

sns.set_theme(style="whitegrid")


def make_lda(k):
    return LatentDirichletAllocation(
        n_components=k, learning_method="online", learning_offset=50.0,
        batch_size=2048, max_iter=10, random_state=RANDOM_STATE,
    )


def top_word_ids(lda, n=TOP_N_WORDS):
    return [np.argsort(row)[::-1][:n] for row in lda.components_]


def npmi_coherence(topics, X_bin):
    """Mean NPMI over each topic's top-word pairs, from document co-occurrence.

    NPMI runs from -1 (the words never appear in the same review) to 1 (they
    always do). Higher means the topic's words genuinely belong together.
    """
    n_docs = X_bin.shape[0]
    scores = []
    for ids in topics:
        sub = X_bin[:, ids]
        p_word = np.asarray(sub.sum(axis=0)).ravel() / n_docs
        p_pair = (sub.T @ sub).toarray() / n_docs
        pair_scores = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                p_ij = p_pair[i, j]
                if p_ij <= 0:
                    pair_scores.append(-1.0)
                    continue
                pmi = np.log(p_ij / (p_word[i] * p_word[j]))
                pair_scores.append(pmi / -np.log(p_ij))
        scores.append(float(np.mean(pair_scores)))
    return scores


def plot_coherence(coherence, best_k):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(coherence)
    ax.plot(ks, [coherence[k] for k in ks], marker="o", color="#42A5F5", linewidth=2)
    ax.axvline(best_k, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Most coherent: {best_k} topics")
    ax.set_title("Choosing the number of complaint themes", fontsize=14, pad=12)
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Mean NPMI coherence (higher is better)")
    ax.legend()
    plt.tight_layout()
    path = ASSETS_DIR / "topic_coherence.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_sector_heatmap(table):
    fig, ax = plt.subplots(figsize=(1.6 * table.shape[1] + 3, 0.6 * table.shape[0] + 2.5))
    sns.heatmap(table, annot=True, fmt=".0%", cmap="Reds", cbar=False, linewidths=0.5, ax=ax)
    ax.set_title("Main theme of 1-2 star reviews, by sector (each row sums to 100%)",
                 fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    path = ASSETS_DIR / "complaint_themes_by_sector.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def main():
    ensure_dirs()
    print(f"Loading {PREPROCESSED_CSV}...")
    df = pd.read_csv(PREPROCESSED_CSV, usecols=["review_stars", "lda_text", "text", "categories"])
    df = df[df["review_stars"] <= 2].copy()
    df["lda_text"] = df["lda_text"].fillna("").astype(str)
    df = df[df["lda_text"].str.len() > 0].reset_index(drop=True)
    print(f"  Complaints (1-2 star reviews): {len(df):,}")

    vectoriser = CountVectorizer(max_df=0.5, min_df=10, max_features=10_000,
                                 stop_words=sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS))
    X = vectoriser.fit_transform(df["lda_text"])
    vocab = vectoriser.get_feature_names_out()
    doc_words = np.asarray(X.sum(axis=1)).ravel()
    X_bin = X.copy()
    X_bin.data[:] = 1
    X_bin = X_bin.tocsc()
    print(f"  Vocabulary: {len(vocab):,} words")

    # --- Choose the number of topics ----------------------------------------
    print(f"\nComparing topic counts {K_RANGE.start}-{K_RANGE.stop - 1} "
          f"on a sample of {min(SELECTION_SAMPLE, X.shape[0]):,} complaints...")
    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(X.shape[0], size=min(SELECTION_SAMPLE, X.shape[0]), replace=False)
    coherence = {}
    for k in K_RANGE:
        lda_k = make_lda(k).fit(X[sample])
        coherence[k] = float(np.mean(npmi_coherence(top_word_ids(lda_k), X_bin)))
        print(f"  {k:2d} topics: coherence {coherence[k]:.3f}")
    best_k = max(coherence, key=coherence.get)
    print(f"  Most coherent: {best_k} topics")

    # --- Final model on all complaints ---------------------------------------
    print(f"\nFitting {best_k} topics on all {X.shape[0]:,} complaints...")
    lda = make_lda(best_k).fit(X)
    doc_topic = lda.transform(X)
    dominant = doc_topic.argmax(axis=1)
    topic_ids = top_word_ids(lda)
    topic_coherence = npmi_coherence(topic_ids, X_bin)

    # Number topics by size: T1 is the most common main theme.
    shares = np.bincount(dominant, minlength=best_k) / len(dominant)
    order = np.argsort(-shares)
    rank = np.empty(best_k, dtype=int)
    rank[order] = np.arange(best_k)

    topics, lines = [], [
        f"Complaint themes: LDA with {best_k} topics (chosen by NPMI coherence)",
        f"Complaints analysed: {len(df):,} reviews rated 1-2 stars",
        "Name each theme from its words and examples, then use the names in the README.",
        "",
    ]
    for r, t in enumerate(order, start=1):
        words = [vocab[i] for i in topic_ids[t]]
        topics.append({
            "topic": r,
            "label": f"T{r}: " + ", ".join(words[:3]),
            "top_words": words,
            "share_of_complaints": float(shares[t]),
            "coherence": topic_coherence[t],
        })
        lines.append(f"T{r}  {shares[t]:.1%} of complaints  (coherence {topic_coherence[t]:.3f})")
        lines.append(f"  words: {', '.join(words)}")
        members = np.where(dominant == t)[0]
        long_enough = members[doc_words[members] >= MIN_EXAMPLE_WORDS]
        if len(long_enough) >= EXAMPLES_PER_TOPIC:
            members = long_enough
        best = members[np.argsort(doc_topic[members, t])[::-1][:EXAMPLES_PER_TOPIC]]
        for i in best:
            snippet = " ".join(str(df.at[i, "text"]).split())[:300]
            lines.append(f"  example: {snippet}")
        lines.append("")
    (OUTPUT_DIR / "lda_topics.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(line for line in lines if not line.startswith("  example:")))
    print(f"Example reviews for each theme saved to {OUTPUT_DIR / 'lda_topics.txt'}")

    # --- Themes by sector -----------------------------------------------------
    labels = [topic["label"] for topic in topics]
    df["theme"] = [labels[rank[t]] for t in dominant]
    df["sector"] = df["categories"].map(sector_of)
    sector_counts = df["sector"].value_counts()
    keep = [s for s in sector_counts.index
            if sector_counts[s] >= MIN_SECTOR_COMPLAINTS and s != "Other"]
    subset = df[df["sector"].isin(keep)]
    table = (pd.crosstab(subset["sector"], subset["theme"], normalize="index")
             .reindex(index=keep, columns=labels, fill_value=0))

    print("\nGenerating charts...")
    plot_coherence(coherence, best_k)
    if len(table):
        plot_sector_heatmap(table)

    save_metrics("04_topics", {
        "n_complaints": len(df),
        "vocabulary_size": len(vocab),
        "coherence_by_k": {str(k): v for k, v in coherence.items()},
        "best_k": best_k,
        "topics": topics,
        "sector_complaints": {s: int(sector_counts[s]) for s in keep},
        "theme_share_by_sector": {
            sector: {label: float(v) for label, v in row.items()} for sector, row in table.iterrows()
        },
    })


if __name__ == "__main__":
    main()
