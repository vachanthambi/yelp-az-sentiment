"""
02_preprocess.py
----------------
Scores every review with VADER and prepares text for topic modelling.

The fix in this version: VADER scores the original review text.
The first version scored the cleaned text, after stopword removal had
already deleted "not", "no", "nor", "but" and "very" (all in NLTK's
stopword list) along with punctuation and capital letters. Those are the
cues VADER is built to read, so "The food was not good and the service was
not friendly" became "food good service friendly" and scored as positive.

The old labels are kept (vader_label_old) so step 03 can measure both
versions against the customers' own star ratings.

The cleaned, lemmatised text (lda_text) is unchanged from the first version
and is used only for topic modelling, where dropping "not" doesn't matter.
If the first version's output (data/vader_preprocessed_output.csv) is
present, its cleaned text and VADER labels are reused: the cleaning is
identical, it skips the slow spaCy step, and the "before" numbers are
exactly the first run's. Otherwise the text is cleaned and lemmatised here.

Usage:
    python src/02_preprocess.py

Input:  data/az_reviews.csv
        data/vader_preprocessed_output.csv   (optional, from the first version)
Output: data/reviews_preprocessed.csv
        outputs/metrics/02_preprocess.json
"""

import re
import time

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

from common import (FIRST_RUN_PREPROCESSED, MERGED_CSV, PREPROCESSED_CSV, SENTIMENT_LABELS,
                    save_metrics, star_sentiment)

for package in ("punkt", "punkt_tab", "stopwords", "vader_lexicon"):
    nltk.download(package, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
URL_RE = re.compile(r"http\S+|www\S+")
NON_LETTERS_RE = re.compile(r"[^a-z\s]")


# ---------------------------------------------------------------------------
# Text cleaning (for topic modelling only)
# ---------------------------------------------------------------------------
def pre_clean(text):
    """Lowercase, strip URLs and non-letters, drop stopwords and 1-letter tokens."""
    text = NON_LETTERS_RE.sub("", URL_RE.sub("", text.lower()))
    return " ".join(t for t in word_tokenize(text) if t not in STOP_WORDS and len(t) > 1)


def lemmatise(texts, batch_size=1_000):
    """Lemmatise with spaCy (tagger only) and drop lemmas that are stopwords."""
    import spacy  # only needed when there is no first-run file to reuse

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    out, start = [], time.time()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size), start=1):
        out.append(" ".join(tok.lemma_ for tok in doc if tok.lemma_ not in STOP_WORDS))
        if i % 50_000 == 0:
            print(f"  lemmatised {i:,} reviews ({time.time() - start:,.0f}s)")
    return out


def first_run_cleaning(review_ids):
    """Cleaned text and VADER labels saved by the first version, in review order."""
    if not FIRST_RUN_PREPROCESSED.exists():
        return None
    print(f"Reusing cleaned text and the old VADER labels from {FIRST_RUN_PREPROCESSED}...")
    old = pd.read_csv(FIRST_RUN_PREPROCESSED,
                      usecols=["review_id", "cleaned_text", "vader_sentiment"])
    old = old.drop_duplicates("review_id").set_index("review_id")
    coverage = float(review_ids.isin(old.index).mean())
    if coverage < 1.0:
        print(f"  Only {coverage:.1%} of reviews are in that file, so cleaning from scratch.")
        return None
    return old.loc[review_ids.to_numpy()]


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------
def vader_compound(texts, sia):
    scores, start = [], time.time()
    for i, text in enumerate(texts, start=1):
        scores.append(sia.polarity_scores(text)["compound"])
        if i % 50_000 == 0:
            print(f"  scored {i:,} reviews ({time.time() - start:,.0f}s)")
    return np.array(scores)


def to_label(compound):
    """VADER's recommended cut-offs: >= 0.05 Positive, <= -0.05 Negative."""
    return np.select([compound >= 0.05, compound <= -0.05], ["Positive", "Negative"], default="Neutral")


def agreement(stars_verdict, labels):
    """How well VADER's labels match what the customer actually rated."""
    complaints = stars_verdict == "Negative"
    return {
        "label_share": {lab: float((labels == lab).mean()) for lab in SENTIMENT_LABELS},
        "complaints_labelled_negative": float((labels[complaints] == "Negative").mean()),
        "complaints_labelled_positive": float((labels[complaints] == "Positive").mean()),
        "agreement_with_stars": float((labels == stars_verdict).mean()),
    }


def main():
    print(f"Loading {MERGED_CSV}...")
    df = pd.read_csv(MERGED_CSV)
    df = df.dropna(subset=["text", "review_stars"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["review_stars"] = df["review_stars"].astype(int)
    print(f"  {len(df):,} reviews, {df['business_id'].nunique():,} businesses")

    sia = SentimentIntensityAnalyzer()
    cached = first_run_cleaning(df["review_id"])
    if cached is not None:
        df["lda_text"] = cached["cleaned_text"].fillna("").astype(str).to_numpy()
        df["vader_label_old"] = cached["vader_sentiment"].astype(str).to_numpy()
        old_source = "first run's saved labels"
    else:
        print("Cleaning text for topic modelling...")
        start = time.time()
        cleaned = [pre_clean(t) for t in df["text"]]
        print(f"  done ({time.time() - start:,.0f}s). Lemmatising with spaCy (the slow step)...")
        df["lda_text"] = lemmatise(cleaned)
        print("Scoring VADER the old way, on the cleaned text, for comparison...")
        df["vader_label_old"] = to_label(vader_compound(df["lda_text"], sia))
        old_source = "recomputed"

    print("Scoring VADER on the original review text...")
    df["vader_compound"] = vader_compound(df["text"], sia)
    df["vader_label"] = to_label(df["vader_compound"].to_numpy())

    verdict = star_sentiment(df["review_stars"]).to_numpy()
    metrics = {
        "n_reviews": len(df),
        "n_businesses": int(df["business_id"].nunique()),
        "star_share": {
            int(k): float(v)
            for k, v in df["review_stars"].value_counts(normalize=True).sort_index().items()
        },
        "old_labels_source": old_source,
        "vader_fixed": agreement(verdict, df["vader_label"].to_numpy()),
        "vader_old": agreement(verdict, df["vader_label_old"].to_numpy()),
    }

    old, new = metrics["vader_old"], metrics["vader_fixed"]
    print(f"\nVADER against the customers' own ratings (old -> fixed; old labels: {old_source}):")
    print(f"  1-2 star reviews labelled negative: {old['complaints_labelled_negative']:.1%} -> "
          f"{new['complaints_labelled_negative']:.1%}")
    print(f"  1-2 star reviews labelled positive: {old['complaints_labelled_positive']:.1%} -> "
          f"{new['complaints_labelled_positive']:.1%}")
    print(f"  Agreement with star rating:         {old['agreement_with_stars']:.1%} -> "
          f"{new['agreement_with_stars']:.1%}")

    print(f"\nSaving {PREPROCESSED_CSV}...")
    df.to_csv(PREPROCESSED_CSV, index=False)
    save_metrics("02_preprocess", metrics)


if __name__ == "__main__":
    main()
