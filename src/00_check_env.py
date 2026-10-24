"""
00_check_env.py
---------------
Checks, in a few seconds, that every package the pipeline needs actually
loads, before anything long-running starts.

On Windows, Smart App Control (or another Application Control policy) can
block a package's compiled files even though the package is installed. That
is what stopped spaCy. spaCy is only needed when there is no first-run file
for step 02 to reuse; everything else here is required.

Usage:
    python src/00_check_env.py
"""

import io
import sys

from common import FIRST_RUN_PREPROCESSED


def check_numpy_pandas():
    import numpy as np
    import pandas as pd
    pd.DataFrame({"a": np.arange(3)}).groupby("a").size()


def check_nltk():
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    SentimentIntensityAnalyzer().polarity_scores("The food was not good.")


def check_matplotlib_seaborn():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap([[0.1, 0.9], [0.5, 0.5]], annot=True, ax=ax)
    fig.savefig(io.BytesIO(), format="png")
    plt.close(fig)


def check_sklearn():
    import numpy as np
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    docs = ["great food friendly staff", "rude staff cold food", "love this place", "never again"] * 6
    y = np.array([1, 0, 1, 0] * 6)
    X = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(docs)
    train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=0)
    pred = LogisticRegression(max_iter=200).fit(X, y).predict(X)
    accuracy_score(y, pred), f1_score(y, pred), confusion_matrix(y, pred)
    classification_report(y, pred, zero_division=0)
    DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y), 1)), y)
    LatentDirichletAllocation(n_components=2, learning_method="online", random_state=0).fit(
        CountVectorizer().fit_transform(docs))
    proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), X.toarray(), y,
                              cv=StratifiedKFold(n_splits=3), method="predict_proba")[:, 1]
    roc_auc_score(y, proba)


def check_spacy():
    if FIRST_RUN_PREPROCESSED.exists():
        return "not needed: step 02 will reuse the first run's cleaned text"
    import spacy
    spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return None


CHECKS = [
    ("numpy + pandas", check_numpy_pandas),
    ("NLTK + VADER", check_nltk),
    ("matplotlib + seaborn", check_matplotlib_seaborn),
    ("scikit-learn + scipy", check_sklearn),
    ("spaCy", check_spacy),
]


def main():
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    failures = []
    for name, fn in CHECKS:
        try:
            note = fn()
            print(f"  OK      {name}" + (f" ({note})" if note else ""))
        except Exception as exc:  # ImportError, OSError (blocked DLL), etc.
            failures.append(name)
            print(f"  FAILED  {name}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\nCan't run the pipeline: {', '.join(failures)} failed to load.")
        print("If an error says 'An Application Control policy has blocked this file', Windows "
              "Smart App Control is blocking that package's compiled files.")
        sys.exit(1)
    print("Everything the pipeline needs loads.")


if __name__ == "__main__":
    main()
