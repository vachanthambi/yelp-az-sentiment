"""
05_star_predictor.py
--------------------
Predicts the customer's 1-5 star rating from the review text alone.

What changed from the first version:
  - Results are reported against a baseline. Most reviews are 5 stars, so
    always guessing 5 already scores well; any model has to beat that.
  - The model reads the original text with negations and two-word phrases
    ("not good", "never again"). The old features came from the cleaned
    text, where "not" had been deleted. The old setup is still trained
    here on the same split for a like-for-like comparison.
  - The TF-IDF vocabulary is learned from training reviews only. Before, it
    was fitted on every review before the train/test split.
  - Metrics that fit how the model would be used: being within one star,
    and how well it flags 1-2 star reviews.
  - Random Forest dropped: it scored 56.8% in the first version, well
    below logistic regression, and almost never predicted 2-4 stars.

Usage:
    python src/05_star_predictor.py

Input:  data/reviews_preprocessed.csv
Output: assets/star_predictor_confusion_matrix.png
        assets/star_predictor_top_phrases.png
        outputs/star_predictor_report.txt
        outputs/star_predictor_errors.txt
        outputs/metrics/05_star_predictor.json
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from common import ASSETS_DIR, OUTPUT_DIR, PREPROCESSED_CSV, RANDOM_STATE, ensure_dirs, save_metrics

VOCAB_SAMPLE = 150_000      # training reviews used to learn the vocabulary (memory)
TUNE_SIZE = 80_000          # training reviews used to pick the regularisation strength
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
TOP_PHRASES = 12
STARS = [1, 2, 3, 4, 5]

MODEL_NAMES = {
    "baseline": "Always guess the most common rating",
    "old": "Old setup: cleaned text, top 3,000 words",
    "new": "New setup: original text, words + 2-word phrases",
}

sns.set_theme(style="whitegrid")


def evaluate(y_true, y_pred):
    error = np.abs(y_pred - y_true)
    flagged, actual = y_pred <= 2, y_true <= 2
    hits = int(np.sum(flagged & actual))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae_stars": float(error.mean()),
        "within_one_star": float((error <= 1).mean()),
        "flag_1_2_recall": hits / max(int(actual.sum()), 1),
        "flag_1_2_precision": hits / max(int(flagged.sum()), 1),
    }


def plot_confusion(y_test, preds, results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key in zip(axes, ["old", "new"]):
        cm = confusion_matrix(y_test, preds[key], labels=STARS, normalize="true")
        ticks = [f"{s} star" for s in STARS]
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=ticks, yticklabels=ticks, linewidths=0.5, ax=ax)
        ax.set_title(f"{MODEL_NAMES[key]}\nAccuracy {results[key]['accuracy']:.1%}", fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Star rating prediction: each row shows where reviews with that rating were predicted",
                 fontsize=13)
    fig.tight_layout()
    path = ASSETS_DIR / "star_predictor_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_top_phrases(model, vectoriser):
    names = vectoriser.get_feature_names_out()
    classes = list(model.classes_)
    phrases = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, star, color in [(axes[0], 1, "#E53935"), (axes[1], 5, "#4CAF50")]:
        weights = model.coef_[classes.index(star)]
        top = np.argsort(weights)[::-1][:TOP_PHRASES]
        phrases[star] = [names[i] for i in top]
        ax.barh([names[i] for i in top][::-1], weights[top][::-1], color=color)
        ax.set_title(f"Strongest signals of a {star}-star review", fontsize=13)
        ax.set_xlabel("Model weight")
    plt.tight_layout()
    path = ASSETS_DIR / "star_predictor_top_phrases.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return phrases


def write_errors(texts, y_test, proba, classes):
    """The model's most confident big misses: useful for explaining its limits."""
    classes = list(classes)
    lines = []
    for actual, predicted in [(5, 1), (1, 5)]:
        confidence = proba[:, classes.index(predicted)]
        candidates = np.where(y_test == actual)[0]
        worst = candidates[np.argsort(confidence[candidates])[::-1][:5]]
        lines.append(f"Rated {actual} stars, model most sure it was {predicted}:")
        for i in worst:
            snippet = " ".join(str(texts[i]).split())[:300]
            lines.append(f"  [{confidence[i]:.2f}] {snippet}")
        lines.append("")
    path = OUTPUT_DIR / "star_predictor_errors.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {path}")


def main():
    ensure_dirs()
    print(f"Loading {PREPROCESSED_CSV}...")
    df = pd.read_csv(PREPROCESSED_CSV, usecols=["text", "lda_text", "review_stars"])
    df = df.dropna(subset=["text", "review_stars"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["lda_text"] = df["lda_text"].fillna("").astype(str)
    y = df["review_stars"].astype(int).to_numpy()

    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2,
                                           random_state=RANDOM_STATE, stratify=y)
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Train: {len(train_idx):,}  Test: {len(test_idx):,}")
    preds = {}

    # 1. Baseline: always predict the most common rating.
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    preds["baseline"] = baseline.predict(np.zeros((len(y_test), 1)))

    # 2. Old setup, refitted on the same split (vocabulary from training data only).
    print("\nTraining the old setup...")
    old_vec = TfidfVectorizer(max_features=3_000, min_df=5, max_df=0.95)
    X_old_train = old_vec.fit_transform(df["lda_text"].iloc[train_idx])
    X_old_test = old_vec.transform(df["lda_text"].iloc[test_idx])
    old_model = LogisticRegression(C=1.0, max_iter=1_000, random_state=RANDOM_STATE)
    preds["old"] = old_model.fit(X_old_train, y_train).predict(X_old_test)

    # 3. New setup: original text, unigrams + bigrams, negations kept.
    print("Building features for the new setup...")
    rng = np.random.default_rng(RANDOM_STATE)
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=100_000,
                                 sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
    vocab_rows = rng.choice(train_idx, size=min(VOCAB_SAMPLE, len(train_idx)), replace=False)
    vectoriser.fit(df["text"].iloc[vocab_rows])
    X_train = vectoriser.transform(df["text"].iloc[train_idx])
    X_test = vectoriser.transform(df["text"].iloc[test_idx])

    print("Choosing regularisation strength on a slice of the training data...")
    tune = rng.permutation(len(train_idx))[:min(TUNE_SIZE, len(train_idx))]
    cut = int(len(tune) * 0.75)
    tune_scores = {}
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=1_000, random_state=RANDOM_STATE)
        m.fit(X_train[tune[:cut]], y_train[tune[:cut]])
        tune_scores[C] = float(accuracy_score(y_train[tune[cut:]], m.predict(X_train[tune[cut:]])))
        print(f"  C={C}: validation accuracy {tune_scores[C]:.3f}")
    best_C = max(tune_scores, key=tune_scores.get)

    print(f"Training the new setup on all training reviews (C={best_C})...")
    model = LogisticRegression(C=best_C, max_iter=2_000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds["new"] = model.predict(X_test)

    # --- Results ----------------------------------------------------------------
    results = {key: evaluate(y_test, p) for key, p in preds.items()}
    print(f"\n{'Model':52} {'Acc':>6} {'Within 1':>8} {'MAE':>5} {'Flag 1-2: recall':>17} {'precision':>10}")
    for key, r in results.items():
        print(f"{MODEL_NAMES[key]:52} {r['accuracy']:6.1%} {r['within_one_star']:8.1%} "
              f"{r['mae_stars']:5.2f} {r['flag_1_2_recall']:17.1%} {r['flag_1_2_precision']:10.1%}")

    report = classification_report(y_test, preds["new"], labels=STARS,
                                   target_names=[f"{s} star" for s in STARS], zero_division=0)
    (OUTPUT_DIR / "star_predictor_report.txt").write_text(
        "\n".join(f"{MODEL_NAMES[k]}: accuracy {r['accuracy']:.4f}" for k, r in results.items())
        + "\n\nNew setup, per class:\n" + report, encoding="utf-8")

    print("\nGenerating charts...")
    plot_confusion(y_test, preds, results)
    phrases = plot_top_phrases(model, vectoriser)
    write_errors(df["text"].iloc[test_idx].to_numpy(), y_test, model.predict_proba(X_test),
                 model.classes_)

    save_metrics("05_star_predictor", {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "test_star_share": {int(s): float((y_test == s).mean()) for s in STARS},
        "best_C": best_C,
        "tuning_accuracy": {str(c): s for c, s in tune_scores.items()},
        "models": {key: {"name": MODEL_NAMES[key], **r} for key, r in results.items()},
        "per_class_recall_new": {
            int(s): float(np.mean(preds["new"][y_test == s] == s)) for s in STARS
        },
        "top_phrases": {f"{s}_star": p for s, p in phrases.items()},
    })


if __name__ == "__main__":
    main()
