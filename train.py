"""Train and evaluate the review-sentiment model.

Fixes versus the original script:
* TF-IDF is fitted inside a Pipeline on training data only (no leakage).
* Negations ("not", "never") are kept: stop-word removal flipped meanings.
* Relative paths, no hard-coded absolute drive paths, no per-row stopword reloads.
* One saved artifact (vectorizer + model) so the app needs no rebuild step.
* All plots use variables that actually exist, with correct axis orientation.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import FeatureUnion, Pipeline

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
OUT = BASE / "reports"
OUT.mkdir(exist_ok=True)
SEED = 0


def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 sublinear_tf=True)),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(C=3, max_iter=2000))])


def main():
    df = pd.read_csv(BASE / "Restaurant_Reviews.tsv", sep="\t", quoting=3)
    assert {"Review", "Liked"} <= set(df.columns)
    df = df.drop_duplicates("Review").reset_index(drop=True)
    X, y = df["Review"], df["Liked"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    cv_scores = cross_val_score(build_pipeline(), X_tr, y_tr, cv=cv, n_jobs=-1)

    model = build_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    metrics = {
        "train_accuracy": round(model.score(X_tr, y_tr), 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
        "test_accuracy": round(accuracy_score(y_te, pred), 4),
        "test_roc_auc": round(roc_auc_score(y_te, prob), 4),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_te, pred, target_names=["Negative", "Positive"]))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix: rows = actual, columns = predicted (sklearn order).
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix (test set)")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(4, 4))
    vals = [metrics["train_accuracy"] * 100, metrics["cv_accuracy_mean"] * 100,
            metrics["test_accuracy"] * 100]
    plt.bar(["Train", "CV", "Test"], vals, color=["#3B6D11", "#888", "#185FA5"], width=0.5)
    for i, v in enumerate(vals):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.ylim(0, 110); plt.ylabel("Accuracy (%)"); plt.title("Train vs CV vs Test")
    plt.tight_layout(); plt.savefig(OUT / "accuracy_bars.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4))
    plt.hist(prob[y_te.values == 1], bins=20, alpha=0.6, color="green", label="Actually positive")
    plt.hist(prob[y_te.values == 0], bins=20, alpha=0.6, color="red", label="Actually negative")
    plt.axvline(0.5, color="black", linestyle="--", label="Threshold")
    plt.xlabel("Predicted P(positive)"); plt.ylabel("Count")
    plt.title("Prediction Probability Distribution"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "probability_hist.png", dpi=150); plt.close()

    # Final model refit on all data for deployment.
    final = build_pipeline().fit(X, y)
    joblib.dump(final, BASE / "sentiment_model.pkl")
    print("Saved sentiment_model.pkl")


if __name__ == "__main__":
    main()
