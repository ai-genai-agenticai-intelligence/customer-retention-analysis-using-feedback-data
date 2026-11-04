from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Restaurant_Reviews.tsv"
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
MAX_REVIEW_LENGTH = 2_000
MAX_BATCH_SIZE = 100

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024


@lru_cache(maxsize=1)
def load_dataset():
    dataset = pd.read_csv(DATA_PATH, sep="\t", quoting=3)
    if not {"Review", "Liked"}.issubset(dataset.columns):
        raise ValueError("The review dataset must contain Review and Liked columns.")
    if dataset.empty or dataset["Review"].isna().any():
        raise ValueError("The review dataset is empty or contains blank reviews.")
    if not set(dataset["Liked"].unique()).issubset({0, 1}):
        raise ValueError("The Liked column must contain only 0 and 1 labels.")
    return dataset


@lru_cache(maxsize=1)
def load_model():
    """Load the single saved pipeline (vectorizer + classifier)."""
    if not MODEL_PATH.exists():
        raise ValueError("sentiment_model.pkl not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)


def analyze_reviews(reviews):
    for index, review in enumerate(reviews, start=1):
        if not any(ch.isalpha() for ch in review):
            raise ValueError(f"Review {index} must contain at least one word.")
    model = load_model()
    probabilities = model.predict_proba(reviews)
    positive_index = list(model.classes_).index(1)
    results = []
    for review, probability in zip(reviews, probabilities):
        positive = float(probability[positive_index])
        results.append({
            "review": review,
            "liked": positive >= 0.5,
            "confidence": max(positive, 1 - positive),
            "positive_probability": positive,
        })
    return results


def render_dashboard(**context):
    dataset = load_dataset()
    positive_count = int((dataset["Liked"] == 1).sum())
    total_count = len(dataset)
    return render_template(
        "index.html",
        total_reviews=total_count,
        positive_count=positive_count,
        negative_count=total_count - positive_count,
        positive_rate=round(positive_count / total_count * 100),
        **context,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "single_result": None,
        "batch_results": None,
        "batch_positive_count": 0,
        "batch_negative_count": 0,
        "error": None,
    }
    if request.method == "POST":
        action = request.form.get("action", "")
        try:
            if action == "single":
                review = request.form.get("review", "").strip()
                if not review:
                    raise ValueError("Enter a review to analyze.")
                if len(review) > MAX_REVIEW_LENGTH:
                    raise ValueError(
                        f"Reviews must be {MAX_REVIEW_LENGTH:,} characters or fewer."
                    )
                context["single_review"] = review
                context["single_result"] = analyze_reviews([review])[0]
            elif action == "batch":
                reviews = [
                    line.strip()
                    for line in request.form.get("reviews", "").splitlines()
                    if line.strip()
                ]
                if not reviews:
                    raise ValueError("Enter at least one review, one per line.")
                if len(reviews) > MAX_BATCH_SIZE:
                    raise ValueError(
                        f"Analyze up to {MAX_BATCH_SIZE} reviews at a time."
                    )
                if any(len(review) > MAX_REVIEW_LENGTH for review in reviews):
                    raise ValueError(
                        f"Each review must be {MAX_REVIEW_LENGTH:,} characters or fewer."
                    )
                context["batch_reviews"] = "\n".join(reviews)
                batch_results = analyze_reviews(reviews)
                context["batch_results"] = batch_results
                context["batch_positive_count"] = sum(
                    result["liked"] for result in batch_results
                )
                context["batch_negative_count"] = (
                    len(batch_results) - context["batch_positive_count"]
                )
            else:
                raise ValueError("Choose a supported analysis action.")
        except (OSError, ValueError) as error:
            context["error"] = str(error)
    return render_dashboard(**context)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
