# Customer Feedback Intelligence
liveDemo - customer-retention-analysis-using-f.vercel.app
Sentiment analysis of restaurant reviews with a Flask interface.

> **Scope note:** the data has only review text and a Liked (1/0) label — no customer IDs, purchase behaviour, churn or retention outcomes. This project classifies **feedback sentiment**; it cannot predict retention or churn. Negative sentiment is a useful *risk signal* to follow up on, not a churn probability.

## Results (1,000 reviews, 996 after removing 4 duplicates)

| Metric | Original | Corrected |
|---|---|---|
| Model | Random Forest (depth 4) | TF-IDF (word 1-2 grams + char 2-5 grams) + Logistic Regression |
| Held-out test accuracy | 69.5% | **84.5%** (ROC-AUC 0.91) |
| Cross-validated accuracy | not measured | 81.6% ± 2.8% |

With only 200 test reviews, treat any single accuracy figure as roughly ±3 points; the cross-validated number is the more honest estimate (about 82%).

## What was fixed
- Training script crashed: `train_acc`, `test_acc`, `y_prob`, `y_test_int` were never defined.
- Data leakage: TF-IDF was fitted on all reviews before the train/test split. It is now inside a `Pipeline`.
- Confusion-matrix arguments were swapped (`y_pred, y_test`), mislabelling axes.
- Stop-word removal deleted negations ("not", "never"), hurting accuracy; they are now kept.
- Stop-words were reloaded on every row (slow); hard-coded `D:\` path removed.
- The app rebuilt the vectorizer from the dataset at start-up because it was never saved. Now one `sentiment_model.pkl` holds vectorizer + model, and NLTK is no longer needed.
- Duplicate reviews removed; stratified split; saved metrics and plots in `reports/`.

## Run
```bash
pip install -r requirements.txt
python train.py     # trains, evaluates, writes sentiment_model.pkl and reports/
python app.py       # http://127.0.0.1:5000
```

## Known limitations
- Small dataset (about 1k short restaurant reviews): train accuracy is 100%, so the model overfits somewhat.
- Double negatives and sarcasm are weak, e.g. "not bad" is classified negative.
- Probabilities are not calibrated.
- Only load `.pkl` files you trained yourself (pickle can execute code).
- To truly study retention, add per-customer data (visit history, recency/frequency, churn flag) and join it with feedback.
