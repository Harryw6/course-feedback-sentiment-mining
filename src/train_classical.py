from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import MULTILABEL_COLUMNS, RANDOM_STATE, TABLE_DIR
from .evaluate import multilabel_metrics, per_label_report, sentiment_metrics


SENTIMENT_BASELINES = {
    "TF-IDF + Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "TF-IDF + Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    "TF-IDF + Random Forest": RandomForestClassifier(
        n_estimators=40,
        max_depth=50,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "TF-IDF + Multinomial Naive Bayes": MultinomialNB(),
}


def _pipeline(classifier: Any, max_features: int = 20000) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_features,
                    stop_words="english",
                ),
            ),
            ("clf", classifier),
        ]
    )


def _not_run_sentiment(reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "status": "not_run",
                "accuracy": np.nan,
                "precision_macro": np.nan,
                "recall_macro": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "notes": reason,
            }
            for model_name in SENTIMENT_BASELINES
        ]
    )


def train_sentiment_baselines(
    splits: dict[str, pd.DataFrame],
    sentiment_info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    reason = sentiment_info.get("reason", "Sentiment labels are unavailable.")
    if not sentiment_info.get("sentiment_available", False):
        return _not_run_sentiment(str(reason)), None

    train = splits["train"].dropna(subset=["clean_text", "sentiment_label"])
    test = splits["test"].dropna(subset=["clean_text", "sentiment_label"])
    if train["sentiment_label"].nunique() < 2 or test.empty:
        return _not_run_sentiment("Not enough labeled classes for sentiment classification."), None

    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name, classifier in SENTIMENT_BASELINES.items():
        model = _pipeline(classifier)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(train["clean_text"], train["sentiment_label"])
        pred = model.predict(test["clean_text"])
        rows.append(sentiment_metrics(test["sentiment_label"], pred, model_name))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "review_text": test["review_text"].to_numpy() if "review_text" in test.columns else test["clean_text"].to_numpy(),
                    "clean_text": test["clean_text"].to_numpy(),
                    "course_id": test["course_id"].to_numpy() if "course_id" in test.columns else "",
                    "true_label": test["sentiment_label"].to_numpy(),
                    "predicted_label": pred,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


def _constant_multilabel_prediction(train: pd.DataFrame, test: pd.DataFrame, labels: list[str]) -> np.ndarray:
    pred = np.zeros((len(test), len(labels)), dtype=int)
    for idx, label in enumerate(labels):
        pred[:, idx] = int(train[label].mean() >= 0.5)
    return pred


def train_multilabel_classifier(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | None]:
    available = [col for col in MULTILABEL_COLUMNS if col in splits["train"].columns]
    if not available:
        result = pd.DataFrame(
            [
                {
                    "model": "TF-IDF + One-vs-Rest Logistic Regression",
                    "status": "not_run",
                    "micro_f1": np.nan,
                    "macro_f1": np.nan,
                    "hamming_loss": np.nan,
                    "subset_accuracy": np.nan,
                    "notes": "No multi-label feedback columns found.",
                }
            ]
        )
        return {"results": result, "predictions": None, "label_report": None}

    train = splits["train"].copy()
    test = splits["test"].copy()
    y_train_all = train[available].fillna(0).astype(int)
    y_test_all = test[available].fillna(0).astype(int)
    active = [label for label in available if y_train_all[label].nunique() > 1]
    if not active:
        pred_all = _constant_multilabel_prediction(train, test, available)
        metrics = multilabel_metrics(y_test_all.to_numpy(), pred_all, "TF-IDF + Majority Multi-label Baseline")
        metrics["notes"] = "All labels were constant in training data."
        return {
            "results": pd.DataFrame([metrics]),
            "predictions": pd.DataFrame(pred_all, columns=[f"pred_{label}" for label in available]),
            "label_report": per_label_report(y_test_all.to_numpy(), pred_all, available),
        }

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=25000,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(train["clean_text"], y_train_all[active])
    pred_active = model.predict(test["clean_text"])
    pred_all = _constant_multilabel_prediction(train, test, available)
    for active_idx, label in enumerate(active):
        pred_all[:, available.index(label)] = pred_active[:, active_idx]

    metrics = multilabel_metrics(
        y_test_all[available].to_numpy(),
        pred_all,
        "TF-IDF + One-vs-Rest Logistic Regression",
    )
    predictions = pd.DataFrame(
        {
            "clean_text": test["clean_text"].to_numpy(),
            "true_labels": [
                "; ".join(label for label in available if row[label] == 1)
                for _, row in y_test_all[available].iterrows()
            ],
            "predicted_labels": [
                "; ".join(label for idx, label in enumerate(available) if pred_all[row_idx, idx] == 1)
                for row_idx in range(len(test))
            ],
        }
    )
    label_report = per_label_report(y_test_all[available].to_numpy(), pred_all, available)
    return {
        "results": pd.DataFrame([metrics]),
        "predictions": predictions,
        "label_report": label_report,
    }


def run_classical_experiments(
    splits: dict[str, pd.DataFrame],
    sentiment_info: dict[str, Any],
) -> dict[str, pd.DataFrame | None]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sentiment_results, sentiment_predictions = train_sentiment_baselines(splits, sentiment_info)
    sentiment_results.to_csv(TABLE_DIR / "classical_baseline_results.csv", index=False)
    if sentiment_predictions is not None:
        sentiment_predictions.to_csv(TABLE_DIR / "sentiment_predictions.csv", index=False)

    multilabel = train_multilabel_classifier(splits)
    multilabel_results = multilabel["results"]
    if isinstance(multilabel_results, pd.DataFrame):
        multilabel_results.to_csv(TABLE_DIR / "multilabel_results.csv", index=False)
    if isinstance(multilabel["predictions"], pd.DataFrame):
        multilabel["predictions"].to_csv(TABLE_DIR / "multilabel_predictions.csv", index=False)
    if isinstance(multilabel["label_report"], pd.DataFrame):
        multilabel["label_report"].to_csv(TABLE_DIR / "multilabel_label_report.csv", index=False)

    return {
        "sentiment_results": sentiment_results,
        "sentiment_predictions": sentiment_predictions,
        "multilabel_results": multilabel.get("results"),
        "multilabel_predictions": multilabel.get("predictions"),
        "multilabel_label_report": multilabel.get("label_report"),
    }
