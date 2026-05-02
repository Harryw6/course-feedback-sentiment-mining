from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TABLE_DIR
from .evaluate import classification_metrics


def _sample(frame: pd.DataFrame, mode: str, max_full_rows: int = 60000) -> pd.DataFrame:
    if mode == "quick" and len(frame) > 5000:
        return frame.sample(n=5000, random_state=RANDOM_STATE).reset_index(drop=True)
    if mode == "full" and len(frame) > max_full_rows:
        return frame.sample(n=max_full_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    return frame.reset_index(drop=True)


def _stratify(frame: pd.DataFrame, label_col: str) -> pd.Series | None:
    counts = frame[label_col].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return frame[label_col]


def _fit_eval(
    frame: pd.DataFrame,
    label_col: str,
    experiment: str,
    ngram_range: tuple[int, int] = (1, 2),
    course_holdout: bool = False,
) -> dict[str, object]:
    data = frame.dropna(subset=["clean_text", label_col]).copy()
    if data[label_col].nunique() < 2 or len(data) < 20:
        return {
            "experiment": experiment,
            "status": "not_run",
            "metric": "macro_f1",
            "value": np.nan,
            "accuracy": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "weighted_f1": np.nan,
            "train_rows": 0,
            "test_rows": 0,
            "notes": f"Insufficient rows/classes for {label_col}.",
        }

    if course_holdout and "course_id" in data.columns and data["course_id"].nunique() >= 5:
        courses = pd.Series(data["course_id"].dropna().unique()).sample(frac=1.0, random_state=RANDOM_STATE)
        holdout_count = max(1, int(round(len(courses) * 0.2)))
        holdout_courses = set(courses.iloc[:holdout_count])
        train = data[~data["course_id"].isin(holdout_courses)]
        test = data[data["course_id"].isin(holdout_courses)]
    else:
        train, test = train_test_split(
            data,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=_stratify(data, label_col),
        )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    min_df=2 if len(train) > 100 else 1,
                    max_df=0.95,
                    max_features=12000,
                    ngram_range=ngram_range,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    model.fit(train["clean_text"], train[label_col])
    pred = model.predict(test["clean_text"])
    metrics = classification_metrics(test[label_col], pred)
    return {
        "experiment": experiment,
        "status": "run",
        "metric": "macro_f1",
        "value": metrics["macro_f1"],
        **metrics,
        "train_rows": len(train),
        "test_rows": len(test),
        "notes": "Executed with TF-IDF + Logistic Regression; rating was not used as a feature.",
    }


def run_main_ablation(
    main_frame: pd.DataFrame,
    weak_frame: pd.DataFrame | None,
    mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sampled_main = _sample(main_frame, mode, max_full_rows=80000)
    rows.append(_fit_eval(sampled_main, "sentiment_label", "3class_random_split_unigram_bigram", (1, 2)))
    rows.append(_fit_eval(sampled_main, "label_5class", "5class_random_split_unigram_bigram", (1, 2)))
    rows.append(_fit_eval(sampled_main, "sentiment_label", "3class_random_split_unigram", (1, 1)))
    rows.append(_fit_eval(sampled_main, "sentiment_label", "3class_course_holdout_split", (1, 2), course_holdout=True))
    rows.append(
        {
            "experiment": "with_topic_features",
            "status": "not_run",
            "metric": "macro_f1",
            "value": np.nan,
            "accuracy": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "weighted_f1": np.nan,
            "train_rows": 0,
            "test_rows": 0,
            "notes": "Topic-feature classifier was documented but not run because topic assignments are generated after supervised splits.",
        }
    )
    if weak_frame is not None and not weak_frame.empty:
        weak_sample = _sample(weak_frame, mode, max_full_rows=60000)
        rows.append(_fit_eval(sampled_main, "sentiment_label", "kaggle_100k_only", (1, 2)))
        weak_result = _fit_eval(weak_sample, "sentiment_label", "kaggle_145m_weak_label_sample", (1, 2))
        weak_result["notes"] += f" Weak-label rows used: {len(weak_sample)} of {len(weak_frame)}."
        rows.append(weak_result)
    else:
        rows.append(
            {
                "experiment": "kaggle_145m_weak_label_sample",
                "status": "not_run",
                "metric": "macro_f1",
                "value": np.nan,
                "accuracy": np.nan,
                "precision_macro": np.nan,
                "recall_macro": np.nan,
                "weighted_f1": np.nan,
                "train_rows": 0,
                "test_rows": 0,
                "notes": "Kaggle 1.45M weak-label dataset was not available.",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "ablation_results.csv", index=False)
    return result


def run_ablation(sentiment_info: dict[str, Any], multilabel_results: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if sentiment_info.get("sentiment_available", False):
        rows.append(
            {
                "experiment": "sentiment_tfidf_unigram_vs_bigram",
                "status": "not_run",
                "metric": "macro_f1",
                "value": np.nan,
                "notes": "Ablation scaffold is present; skipped to keep runtime bounded.",
            }
        )
    else:
        rows.append(
            {
                "experiment": "sentiment_ablation",
                "status": "not_run",
                "metric": "macro_f1",
                "value": np.nan,
                "notes": "No sentiment labels are available in the downloaded fallback dataset.",
            }
        )
    if multilabel_results is not None and not multilabel_results.empty:
        rows.append(
            {
                "experiment": "multilabel_tfidf_one_vs_rest_baseline",
                "status": multilabel_results.iloc[0].get("status", "unknown"),
                "metric": "macro_f1",
                "value": multilabel_results.iloc[0].get("macro_f1", np.nan),
                "notes": "Executed multi-label feedback-type baseline on available public fallback dataset.",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "ablation_results.csv", index=False)
    return result
