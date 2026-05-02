from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from .config import MULTILABEL_COLUMNS, TABLE_DIR
from .evaluate import multilabel_metrics


def train_multilabel_feedback_classifier(
    splits: dict[str, pd.DataFrame],
    mode: str,
    output_path: Path = TABLE_DIR / "multilabel_results.csv",
) -> pd.DataFrame:
    train = splits.get("train", pd.DataFrame()).copy()
    val = splits.get("val", pd.DataFrame()).copy()
    test = splits.get("test", pd.DataFrame()).copy()
    available = [col for col in MULTILABEL_COLUMNS if col in train.columns and train[col].nunique(dropna=True) > 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not available:
        result = pd.DataFrame(
            [
                {
                    "task": "feedback_type_multilabel",
                    "status": "not_run",
                    "reason": "No multi-label feedback columns with positive and negative examples were available.",
                    "micro_f1": pd.NA,
                    "macro_f1": pd.NA,
                    "hamming_loss": pd.NA,
                    "subset_accuracy": pd.NA,
                    "labels_used": "",
                    "train_rows": 0,
                    "test_rows": 0,
                }
            ]
        )
        result.to_csv(output_path, index=False)
        return result

    if test.empty:
        result = pd.DataFrame(
            [
                {
                    "task": "feedback_type_multilabel",
                    "status": "not_run",
                    "reason": "Test split is empty.",
                    "micro_f1": pd.NA,
                    "macro_f1": pd.NA,
                    "hamming_loss": pd.NA,
                    "subset_accuracy": pd.NA,
                    "labels_used": ",".join(available),
                    "train_rows": len(train),
                    "test_rows": 0,
                }
            ]
        )
        result.to_csv(output_path, index=False)
        return result

    train_data = pd.concat([train, val], ignore_index=True) if not val.empty else train
    max_features = 5000 if mode == "quick" else 20000
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=1 if len(train_data) < 100 else 2,
                    strip_accents="unicode",
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1))),
        ]
    )
    y_train = train_data[available].astype(int)
    y_test = test[available].astype(int)
    pipeline.fit(train_data["clean_text"], y_train)
    predictions = pipeline.predict(test["clean_text"])
    metrics = multilabel_metrics(y_test, predictions)
    result = pd.DataFrame(
        [
            {
                "task": "feedback_type_multilabel",
                "status": "run",
                "reason": "",
                **metrics,
                "labels_used": ",".join(available),
                "train_rows": len(train_data),
                "test_rows": len(test),
            }
        ]
    )
    result.to_csv(output_path, index=False)

    pred_frame = test[["review_text", "clean_text"]].copy()
    for idx, column in enumerate(available):
        pred_frame[f"true_{column}"] = y_test[column].to_numpy()
        pred_frame[f"pred_{column}"] = predictions[:, idx]
    pred_frame.to_csv(TABLE_DIR / "multilabel_predictions.csv", index=False)
    return result
