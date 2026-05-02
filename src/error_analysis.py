from __future__ import annotations

import pandas as pd

from .config import TABLE_DIR


def generate_error_cases(
    sentiment_predictions: pd.DataFrame | None,
    multilabel_predictions: pd.DataFrame | None,
    max_cases: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if sentiment_predictions is not None and not sentiment_predictions.empty:
        errors = sentiment_predictions[
            sentiment_predictions["true_label"] != sentiment_predictions["predicted_label"]
        ].head(max_cases)
        for _, row in errors.iterrows():
            true_label = str(row.get("true_label", ""))
            pred_label = str(row.get("predicted_label", ""))
            text = str(row.get("clean_text", ""))
            if true_label == "positive" and pred_label == "negative":
                category = "positive_predicted_as_negative"
            elif true_label == "negative" and pred_label == "positive":
                category = "negative_predicted_as_positive"
            elif true_label == "neutral" or pred_label == "neutral":
                category = "neutral_confusion"
            elif len(text.split()) <= 5:
                category = "short_review"
            elif any(token in text.lower() for token in ["but", "however", "although", "except"]):
                category = "mixed_sentiment"
            else:
                category = "other_sentiment_error"
            rows.append(
                {
                    "task": "sentiment",
                    "model": row.get("model", ""),
                    "course_id": row.get("course_id", ""),
                    "clean_text": row.get("clean_text", ""),
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "error_type": category,
                    "analysis_note": "Review for ambiguous wording, mixed sentiment, or preprocessing loss.",
                }
            )
    if multilabel_predictions is not None and not multilabel_predictions.empty:
        errors = multilabel_predictions[
            multilabel_predictions["true_labels"] != multilabel_predictions["predicted_labels"]
        ].head(max_cases)
        for _, row in errors.iterrows():
            rows.append(
                {
                    "task": "multi_label_feedback_type",
                    "model": "TF-IDF + One-vs-Rest Logistic Regression",
                    "clean_text": row.get("clean_text", ""),
                    "true_label": row.get("true_labels", ""),
                    "predicted_label": row.get("predicted_labels", ""),
                    "error_type": "label_set_mismatch",
                    "analysis_note": "Review for overlapping feedback functions or sparse label evidence.",
                }
            )
    if not rows:
        rows.append(
            {
                "task": "not_available",
                "model": "",
                "clean_text": "",
                "true_label": "",
                "predicted_label": "",
                "error_type": "not_run",
                "analysis_note": "No predictions were available for error analysis.",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "error_cases.csv", index=False)
    return result
