from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def confusion_matrix_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])


def sentiment_metrics(y_true, y_pred, model_name: str) -> dict[str, object]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "model": model_name,
        "status": "run",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "notes": "",
    }


def multilabel_metrics(y_true, y_pred, model_name: str) -> dict[str, object]:
    return {
        "model": model_name,
        "status": "run",
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "notes": "",
    }


def per_label_report(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in labels:
        values = report.get(label, {})
        rows.append(
            {
                "label": label,
                "precision": values.get("precision", 0.0),
                "recall": values.get("recall", 0.0),
                "f1_score": values.get("f1-score", 0.0),
                "support": values.get("support", 0.0),
            }
        )
    return pd.DataFrame(rows)
