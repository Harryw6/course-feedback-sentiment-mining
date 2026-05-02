from __future__ import annotations

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from .config import FIGURE_DIR


def _save_message_figure(path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(label_distribution: pd.DataFrame) -> None:
    path = FIGURE_DIR / "label_distribution.png"
    if label_distribution.empty or "count" not in label_distribution.columns:
        _save_message_figure(path, "Label Distribution", "No label distribution could be generated.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=label_distribution, x="label", y="count", hue="label", legend=False, ax=ax)
    ax.set_title("Sentiment Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_comparison(sentiment_results: pd.DataFrame, multilabel_results: pd.DataFrame | None) -> None:
    path = FIGURE_DIR / "model_comparison.png"
    rows = []
    if sentiment_results is not None and not sentiment_results.empty:
        run_rows = sentiment_results[sentiment_results["status"] == "run"]
        for _, row in run_rows.iterrows():
            rows.append({"model": row["model"], "metric": "sentiment_macro_f1", "value": row["macro_f1"]})
    if multilabel_results is not None and not multilabel_results.empty:
        run_rows = multilabel_results[multilabel_results["status"] == "run"]
        for _, row in run_rows.iterrows():
            rows.append({"model": row["model"], "metric": "multilabel_macro_f1", "value": row["macro_f1"]})
            rows.append({"model": row["model"], "metric": "multilabel_micro_f1", "value": row["micro_f1"]})
    if not rows:
        _save_message_figure(
            path,
            "Model Comparison",
            "No supervised model comparison was available from the downloaded data.",
        )
        return
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=data, x="value", y="model", hue="metric", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title("Executed Model Metrics")
    ax.set_xlabel("Score")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(sentiment_predictions: pd.DataFrame | None) -> None:
    path = FIGURE_DIR / "confusion_matrix_best_model.png"
    if sentiment_predictions is None or sentiment_predictions.empty:
        _save_message_figure(
            path,
            "Confusion Matrix",
            "Sentiment labels were unavailable, so no sentiment confusion matrix was generated.",
        )
        return
    error_rates = (
        sentiment_predictions.assign(correct=sentiment_predictions["true_label"] == sentiment_predictions["predicted_label"])
        .groupby("model")["correct"]
        .mean()
        .sort_values(ascending=False)
    )
    model_name = error_rates.index[0]
    subset = sentiment_predictions[sentiment_predictions["model"] == model_name]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        subset["true_label"],
        subset["predicted_label"],
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_topic_sentiment_heatmap(topic_sentiment: pd.DataFrame) -> None:
    path = FIGURE_DIR / "topic_sentiment_heatmap.png"
    if topic_sentiment.empty:
        _save_message_figure(path, "Topic-Sentiment Heatmap", "Topic modeling did not produce topic assignments.")
        return
    pivot = topic_sentiment.pivot_table(
        index="topic_id",
        columns="sentiment_label",
        values="document_count",
        fill_value=0,
        aggfunc="sum",
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Topic Distribution by Sentiment Availability")
    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Topic")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_course_sentiment_distribution(frame: pd.DataFrame) -> None:
    path = FIGURE_DIR / "course_sentiment_distribution.png"
    if "course_id" not in frame.columns or "sentiment_label" not in frame.columns:
        _save_message_figure(
            path,
            "Course Sentiment Distribution",
            "course_id and/or sentiment labels were unavailable in the downloaded dataset.",
        )
        return
    counts = (
        frame.groupby(["course_id", "sentiment_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(40)
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=counts, x="count", y="course_id", hue="sentiment_label", ax=ax)
    ax.set_title("Course Sentiment Distribution")
    ax.set_xlabel("Review Count")
    ax.set_ylabel("Course")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_all_figures(
    label_distribution: pd.DataFrame,
    sentiment_results: pd.DataFrame,
    sentiment_predictions: pd.DataFrame | None,
    multilabel_results: pd.DataFrame | None,
    topic_sentiment: pd.DataFrame,
    frame: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_label_distribution(label_distribution)
    plot_model_comparison(sentiment_results, multilabel_results)
    plot_confusion_matrix(sentiment_predictions)
    plot_topic_sentiment_heatmap(topic_sentiment)
    plot_course_sentiment_distribution(frame)
