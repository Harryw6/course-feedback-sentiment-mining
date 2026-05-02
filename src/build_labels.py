from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE


EXPLICIT_SENTIMENT_MAP = {
    "very positive": "positive",
    "positive": "positive",
    "pos": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "very negative": "negative",
    "neg": "negative",
}

FIVE_CLASS_LABEL_MAP = {
    "5": "Very Positive",
    "4": "Positive",
    "3": "Neutral",
    "2": "Negative",
    "1": "Very Negative",
    "very positive": "Very Positive",
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "very negative": "Very Negative",
}

THREE_CLASS_FROM_FIVE = {
    "Very Positive": "positive",
    "Positive": "positive",
    "Neutral": "neutral",
    "Negative": "negative",
    "Very Negative": "negative",
}


def map_label_to_5class(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    key = str(value).strip()
    if key.endswith(".0"):
        key = key[:-2]
    return FIVE_CLASS_LABEL_MAP.get(key.lower())


def map_label_to_3class(value: object) -> str | None:
    five_class = map_label_to_5class(value)
    if five_class is None:
        return None
    return THREE_CLASS_FROM_FIVE[five_class]


def add_3class_and_5class_labels(frame: pd.DataFrame, source_column: str = "label_raw") -> pd.DataFrame:
    labeled = frame.copy()
    if source_column not in labeled.columns:
        raise ValueError(f"Missing label source column: {source_column}")
    labeled["label_5class"] = labeled[source_column].map(map_label_to_5class)
    labeled["sentiment_label"] = labeled[source_column].map(map_label_to_3class)
    return labeled[labeled["label_5class"].notna() & labeled["sentiment_label"].notna()].reset_index(drop=True)


def _normalize_explicit_label(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return EXPLICIT_SENTIMENT_MAP.get(text)


def construct_sentiment_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    labeled = frame.copy()
    info: dict[str, Any] = {
        "sentiment_available": False,
        "sentiment_source": "none",
        "reason": "No explicit sentiment labels or numeric ratings were found.",
    }

    if "explicit_label" in labeled.columns:
        mapped = labeled["explicit_label"].map(map_label_to_3class)
        if mapped.notna().any():
            labeled["sentiment_label"] = mapped
            labeled["label_5class"] = labeled["explicit_label"].map(map_label_to_5class)
            labeled = labeled[labeled["sentiment_label"].notna()].copy()
            info.update(
                {
                    "sentiment_available": True,
                    "sentiment_source": "explicit_label",
                    "reason": "Explicit sentiment labels were mapped to three classes.",
                }
            )
            return labeled.reset_index(drop=True), info

    if "rating" in labeled.columns:
        ratings = pd.to_numeric(labeled["rating"], errors="coerce")
        if ratings.notna().any():
            labeled["rating"] = ratings
            labeled["sentiment_label"] = pd.NA
            labeled.loc[ratings >= 4, "sentiment_label"] = "positive"
            labeled.loc[ratings == 3, "sentiment_label"] = "neutral"
            labeled.loc[ratings <= 2, "sentiment_label"] = "negative"
            labeled = labeled[labeled["sentiment_label"].notna()].copy()
            info.update(
                {
                    "sentiment_available": True,
                    "sentiment_source": "rating",
                    "reason": "Ratings were mapped to three sentiment classes by AGENTS.md thresholds.",
                }
            )
            return labeled.reset_index(drop=True), info

    return labeled.reset_index(drop=True), info


def feature_columns(frame: pd.DataFrame, sentiment_source: str) -> list[str]:
    columns = ["clean_text"] if "clean_text" in frame.columns else ["review_text"]
    if sentiment_source == "rating":
        return columns
    return columns


def _stratify_or_none(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if not column or column not in frame.columns:
        return None
    counts = frame[column].value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return frame[column]


def create_splits(
    frame: pd.DataFrame,
    split_dir: Path,
    stratify_col: str | None = "sentiment_label",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> dict[str, pd.DataFrame]:
    split_dir.mkdir(parents=True, exist_ok=True)
    working = frame.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    stratify = _stratify_or_none(working, stratify_col)
    train_val, test = train_test_split(
        working,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_val_stratify = _stratify_or_none(train_val, stratify_col)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=train_val_stratify,
    )
    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    for name, part in splits.items():
        part.to_csv(split_dir / f"{name}.csv", index=False)
    return splits


def create_course_holdout(frame: pd.DataFrame, split_dir: Path) -> dict[str, pd.DataFrame] | None:
    if "course_id" not in frame.columns or frame["course_id"].dropna().nunique() < 2:
        return None
    courses = pd.Series(frame["course_id"].dropna().unique()).sample(frac=1.0, random_state=RANDOM_STATE)
    holdout_count = max(1, int(round(len(courses) * 0.2)))
    holdout_courses = set(courses.iloc[:holdout_count])
    holdout = frame[frame["course_id"].isin(holdout_courses)].copy()
    train = frame[~frame["course_id"].isin(holdout_courses)].copy()
    train.to_csv(split_dir / "course_holdout_train.csv", index=False)
    holdout.to_csv(split_dir / "course_holdout_test.csv", index=False)
    return {"course_holdout_train": train, "course_holdout_test": holdout}
