from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BASE_DIR, MANIFEST_PATH, MULTILABEL_COLUMNS
from .build_labels import add_3class_and_5class_labels


TEXT_CANDIDATES = ["review_text", "review", "Review", "reviews", "Reviews", "comment", "text", "feedback"]
RATING_CANDIDATES = ["rating", "Rating", "ratings", "star", "stars", "score"]
LABEL_CANDIDATES = ["sentiment", "Sentiment", "label", "Label", "polarity"]
COURSE_CANDIDATES = ["course_id", "CourseId", "course id", "course", "course_name", "Course Name", "CourseTitle"]


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data manifest at {path}. Run `python scripts/download_public_data.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def _normalize_frame(frame: pd.DataFrame, record: dict[str, Any], source_file: str) -> pd.DataFrame:
    text_col = _first_existing(list(frame.columns), TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError(f"No review text column found in {source_file}")
    normalized = pd.DataFrame()
    normalized["review_text"] = frame[text_col]
    normalized["source_dataset"] = record.get("dataset_id", "unknown")
    normalized["source_platform"] = record.get("source_platform", "unknown")
    normalized["source_file"] = source_file
    normalized["source_row_id"] = range(len(frame))

    rating_col = _first_existing(list(frame.columns), RATING_CANDIDATES)
    if rating_col is not None:
        normalized["rating"] = frame[rating_col]

    label_col = _first_existing(list(frame.columns), LABEL_CANDIDATES)
    if label_col is not None:
        label_values = frame[label_col]
        numeric_fraction = pd.to_numeric(label_values, errors="coerce").notna().mean()
        if numeric_fraction > 0.9 and "rating" not in normalized.columns:
            normalized["rating"] = label_values
        else:
            normalized["explicit_label"] = label_values

    course_col = _first_existing(list(frame.columns), COURSE_CANDIDATES)
    if course_col is not None:
        normalized["course_id"] = frame[course_col].astype(str)

    for col in MULTILABEL_COLUMNS:
        if col in frame.columns:
            normalized[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)

    return normalized


def load_public_reviews(manifest_path: Path = MANIFEST_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    frames: list[pd.DataFrame] = []
    for record in manifest.get("records", []):
        for file_path in record.get("files_downloaded", []):
            path = BASE_DIR / file_path
            if path.suffix.lower() != ".csv":
                continue
            frame = pd.read_csv(path)
            frames.append(_normalize_frame(frame, record, file_path))
    if not frames:
        raise RuntimeError("No CSV review files were found in downloaded public datasets.")
    combined = pd.concat(frames, ignore_index=True)
    return combined, manifest


def _record_by_id(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any] | None:
    for record in manifest.get("records", []):
        if record.get("dataset_id") == dataset_id:
            return record
    return None


def _manifest_file(record: dict[str, Any], filename: str) -> Path | None:
    for relative in record.get("files_downloaded", []):
        path = BASE_DIR / relative
        if path.name.lower() == filename.lower() and path.exists():
            return path
    return None


def load_kaggle_100k_reviews(manifest: dict[str, Any] | None = None) -> pd.DataFrame:
    manifest = manifest or load_manifest()
    record = _record_by_id(manifest, "septa97/100k-courseras-course-reviews-dataset")
    if record is None:
        return pd.DataFrame()
    path = _manifest_file(record, "reviews_by_course.csv") or _manifest_file(record, "reviews.tsv")
    if path is None:
        path = _manifest_file(record, "reviews.csv")
    if path is None:
        return pd.DataFrame()

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    raw = pd.read_csv(path, sep=sep)
    text_col = _first_existing(list(raw.columns), ["Review", "review"])
    label_col = _first_existing(list(raw.columns), ["Label", "label"])
    if text_col is None or label_col is None:
        raise ValueError(f"Kaggle 100K file missing Review/Label fields: {path}")
    normalized = pd.DataFrame(
        {
            "review_text": raw[text_col],
            "label_raw": raw[label_col],
            "source_dataset": record["dataset_id"],
            "source_platform": "Kaggle",
            "source_file": str(path.relative_to(BASE_DIR)),
        }
    )
    course_col = _first_existing(list(raw.columns), ["CourseId", "course_id", "course"])
    if course_col is not None:
        normalized["course_id"] = raw[course_col].astype(str)
    if "Id" in raw.columns:
        normalized["review_id"] = raw["Id"]
    labeled = add_3class_and_5class_labels(normalized, "label_raw")
    labeled["label_source"] = "kaggle_100k_explicit_label"
    return labeled


def load_kaggle_145m_reviews(manifest: dict[str, Any] | None = None) -> pd.DataFrame:
    manifest = manifest or load_manifest()
    record = _record_by_id(manifest, "imuhammad/course-reviews-on-coursera")
    if record is None:
        return pd.DataFrame()
    path = _manifest_file(record, "Coursera_reviews.csv")
    if path is None:
        return pd.DataFrame()
    raw = pd.read_csv(path)
    required = {"reviews", "rating", "course_id"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Kaggle 1.45M file missing fields {sorted(missing)}: {path}")
    normalized = pd.DataFrame(
        {
            "review_text": raw["reviews"],
            "label_raw": raw["rating"],
            "rating": pd.to_numeric(raw["rating"], errors="coerce"),
            "course_id": raw["course_id"].astype(str),
            "review_date": raw["date_reviews"] if "date_reviews" in raw.columns else pd.NA,
            "source_dataset": record["dataset_id"],
            "source_platform": "Kaggle",
            "source_file": str(path.relative_to(BASE_DIR)),
        }
    )
    labeled = add_3class_and_5class_labels(normalized, "label_raw")
    labeled["label_source"] = "kaggle_145m_rating_weak_label"
    return labeled


def load_hf_auxiliary_reviews(manifest: dict[str, Any] | None = None) -> pd.DataFrame:
    manifest = manifest or load_manifest()
    record = _record_by_id(manifest, "chillies/course-review-multilabel-sentiment-analysis")
    if record is None:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for relative in record.get("files_downloaded", []):
        path = BASE_DIR / relative
        if path.suffix.lower() != ".csv" or not path.exists():
            continue
        raw = pd.read_csv(path)
        if "review" not in raw.columns:
            continue
        normalized = pd.DataFrame(
            {
                "review_text": raw["review"],
                "source_dataset": record["dataset_id"],
                "source_platform": "Hugging Face",
                "source_file": str(path.relative_to(BASE_DIR)),
            }
        )
        for column in MULTILABEL_COLUMNS:
            if column in raw.columns:
                normalized[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0).astype(int)
        frames.append(normalized)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_experiment_datasets(manifest_path: Path = MANIFEST_PATH) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    datasets = {
        "main_kaggle_100k": load_kaggle_100k_reviews(manifest),
        "weak_kaggle_145m": load_kaggle_145m_reviews(manifest),
        "aux_hf_multilabel": load_hf_auxiliary_reviews(manifest),
    }
    return datasets, manifest
