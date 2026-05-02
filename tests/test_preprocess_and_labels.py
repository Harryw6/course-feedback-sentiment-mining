from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.build_labels import (
    construct_sentiment_labels,
    create_splits,
    feature_columns,
    map_label_to_3class,
    map_label_to_5class,
)
from src.preprocess import clean_text, preprocess_reviews


class PreprocessAndLabelTests(unittest.TestCase):
    def test_clean_text_masks_private_contact_patterns(self) -> None:
        raw = "Email me at student@example.com or visit https://example.com. Call +1 555-123-4567."
        cleaned = clean_text(raw)
        self.assertIn("[EMAIL]", cleaned)
        self.assertIn("[URL]", cleaned)
        self.assertIn("[PHONE]", cleaned)
        self.assertNotIn("student@example.com", cleaned)
        self.assertNotIn("https://example.com", cleaned)

    def test_construct_sentiment_labels_from_explicit_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "review_text": ["a", "b", "c", "d", "e"],
                "explicit_label": ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"],
            }
        )
        labeled, info = construct_sentiment_labels(frame)
        self.assertEqual(info["sentiment_source"], "explicit_label")
        self.assertEqual(labeled["sentiment_label"].tolist(), ["positive", "positive", "neutral", "negative", "negative"])

    def test_construct_sentiment_labels_from_rating(self) -> None:
        frame = pd.DataFrame({"review_text": ["a", "b", "c"], "rating": [5, 3, 1]})
        labeled, info = construct_sentiment_labels(frame)
        self.assertEqual(info["sentiment_source"], "rating")
        self.assertEqual(labeled["sentiment_label"].tolist(), ["positive", "neutral", "negative"])

    def test_numeric_kaggle_labels_map_to_5class_and_3class(self) -> None:
        self.assertEqual(map_label_to_5class(5), "Very Positive")
        self.assertEqual(map_label_to_5class(4), "Positive")
        self.assertEqual(map_label_to_5class(3), "Neutral")
        self.assertEqual(map_label_to_5class(2), "Negative")
        self.assertEqual(map_label_to_5class(1), "Very Negative")
        self.assertEqual(map_label_to_3class("Very Positive"), "positive")
        self.assertEqual(map_label_to_3class("Positive"), "positive")
        self.assertEqual(map_label_to_3class("Neutral"), "neutral")
        self.assertEqual(map_label_to_3class("Negative"), "negative")
        self.assertEqual(map_label_to_3class("Very Negative"), "negative")

    def test_rating_is_not_feature_when_used_for_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "clean_text": ["great", "fine", "bad"],
                "rating": [5, 3, 1],
                "sentiment_label": ["positive", "neutral", "negative"],
            }
        )
        self.assertEqual(feature_columns(frame, sentiment_source="rating"), ["clean_text"])

    def test_preprocess_removes_empty_and_duplicate_reviews(self) -> None:
        frame = pd.DataFrame(
            {
                "review_text": ["Great course!", "Great course!", "   ", None],
                "source_dataset": ["x", "x", "x", "x"],
            }
        )
        cleaned = preprocess_reviews(frame)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["clean_text"], "Great course!")

    def test_create_splits_writes_train_val_test(self) -> None:
        rows = []
        for label in ["positive", "neutral", "negative"]:
            for idx in range(12):
                rows.append({"clean_text": f"{label} {idx}", "sentiment_label": label, "review_text": f"{label} {idx}"})
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp)
            splits = create_splits(frame, split_dir=split_dir, stratify_col="sentiment_label")
            self.assertEqual(set(splits), {"train", "val", "test"})
            self.assertTrue((split_dir / "train.csv").exists())
            self.assertTrue((split_dir / "val.csv").exists())
            self.assertTrue((split_dir / "test.csv").exists())
            self.assertEqual(sum(len(part) for part in splits.values()), len(frame))


if __name__ == "__main__":
    unittest.main()
