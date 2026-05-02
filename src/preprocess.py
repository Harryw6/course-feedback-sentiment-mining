from __future__ import annotations

import re

import pandas as pd


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
NAME_RE = re.compile(r"\b(my name is|i am|i'm)\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = URL_RE.sub("[URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = NAME_RE.sub(lambda match: f"{match.group(1)} [NAME]", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def preprocess_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    if "review_text" not in frame.columns:
        raise ValueError("preprocess_reviews requires a review_text column")
    cleaned = frame.copy()
    cleaned["clean_text"] = cleaned["review_text"].map(clean_text)
    before = len(cleaned)
    cleaned = cleaned[cleaned["clean_text"].str.len() > 0].copy()
    cleaned = cleaned.drop_duplicates(subset=["clean_text"]).copy()
    cleaned["preprocess_removed_rows"] = before - len(cleaned)
    return cleaned.reset_index(drop=True)
