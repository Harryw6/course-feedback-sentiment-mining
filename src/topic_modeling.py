from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .config import MIN_TOPIC_DOCUMENTS, RANDOM_STATE, TABLE_DIR


GENERIC_ISSUES = {
    "assignment": ("Assessment design or peer review clarity", "Review assignment instructions, rubrics, examples, and feedback timing."),
    "quiz": ("Quiz feedback or assessment difficulty", "Audit quiz wording and add targeted explanations for common incorrect answers."),
    "video": ("Lecture pacing or video clarity", "Revise short lecture segments, captions, and pacing where learners report confusion."),
    "forum": ("Discussion/community support", "Increase instructor or teaching-assistant presence in forums and seed discussion prompts."),
    "technical": ("Platform or tool friction", "Provide setup checks, troubleshooting notes, and alternative tool paths."),
    "content": ("Course content relevance or depth", "Align examples and readings with learner goals and add optional advanced material."),
}


def _infer_issue_action(keywords: str) -> tuple[str, str]:
    lowered = keywords.lower()
    for token, value in GENERIC_ISSUES.items():
        if token in lowered:
            return value
    return (
        "General learning experience issue",
        "Review representative comments and prioritize revisions with frequent negative or improvement-oriented feedback.",
    )


def run_lda_topics(frame: pd.DataFrame, max_topics: int = 8) -> dict[str, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if len(frame) < MIN_TOPIC_DOCUMENTS:
        empty = pd.DataFrame(
            [
                {
                    "topic_id": "not_run",
                    "keywords": "",
                    "document_count": len(frame),
                    "representative_comment": "",
                    "topic_diversity": np.nan,
                    "notes": "Not enough documents for topic modeling.",
                }
            ]
        )
        empty.to_csv(TABLE_DIR / "topic_summary.csv", index=False)
        return {
            "topic_summary": empty,
            "document_topics": pd.DataFrame(),
            "topic_sentiment": pd.DataFrame(),
            "teaching_matrix": pd.DataFrame(),
        }

    docs = frame["clean_text"].astype(str)
    n_topics = max(2, min(max_topics, len(frame) // 50 if len(frame) >= 100 else 2))
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=2,
        max_df=0.9,
        max_features=5000,
    )
    matrix = vectorizer.fit_transform(docs)
    if matrix.shape[1] < n_topics:
        n_topics = max(1, matrix.shape[1])
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        learning_method="batch",
        max_iter=10,
    )
    doc_topic = lda.fit_transform(matrix)
    dominant_topics = doc_topic.argmax(axis=1)
    feature_names = np.array(vectorizer.get_feature_names_out())

    summary_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    keyword_sets: list[set[str]] = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        top_indices = topic_weights.argsort()[-10:][::-1]
        keywords = feature_names[top_indices].tolist()
        keyword_sets.append(set(keywords))
        topic_docs = frame.loc[dominant_topics == topic_idx].copy()
        if topic_docs.empty:
            representative = ""
        else:
            local_indices = np.where(dominant_topics == topic_idx)[0]
            best_local = local_indices[doc_topic[local_indices, topic_idx].argmax()]
            representative = frame.iloc[int(best_local)]["clean_text"]
        issue, action = _infer_issue_action(", ".join(keywords))
        sentiment_col = "sentiment_label" if "sentiment_label" in topic_docs.columns else None
        dominant_sentiment = (
            str(topic_docs[sentiment_col].mode().iloc[0])
            if sentiment_col and not topic_docs[sentiment_col].dropna().empty
            else "sentiment_not_available"
        )
        negative_ratio = (
            float((topic_docs[sentiment_col] == "negative").mean())
            if sentiment_col and not topic_docs.empty
            else np.nan
        )
        summary_rows.append(
            {
                "topic_id": topic_idx,
                "keywords": ", ".join(keywords),
                "document_count": int((dominant_topics == topic_idx).sum()),
                "representative_comment": representative[:500],
                "dominant_sentiment": dominant_sentiment,
                "negative_ratio": negative_ratio,
                "topic_diversity": np.nan,
                "bertopic_status": "not_run_dependency_unavailable",
                "notes": "LDA topic generated from cleaned public review text.",
            }
        )
        matrix_rows.append(
            {
                "topic_id": topic_idx,
                "keywords": ", ".join(keywords[:6]),
                "dominant_sentiment": dominant_sentiment,
                "negative_ratio": negative_ratio,
                "representative_comments": representative[:500],
                "likely_teaching_issue": issue,
                "suggested_teaching_improvement_action": action,
            }
        )

    unique_keywords = len(set().union(*keyword_sets)) if keyword_sets else 0
    total_keywords = sum(len(keywords) for keywords in keyword_sets)
    topic_diversity = unique_keywords / total_keywords if total_keywords else np.nan
    topic_summary = pd.DataFrame(summary_rows)
    topic_summary["topic_diversity"] = topic_diversity

    document_topics = frame[["clean_text"]].copy()
    document_topics["topic_id"] = dominant_topics
    if "sentiment_label" in frame.columns:
        document_topics["sentiment_label"] = frame["sentiment_label"].values
    else:
        document_topics["sentiment_label"] = "sentiment_not_available"
    if "course_id" in frame.columns:
        document_topics["course_id"] = frame["course_id"].values

    topic_sentiment = (
        document_topics.groupby(["topic_id", "sentiment_label"], dropna=False)
        .size()
        .reset_index(name="document_count")
    )
    teaching_matrix = pd.DataFrame(matrix_rows)

    topic_summary.to_csv(TABLE_DIR / "topic_summary.csv", index=False)
    topic_sentiment.to_csv(TABLE_DIR / "topic_sentiment_distribution.csv", index=False)
    document_topics.to_csv(TABLE_DIR / "document_topics.csv", index=False)
    teaching_matrix.to_csv(TABLE_DIR / "teaching_improvement_matrix.csv", index=False)
    return {
        "topic_summary": topic_summary,
        "document_topics": document_topics,
        "topic_sentiment": topic_sentiment,
        "teaching_matrix": teaching_matrix,
    }
