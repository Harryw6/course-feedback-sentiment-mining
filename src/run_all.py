from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .ablation import run_main_ablation
from .build_labels import construct_sentiment_labels, create_course_holdout, create_splits
from .config import (
    BASE_DIR,
    CLEANED_DIR,
    FIGURE_DIR,
    MULTILABEL_COLUMNS,
    PAPER_DIR,
    PROVENANCE_PATH,
    QUICK_SAMPLE_SIZE,
    RANDOM_STATE,
    REPORT_DIR,
    REQUIRED_FIGURES,
    REQUIRED_REPORTS,
    REQUIRED_TABLES,
    SPLITS_DIR,
    TABLE_DIR,
    ensure_project_dirs,
)
from .error_analysis import generate_error_cases
from .load_data import load_experiment_datasets
from .preprocess import preprocess_reviews
from .topic_modeling import run_lda_topics
from .train_classical import run_classical_experiments, train_multilabel_classifier
from .train_transformer import run_transformer_baselines
from .visualization import generate_all_figures


def _run_source_verifier() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/verify_data_sources.py"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)


def _sample_for_mode(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "quick" and len(frame) > QUICK_SAMPLE_SIZE:
        return frame.sample(n=QUICK_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    return frame.reset_index(drop=True)


def _write_data_summary(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    manifest: dict[str, Any],
    sentiment_info: dict[str, Any],
    mode: str,
) -> pd.DataFrame:
    rows = []
    for record in manifest.get("records", []):
        dataset_id = record.get("dataset_id", "unknown")
        rows.append(
            {
                "mode": mode,
                "dataset_id": dataset_id,
                "source_platform": record.get("source_platform", ""),
                "license": record.get("license", ""),
                "license_status": record.get("license_status", ""),
                "raw_rows_loaded": int((raw["source_dataset"] == dataset_id).sum()) if "source_dataset" in raw else len(raw),
                "cleaned_rows_used": int((cleaned["source_dataset"] == dataset_id).sum())
                if "source_dataset" in cleaned
                else len(cleaned),
                "sentiment_source": sentiment_info.get("sentiment_source", "none"),
                "sentiment_available": sentiment_info.get("sentiment_available", False),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "data_summary.csv", index=False)
    return summary


def _write_label_distributions(frame: pd.DataFrame, sentiment_info: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "sentiment_label" in frame.columns:
        for label, count in frame["sentiment_label"].value_counts().items():
            rows.append({"label_type": "3class", "label": label, "count": int(count), "source": sentiment_info.get("sentiment_source", ""), "notes": ""})
    if "label_5class" in frame.columns:
        for label, count in frame["label_5class"].value_counts().items():
            rows.append({"label_type": "5class", "label": label, "count": int(count), "source": sentiment_info.get("sentiment_source", ""), "notes": ""})
    if not rows:
        rows.append(
            {
                "label_type": "sentiment",
                "label": "sentiment_not_available",
                "count": len(frame),
                "source": "none",
                "notes": sentiment_info.get("reason", "Sentiment labels unavailable."),
            }
        )
    label_distribution = pd.DataFrame(rows)
    label_distribution.to_csv(TABLE_DIR / "label_distribution.csv", index=False)

    multilabel_rows = []
    for col in MULTILABEL_COLUMNS:
        if col in frame.columns:
            multilabel_rows.append({"label": col, "positive_count": int(frame[col].fillna(0).astype(int).sum())})
    if multilabel_rows:
        pd.DataFrame(multilabel_rows).to_csv(TABLE_DIR / "multilabel_label_distribution.csv", index=False)
    return label_distribution


def _split_frame(frame: pd.DataFrame, sentiment_info: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if sentiment_info.get("sentiment_available", False):
        stratify_col = "sentiment_label"
    elif "not_praise" in frame.columns and frame["not_praise"].nunique() > 1:
        stratify_col = "not_praise"
    else:
        stratify_col = None
    splits = create_splits(frame, SPLITS_DIR, stratify_col=stratify_col)
    create_course_holdout(frame, SPLITS_DIR)
    return splits


def _format_table(frame: pd.DataFrame, max_rows: int = 10) -> str:
    if frame.empty:
        return "Empty table."
    return "```\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def _markdown_table(path: Path) -> str:
    if not path.exists():
        return "Not generated."
    frame = pd.read_csv(path)
    if frame.empty:
        return "Empty table."
    return _format_table(frame)


def _write_ethics_report(manifest: dict[str, Any]) -> None:
    lines = [
        "# Data Ethics and License Check",
        "",
        "This report was generated by executed project code.",
        "",
        "## Policy Checks",
        "",
        "- No website scraping was performed by this project.",
        "- Raw datasets are stored under `data/raw/`, which is ignored by Git.",
        "- Kaggle credentials, API tokens, model weights, and raw data are ignored by Git.",
        "- Sentiment labels are only constructed from explicit labels or ratings when present.",
        "- Rating is excluded from model features whenever it is used to construct labels.",
        "",
        "## Dataset License Status",
        "",
    ]
    for record in manifest.get("records", []):
        lines.extend(
            [
                f"### {record.get('name', record.get('dataset_id', 'dataset'))}",
                "",
                f"- Dataset ID: `{record.get('dataset_id', '')}`",
                f"- Source: {record.get('source_url', '')}",
                f"- License: {record.get('license', '')}",
                f"- License status: {record.get('license_status', '')}",
                "",
            ]
        )
        if "not specified" in str(record.get("license", "")).lower():
            lines.append(
                "Publication caution: this dataset is public, but an explicit license was not found in the available metadata. Verify reuse rights before EI paper submission."
            )
            lines.append("")
    errors = manifest.get("download_errors", [])
    if errors:
        lines.extend(["## Download Attempt Notes", ""])
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")
    (REPORT_DIR / "data_ethics_and_license_check.md").write_text("\n".join(lines), encoding="utf-8")


def _write_experiment_summary(
    mode: str,
    data_summary: pd.DataFrame,
    label_distribution: pd.DataFrame,
    classical: dict[str, pd.DataFrame | None],
    transformer_results: pd.DataFrame,
    topics: dict[str, pd.DataFrame],
    ablation_results: pd.DataFrame,
    error_cases: pd.DataFrame,
    sentiment_info: dict[str, Any],
) -> None:
    lines = [
        "# Experiment Summary",
        "",
        f"Mode: `{mode}`",
        "",
        "All numbers in this report were generated by executed code in `python -m src.run_all`.",
        "",
        "## Data Summary",
        "",
        _format_table(data_summary),
        "",
        "## Label Availability",
        "",
        _format_table(label_distribution),
        "",
        f"Sentiment source: `{sentiment_info.get('sentiment_source', 'none')}`.",
        f"Sentiment note: {sentiment_info.get('reason', '')}",
        "",
        "## Sentiment Baselines",
        "",
        _markdown_table(TABLE_DIR / "classical_baseline_results.csv"),
        "",
        "## Auxiliary Multi-label Feedback Baseline",
        "",
        _markdown_table(TABLE_DIR / "auxiliary_multilabel_results.csv"),
        "",
        "## Transformer Baselines",
        "",
        _format_table(transformer_results),
        "",
        "## Topic Modeling",
        "",
        _format_table(topics["topic_summary"]),
        "",
        "## Ablation",
        "",
        _format_table(ablation_results),
        "",
        "## Error Analysis Sample",
        "",
        _format_table(error_cases),
        "",
        "## Honest Limitations",
        "",
        "- Kaggle 100K is the main sentiment dataset; labels are mapped from the dataset's 1-5 Label field into 5-class and 3-class targets.",
        "- Kaggle 1.45M is treated as an optional weak-label rating dataset; rating is not used as an input feature.",
        "- The Hugging Face dataset is auxiliary only and is not reported as sentiment analysis.",
        "- Transformer fine-tuning is run only when torch/transformers and suitable hardware are available; otherwise `transformer_results.csv` records the limitation.",
        "- Topic modeling uses LDA; BERTopic was not run because it is not installed in the environment.",
        "",
    ]
    (REPORT_DIR / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_error_report(error_cases: pd.DataFrame) -> None:
    lines = [
        "# Error Analysis",
        "",
        "Error cases below are sampled from executed model predictions where available.",
        "",
        _format_table(error_cases, max_rows=30),
        "",
        "Error categories include polarity flips, neutral confusion, short reviews, mixed-sentiment cues, and course-specific errors where course_id is available.",
        "",
    ]
    (REPORT_DIR / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _write_auxiliary_report(auxiliary_results: pd.DataFrame, aux_frame: pd.DataFrame) -> None:
    lines = [
        "# Auxiliary Multi-label Feedback-type Experiment",
        "",
        "This is an auxiliary experiment using the Hugging Face `chillies/course-review-multilabel-sentiment-analysis` dataset.",
        "It is not presented as the main sentiment analysis experiment.",
        "",
        f"Rows used: {len(aux_frame)}",
        "",
        "## Results",
        "",
        _format_table(auxiliary_results),
        "",
        "## Limitation",
        "",
        "The dataset has feedback-type labels and no explicit sentiment labels or ratings. Its license was not explicit in available metadata and must be verified before publication reuse.",
        "",
    ]
    (REPORT_DIR / "auxiliary_multilabel_experiment.md").write_text("\n".join(lines), encoding="utf-8")


def _write_publication_results(
    data_summary: pd.DataFrame,
    classical: dict[str, pd.DataFrame | None],
    topics: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# Publication-ready Results",
        "",
        "These tables are generated from executed code and should be copied into the paper only after license review.",
        "",
        "## Dataset Table",
        "",
        _format_table(data_summary),
        "",
        "## Sentiment Classification Table",
        "",
        _markdown_table(TABLE_DIR / "classical_baseline_results.csv"),
        "",
        "## Auxiliary Multi-label Feedback Classification Table",
        "",
        _markdown_table(TABLE_DIR / "auxiliary_multilabel_results.csv"),
        "",
        "## Topic Summary Table",
        "",
        _format_table(topics["topic_summary"]),
        "",
        "## Teaching Improvement Matrix",
        "",
        _markdown_table(TABLE_DIR / "teaching_improvement_matrix.csv"),
        "",
    ]
    (REPORT_DIR / "publication_ready_results.md").write_text("\n".join(lines), encoding="utf-8")


def _write_paper_draft(manifest: dict[str, Any], sentiment_info: dict[str, Any]) -> None:
    datasets = ", ".join(record.get("dataset_id", "") for record in manifest.get("records", []))
    lines = [
        "# Experimental Section Draft",
        "",
        "## Data",
        "",
        f"The experiment used the public dataset(s): {datasets}. Dataset provenance, source URLs, license status, downloaded files, and limitations are recorded in `outputs/reports/DATA_PROVENANCE.md`.",
        "",
        "## Preprocessing",
        "",
        "Reviews were cleaned by removing empty entries, deduplicating identical cleaned text, normalizing whitespace, and masking URLs, email addresses, phone numbers, and simple self-identification patterns.",
        "",
        "## Label Construction",
        "",
    ]
    if sentiment_info.get("sentiment_available", False):
        lines.append(
            "The main Kaggle Coursera dataset provides a 1-5 `Label` field. We map 5/4/3/2/1 to Very Positive, Positive, Neutral, Negative, and Very Negative for the 5-class task, and then map Very Positive/Positive to positive, Neutral to neutral, and Negative/Very Negative to negative for the 3-class task."
        )
    else:
        lines.append(
        "The main Kaggle Coursera dataset provides a 1-5 `Label` field. We map 5/4/3/2/1 to Very Positive, Positive, Neutral, Negative, and Very Negative for the 5-class task, and then map Very Positive/Positive to positive, Neutral to neutral, and Negative/Very Negative to negative for the 3-class task."
        )
    lines.extend(
        [
            "",
            "## Models",
            "",
            "The main experiment trains TF-IDF classical sentiment baselines on Kaggle Coursera review data. Transformer baselines are implemented and reported when dependencies and hardware allow. The Hugging Face dataset is used only as an auxiliary multi-label feedback-type experiment. LDA is used for topic mining and the teaching improvement matrix.",
            "",
            "## Results",
            "",
            "Use `outputs/reports/publication_ready_results.md` for the generated tables. Do not report metrics that are marked `not_run`.",
            "",
            "## Limitations",
            "",
            "The optional 1.45M Kaggle dataset uses rating-derived weak labels, so rating is never used as a model feature. The Hugging Face auxiliary dataset is public but its license was not explicit in available metadata, so reuse rights must be checked before submission.",
            "",
        ]
    )
    (PAPER_DIR / "experiment_section_draft.md").write_text("\n".join(lines), encoding="utf-8")


def _verify_required_outputs() -> pd.DataFrame:
    rows = []
    for name in REQUIRED_TABLES:
        path = TABLE_DIR / name
        rows.append({"path": str(path.relative_to(BASE_DIR)), "exists": path.exists(), "type": "table"})
    for name in REQUIRED_FIGURES:
        path = FIGURE_DIR / name
        rows.append({"path": str(path.relative_to(BASE_DIR)), "exists": path.exists(), "type": "figure"})
    for name in REQUIRED_REPORTS:
        path = REPORT_DIR / name
        rows.append({"path": str(path.relative_to(BASE_DIR)), "exists": path.exists(), "type": "report"})
    paper_path = PAPER_DIR / "experiment_section_draft.md"
    rows.append({"path": str(paper_path.relative_to(BASE_DIR)), "exists": paper_path.exists(), "type": "paper"})
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "output_manifest.csv", index=False)
    missing = result[~result["exists"]]
    if not missing.empty:
        raise RuntimeError(f"Missing required outputs: {missing['path'].tolist()}")
    return result


def run_pipeline(mode: str) -> dict[str, Any]:
    ensure_project_dirs()
    _run_source_verifier()
    datasets, manifest = load_experiment_datasets()
    main_raw = datasets["main_kaggle_100k"]
    weak_raw = datasets["weak_kaggle_145m"]
    aux_raw = datasets["aux_hf_multilabel"]

    if main_raw.empty:
        raise RuntimeError(
            "Kaggle 100K Coursera dataset is not available. Main sentiment experiment stopped; "
            "see outputs/reports/DATA_PROVENANCE.md for download errors."
        )

    main_selected = _sample_for_mode(main_raw, mode)
    main_cleaned = preprocess_reviews(main_selected)
    sentiment_info = {
        "sentiment_available": True,
        "sentiment_source": "kaggle_100k_label",
        "reason": "Kaggle 100K Label field mapped to 5-class and 3-class sentiment targets.",
    }
    experiment_frame = main_cleaned
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    experiment_frame.to_csv(CLEANED_DIR / "cleaned_feedback.csv", index=False)
    experiment_frame.to_csv(CLEANED_DIR / "cleaned_feedback_3class.csv", index=False)
    experiment_frame.to_csv(CLEANED_DIR / "cleaned_feedback_5class.csv", index=False)

    if weak_raw.empty:
        weak_cleaned = pd.DataFrame()
    else:
        weak_for_ablation = weak_raw.sample(
            n=min(len(weak_raw), 5000 if mode == "quick" else 60000),
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)
        weak_cleaned = preprocess_reviews(weak_for_ablation)

    aux_cleaned = preprocess_reviews(aux_raw) if not aux_raw.empty else pd.DataFrame()

    summary_raw = pd.concat([frame for frame in [main_raw, weak_raw, aux_raw] if not frame.empty], ignore_index=True)
    summary_cleaned = pd.concat(
        [frame for frame in [main_cleaned, weak_cleaned, aux_cleaned] if not frame.empty],
        ignore_index=True,
    )

    data_summary = _write_data_summary(summary_raw, summary_cleaned, manifest, sentiment_info, mode)
    label_distribution = _write_label_distributions(experiment_frame, sentiment_info)
    splits = _split_frame(experiment_frame, sentiment_info)
    classical = run_classical_experiments(splits, sentiment_info)
    transformer_results = run_transformer_baselines(sentiment_info, mode)
    topics = run_lda_topics(experiment_frame, max_topics=5 if mode == "quick" else 8)
    ablation_results = run_main_ablation(experiment_frame, weak_cleaned, mode)

    if aux_cleaned.empty:
        auxiliary_results = pd.DataFrame(
            [
                {
                    "model": "TF-IDF + One-vs-Rest Logistic Regression",
                    "status": "not_run",
                    "micro_f1": pd.NA,
                    "macro_f1": pd.NA,
                    "hamming_loss": pd.NA,
                    "subset_accuracy": pd.NA,
                    "notes": "HF auxiliary dataset was not available.",
                }
            ]
        )
        auxiliary_predictions = None
        auxiliary_results.to_csv(TABLE_DIR / "auxiliary_multilabel_results.csv", index=False)
    else:
        aux_splits = create_splits(aux_cleaned, SPLITS_DIR / "auxiliary_multilabel", stratify_col="not_praise")
        auxiliary = train_multilabel_classifier(aux_splits)
        auxiliary_results = auxiliary["results"] if isinstance(auxiliary["results"], pd.DataFrame) else pd.DataFrame()
        auxiliary_predictions = auxiliary.get("predictions")
        auxiliary_results.to_csv(TABLE_DIR / "auxiliary_multilabel_results.csv", index=False)
        if isinstance(auxiliary_predictions, pd.DataFrame):
            auxiliary_predictions.to_csv(TABLE_DIR / "auxiliary_multilabel_predictions.csv", index=False)
        if isinstance(auxiliary.get("label_report"), pd.DataFrame):
            auxiliary["label_report"].to_csv(TABLE_DIR / "auxiliary_multilabel_label_report.csv", index=False)

    error_cases = generate_error_cases(
        classical.get("sentiment_predictions"),
        auxiliary_predictions,
    )
    generate_all_figures(
        label_distribution,
        classical["sentiment_results"],
        classical.get("sentiment_predictions"),
        classical.get("multilabel_results"),
        topics["topic_sentiment"],
        experiment_frame,
    )
    _write_ethics_report(manifest)
    _write_experiment_summary(
        mode,
        data_summary,
        label_distribution,
        classical,
        transformer_results,
        topics,
        ablation_results,
        error_cases,
        sentiment_info,
    )
    _write_error_report(error_cases)
    _write_auxiliary_report(auxiliary_results, aux_cleaned)
    _write_publication_results(data_summary, classical, topics)
    _write_paper_draft(manifest, sentiment_info)
    output_manifest = _verify_required_outputs()

    return {
        "mode": mode,
        "manifest": manifest,
        "data_summary": data_summary,
        "label_distribution": label_distribution,
        "classical": classical,
        "transformer_results": transformer_results,
        "topics": topics,
        "ablation_results": ablation_results,
        "auxiliary_results": auxiliary_results,
        "error_cases": error_cases,
        "output_manifest": output_manifest,
        "sentiment_info": sentiment_info,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible course-feedback experiment pipeline.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args(argv)
    result = run_pipeline(args.mode)
    summary = {
        "mode": result["mode"],
        "datasets": [record["dataset_id"] for record in result["manifest"].get("records", [])],
        "sentiment_available": result["sentiment_info"].get("sentiment_available", False),
        "outputs_verified": int(result["output_manifest"]["exists"].sum()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
