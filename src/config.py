from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = OUTPUT_DIR / "models"
PAPER_DIR = BASE_DIR / "paper"

MANIFEST_PATH = RAW_DIR / "download_manifest.json"
PROVENANCE_PATH = REPORT_DIR / "DATA_PROVENANCE.md"

RANDOM_STATE = 42
QUICK_SAMPLE_SIZE = 1500
MIN_TOPIC_DOCUMENTS = 20

TEXT_COLUMNS = ["review_text", "clean_text"]
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
MULTILABEL_COLUMNS = [
    "Improvement Suggestions",
    "Questions and Answers",
    "Experience Sharing",
    "Technical Feedback",
    "Support Request",
    "Community Interaction",
    "Course Comparison",
    "Related Course Suggestions",
    "not_praise",
]

REQUIRED_TABLES = [
    "data_summary.csv",
    "label_distribution.csv",
    "classical_baseline_results.csv",
    "transformer_results.csv",
    "topic_summary.csv",
    "teaching_improvement_matrix.csv",
    "ablation_results.csv",
    "error_cases.csv",
    "auxiliary_multilabel_results.csv",
]

REQUIRED_FIGURES = [
    "label_distribution.png",
    "model_comparison.png",
    "confusion_matrix_best_model.png",
    "topic_sentiment_heatmap.png",
    "course_sentiment_distribution.png",
]

REQUIRED_REPORTS = [
    "DATA_PROVENANCE.md",
    "experiment_summary.md",
    "data_ethics_and_license_check.md",
    "error_analysis.md",
    "publication_ready_results.md",
    "auxiliary_multilabel_experiment.md",
]


def ensure_project_dirs() -> None:
    for path in [
        RAW_DIR,
        CLEANED_DIR,
        SPLITS_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        REPORT_DIR,
        MODEL_DIR,
        PAPER_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
