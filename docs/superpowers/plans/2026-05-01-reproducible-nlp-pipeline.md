# Reproducible NLP Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible MOOC course-review sentiment, topic, and teaching-improvement pipeline using only public datasets and executed results.

**Architecture:** The project uses a small Python package under `src/` plus two public-data scripts under `scripts/`. The downloader records provenance before modeling; `src.run_all` orchestrates cleaning, label construction, splits, classical baselines, optional Transformer reporting, topic modeling, ablations, error analysis, figures, reports, and the paper draft.

**Tech Stack:** Python 3, pandas, scikit-learn, matplotlib/seaborn, Kaggle CLI when credentials exist, Hugging Face direct file endpoints as fallback.

---

### Task 1: Public Data Acquisition

**Files:**
- Create: `scripts/download_public_data.py`
- Create: `scripts/verify_data_sources.py`

- [x] Implement a downloader that tries Kaggle `septa97/100k-courseras-course-reviews-dataset` first.
- [x] If Kaggle credentials are available, also try `imuhammad/course-reviews-on-coursera`.
- [x] If Kaggle is unavailable, download the Hugging Face fallback through official file URLs.
- [x] Record source page, dataset ID, license status, command, time, downloaded files, fields, and limitations in `outputs/reports/DATA_PROVENANCE.md`.
- [x] Add a verifier that reads the manifest and fails if source or license metadata is absent.

### Task 2: Core Pipeline

**Files:**
- Create: `src/config.py`
- Create: `src/load_data.py`
- Create: `src/preprocess.py`
- Create: `src/build_labels.py`
- Create: `src/evaluate.py`
- Create: `src/train_classical.py`
- Create: `src/train_transformer.py`
- Create: `src/topic_modeling.py`
- Create: `src/ablation.py`
- Create: `src/error_analysis.py`
- Create: `src/visualization.py`
- Create: `src/run_all.py`

- [ ] Load Kaggle and Hugging Face schemas into a normalized `review_text`, optional `rating`, optional `explicit_label`, optional `course_id` format.
- [ ] Clean text by masking URLs, emails, phone numbers, and obvious self-identifying phrases.
- [ ] Construct sentiment labels only from explicit labels or ratings, never from text.
- [ ] Split data into train/validation/test and course-level holdout where `course_id` exists.
- [ ] Train classical baselines when sentiment labels exist; otherwise emit executed `not_run` tables with reasons.
- [ ] Run LDA topic modeling and topic-summary artifacts when dependencies and data exist.
- [ ] Generate teaching-improvement, ablation, error-analysis, ethics, summary, publication, and paper-draft outputs from executed artifacts.

### Task 3: Verification

**Files:**
- Create: `tests/test_preprocess_and_labels.py`

- [ ] Add minimal unit tests for cleaning, label mapping, leakage-safe feature selection, and split generation.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python scripts/download_public_data.py`.
- [ ] Run `python -m src.run_all --mode quick`.
- [ ] Run `python -m src.run_all --mode full` if dependencies and runtime allow.
