# Course Feedback Sentiment Mining

Reproducible academic NLP pipeline for:

**Sentiment Analysis, Topic Mining, and Teaching Improvement Recommendation from Public MOOC Course Reviews.**

The project downloads legal public course-review datasets, records provenance, cleans review text, constructs labels only when labels or ratings exist, trains feasible baselines, mines topics, generates a teaching-improvement matrix, and writes reports for an education analytics paper.

## Dataset Sources

Priority order:

1. Kaggle `septa97/100k-courseras-course-reviews-dataset`
2. Kaggle `imuhammad/course-reviews-on-coursera`
3. Hugging Face `chillies/course-review-multilabel-sentiment-analysis`

The downloader first tries Kaggle. The Kaggle 100K dataset is the main sentiment experiment. The Kaggle 1.45M dataset is an optional weak-label rating extension. The Hugging Face dataset is used only as an auxiliary multi-label feedback-type experiment, not as sentiment analysis. The code does not scrape websites.

## License Status

Dataset provenance is generated at:

`outputs/reports/DATA_PROVENANCE.md`

The Kaggle source pages identify licenses for the preferred Kaggle datasets. The Hugging Face fallback is public, but its license was not explicit in the available metadata at download time, so publication reuse requires manual license confirmation before EI submission.

## Kaggle Credentials

Do not commit credentials. Configure one of:

```bash
mkdir -p ~/.kaggle
# place kaggle.json at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

or set the following in a local `.env` file:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

`.env`, `kaggle.json`, raw data, and model artifacts are ignored by Git.

## Download Data

```bash
python scripts/download_public_data.py
python scripts/verify_data_sources.py
```

If Kaggle credentials are unavailable, the script records the failure and downloads the Hugging Face auxiliary dataset. The main sentiment experiment stops unless Kaggle 100K is available.

## Quick Mode

Quick mode uses a manageable sample, trains feasible baselines, runs LDA topic modeling, and generates all core artifacts.

```bash
python -m src.run_all --mode quick
```

## Full Mode

Full mode uses all downloaded public rows. Transformer fine-tuning runs only if labels, dependencies, and hardware are available; otherwise the limitation is documented in `outputs/tables/transformer_results.csv`.

```bash
python -m src.run_all --mode full
```

## Reproduce Tables and Figures

Run:

```bash
python scripts/download_public_data.py
python -m src.run_all --mode quick
python -m src.run_all --mode full
```

Generated tables:

`outputs/tables/`

Generated figures:

`outputs/figures/`

Generated reports:

`outputs/reports/`

Paper draft:

`paper/experiment_section_draft.md`

The manifest `outputs/tables/output_manifest.csv` records whether required artifacts exist.

## Known Limitations

- Do not report metrics marked `not_run`.
- If only the Hugging Face fallback is available, three-class sentiment classification is not run because the dataset has no explicit sentiment labels or ratings.
- Rating is never used as an input feature when it is used to construct labels.
- BERTopic and Transformer fine-tuning are skipped unless the required packages and hardware are available.
- The Hugging Face fallback license must be verified before publication use.
