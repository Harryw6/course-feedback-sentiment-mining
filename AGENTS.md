# AGENTS.md

You are working on a reproducible academic NLP experiment for an education analytics paper.

Project title:
Sentiment Analysis, Topic Mining, and Teaching Improvement Recommendation from Public MOOC Course Reviews.

Research goal:
Build a complete, reproducible experimental pipeline using legal public course-review datasets. The pipeline should support an EI-indexed education analytics paper.

Core academic rules:
1. Do not fabricate data.
2. Do not fabricate labels.
3. Do not fabricate results.
4. Do not scrape websites.
5. Only use clearly public datasets with identifiable source pages and licenses.
6. All reported numbers must come from executed code.
7. Record dataset source, license, dataset ID, download command, and download time.
8. Do not commit raw datasets, model weights, Kaggle credentials, API tokens, or personal information to Git.

Preferred public datasets:
1. Kaggle 100K Coursera Course Reviews Dataset
   Dataset ID: septa97/100k-courseras-course-reviews-dataset

2. Kaggle Course Reviews on Coursera
   Dataset ID: imuhammad/course-reviews-on-coursera

3. Hugging Face course-review-multilabel-sentiment-analysis
   Dataset ID: chillies/course-review-multilabel-sentiment-analysis

Download policy:
- First try Kaggle 100K Coursera Course Reviews.
- If Kaggle credentials are available, also try the 1.45M Coursera Reviews dataset.
- If Kaggle is unavailable, use the Hugging Face dataset as fallback.
- If no public dataset can be downloaded legally, create manual download instructions and stop.
- Do not create fake research data.
- Dummy data may only be used to test code execution, and must be clearly marked as dummy.

Data provenance:
Create:
outputs/reports/DATA_PROVENANCE.md

It must include:
- dataset name
- dataset ID
- source platform
- license
- download command
- download date/time
- files downloaded
- fields used
- known limitations

Expected research tasks:
1. Download public data.
2. Validate dataset source and license.
3. Clean and preprocess course-review text.
4. Construct sentiment labels.
5. Train classical ML baselines.
6. Train Transformer-based baselines if feasible.
7. Run topic modeling.
8. Generate topic-sentiment analysis.
9. Generate teaching improvement matrix.
10. Run ablation experiments where feasible.
11. Run error analysis.
12. Generate publication-ready tables, figures, and reports.
13. Draft the experimental section for a paper.

Label construction:
If explicit labels exist:
- Very Positive and Positive -> positive
- Neutral -> neutral
- Negative and Very Negative -> negative

If only ratings exist:
- rating >= 4 -> positive
- rating == 3 -> neutral
- rating <= 2 -> negative

Important leakage rule:
If rating is used to construct sentiment labels, rating must not be used as a model input feature.

Preprocessing:
- remove empty reviews
- remove duplicated reviews
- normalize whitespace
- mask emails
- mask phone numbers
- mask URLs
- mask obvious personal identifiers where possible
- save cleaned data to data/cleaned/cleaned_feedback.csv

Splits:
Create:
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv

Use stratified splitting by sentiment label.
If course_id exists, also create a course-level holdout split.

Classical baselines:
Train:
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Random Forest

Transformer baselines:
If runtime and hardware allow, train:
- distilbert-base-uncased
- bert-base-uncased
- roberta-base

If GPU is unavailable:
- implement scripts
- run quick mode on a small subset
- document the limitation clearly

Topic modeling:
Run:
- LDA
- BERTopic if available

Generate:
- topic keywords
- representative comments
- topic distribution by sentiment
- topic-sentiment heatmap
- topic distribution by course if course_id exists

Teaching improvement matrix:
Create a matrix mapping:
topic -> dominant sentiment -> representative comments -> likely teaching issue -> suggested teaching improvement action

Multi-label experiment:
If the Hugging Face multi-label dataset is available, train a multi-label feedback-type classifier.

Possible feedback labels:
- Improvement Suggestions
- Questions and Answers
- Experience Sharing
- Technical Feedback
- Support Request
- Community Interaction
- Course Comparison
- Related Course Suggestions
- not_praise

Metrics:
Sentiment classification:
- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- Confusion matrix

Multi-label classification:
- Micro-F1
- Macro-F1
- Hamming Loss
- Subset Accuracy

Topic modeling:
- topic keywords
- topic diversity if feasible
- topic coherence if feasible
- representative comments

Required code files:
scripts/download_public_data.py
scripts/verify_data_sources.py

src/config.py
src/load_data.py
src/preprocess.py
src/build_labels.py
src/train_classical.py
src/train_transformer.py
src/topic_modeling.py
src/evaluate.py
src/ablation.py
src/error_analysis.py
src/visualization.py
src/run_all.py

Required output files:
outputs/tables/data_summary.csv
outputs/tables/label_distribution.csv
outputs/tables/classical_baseline_results.csv
outputs/tables/transformer_results.csv
outputs/tables/topic_summary.csv
outputs/tables/teaching_improvement_matrix.csv
outputs/tables/ablation_results.csv
outputs/tables/error_cases.csv

outputs/figures/label_distribution.png
outputs/figures/model_comparison.png
outputs/figures/confusion_matrix_best_model.png
outputs/figures/topic_sentiment_heatmap.png
outputs/figures/course_sentiment_distribution.png

outputs/reports/DATA_PROVENANCE.md
outputs/reports/experiment_summary.md
outputs/reports/data_ethics_and_license_check.md
outputs/reports/error_analysis.md
outputs/reports/publication_ready_results.md

paper/experiment_section_draft.md

README requirements:
README.md must explain:
- project objective
- dataset sources
- license status
- how to configure Kaggle credentials
- how to download data
- how to run quick mode
- how to run full mode
- how to reproduce every table and figure
- known limitations

Main commands:
The following commands should work:

python scripts/download_public_data.py
python -m src.run_all --mode quick
python -m src.run_all --mode full

Quick mode:
- Use a manageable sample size.
- Train classical baselines.
- Run topic modeling if feasible.
- Generate all core tables and reports.

Full mode:
- Use full available dataset.
- Train classical models.
- Train Transformer models if hardware allows.
- Run ablation, topic modeling, and error analysis.

Acceptance criteria:
1. The pipeline runs from public data.
2. Dataset source and license are documented.
3. No fake data or fake results are created.
4. Rating is not used as a feature when it is used to create sentiment labels.
5. Raw data and credentials are not committed to Git.
6. All output tables and figures come from executed code.
7. The experiment summary documents limitations honestly.
8. The paper draft uses only generated results.

Final response expected:
After finishing, summarize:
1. Datasets downloaded.
2. License status.
3. Files created.
4. Commands run.
5. Main experimental results.
6. Generated output paths.
7. Known limitations.
8. Next steps for turning this into an EI paper.