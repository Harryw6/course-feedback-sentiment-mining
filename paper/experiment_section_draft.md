# Experimental Section Draft

## Data

The experiment used the public dataset(s): septa97/100k-courseras-course-reviews-dataset, imuhammad/course-reviews-on-coursera, chillies/course-review-multilabel-sentiment-analysis. Dataset provenance, source URLs, license status, downloaded files, and limitations are recorded in `outputs/reports/DATA_PROVENANCE.md`.

## Preprocessing

Reviews were cleaned by removing empty entries, deduplicating identical cleaned text, normalizing whitespace, and masking URLs, email addresses, phone numbers, and simple self-identification patterns.

## Label Construction

The main Kaggle Coursera dataset provides a 1-5 `Label` field. We map 5/4/3/2/1 to Very Positive, Positive, Neutral, Negative, and Very Negative for the 5-class task, and then map Very Positive/Positive to positive, Neutral to neutral, and Negative/Very Negative to negative for the 3-class task.

## Models

The main experiment trains TF-IDF classical sentiment baselines on Kaggle Coursera review data. Transformer baselines are implemented and reported when dependencies and hardware allow. The Hugging Face dataset is used only as an auxiliary multi-label feedback-type experiment. LDA is used for topic mining and the teaching improvement matrix.

## Results

Use `outputs/reports/publication_ready_results.md` for the generated tables. Do not report metrics that are marked `not_run`.

## Limitations

The optional 1.45M Kaggle dataset uses rating-derived weak labels, so rating is never used as a model feature. The Hugging Face auxiliary dataset is public but its license was not explicit in available metadata, so reuse rights must be checked before submission.
