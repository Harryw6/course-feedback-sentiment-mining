# Auxiliary Multi-label Feedback-type Experiment

This is an auxiliary experiment using the Hugging Face `chillies/course-review-multilabel-sentiment-analysis` dataset.
It is not presented as the main sentiment analysis experiment.

Rows used: 8152

## Results

```
                                   model status  micro_f1  macro_f1  hamming_loss  subset_accuracy notes
TF-IDF + One-vs-Rest Logistic Regression    run  0.687119  0.499878      0.108386         0.445126      
```

## Limitation

The dataset has feedback-type labels and no explicit sentiment labels or ratings. Its license was not explicit in available metadata and must be verified before publication reuse.
