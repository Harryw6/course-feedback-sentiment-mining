# Manual Download Instructions

Automatic public dataset download did not complete.

1. Configure Kaggle credentials without committing them:
   - Create `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`.
   - Keep credentials out of Git.
2. Run:
   `python scripts/download_public_data.py`
3. If automatic Kaggle download remains unavailable, manually download one of these licensed source datasets:
   - https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset
   - https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera
4. Place extracted files under `data/raw/<dataset-name>/` and rerun `python scripts/verify_data_sources.py`.

Errors observed:
- septa97/100k-courseras-course-reviews-dataset: Dataset URL: https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset
License(s): DbCL-1.0
HTTPSConnectionPool(host='storage.googleapis.com', port=443): Max retries exceeded with url: /kaggle-data-sets/1852/62952/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260502%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260502T013409Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4f00f747b9022eade2e3ae2612da46260b36c3e7c7b3ffe8af2c5e002a5bce0d3a795ae35806c968781742d6dc26058e432a05684d95bdd6df757839ef4cd673e2d1acfcdbdac7631bd562b5e0b6e2d93173c40b6f1b5d8b1b9fa69789d0b961a32344f8d4dfef60b60d1c9a0f4dd9951806dcb44e282c4f0f64f02e963875995dca2baf02da873b5daa065e1e2693bf5b3b169863c7676d81c03c6ef446621d78f5ee803d00ca7f1bbf936cf8794b3daf12a2ef31075449169b33439a1b4822620b83586a2ffa011c26adc1f66275057ed6704fa0b939981c7a6ea3e2f0aed6f2b4ef926e05a1cc90ff7df51ae176e5942c82c4899786a9137837005ac1abbd (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))
- imuhammad/course-reviews-on-coursera: HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: /v1/datasets.DatasetApiService/GetDatasetMetadata (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))
- Hugging Face fallback download failed: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)>
