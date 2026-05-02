from __future__ import annotations

import importlib.util
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import SPLITS_DIR, TABLE_DIR


TRANSFORMER_MODELS = [
    ("distilbert-base-uncased", "distilbert-base-uncased"),
    ("bert-base-uncased", "bert-base-uncased"),
    ("roberta-base", "roberta-base"),
]

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def _packages_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("datasets") is not None
    )


def _select_free_gpu() -> str | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return None
    candidates: list[tuple[int, int]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            candidates.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    if not candidates:
        return None
    gpu_index, _ = min(candidates, key=lambda item: item[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return str(gpu_index)


def _load_splits(splits: dict[str, pd.DataFrame] | None) -> dict[str, pd.DataFrame]:
    if splits is not None:
        return splits
    return {name: pd.read_csv(SPLITS_DIR / f"{name}.csv") for name in ["train", "val", "test"]}


def _balanced_sample(frame: pd.DataFrame, label_col: str, per_class: int, random_state: int = 42) -> pd.DataFrame:
    parts = []
    for _, group in frame.dropna(subset=["clean_text", label_col]).groupby(label_col):
        parts.append(group.sample(n=min(per_class, len(group)), random_state=random_state))
    if not parts:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _prepare_data(splits: dict[str, pd.DataFrame], mode: str) -> dict[str, pd.DataFrame]:
    caps = {"train": 1200, "val": 300, "test": 600} if mode == "quick" else {"train": 5000, "val": 1000, "test": 2000}
    return {name: _balanced_sample(splits[name], "sentiment_label", cap) for name, cap in caps.items()}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _not_run_rows(reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": display_name,
                "status": "not_run",
                "accuracy": np.nan,
                "precision_macro": np.nan,
                "recall_macro": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "train_rows": 0,
                "val_rows": 0,
                "test_rows": 0,
                "epochs": 0,
                "device": "",
                "runtime_seconds": np.nan,
                "notes": reason,
            }
            for display_name, _ in TRANSFORMER_MODELS
        ]
    )


def _train_one(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    output_root: Path,
    device_note: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(frame: pd.DataFrame) -> Dataset:
        data = pd.DataFrame(
            {
                "text": frame["clean_text"].astype(str).tolist(),
                "label": frame["sentiment_label"].map(LABEL_TO_ID).astype(int).tolist(),
            }
        )
        dataset = Dataset.from_pandas(data, preserve_index=False)
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=192),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = encode(train_df)
    val_ds = encode(val_df)
    test_ds = encode(test_df)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    batch_size = 32 if torch.cuda.is_available() else 8
    epochs = 1 if mode == "quick" else 2
    args = TrainingArguments(
        output_dir=str(output_root / model_name.replace("/", "_")),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        fp16=bool(torch.cuda.is_available()),
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    start = time.time()
    trainer.train()
    predictions = trainer.predict(test_ds)
    runtime = time.time() - start
    y_true = np.asarray(predictions.label_ids)
    y_pred = np.argmax(predictions.predictions, axis=1)
    metrics = _metrics(y_true, y_pred)
    pred_frame = pd.DataFrame(
        {
            "model": model_name,
            "clean_text": test_df["clean_text"].tolist(),
            "true_label": [ID_TO_LABEL[int(i)] for i in y_true],
            "predicted_label": [ID_TO_LABEL[int(i)] for i in y_pred],
        }
    )
    row = {
        "model": model_name,
        "status": "run",
        **metrics,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "epochs": epochs,
        "device": device_note,
        "runtime_seconds": round(runtime, 2),
        "notes": "Fine-tuned locally with Hugging Face Trainer on balanced Kaggle 100K sentiment subset.",
    }
    return row, pred_frame


def run_transformer_baselines(sentiment_info: dict[str, Any], mode: str, splits: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if not sentiment_info.get("sentiment_available", False):
        result = _not_run_rows("No sentiment labels are available; Transformer sentiment baselines were not run.")
        result.to_csv(TABLE_DIR / "transformer_results.csv", index=False)
        return result
    if not _packages_available():
        result = _not_run_rows("torch/transformers/datasets are not installed.")
        result.to_csv(TABLE_DIR / "transformer_results.csv", index=False)
        return result

    gpu = _select_free_gpu()
    import torch

    device_note = f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"
    split_frames = _prepare_data(_load_splits(splits), mode)
    output_root = Path("outputs/models/transformers")
    output_root.mkdir(parents=True, exist_ok=True)
    models_to_run = TRANSFORMER_MODELS[:1] if mode == "quick" else TRANSFORMER_MODELS

    rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    for display_name, model_name in TRANSFORMER_MODELS:
        if (display_name, model_name) not in models_to_run:
            rows.append(
                {
                    "model": display_name,
                    "status": "not_run",
                    "accuracy": np.nan,
                    "precision_macro": np.nan,
                    "recall_macro": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "train_rows": 0,
                    "val_rows": 0,
                    "test_rows": 0,
                    "epochs": 0,
                    "device": device_note,
                    "runtime_seconds": np.nan,
                    "notes": "Skipped in quick mode; full mode runs all configured Transformer baselines.",
                }
            )
            continue
        try:
            row, predictions = _train_one(model_name, split_frames["train"], split_frames["val"], split_frames["test"], mode, output_root, device_note)
            row["model"] = display_name
            predictions["model"] = display_name
            rows.append(row)
            pred_frames.append(predictions)
        except Exception as exc:
            rows.append(
                {
                    "model": display_name,
                    "status": "not_run",
                    "accuracy": np.nan,
                    "precision_macro": np.nan,
                    "recall_macro": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "train_rows": len(split_frames["train"]),
                    "val_rows": len(split_frames["val"]),
                    "test_rows": len(split_frames["test"]),
                    "epochs": 0,
                    "device": device_note,
                    "runtime_seconds": np.nan,
                    "notes": f"Transformer training failed: {type(exc).__name__}: {exc}",
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "transformer_results.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(TABLE_DIR / "transformer_predictions.csv", index=False)
    return result
