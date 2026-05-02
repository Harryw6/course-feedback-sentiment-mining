from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from .config import TABLE_DIR


TRANSFORMER_MODELS = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]


def _torch_gpu_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run_transformer_baselines(
    sentiment_info: dict[str, Any],
    mode: str,
) -> pd.DataFrame:
    if not sentiment_info.get("sentiment_available", False):
        reason = "No sentiment labels are available; Transformer sentiment baselines were not run."
    elif importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        reason = "transformers/torch are not installed; Transformer training script path is implemented but DistilBERT quick mode could not run in this environment."
    elif not _torch_gpu_available():
        reason = "No CUDA GPU is available; CPU DistilBERT quick mode is documented for later rerun but skipped here to keep the reproducible run bounded."
    elif mode == "quick":
        reason = "Quick mode skips expensive Transformer fine-tuning."
    else:
        reason = "Transformer training is not enabled in this CPU-oriented reproducibility script."

    rows = [
        {
            "model": model_name,
            "status": "not_run",
            "accuracy": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "notes": reason,
        }
        for model_name in TRANSFORMER_MODELS
    ]
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "transformer_results.csv", index=False)
    return result


try:
    from .train_transformer_real import run_transformer_baselines as run_transformer_baselines
except Exception:
    pass
