"""Threshold analysis for the Hybrid RF system."""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
TABLE_DIR = Path("outputs/tables")


def run_threshold_analysis() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    split_file = Path("outputs/models") / "split_amazon.joblib"
    vectorizer_file = Path("outputs/models") / "tfidf_amazon.joblib"
    scaler_file = Path("outputs/models") / "scaler_amazon.joblib"
    model_file = Path("outputs/models") / "models_amazon.joblib"

    if not split_file.exists() or not vectorizer_file.exists() or not scaler_file.exists() or not model_file.exists():
        raise FileNotFoundError("Required Amazon artifacts missing for threshold analysis")

    train_df, test_df = joblib.load(split_file)
    vectorizer = joblib.load(vectorizer_file)
    scaler = joblib.load(scaler_file)
    models = joblib.load(model_file)
    hybrid_model = models["hybrid_rf"]

    from src.features import fit_transform_features

    _, _, _ = fit_transform_features(train_df, vectorizer, scaler, fit=False)
    X_test_hybrid, _, _ = fit_transform_features(test_df, vectorizer, scaler, fit=False)
    proba = hybrid_model.predict_proba(X_test_hybrid)[:, 1]

    records: List[Dict[str, float]] = []
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        preds = (proba >= thresh).astype(int)
        records.append({
            "threshold": thresh,
            "precision": float((test_df["label"] & preds).sum() / max(preds.sum(), 1)),
            "recall": float((test_df["label"] & preds).sum() / max(test_df["label"].sum(), 1)),
        })

    pd.DataFrame(records).to_csv(TABLE_DIR / "table_4_3_threshold.csv", index=False)
    LOGGER.info("Saved threshold analysis table to %s", TABLE_DIR / "table_4_3_threshold.csv")


if __name__ == "__main__":
    run_threshold_analysis()
