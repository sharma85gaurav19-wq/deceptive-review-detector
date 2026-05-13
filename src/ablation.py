"""Ablation study for hybrid, text-only, and behavioral-only systems."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.features import build_tfidf_vectorizer, fit_transform_features, _compute_behavioral_features
from src.models import build_text_random_forest, build_hybrid_random_forest

LOGGER = logging.getLogger(__name__)
TABLE_DIR = Path("outputs/tables")


def _train_rf(X, y) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X, y)
    return rf


def run_ablation() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model_file = Path("outputs/models") / "split_amazon.joblib"
    if not model_file.exists():
        raise FileNotFoundError("Amazon split not found. Train the models first.")

    train_df, test_df = joblib.load(model_file)
    vectorizer = build_tfidf_vectorizer()
    scaler = StandardScaler()
    X_train_hybrid, _, scaler = fit_transform_features(train_df, vectorizer, scaler, fit=True)
    X_test_hybrid, _, _ = fit_transform_features(test_df, vectorizer, scaler, fit=False)
    X_train_text = X_train_hybrid[:, : X_train_hybrid.shape[1] - 10]
    X_test_text = X_test_hybrid[:, : X_test_hybrid.shape[1] - 10]

    X_train_behavioral = X_train_hybrid[:, -10:]
    X_test_behavioral = X_test_hybrid[:, -10:]

    hybrid_rf = _train_rf(X_train_hybrid, train_df["label"].to_numpy())
    text_rf = _train_rf(X_train_text, train_df["label"].to_numpy())
    behavior_rf = _train_rf(X_train_behavioral, train_df["label"].to_numpy())

    def score_model(model, X, y):
        from sklearn.metrics import f1_score
        pred = model.predict(X)
        return f1_score(y, pred, zero_division=0)

    results = [
        {"model": "Hybrid", "f1": score_model(hybrid_rf, X_test_hybrid, test_df["label"].to_numpy())},
        {"model": "Text only", "f1": score_model(text_rf, X_test_text, test_df["label"].to_numpy())},
        {"model": "Behavioral only", "f1": score_model(behavior_rf, X_test_behavioral, test_df["label"].to_numpy())},
    ]

    df = pd.DataFrame(results)
    df.to_csv(TABLE_DIR / "table_4_2_ablation.csv", index=False)
    LOGGER.info("Saved ablation results to %s", TABLE_DIR / "table_4_2_ablation.csv")


if __name__ == "__main__":
    run_ablation()
