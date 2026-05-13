"""Smoke tests for the deceptive-review-detector pipeline."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data_generator import _generate_dataset
from src.features import build_tfidf_vectorizer, fit_transform_features
from src.models import build_text_random_forest


def test_pipeline_smoke() -> None:
    rng = np.random.default_rng(42)
    df = _generate_dataset("amazon", 1000, 20, rng)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["label"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    vectorizer = build_tfidf_vectorizer()
    scaler = StandardScaler()
    X_train, _, scaler = fit_transform_features(train_df, vectorizer, scaler, fit=True)
    X_test, _, _ = fit_transform_features(test_df, vectorizer, scaler, fit=False)

    model = build_text_random_forest()
    model.set_params(n_estimators=10)
    model.fit(X_train[:, : X_train.shape[1] - 10], train_df["label"].to_numpy())

    pred = model.predict(X_test[:, : X_test.shape[1] - 10])
    assert len(pred) == len(test_df)
    assert set(pred).issubset({0, 1})
