"""Training and persistence utilities for deceptive review detection."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src import data_generator
from src.features import (
    TFIDF_CONFIG,
    build_tfidf_vectorizer,
    fit_transform_features,
    transform_text_features,
    transform_behavioral_features,
)
from src.models import (
    build_bert_proxy,
    build_hybrid_random_forest,
    build_linear_svm,
    build_logistic_regression,
    build_lstm_proxy,
    build_naive_bayes,
    build_text_random_forest,
    fit_bert_proxy,
    fit_lstm_proxy,
    prepare_text_matrix,
)


LOGGER = logging.getLogger(__name__)

MODEL_OUTPUT_DIR = Path("outputs/models")


def _ensure_directories() -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)


def _train_baselines(train_df: pd.DataFrame, X_text_train, y_train, dataset_name: str) -> Dict[str, object]:
    models = {}
    texts = train_df["review_text"].fillna("").tolist()

    # Naive Bayes on TF-IDF only
    nb = build_naive_bayes()
    nb.fit(X_text_train, y_train)
    models["naive_bayes"] = nb

    lr = build_logistic_regression()
    lr.fit(X_text_train, y_train)
    models["logistic_regression"] = lr

    svm = build_linear_svm()
    svm.fit(X_text_train, y_train)
    models["linear_svm"] = svm

    rf_text = build_text_random_forest()
    rf_text.fit(X_text_train, y_train)
    models["text_random_forest"] = rf_text

    lstm_vectorizer, lstm_classifier = build_lstm_proxy()
    fit_lstm_proxy(lstm_vectorizer, lstm_classifier, texts, y_train)
    models["lstm_proxy"] = (lstm_vectorizer, lstm_classifier)

    word_vectorizer, char_vectorizer, bert_classifier = build_bert_proxy()
    fit_bert_proxy(word_vectorizer, char_vectorizer, bert_classifier, texts, y_train)
    models["bert_proxy"] = (word_vectorizer, char_vectorizer, bert_classifier)

    return models


def _train_hybrid_rf(X_hybrid_train, y_train) -> Tuple[RandomForestClassifier, dict]:
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }
    base_rf = build_hybrid_random_forest()
    search = GridSearchCV(
        base_rf,
        param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_hybrid_train, y_train)
    return search.best_estimator_, search.best_params_


def _split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    y = df["label"].to_numpy()
    for train_idx, test_idx in splitter.split(df, y):
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    raise ValueError("Unable to split dataset")


def train_dataset(dataset_name: str) -> None:
    """Train the full suite of models for a single dataset."""
    _ensure_directories()
    path = Path("data") / f"{dataset_name}_reviews.csv"
    if not path.exists():
        LOGGER.error("Dataset %s not found. Generate it first.", path)
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    train_df, test_df = _split_dataset(df)

    vectorizer = build_tfidf_vectorizer()
    scaler = StandardScaler()
    X_train_hybrid, _, scaler = fit_transform_features(train_df, vectorizer, scaler, fit=True)
    X_train_text = transform_text_features(train_df, vectorizer)

    models = _train_baselines(train_df, X_train_text, train_df["label"].to_numpy(), dataset_name)
    hybrid_rf, best_params = _train_hybrid_rf(X_train_hybrid, train_df["label"].to_numpy())
    models["hybrid_rf"] = hybrid_rf

    # Persist both train/test split objects so other modules can load them if needed
    joblib.dump(models, MODEL_OUTPUT_DIR / f"models_{dataset_name}.joblib")
    joblib.dump(vectorizer, MODEL_OUTPUT_DIR / f"tfidf_{dataset_name}.joblib")
    joblib.dump(scaler, MODEL_OUTPUT_DIR / f"scaler_{dataset_name}.joblib")
    joblib.dump((train_df, test_df), MODEL_OUTPUT_DIR / f"split_{dataset_name}.joblib")
    joblib.dump(best_params, MODEL_OUTPUT_DIR / f"hybrid_params_{dataset_name}.joblib")

    LOGGER.info("Trained %s models and saved artifacts for %s", len(models), dataset_name)


def train_all(datasets: str = "both") -> None:
    if datasets in ("amazon", "both"):
        train_dataset("amazon")
    if datasets in ("yelp", "both"):
        train_dataset("yelp")


if __name__ == "__main__":
    train_all("both")
