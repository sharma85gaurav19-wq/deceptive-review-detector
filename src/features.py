"""Feature engineering for deceptive review detection."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "stop_words": "english",
}

BEHAVIORAL_COLUMNS = [
    "verified_purchase",
    "rating",
    "rating_deviation",
    "review_length_chars",
    "review_length_words",
    "avg_word_length",
    "capital_ratio",
    "exclamation_count",
    "question_count",
    "reviewer_review_count",
]


def _compute_behavioral_features(df: pd.DataFrame) -> np.ndarray:
    product_mean = df.groupby("product_id")["rating"].transform("mean")
    review_text = df["review_text"].fillna("")
    lengths = review_text.str.len()
    tokens = review_text.str.split()
    avg_word_length = tokens.apply(lambda token_list: np.mean([len(tok) for tok in token_list]) if token_list else 0.0)
    capital_letters = review_text.str.count(r"[A-Z]")
    total_letters = review_text.str.count(r"[A-Za-z]")

    feature_matrix = np.vstack([
        df["verified_purchase"].astype(float).to_numpy(),
        df["rating"].astype(float).to_numpy(),
        np.abs(df["rating"].astype(float) - product_mean.astype(float)).to_numpy(),
        lengths.astype(float).to_numpy(),
        tokens.apply(len).astype(float).to_numpy(),
        avg_word_length.astype(float).to_numpy(),
        (capital_letters / np.maximum(total_letters, 1)).astype(float).to_numpy(),
        review_text.str.count("!").astype(float).to_numpy(),
        review_text.str.count(r"\?").astype(float).to_numpy(),
        df["reviewer_review_count"].astype(float).to_numpy(),
    ]).T

    return feature_matrix


def compute_behavioral_features(df: pd.DataFrame) -> np.ndarray:
    """Compute the standardized behavioral feature matrix for a dataframe."""
    return _compute_behavioral_features(df)


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with the exact thesis configuration."""
    return TfidfVectorizer(**TFIDF_CONFIG)


def fit_transform_features(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    fit: bool = True,
) -> Tuple[sparse.csr_matrix, np.ndarray, StandardScaler]:
    """Fit or transform TF-IDF and behavioral features for a dataset."""
    text = df["review_text"].fillna("")
    if fit:
        text_matrix = vectorizer.fit_transform(text)
    else:
        text_matrix = vectorizer.transform(text)

    behavioral = _compute_behavioral_features(df)
    if fit:
        behavioral_scaled = scaler.fit_transform(behavioral)
    else:
        behavioral_scaled = scaler.transform(behavioral)

    behavioral_sparse = sparse.csr_matrix(behavioral_scaled)
    hybrid_matrix = sparse.hstack([text_matrix, behavioral_sparse], format="csr")
    return hybrid_matrix, behavioral_scaled, scaler


def transform_text_features(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> sparse.csr_matrix:
    """Transform review text using a fitted TF-IDF vectorizer."""
    return vectorizer.transform(df["review_text"].fillna(""))


def transform_behavioral_features(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform behavioral features using a fitted scaler."""
    behavioral = _compute_behavioral_features(df)
    return scaler.transform(behavioral)


def save_vectorizer_and_scaler(vectorizer: TfidfVectorizer, scaler: StandardScaler, prefix: str) -> None:
    """Placeholder if persistence is required in higher-level code."""
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    return
