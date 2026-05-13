"""LIME explanation utilities for deceptive review detection."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
from scipy import sparse

LOGGER = logging.getLogger(__name__)
LOG_FILE = Path("outputs/logs/lime_examples.txt")
GLOBAL_VECTORIZER = None

FEATURE_NAMES = [
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


def explain_text(model: Any, vectorizer: Any, scaler: Any, review_record: Dict[str, Any], num_features: int = 5) -> List[Tuple[str, float]]:
    from src.features import compute_behavioral_features

    explainer = LimeTextExplainer(class_names=["genuine", "deceptive"], split_expression=r"\s+")
    text = review_record["review_text"]
    behavior = compute_behavioral_features(pd.DataFrame([review_record]))
    behavior_scaled = scaler.transform(behavior)
    behavior_sparse = sparse.csr_matrix(behavior_scaled)

    def predict_proba(texts):
        X_text = vectorizer.transform(texts)
        repeated_behavior = sparse.vstack([behavior_sparse] * len(texts), format="csr")
        X_hybrid = sparse.hstack([X_text, repeated_behavior], format="csr")
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_hybrid)
        decisions = model.decision_function(X_hybrid)
        probs = 1 / (1 + np.exp(-decisions))
        return np.vstack([1 - probs, probs]).T

    exp = explainer.explain_instance(text, predict_proba, num_features=num_features, num_samples=500)
    return exp.as_list(label=1)


def explain_behavioral(model: Any, scaler: Any, train_df: pd.DataFrame, review_record: Dict[str, Any], num_features: int = 5) -> List[Tuple[str, float]]:
    from src.features import compute_behavioral_features

    if GLOBAL_VECTORIZER is None:
        raise RuntimeError("GLOBAL_VECTORIZER must be initialized for behavioral explanation")

    train_data = compute_behavioral_features(train_df)
    explainer = LimeTabularExplainer(
        train_data,
        feature_names=FEATURE_NAMES,
        class_names=["genuine", "deceptive"],
        discretize_continuous=True,
        random_state=42,
    )

    record_behavior = compute_behavioral_features(pd.DataFrame([review_record]))
    record_behavior_scaled = scaler.transform(record_behavior)
    text_vector = GLOBAL_VECTORIZER.transform([review_record["review_text"]])

    def predict_proba(X):
        X_arr = np.atleast_2d(X)
        repeated_text = sparse.vstack([text_vector] * X_arr.shape[0], format="csr")
        behavior_sparse = sparse.csr_matrix(X_arr)
        X_hybrid = sparse.hstack([repeated_text, behavior_sparse], format="csr")
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_hybrid)
        decisions = model.decision_function(X_hybrid)
        probs = 1 / (1 + np.exp(-decisions))
        return np.vstack([1 - probs, probs]).T

    exp = explainer.explain_instance(record_behavior_scaled[0], predict_proba, num_features=num_features)
    return exp.as_list(label=1)


def generate_lime_examples() -> None:
    global GLOBAL_VECTORIZER
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    split_file = Path("outputs/models") / "split_amazon.joblib"
    vectorizer_file = Path("outputs/models") / "tfidf_amazon.joblib"
    scaler_file = Path("outputs/models") / "scaler_amazon.joblib"
    model_file = Path("outputs/models") / "models_amazon.joblib"

    if not split_file.exists() or not vectorizer_file.exists() or not scaler_file.exists() or not model_file.exists():
        raise FileNotFoundError("Amazon artifacts missing for LIME explanations")

    train_df, test_df = joblib.load(split_file)
    vectorizer = joblib.load(vectorizer_file)
    scaler = joblib.load(scaler_file)
    models = joblib.load(model_file)
    hybrid_model = models["hybrid_rf"]
    GLOBAL_VECTORIZER = vectorizer

    from src.features import fit_transform_features

    X_test_hybrid, _, _ = fit_transform_features(test_df, vectorizer, scaler, fit=False)
    proba = hybrid_model.predict_proba(X_test_hybrid)[:, 1]
    preds = (proba >= 0.5).astype(int)
    actual = test_df["label"].to_numpy()

    cases = {
        "TP": (preds == 1) & (actual == 1),
        "TN": (preds == 0) & (actual == 0),
        "FP": (preds == 1) & (actual == 0),
        "FN": (preds == 0) & (actual == 1),
    }

    with LOG_FILE.open("w", encoding="utf-8") as log_handle:
        log_handle.write("LIME explanation examples for Hybrid RF\n\n")
        for label, mask in cases.items():
            indices = np.where(mask)[0]
            if len(indices) == 0:
                log_handle.write(f"{label}: no examples found\n\n")
                continue
            idx = int(indices[0])
            record = test_df.iloc[idx].to_dict()
            text_explanation = explain_text(hybrid_model, vectorizer, scaler, record, num_features=5)
            behavior_explanation = explain_behavioral(hybrid_model, scaler, train_df, record, num_features=5)
            log_handle.write(f"{label} example index: {idx}\n")
            log_handle.write(f"Review text: {record['review_text']}\n")
            log_handle.write(f"Label: {record['label']}\n")
            log_handle.write("Top text features:\n")
            for term, weight in text_explanation:
                log_handle.write(f"  {term}: {weight:.4f}\n")
            log_handle.write("Top behavioral features:\n")
            for term, weight in behavior_explanation:
                log_handle.write(f"  {term}: {weight:.4f}\n")
            log_handle.write("\n")

    LOGGER.info("Saved LIME explanations to %s", LOG_FILE)


if __name__ == "__main__":
    generate_lime_examples()
