"""Cross-domain evaluation for Amazon and Yelp datasets."""

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features import build_tfidf_vectorizer, fit_transform_features, transform_text_features, transform_behavioral_features
from src.models import build_hybrid_random_forest, build_text_random_forest

LOGGER = logging.getLogger(__name__)
TABLE_DIR = Path("outputs/tables")


def _evaluate_model(model, X, y):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        decisions = model.decision_function(X)
        proba = 1 / (1 + np.exp(-decisions))
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, proba),
    }


def run_cross_domain() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    source_target = [("amazon", "yelp"), ("yelp", "amazon")]
    records = []

    for source, target in source_target:
        source_split = joblib.load(Path("outputs/models") / f"split_{source}.joblib")
        target_split = joblib.load(Path("outputs/models") / f"split_{target}.joblib")
        source_train, _ = source_split
        _, target_test = target_split

        vectorizer = build_tfidf_vectorizer()
        scaler = StandardScaler()
        X_train_hybrid, _, scaler = fit_transform_features(source_train, vectorizer, scaler, fit=True)
        X_target_hybrid, _, _ = fit_transform_features(target_test, vectorizer, scaler, fit=False)

        hybrid_model = build_hybrid_random_forest()
        hybrid_model.fit(X_train_hybrid, source_train["label"].to_numpy())
        result = _evaluate_model(hybrid_model, X_target_hybrid, target_test["label"].to_numpy())
        records.append({
            "train_source": source,
            "test_target": target,
            "model": "Hybrid RF",
            **result,
        })

    pd.DataFrame(records).to_csv(TABLE_DIR / "table_4_4_cross_domain.csv", index=False)
    LOGGER.info("Saved cross-domain results to %s", TABLE_DIR / "table_4_4_cross_domain.csv")


if __name__ == "__main__":
    run_cross_domain()
