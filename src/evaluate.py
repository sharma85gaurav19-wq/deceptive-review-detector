"""Evaluation metrics and result tables for deceptive review detection."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from scipy import sparse

LOGGER = logging.getLogger(__name__)

TABLE_DIR = Path("outputs/tables")


def _make_markdown_table(metrics: List[Dict[str, Any]]) -> str:
    header = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    rows = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for row in metrics:
        rows.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['auc']:.4f} |"
        )
    return "\n".join(rows)


def _predict_text_model(model: Any, df: pd.DataFrame, vectorizer: Any) -> Tuple[np.ndarray, np.ndarray]:
    texts = df["review_text"].fillna("")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vectorizer.transform(texts))[:, 1]
    else:
        proba = model.decision_function(vectorizer.transform(texts))
        proba = 1 / (1 + np.exp(-proba))
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def _predict_lstm_proxy(model_tuple: Any, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer, classifier = model_tuple
    X = vectorizer.transform(df["review_text"].fillna(""))
    proba = classifier.predict_proba(X)[:, 1]
    return classifier.predict(X), proba


def _predict_bert_proxy(model_tuple: Any, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    word_vectorizer, char_vectorizer, classifier = model_tuple
    X_word = word_vectorizer.transform(df["review_text"].fillna(""))
    X_char = char_vectorizer.transform(df["review_text"].fillna(""))
    X = sparse.hstack([X_word, X_char], format="csr")
    proba = classifier.predict_proba(X)[:, 1]
    return classifier.predict(X), proba


def _predict_hybrid(model: Any, df: pd.DataFrame, vectorizer: Any, scaler: Any) -> Tuple[np.ndarray, np.ndarray]:
    from src.features import fit_transform_features

    X_hybrid, _, _ = fit_transform_features(df, vectorizer, scaler, fit=False)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_hybrid)[:, 1]
    else:
        proba = model.predict(X_hybrid)
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def _evaluate_model(model_name: str, model: Any, df: pd.DataFrame, dataset_name: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    if model_name == "lstm_proxy":
        pred, proba = _predict_lstm_proxy(model, df)
    elif model_name == "bert_proxy":
        pred, proba = _predict_bert_proxy(model, df)
    elif model_name == "hybrid_rf":
        pred, proba = _predict_hybrid(model, df, artifacts["vectorizer"], artifacts["scaler"])
    else:
        pred, proba = _predict_text_model(model, df, artifacts["vectorizer"])

    return {
        "model": model_name,
        "accuracy": accuracy_score(df["label"], pred),
        "precision": precision_score(df["label"], pred, zero_division=0),
        "recall": recall_score(df["label"], pred, zero_division=0),
        "f1": f1_score(df["label"], pred, zero_division=0),
        "auc": roc_auc_score(df["label"], proba),
        "confusion": confusion_matrix(df["label"], pred).tolist(),
    }


def evaluate_dataset(dataset_name: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model_file = Path("outputs/models") / f"models_{dataset_name}.joblib"
    split_file = Path("outputs/models") / f"split_{dataset_name}.joblib"
    vectorizer_file = Path("outputs/models") / f"tfidf_{dataset_name}.joblib"
    scaler_file = Path("outputs/models") / f"scaler_{dataset_name}.joblib"

    if not model_file.exists() or not split_file.exists() or not vectorizer_file.exists() or not scaler_file.exists():
        LOGGER.error("Required artifacts missing for %s evaluation", dataset_name)
        raise FileNotFoundError(dataset_name)

    models = joblib.load(model_file)
    train_df, test_df = joblib.load(split_file)
    vectorizer = joblib.load(vectorizer_file)
    scaler = joblib.load(scaler_file)

    artifacts = {"vectorizer": vectorizer, "scaler": scaler}
    results = []
    for model_name, model in models.items():
        results.append(_evaluate_model(model_name, model, test_df, dataset_name, artifacts))

    df_results = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "AUC-ROC": r["auc"],
    } for r in results])

    file_name = TABLE_DIR / f"table_4_1a_{dataset_name}.csv" if dataset_name == "amazon" else TABLE_DIR / f"table_4_1b_{dataset_name}.csv"
    df_results.to_csv(file_name, index=False)
    LOGGER.info("Saved evaluation table for %s to %s", dataset_name, file_name)

    if dataset_name == "amazon":
        hybrid = next(r for r in results if r["model"] == "hybrid_rf")
        cm = pd.DataFrame(hybrid["confusion"], index=["actual_genuine", "actual_deceptive"], columns=["pred_genuine", "pred_deceptive"])
        cm.to_csv(TABLE_DIR / "confusion_matrix_amazon.csv")
        LOGGER.info("Saved Amazon confusion matrix")

    print(_make_markdown_table(results))


def evaluate_all(datasets: str = "both") -> None:
    if datasets in ("amazon", "both"):
        evaluate_dataset("amazon")
    if datasets in ("yelp", "both"):
        evaluate_dataset("yelp")


if __name__ == "__main__":
    evaluate_all("both")
