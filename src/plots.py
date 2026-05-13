"""Figure generation for deceptive review detection results."""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from scipy import sparse

from src.evaluate import _predict_text_model, _predict_lstm_proxy, _predict_bert_proxy, _predict_hybrid

LOGGER = logging.getLogger(__name__)
FIG_DIR = Path("outputs/figures")


def _load_artifacts(dataset_name: str):
    models = joblib.load(Path("outputs/models") / f"models_{dataset_name}.joblib")
    train_df, test_df = joblib.load(Path("outputs/models") / f"split_{dataset_name}.joblib")
    vectorizer = joblib.load(Path("outputs/models") / f"tfidf_{dataset_name}.joblib")
    scaler = joblib.load(Path("outputs/models") / f"scaler_{dataset_name}.joblib")
    return models, train_df, test_df, vectorizer, scaler


def _predict(model_name: str, model, df, vectorizer, scaler):
    from src.evaluate import _predict_text_model, _predict_lstm_proxy, _predict_bert_proxy, _predict_hybrid

    if model_name == "lstm_proxy":
        return _predict_lstm_proxy(model, df)
    if model_name == "bert_proxy":
        return _predict_bert_proxy(model, df)
    if model_name == "hybrid_rf":
        return _predict_hybrid(model, df, vectorizer, scaler)
    return _predict_text_model(model, df, vectorizer)


def figure_performance() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table_amazon = pd.read_csv(Path("outputs/tables") / "table_4_1a_amazon.csv")
    table_yelp = pd.read_csv(Path("outputs/tables") / "table_4_1b_yelp.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, table, title in zip(axes, [table_amazon, table_yelp], ["Amazon", "Yelp"]):
        melted = table.melt(id_vars=["Model"], value_vars=["Accuracy", "F1"], var_name="Metric", value_name="Score")
        sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax)
        ax.set_title(f"Performance on {title} Test Set")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "fig_4_1_performance.png", dpi=300)
    LOGGER.info("Saved performance figure")


def figure_roc_amazon() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models, _, test_df, vectorizer, scaler = _load_artifacts("amazon")
    top_models = ["hybrid_rf", "bert_proxy", "text_random_forest", "lstm_proxy"]
    plt.figure(figsize=(8, 6))
    for model_name in top_models:
        model = models[model_name]
        if model_name == "lstm_proxy":
            _, proba = _predict_lstm_proxy(model, test_df)
        elif model_name == "bert_proxy":
            _, proba = _predict_bert_proxy(model, test_df)
        elif model_name == "hybrid_rf":
            _, proba = _predict_hybrid(model, test_df, vectorizer, scaler)
        else:
            _, proba = _predict_text_model(model, test_df, vectorizer)
        fpr, tpr, _ = roc_curve(test_df["label"], proba)
        plt.plot(fpr, tpr, label=f"{model_name}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves on Amazon Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_4_2_roc_amazon.png", dpi=300)
    LOGGER.info("Saved ROC figure")


def figure_confusion_matrix() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm = pd.read_csv(Path("outputs/tables") / "confusion_matrix_amazon.csv", index_col=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix for Hybrid RF on Amazon Test Set")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_4_3_confusion_matrix.png", dpi=300)
    LOGGER.info("Saved confusion matrix figure")


def figure_feature_importance() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # train behavioral-only RF on Amazon training set
    models, train_df, _, vectorizer, scaler = _load_artifacts("amazon")
    from sklearn.ensemble import RandomForestClassifier
    from src.features import fit_transform_features
    X_train_hybrid, _, _ = fit_transform_features(train_df, vectorizer, scaler, fit=False)
    X_train_behavioral = X_train_hybrid[:, -10:]
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X_train_behavioral, train_df["label"].to_numpy())
    importances = rf.feature_importances_
    feature_names = [
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
    imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    imp_df.plot(kind="barh", color="steelblue")
    plt.title("Behavioral Feature Importances from Random Forest")
    plt.xlabel("Gini Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_4_4_feature_importance.png", dpi=300)
    LOGGER.info("Saved feature importance figure")


def generate_all_plots() -> None:
    figure_performance()
    figure_roc_amazon()
    figure_confusion_matrix()
    figure_feature_importance()


if __name__ == "__main__":
    generate_all_plots()
