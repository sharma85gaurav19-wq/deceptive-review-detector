"""Smoke tests for the deceptive-review-detector pipeline."""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data_generator import _generate_dataset
from src.features import build_tfidf_vectorizer, fit_transform_features
from src.models import build_hybrid_random_forest, build_text_random_forest


def _build_train_test_features():
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
    return X_train, X_test, train_df, test_df


def test_pipeline_smoke() -> None:
    X_train, X_test, train_df, test_df = _build_train_test_features()

    model = build_text_random_forest()
    model.set_params(n_estimators=10)
    model.fit(X_train[:, : X_train.shape[1] - 10], train_df["label"].to_numpy())

    pred = model.predict(X_test[:, : X_test.shape[1] - 10])
    assert len(pred) == len(test_df)
    assert set(pred).issubset({0, 1})


def test_hybrid_rf_predicts_both_classes() -> None:
    """Regression test for the 'always-deceptive' bug.

    The hybrid Random Forest used to collapse to a single class (predicting
    every review as deceptive) because behavioural features were computed
    across the whole frame instead of row-locally. This test guards against a
    regression by asserting the model predicts both classes on held-out data
    and beats the trivial majority-class baseline.
    """
    X_train, X_test, train_df, test_df = _build_train_test_features()

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    model = build_hybrid_random_forest()
    model.set_params(n_estimators=50)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    assert set(np.unique(pred)) == {0, 1}, (
        "Hybrid RF collapsed to a single predicted class (always-deceptive bug)."
    )

    majority_acc = max(np.mean(y_test == 0), np.mean(y_test == 1))
    model_acc = np.mean(pred == y_test)
    assert model_acc > majority_acc, (
        f"Hybrid RF accuracy {model_acc:.3f} did not beat majority baseline {majority_acc:.3f}."
    )
