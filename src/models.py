"""Model definitions and baseline classifiers for deceptive review detection."""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


def _clip_sparse_matrix(X: sparse.spmatrix) -> sparse.spmatrix:
    X = X.copy().tolil()
    X[X < 0] = 0
    return X.tocsr()


def build_naive_bayes() -> MultinomialNB:
    return MultinomialNB()


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)


def build_linear_svm() -> CalibratedClassifierCV:
    base = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
    return CalibratedClassifierCV(base_estimator=base, cv=3)


def build_text_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )


def build_lstm_proxy() -> Tuple[HashingVectorizer, LogisticRegression]:
    vectorizer = HashingVectorizer(n_features=512, alternate_sign=False, norm=None)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return vectorizer, classifier


def build_bert_proxy() -> Tuple[TfidfVectorizer, TfidfVectorizer, LogisticRegression]:
    word_vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
    )
    char_vectorizer = TfidfVectorizer(
        max_features=2500,
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return word_vectorizer, char_vectorizer, classifier


def build_hybrid_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )


def prepare_text_matrix(X_text: sparse.spmatrix) -> sparse.spmatrix:
    return _clip_sparse_matrix(X_text)


def combine_char_word_features(word_matrix: sparse.spmatrix, char_matrix: sparse.spmatrix) -> sparse.spmatrix:
    return hstack([word_matrix, char_matrix], format="csr")


def fit_lstm_proxy(
    vectorizer: HashingVectorizer,
    classifier: LogisticRegression,
    texts: List[str],
    y: np.ndarray,
) -> Tuple[HashingVectorizer, LogisticRegression]:
    X = vectorizer.transform(texts)
    classifier.fit(X, y)
    return vectorizer, classifier


def fit_bert_proxy(
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    texts: List[str],
    y: np.ndarray,
) -> Tuple[TfidfVectorizer, TfidfVectorizer, LogisticRegression]:
    X_word = word_vectorizer.fit_transform(texts)
    X_char = char_vectorizer.fit_transform(texts)
    X = combine_char_word_features(X_word, X_char)
    classifier.fit(X, y)
    return word_vectorizer, char_vectorizer, classifier
