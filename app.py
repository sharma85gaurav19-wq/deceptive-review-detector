"""
Deceptive Review Detector — Streamlit Web Application
======================================================
Analyses product reviews for signs of fake / spam content using a
Hybrid Random Forest model (TF-IDF n-grams + behavioural signals).

Usage:
    streamlit run app.py
"""

import io
import textwrap
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import sparse

from src.features import compute_behavioral_features
from src.scraper import extract_asin, scrape_amazon_reviews
from src.database import (
    db_available,
    log_prediction,
    update_feedback,
    get_statistics,
    get_recent_predictions,
)

# ── App-level constants ────────────────────────────────────────────────────────
_APP_VERSION    = "1.3.0"
_EXAMPLE_URL    = "https://www.amazon.com/All-New-Echo-Dot-4th-Gen/dp/B07XJ8C8F7"
_THRESHOLD_DEF  = 0.55
_MODEL_DATASETS = ["amazon", "yelp"]

# Text-only models were trained on TF-IDF (5 000 features) only.
# hybrid_rf uses TF-IDF + 10 behavioural features (5 010 total).
_TEXT_ONLY_MODELS = {"naive_bayes", "logistic_regression", "linear_svm", "text_random_forest"}
_CLASSIFIER_OPTIONS = {
    "Logistic Regression (recommended)": "logistic_regression",
    "Naive Bayes":                        "naive_bayes",
    "Text Random Forest":                 "text_random_forest",
    "Hybrid Random Forest":               "hybrid_rf",
}

# Resolve all file paths relative to this file so they work on any machine /
# Streamlit Cloud regardless of the working directory.
_ROOT = Path(__file__).resolve().parent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deceptive Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Cards ── */
.card {
    padding: 1.25rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.card-red    { background: #fef2f2; border-left: 6px solid #ef4444; }
.card-green  { background: #f0fdf4; border-left: 6px solid #22c55e; }
.card-yellow { background: #fffbeb; border-left: 6px solid #f59e0b; }
.card-blue   { background: #eff6ff; border-left: 6px solid #3b82f6; }
.card-title  { font-size: 1.5rem; font-weight: 800; margin: 0 0 .25rem 0; }
.card-body   { font-size: .95rem; color: #555; margin: 0; }

/* ── Trust badge ── */
.trust-badge {
    display: inline-block;
    padding: .3rem .9rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: .9rem;
    color: #fff;
}
.badge-green  { background: #16a34a; }
.badge-yellow { background: #d97706; }
.badge-red    { background: #dc2626; }

/* ── Footer ── */
.footer {
    font-size: .78rem;
    color: #888;
    text-align: center;
    padding: 2rem 0 .5rem 0;
    border-top: 1px solid #e5e7eb;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts(dataset: str, model_key: str = "logistic_regression"):
    base = _ROOT / "outputs" / "models"
    all_models = joblib.load(base / f"models_{dataset}.joblib")
    vectorizer = joblib.load(base / f"tfidf_{dataset}.joblib")
    scaler     = joblib.load(base / f"scaler_{dataset}.joblib")
    return all_models[model_key], vectorizer, scaler


def _build_X(text_vecs, df, vectorizer, scaler, model_key: str):
    """Return the correct feature matrix for the chosen classifier."""
    if model_key in _TEXT_ONLY_MODELS:
        return text_vecs
    beh_scaled = scaler.transform(compute_behavioral_features(df))
    return sparse.hstack([text_vecs, sparse.csr_matrix(beh_scaled)], format="csr")


def _safe_prob(p) -> float:
    p = float(p)
    return 0.0 if np.isnan(p) else max(0.0, min(1.0, p))


def _predict_proba_deceptive(model, X) -> np.ndarray:
    """Return P(deceptive) per row, re-normalising each row to sum to 1.

    Guards against scikit-learn version skew: RandomForest models pickled
    under sklearn <1.4 and loaded under >=1.4 return raw leaf counts from
    predict_proba instead of probabilities, which made every review appear
    100% deceptive once clamped.
    """
    proba = np.asarray(model.predict_proba(X), dtype=float)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / np.where(row_sums > 0, row_sums, 1.0)
    return proba[:, 1]


def _trust_score(prob_deceptive: float) -> float:
    return round((1.0 - prob_deceptive) * 100, 1)


def _trust_label(score: float) -> tuple[str, str, str]:
    if score >= 70:
        return "High Trust", "badge-green",  "card-green"
    if score >= 40:
        return "Moderate",   "badge-yellow", "card-yellow"
    return     "Low Trust",  "badge-red",    "card-red"


def _verdict(prob: float, threshold: float) -> str:
    return "Deceptive" if prob >= threshold else "Genuine"


def predict_single(text: str, rating: int, verified: bool,
                   review_count: int, dataset: str, threshold: float,
                   model_key: str = "logistic_regression"):
    """Return (prob_deceptive, trust_score, top_words) for one review."""
    model, vectorizer, scaler = load_artifacts(dataset, model_key)

    df = pd.DataFrame([{
        "review_text": text, "rating": rating,
        "verified_purchase": int(verified),
        "product_id": "P00000", "reviewer_review_count": review_count,
    }])
    text_vec = vectorizer.transform([text])
    X = _build_X(text_vec, df, vectorizer, scaler, model_key)
    prob = _safe_prob(_predict_proba_deceptive(model, X)[0])

    # Word contributions: feature_importances_ for RF, coef_ for LR/SVM
    vocab = vectorizer.get_feature_names_out()
    tfidf_vals = text_vec.toarray()[0]
    top_words = []
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_[:len(vocab)]
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_).ravel()[:len(vocab)])
        else:
            imp = None
        if imp is not None:
            sc      = tfidf_vals * imp
            top_idx = sc.argsort()[::-1][:12]
            top_words = [{"word": vocab[i], "importance": float(sc[i])}
                         for i in top_idx if sc[i] > 0]
    except Exception:
        pass

    return prob, _trust_score(prob), top_words


def predict_batch(reviews: list[dict], dataset: str, threshold: float,
                  model_key: str = "logistic_regression") -> pd.DataFrame:
    """Batch inference; returns a display-ready DataFrame with hidden _prob column."""
    model, vectorizer, scaler = load_artifacts(dataset, model_key)
    df = pd.DataFrame(reviews)
    for col, default in [("rating", 5), ("verified_purchase", 0),
                          ("reviewer_review_count", 1), ("product_id", "P00000")]:
        if col not in df.columns:
            df[col] = default

    text_vecs = vectorizer.transform(df["review_text"].fillna(""))
    X = _build_X(text_vecs, df, vectorizer, scaler, model_key)
    probs = _predict_proba_deceptive(model, X)
    probs = np.where(np.isnan(probs), 0.0, np.clip(probs, 0.0, 1.0))

    trust = [_trust_score(p) for p in probs]
    labels = [_verdict(p, threshold) for p in probs]

    result = pd.DataFrame({
        "Reviewer":         df.get("reviewer_name",      pd.Series(["—"] * len(df))),
        "Rating":           df["rating"].apply(lambda r: "⭐" * int(r)),
        "Verified":         df["verified_purchase"].apply(lambda v: "✓" if v else "—"),
        "Review (preview)": df["review_text"].str[:110].str.strip() + "…",
        "Verdict":          labels,
        "Trust Score":      trust,
        "Deceptive %":      (probs * 100).round(1),
        # hidden columns used internally
        "_prob":            probs,
        "_full_text":       df["review_text"],
        "_title":           df.get("title", pd.Series([""] * len(df))),
        "_rating":          df["rating"],
        "_verified":        df["verified_purchase"],
        "_review_count":    df["reviewer_review_count"],
    })
    return result


# ── Chart helpers ─────────────────────────────────────────────────────────────
def gauge_chart(trust: float) -> go.Figure:
    label, _, _ = _trust_label(trust)
    color = "#16a34a" if trust >= 70 else "#d97706" if trust >= 40 else "#dc2626"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=trust,
        number={"suffix": " / 100", "font": {"size": 26}},
        title={"text": f"Trust Score — <b>{label}</b>", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100],"color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": trust,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def pie_chart(n_genuine: int, n_deceptive: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Genuine", "Deceptive"],
        values=[n_genuine, n_deceptive],
        marker_colors=["#22c55e", "#ef4444"],
        hole=0.45,
        textinfo="label+percent",
        hoverinfo="label+value",
    ))
    fig.update_layout(
        showlegend=True,
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def histogram_chart(probs: list[float]) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=[p * 100 for p in probs],
        nbinsx=10,
        marker_color="#6366f1",
        opacity=0.8,
    ))
    fig.update_layout(
        xaxis_title="Deceptive probability (%)",
        yaxis_title="Number of reviews",
        height=240,
        margin=dict(l=20, r=10, t=20, b=40),
        bargap=0.05,
    )
    return fig


# ── Export helper ─────────────────────────────────────────────────────────────
def results_to_csv(df: pd.DataFrame) -> bytes:
    export = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    return export.to_csv(index=False).encode("utf-8")


# ── Footer ────────────────────────────────────────────────────────────────────
def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer">
            🔍 <b>Deceptive Review Detector</b> v{_APP_VERSION} &nbsp;|&nbsp;
            For research &amp; educational use only &nbsp;|&nbsp;
            Web scraping may be subject to third-party Terms of Service &nbsp;|&nbsp;
            Not intended as legal evidence &nbsp;|&nbsp;
            © {datetime.now().year}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search.png", width=56)
    st.title("Review Detector")
    st.caption(f"v{_APP_VERSION}")
    st.divider()

    st.subheader("⚙️ Model Settings")
    dataset = st.selectbox(
        "Trained dataset", _MODEL_DATASETS,
        help="Amazon model is better for e-commerce. Yelp for hospitality/local businesses.",
    )
    classifier_label = st.selectbox(
        "Classifier", list(_CLASSIFIER_OPTIONS.keys()), index=0,
        help="Logistic Regression generalises best to real reviews. "
             "Hybrid RF may over-flag positive language as deceptive.",
    )
    model_key = _CLASSIFIER_OPTIONS[classifier_label]
    st.caption("✅ Text-only" if model_key in _TEXT_ONLY_MODELS else "🔬 Text + behavioural")

    threshold = st.slider(
        "Classification threshold",
        min_value=0.30, max_value=0.80,
        value=_THRESHOLD_DEF, step=0.05,
        format="%.2f",
    )
    st.caption(f"Reviews with deceptive probability ≥ **{threshold:.0%}** are flagged.")

    st.divider()
    st.subheader("👤 Reviewer Metadata")
    st.caption("Applied when metadata is not scraped automatically.")
    rating       = st.slider("Star rating", 1, 5, 5)
    verified     = st.checkbox("Verified Purchase", value=False)
    review_count = st.number_input("Total reviews by reviewer", 1, 50000, 1)

    st.divider()
    # Database connection indicator
    if db_available():
        st.success("🗄️ Database connected", icon="✅")
    else:
        st.caption("🗄️ Database: offline (predictions not logged)")

    st.divider()
    st.subheader("ℹ️ About")
    st.caption(
        "Logistic Regression on TF-IDF (5 000 n-gram features) "
        "is the most reliable classifier for real-world reviews."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
col_h1, col_h2 = st.columns([5, 2])
with col_h1:
    st.title("🔍 Deceptive Review Detector")
    st.markdown(
        "Detect fake, paid, or bot-generated reviews using machine learning. "
        "Combines **TF-IDF text analysis** with **behavioural signals** "
        "(star rating, verification status, reviewer history)."
    )
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"**Threshold:** {threshold:.0%}  \n"
        f"**Model:** {dataset.capitalize()}  \n"
        f"**Version:** {_APP_VERSION}",
        icon="⚙️",
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_single, tab_url, tab_bulk, tab_analytics, tab_about = st.tabs([
    "✍️  Single Review",
    "🔗  Amazon Product URL",
    "📂  Bulk CSV Upload",
    "📊  Analytics",
    "📖  About & Methodology",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single review
# ─────────────────────────────────────────────────────────────────────────────
with tab_single:
    EXAMPLES = {
        "genuine": (
            "The laptop arrived with a small scratch on the bezel. The keyboard has "
            "good travel and the battery lasts roughly 6 hours under normal office use. "
            "Fan gets loud under load but expected at this price. "
            "Solid mid-range choice for students, not for power users."
        ),
        "deceptive": (
            "AMAZING!!! The BEST product I have EVER used in my ENTIRE life!! "
            "Everything was absolutely PERFECT and I cannot imagine life without it. "
            "You MUST buy this immediately if you want a perfect experience!!! 10/10!!!"
        ),
    }

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button("📋 Try genuine example", use_container_width=True):
            st.session_state.review_text = EXAMPLES["genuine"]
            st.session_state.pop("single_result", None)
            st.rerun()
    with c2:
        if st.button("🚩 Try deceptive example", use_container_width=True):
            st.session_state.review_text = EXAMPLES["deceptive"]
            st.session_state.pop("single_result", None)
            st.rerun()
    with c3:
        if st.button("Clear", use_container_width=True):
            st.session_state.review_text = ""
            st.session_state.pop("single_result", None)
            st.rerun()

    review_text = st.text_area(
        "Review text",
        value=st.session_state.review_text,
        placeholder="Paste or type a product review here…",
        height=160,
        label_visibility="collapsed",
    )
    wc = len(review_text.split()) if review_text.strip() else 0
    st.caption(f"{wc} words · {len(review_text)} characters")

    if st.button("Analyze Review →", type="primary",
                 disabled=not review_text.strip(), use_container_width=True):
        with st.spinner("Analysing…"):
            prob, trust, top_words = predict_single(
                review_text, rating, bool(verified),
                int(review_count), dataset, threshold, model_key,
            )
        verdict = _verdict(prob, threshold)

        # Log to database (fire-and-forget; never raises)
        record_id = log_prediction(
            review_text=review_text,
            predicted_label=verdict,
            confidence_score=prob,
            trust_score=trust,
            model_used=model_key,
            dataset_used=dataset,
            source="manual",
            rating=rating,
            verified_purchase=int(verified),
            reviewer_review_count=int(review_count),
        )

        # Store results so they survive reruns (e.g. feedback button click)
        st.session_state.single_result = {
            "prob": prob, "trust": trust, "top_words": top_words,
            "text": review_text, "verdict": verdict,
            "record_id": record_id,
            "threshold": threshold,
            "rating": rating, "verified": verified,
        }
        st.session_state.single_feedback_done = False

    # ── Display results (persists across reruns) ──────────────────────────────
    if st.session_state.get("single_result"):
        r = st.session_state.single_result
        prob     = r["prob"]
        trust    = r["trust"]
        top_words = r["top_words"]
        verdict  = r["verdict"]
        t_label, t_badge, t_card = _trust_label(trust)
        threshold_used = r["threshold"]
        rating_used    = r["rating"]
        verified_used  = r["verified"]
        record_id      = r.get("record_id")
        review_text_shown = r["text"]

        st.divider()
        left, right = st.columns([3, 2])

        with left:
            icon = "🔴" if verdict == "Deceptive" else "🟢"
            desc = (
                "Patterns consistent with fake, paid, or bot-generated content."
                if verdict == "Deceptive" else
                "Patterns consistent with an authentic, first-hand review."
            )
            st.markdown(
                f'<div class="card {t_card}">'
                f'<p class="card-title">{icon} {verdict}</p>'
                f'<p class="card-body">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(max(0.0, min(1.0, float(prob))),
                        text=f"Deceptive probability: **{prob*100:.1f}%**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Verdict",          verdict)
            m2.metric("Deceptive prob.",  f"{prob*100:.1f}%")
            m3.metric("Threshold used",   f"{threshold_used:.0%}")

        with right:
            st.plotly_chart(gauge_chart(trust), use_container_width=True,
                            config={"displayModeBar": False})

        st.divider()
        col_words, col_beh = st.columns(2)

        with col_words:
            st.subheader("📊 Top contributing words")
            if top_words:
                chart_df = (
                    pd.DataFrame(top_words)
                    .set_index("word")
                    .sort_values("importance", ascending=False)
                )
                st.bar_chart(chart_df, height=240)
            else:
                st.info("No significant word signals found.")

        with col_beh:
            st.subheader("🧠 Behavioural signals")
            exc   = review_text_shown.count("!")
            ques  = review_text_shown.count("?")
            caps  = sum(1 for c in review_text_shown if c.isupper())
            lets  = sum(1 for c in review_text_shown if c.isalpha())
            cap_r = caps / max(lets, 1)
            awl   = (np.mean([len(w) for w in review_text_shown.split()])
                     if review_text_shown.strip() else 0)
            b1, b2 = st.columns(2)
            b1.metric("Exclamations", exc)
            b2.metric("Questions",    ques)
            b3, b4 = st.columns(2)
            b3.metric("Caps ratio",   f"{cap_r:.1%}")
            b4.metric("Avg word len", f"{awl:.1f}")
            b5, b6 = st.columns(2)
            b5.metric("Star rating",  rating_used)
            b6.metric("Verified",     "Yes" if verified_used else "No")

        # Single-review download
        single_df = pd.DataFrame([{
            "review_text": review_text_shown,
            "verdict": verdict,
            "deceptive_probability_%": round(prob * 100, 2),
            "trust_score": trust,
            "threshold_used": threshold_used,
        }])
        st.download_button(
            "⬇️ Download result as CSV",
            data=results_to_csv(single_df),
            file_name="review_analysis.csv",
            mime="text/csv",
        )

        # ── Feedback section ──────────────────────────────────────────────────
        if record_id:
            st.divider()
            if not st.session_state.get("single_feedback_done"):
                st.markdown("**Was this prediction correct?** Help us improve the model.")
                fb1, fb2, _ = st.columns([1, 1, 4])
                with fb1:
                    if st.button("✅ Correct", key="fb_correct", use_container_width=True):
                        update_feedback(record_id, "correct")
                        st.session_state.single_feedback_done = True
                        st.rerun()
                with fb2:
                    if st.button("❌ Incorrect", key="fb_incorrect", use_container_width=True):
                        update_feedback(record_id, "incorrect")
                        st.session_state.single_feedback_done = True
                        st.rerun()
            else:
                st.success("✓ Feedback recorded — thank you for helping improve the model!")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Amazon product URL
# ─────────────────────────────────────────────────────────────────────────────
with tab_url:
    # Legal notice
    st.markdown(
        '<div class="card card-yellow">'
        '<p class="card-title">⚖️ Legal Notice</p>'
        '<p class="card-body">'
        "This feature fetches publicly visible review text for <b>personal, "
        "research, and educational purposes only</b>. "
        "Automated scraping of Amazon may conflict with their "
        "<a href='https://www.amazon.com/gp/help/customer/display.html?nodeId=508088' "
        "target='_blank'>Conditions of Use</a>. "
        "For commercial use, integrate the official "
        "<a href='https://affiliate-program.amazon.com/help/node/topic/GED6338G5J8AJSKQ' "
        "target='_blank'>Amazon Product Advertising API</a> instead."
        "</p></div>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Supported URL formats"):
        st.markdown("""
| Format | Example |
|---|---|
| Standard product page | `https://www.amazon.com/dp/B07XJ8C8F7` |
| Full product URL | `https://www.amazon.com/Echo-Dot/dp/B07XJ8C8F7/ref=...` |
| Amazon India | `https://www.amazon.in/dp/B09W9FND7K` |
| Reviews page | `https://www.amazon.com/product-reviews/B07XJ8C8F7` |
        """)

    # Input row
    if "product_url" not in st.session_state:
        st.session_state.product_url = _EXAMPLE_URL

    u1, u2 = st.columns([4, 1])
    with u1:
        st.markdown("**Amazon product URL**")
    with u2:
        if st.button("🔁 Reset example", use_container_width=True):
            st.session_state.product_url = _EXAMPLE_URL
            st.rerun()

    product_url = st.text_input(
        "URL",
        value=st.session_state.product_url,
        label_visibility="collapsed",
        placeholder="https://www.amazon.com/dp/B07XJ8C8F7",
    )
    st.session_state.product_url = product_url

    asin_preview = extract_asin(product_url) if product_url else None
    if product_url and asin_preview:
        st.caption(f"✅ Detected ASIN: `{asin_preview}`")
    elif product_url:
        st.warning("Could not detect a valid ASIN. Please check the URL.")

    max_reviews = st.slider(
        "Reviews to fetch", min_value=5, max_value=30, value=10, step=5,
    )

    fetch_clicked = st.button(
        "🔍 Fetch & Analyse Reviews",
        type="primary",
        disabled=not (product_url and asin_preview),
        use_container_width=True,
    )

    if fetch_clicked and asin_preview:
        progress = st.progress(0, text="Connecting to Amazon…")
        try:
            progress.progress(20, text="Fetching reviews…")
            reviews = scrape_amazon_reviews(product_url, max_reviews=max_reviews)
            progress.progress(60, text=f"Analysing {len(reviews)} reviews…")
            results_df = predict_batch(reviews, dataset, threshold, model_key)
            progress.progress(90, text="Logging to database…")

            # Log each scraped review to the database
            for _, row in results_df.iterrows():
                log_prediction(
                    review_text=str(row["_full_text"])[:2000],
                    predicted_label=row["Verdict"],
                    confidence_score=float(row["_prob"]),
                    trust_score=float(row["Trust Score"]),
                    model_used=model_key,
                    dataset_used=dataset,
                    source="url_scrape",
                    product_asin=asin_preview,
                    rating=int(row["_rating"]),
                    verified_purchase=int(row["_verified"]),
                    reviewer_review_count=int(row["_review_count"]),
                )

            progress.progress(100, text="Done!")
            progress.empty()
        except PermissionError as exc:
            progress.empty()
            st.error(f"🚫 **CAPTCHA detected:** {exc}")
            st.info("Open the product page in your browser, solve the CAPTCHA, then retry.")
            st.stop()
        except (ValueError, ConnectionError, RuntimeError) as exc:
            progress.empty()
            st.error(f"**Error:** {exc}")
            st.stop()

        # ── Summary ──────────────────────────────────────────────────────────
        total     = len(results_df)
        n_dec     = int((results_df["_prob"] >= threshold).sum())
        n_gen     = total - n_dec
        pct_dec   = n_dec / total * 100
        avg_trust = float(results_df["_prob"]
                          .apply(lambda p: (1 - p) * 100).mean())

        st.divider()
        st.subheader(f"Results — ASIN `{asin_preview}`")

        if pct_dec >= 50:
            st.markdown(
                '<div class="card card-red">'
                '<p class="card-title">🔴 Suspicious Product</p>'
                '<p class="card-body">Over half the analysed reviews show deceptive patterns.</p>'
                '</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="card card-green">'
                '<p class="card-title">🟢 Mostly Genuine</p>'
                '<p class="card-body">The majority of analysed reviews appear authentic.</p>'
                '</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Reviews analysed", total)
        s2.metric("🔴 Deceptive",      n_dec)
        s3.metric("🟢 Genuine",        n_gen)
        s4.metric("Avg trust score",   f"{avg_trust:.1f}/100")

        st.progress(max(0.0, min(1.0, (pct_dec or 0) / 100)),
                    text=f"Deceptive reviews: **{pct_dec:.1f}%**")

        # ── Charts ────────────────────────────────────────────────────────────
        st.divider()
        ch1, ch2 = st.columns(2)
        with ch1:
            st.subheader("Genuine vs Deceptive")
            st.plotly_chart(pie_chart(n_gen, n_dec),
                            use_container_width=True, config={"displayModeBar": False})
        with ch2:
            st.subheader("Confidence distribution")
            st.plotly_chart(histogram_chart(list(results_df["_prob"])),
                            use_container_width=True, config={"displayModeBar": False})

        # ── Table ─────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Review-by-review breakdown")
        display_df = results_df.drop(columns=["_full_text", "_prob", "_title",
                                               "_rating", "_verified", "_review_count"])
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Trust Score": st.column_config.ProgressColumn(
                    "Trust Score", min_value=0, max_value=100, format="%.1f"
                ),
                "Deceptive %": st.column_config.NumberColumn(
                    "Deceptive %", format="%.1f%%"
                ),
                "Review (preview)": st.column_config.TextColumn(
                    "Review preview", width="large"
                ),
            },
            hide_index=True,
        )

        # ── Expandable full reviews ───────────────────────────────────────────
        st.divider()
        st.subheader("Full review details")
        for i, row in results_df.iterrows():
            p        = row["_prob"]
            is_dec   = p >= threshold
            icon     = "🔴" if is_dec else "🟢"
            label    = "Deceptive" if is_dec else "Genuine"
            conf_pct = p * 100 if is_dec else (1 - p) * 100
            title    = row["_title"] or f"Review {i + 1}"
            header   = f"{icon} {label} ({conf_pct:.1f}%) — {row['Reviewer']} · {row['Rating']}"
            with st.expander(f"Review {i + 1}: {header}"):
                if title:
                    st.markdown(f"**{title}**")
                st.write(row["_full_text"])
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Deceptive prob.", f"{p*100:.1f}%")
                mc2.metric("Trust Score",     row["Trust Score"])
                mc3.metric("Verified",        row["Verified"])

        # ── Export ────────────────────────────────────────────────────────────
        st.divider()
        st.download_button(
            "⬇️ Download all results as CSV",
            data=results_to_csv(results_df),
            file_name=f"reviews_{asin_preview}_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Bulk CSV upload
# ─────────────────────────────────────────────────────────────────────────────
with tab_bulk:
    st.markdown(
        "Upload a CSV file of reviews for **batch analysis**. "
        "Ideal for e-commerce sellers, researchers, or platform moderators "
        "who need to screen large review datasets."
    )

    # Template download
    template_df = pd.DataFrame({
        "review_text": [
            "The product arrived on time and works as described. Battery life is great.",
            "BEST PRODUCT EVER!! BUY NOW!! Amazing quality 10/10 perfection!!!",
        ],
        "rating":              [4, 5],
        "verified_purchase":   [1, 0],
        "reviewer_review_count": [23, 1],
    })
    st.download_button(
        "⬇️ Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="review_template.csv",
        mime="text/csv",
    )

    st.markdown("**Required column:** `review_text`  \n"
                "**Optional columns:** `rating`, `verified_purchase`, `reviewer_review_count`")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        if "review_text" not in raw_df.columns:
            st.error("The uploaded file must contain a **`review_text`** column.")
            st.stop()

        st.success(f"Loaded **{len(raw_df):,} reviews**.")

        with st.expander("Preview uploaded data (first 5 rows)"):
            st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("🔍 Analyse All Reviews", type="primary",
                     use_container_width=True):
            with st.spinner(f"Analysing {len(raw_df):,} reviews…"):
                results_df = predict_batch(
                    raw_df.to_dict("records"), dataset, threshold, model_key
                )

            # Log each review to database
            for _, row in results_df.iterrows():
                log_prediction(
                    review_text=str(row["_full_text"])[:2000],
                    predicted_label=row["Verdict"],
                    confidence_score=float(row["_prob"]),
                    trust_score=float(row["Trust Score"]),
                    model_used=model_key,
                    dataset_used=dataset,
                    source="csv_upload",
                    rating=int(row["_rating"]),
                    verified_purchase=int(row["_verified"]),
                    reviewer_review_count=int(row["_review_count"]),
                )

            total   = len(results_df)
            n_dec   = int((results_df["_prob"] >= threshold).sum())
            n_gen   = total - n_dec
            pct_dec = n_dec / total * 100
            avg_t   = float(results_df["_prob"]
                            .apply(lambda p: (1 - p) * 100).mean())

            # ── Summary ───────────────────────────────────────────────────────
            st.divider()
            st.subheader("📊 Analysis Summary")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total reviews",     total)
            s2.metric("🔴 Deceptive",      f"{n_dec} ({pct_dec:.1f}%)")
            s3.metric("🟢 Genuine",        f"{n_gen} ({100-pct_dec:.1f}%)")
            s4.metric("Avg trust score",   f"{avg_t:.1f}/100")

            st.progress(max(0.0, min(1.0, (pct_dec or 0) / 100)),
                        text=f"Deceptive: **{pct_dec:.1f}%** · "
                             f"Genuine: **{100-pct_dec:.1f}%**")

            ch1, ch2 = st.columns(2)
            with ch1:
                st.subheader("Genuine vs Deceptive")
                st.plotly_chart(pie_chart(n_gen, n_dec),
                                use_container_width=True,
                                config={"displayModeBar": False})
            with ch2:
                st.subheader("Confidence distribution")
                st.plotly_chart(histogram_chart(list(results_df["_prob"])),
                                use_container_width=True,
                                config={"displayModeBar": False})

            # ── Full results table ─────────────────────────────────────────────
            st.divider()
            st.subheader("Detailed results")
            display_df = results_df.drop(columns=[c for c in results_df.columns
                                                   if c.startswith("_") or c == "Reviewer"],
                                         errors="ignore")
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Trust Score": st.column_config.ProgressColumn(
                        "Trust Score", min_value=0, max_value=100, format="%.1f"
                    ),
                    "Deceptive %": st.column_config.NumberColumn(
                        "Deceptive %", format="%.1f%%"
                    ),
                    "Review (preview)": st.column_config.TextColumn(
                        "Review preview", width="large"
                    ),
                },
                hide_index=True,
            )

            # Add original columns back for export
            export_df = raw_df.copy()
            export_df["verdict"]               = results_df["Verdict"].values
            export_df["trust_score"]           = results_df["Trust Score"].values
            export_df["deceptive_probability_%"] = results_df["Deceptive %"].values

            st.download_button(
                "⬇️ Download full results as CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"bulk_analysis_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Analytics dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("📊 Usage Analytics")

    if not db_available():
        st.info(
            "**Database not connected.** "
            "Add `SUPABASE_URL` and `SUPABASE_KEY` to `.streamlit/secrets.toml` "
            "(local) or the Streamlit Cloud Secrets panel to enable analytics.",
            icon="🗄️",
        )
    else:
        col_refresh, _ = st.columns([1, 5])
        with col_refresh:
            refresh = st.button("🔄 Refresh", use_container_width=True)

        stats = get_statistics()
        recent = get_recent_predictions(limit=100)

        if not stats:
            st.warning("No predictions logged yet. Analyse some reviews first!")
        else:
            # ── Overview metrics ─────────────────────────────────────────────
            st.divider()
            st.subheader("Overview")
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Total predictions",  f"{stats.get('total', 0):,}")
            a2.metric("🔴 Deceptive",       f"{stats.get('deceptive', 0):,}")
            a3.metric("🟢 Genuine",         f"{stats.get('genuine', 0):,}")
            a4.metric("Deceptive rate",     f"{stats.get('pct_deceptive', 0):.1f}%")
            a5.metric("With feedback",      f"{stats.get('with_feedback', 0):,}")

            total = stats.get("total", 1)
            pct_dec = stats.get("pct_deceptive", 0) / 100
            st.progress(max(0.0, min(1.0, pct_dec)),
                        text=f"Overall deceptive rate: **{stats.get('pct_deceptive', 0):.1f}%**")

            # ── Charts ────────────────────────────────────────────────────────
            st.divider()
            ch1, ch2 = st.columns(2)
            with ch1:
                st.subheader("Genuine vs Deceptive (all time)")
                st.plotly_chart(
                    pie_chart(stats.get("genuine", 0), stats.get("deceptive", 0)),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with ch2:
                if recent:
                    st.subheader("Confidence distribution (recent 100)")
                    probs = [r.get("confidence_score", 0) for r in recent]
                    st.plotly_chart(
                        histogram_chart(probs),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            # ── Recent predictions table ──────────────────────────────────────
            if recent:
                st.divider()
                st.subheader("Recent predictions")
                recent_df = pd.DataFrame(recent)

                # Clean up for display
                display_cols = ["created_at", "predicted_label", "confidence_score",
                                "trust_score", "model_used", "source", "user_feedback",
                                "review_text"]
                display_cols = [c for c in display_cols if c in recent_df.columns]
                display_recent = recent_df[display_cols].copy()
                if "review_text" in display_recent.columns:
                    display_recent["review_text"] = display_recent["review_text"].str[:80] + "…"
                if "created_at" in display_recent.columns:
                    display_recent["created_at"] = pd.to_datetime(
                        display_recent["created_at"]
                    ).dt.strftime("%Y-%m-%d %H:%M")

                st.dataframe(
                    display_recent,
                    use_container_width=True,
                    column_config={
                        "confidence_score": st.column_config.NumberColumn(
                            "Confidence", format="%.3f"
                        ),
                        "trust_score": st.column_config.ProgressColumn(
                            "Trust", min_value=0, max_value=100, format="%.1f"
                        ),
                        "review_text": st.column_config.TextColumn(
                            "Review preview", width="large"
                        ),
                    },
                    hide_index=True,
                )

                # Export analytics data
                st.download_button(
                    "⬇️ Export analytics data (CSV)",
                    data=recent_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"analytics_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv",
                )

        # ── Feedback quality ──────────────────────────────────────────────────
        if stats and stats.get("with_feedback", 0) > 0:
            st.divider()
            st.subheader("Model feedback")
            st.markdown(
                f"**{stats['with_feedback']:,}** predictions have been reviewed by users. "
                "This labeled data will be used to retrain and improve the model. "
                "The retraining pipeline runs automatically every week via GitHub Actions."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — About & Methodology
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("🎯 What this tool does")
        st.markdown(
            "The Deceptive Review Detector uses a **Hybrid Random Forest** classifier "
            "that combines two complementary feature sets:\n\n"
            "1. **TF-IDF text features** — 5 000 unigram & bigram weights that capture "
            "vocabulary patterns associated with fake reviews (e.g., excessive superlatives, "
            "promotional language, lack of specific detail).\n\n"
            "2. **Behavioural features** — 10 metadata signals extracted from the reviewer "
            "and their review (star rating deviation, caps ratio, exclamation count, "
            "review length, verified-purchase flag, reviewer history)."
        )

        st.subheader("🏗️ Model architecture")
        st.markdown("""
| Component | Detail |
|---|---|
| Text vectoriser | TF-IDF, max 5 000 features, (1,2)-grams, sublinear TF |
| Behavioural scaler | StandardScaler (zero mean, unit variance) |
| Classifier | Hybrid Random Forest (GridSearchCV tuned) |
| Training datasets | Amazon product reviews · Yelp restaurant reviews |
| Classification threshold | Adjustable (default 0.55) |
| Database | Supabase (PostgreSQL) — predictions logged for retraining |
        """)

        st.subheader("🚩 Deceptive review signals")
        st.markdown("""
- Excessive superlatives and all-caps words ("BEST EVER", "PERFECT")
- Unusual punctuation density (multiple `!` or `?`)
- Unverified purchase combined with high rating
- Very short review history (first-time or single-review accounts)
- Significant deviation from the product's average star rating
- Generic promotional phrasing lacking specific product detail
        """)

    with col_b:
        st.subheader("📐 Trust Score scale")
        st.markdown("""
| Score | Label | Meaning |
|---|---|---|
| 70 – 100 | 🟢 High Trust | Review appears authentic |
| 40 – 69  | 🟡 Moderate   | Some suspicious signals |
| 0 – 39   | 🔴 Low Trust  | Strong deceptive indicators |
        """)

        st.subheader("⚠️ Limitations")
        st.markdown(
            "- The model is probabilistic — a 'Deceptive' verdict is a **signal, "
            "not a certainty**.\n"
            "- It was trained on English-language reviews; accuracy may be lower "
            "for other languages.\n"
            "- Sophisticated AI-generated reviews may evade detection.\n"
            "- Metadata defaults are used when reviewer history is unavailable.\n"
            "- Do not use the output as sole evidence in legal or business decisions."
        )

        st.subheader("🔄 Continuous improvement")
        st.markdown(
            "Every prediction is logged to a **Supabase** database (when connected). "
            "User feedback (✅ / ❌) creates labeled training data. "
            "A GitHub Actions workflow retrains the model weekly using this feedback, "
            "making the detector progressively more accurate over time."
        )

        st.subheader("🏢 Commercial & API use")
        st.markdown(
            "For commercial deployment or high-volume API access, "
            "consider wrapping the inference pipeline with **FastAPI** "
            "and hosting on a cloud provider. "
            "Replace the Amazon scraper with the official "
            "[Product Advertising API](https://affiliate-program.amazon.com/) "
            "to remain ToS-compliant."
        )

    st.divider()
    st.subheader("⚖️ Legal & Ethical Disclaimer")
    st.markdown(
        '<div class="card card-yellow">'
        "<p class='card-body'>"
        "This tool is provided <b>for research and educational purposes only</b>. "
        "Results must not be used as the sole basis for legal action, business decisions, "
        "or public accusations against individual reviewers or products. "
        "Web scraping of third-party platforms (including Amazon) may be subject to "
        "their Terms of Service and applicable law — users are responsible for "
        "ensuring their use is lawful. "
        "No personally identifiable information is stored or transmitted by this application."
        "</p></div>",
        unsafe_allow_html=True,
    )

# ── Footer (all tabs) ─────────────────────────────────────────────────────────
render_footer()
