"""
Supabase database integration.
Logs every prediction, collects user feedback, and supplies
labeled data for periodic model retraining.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# ── Lazy client (imported only when credentials are present) ──────────────────
try:
    from supabase import create_client, Client as SupabaseClient
    _SUPABASE_LIB = True
except ImportError:
    _SUPABASE_LIB = False
    LOGGER.warning("supabase-py not installed — DB logging disabled.")


def _get_client() -> Optional[Any]:
    """Return a Supabase client, or None if credentials are missing."""
    if not _SUPABASE_LIB:
        return None
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", "") or os.getenv("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "") or os.getenv("SUPABASE_KEY", "")
    except Exception:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as exc:
        LOGGER.error("Supabase connect failed: %s", exc)
        return None


def db_available() -> bool:
    """True if a Supabase connection can be established."""
    return _get_client() is not None


# ── Write operations ──────────────────────────────────────────────────────────

def log_prediction(
    review_text: str,
    predicted_label: str,
    confidence_score: float,
    trust_score: float,
    model_used: str,
    dataset_used: str,
    source: str = "manual",
    product_asin: Optional[str] = None,
    rating: int = 5,
    verified_purchase: int = 0,
    reviewer_review_count: int = 1,
) -> Optional[str]:
    """
    Insert one prediction row.  Returns the row UUID, or None on failure.
    Never raises — database errors must not crash the UI.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        row = {
            "review_text":           review_text[:2000],
            "predicted_label":       predicted_label,
            "confidence_score":      round(float(confidence_score), 4),
            "trust_score":           round(float(trust_score), 1),
            "model_used":            model_used,
            "dataset_used":          dataset_used,
            "source":                source,
            "product_asin":          product_asin,
            "rating":                int(rating),
            "verified_purchase":     bool(verified_purchase),
            "reviewer_review_count": int(reviewer_review_count),
        }
        res = client.table("predictions").insert(row).execute()
        return res.data[0]["id"] if res.data else None
    except Exception as exc:
        LOGGER.error("log_prediction failed: %s", exc)
        return None


def update_feedback(record_id: str, feedback: str) -> bool:
    """
    Attach user feedback ('correct' | 'incorrect') to a prediction row.
    This labeled data is used for retraining.
    """
    if not record_id:
        return False
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("predictions").update(
            {"user_feedback": feedback}
        ).eq("id", record_id).execute()
        return True
    except Exception as exc:
        LOGGER.error("update_feedback failed: %s", exc)
        return False


# ── Read operations ───────────────────────────────────────────────────────────

def get_statistics() -> Dict[str, Any]:
    """Return aggregate counts for the analytics dashboard."""
    client = _get_client()
    if client is None:
        return {}
    try:
        total_res = client.table("predictions").select("id", count="exact").execute()
        total     = total_res.count or 0

        dec_res   = (client.table("predictions")
                     .select("id", count="exact")
                     .eq("predicted_label", "Deceptive")
                     .execute())
        deceptive = dec_res.count or 0

        fb_res    = (client.table("predictions")
                     .select("id", count="exact")
                     .not_.is_("user_feedback", "null")
                     .execute())
        with_feedback = fb_res.count or 0

        return {
            "total":           total,
            "deceptive":       deceptive,
            "genuine":         total - deceptive,
            "pct_deceptive":   round(deceptive / max(total, 1) * 100, 1),
            "with_feedback":   with_feedback,
        }
    except Exception as exc:
        LOGGER.error("get_statistics failed: %s", exc)
        return {}


def get_recent_predictions(limit: int = 100) -> List[Dict]:
    """Return the *limit* most recent predictions for the analytics table."""
    client = _get_client()
    if client is None:
        return []
    try:
        res = (client.table("predictions")
               .select("id,review_text,predicted_label,confidence_score,"
                       "trust_score,model_used,source,user_feedback,created_at")
               .order("created_at", desc=True)
               .limit(limit)
               .execute())
        return res.data or []
    except Exception as exc:
        LOGGER.error("get_recent_predictions failed: %s", exc)
        return []


def get_labeled_reviews_for_training() -> List[Dict]:
    """
    Return all predictions where the user confirmed or corrected the label.
    Used by the retraining pipeline to build a curated training set.
    """
    client = _get_client()
    if client is None:
        return []
    try:
        res = (client.table("predictions")
               .select("review_text,predicted_label,user_feedback,"
                       "rating,verified_purchase,reviewer_review_count")
               .not_.is_("user_feedback", "null")
               .execute())
        rows = res.data or []
        labeled = []
        for r in rows:
            # If user said 'correct', keep original label; 'incorrect' → flip
            orig  = r["predicted_label"]
            fb    = r["user_feedback"]
            label = orig if fb == "correct" else (
                "Genuine" if orig == "Deceptive" else "Deceptive"
            )
            labeled.append({
                "review_text":           r["review_text"],
                "label":                 1 if label == "Deceptive" else 0,
                "rating":                r.get("rating", 5),
                "verified_purchase":     int(r.get("verified_purchase", 0)),
                "reviewer_review_count": r.get("reviewer_review_count", 1),
                "product_id":            "P00000",
            })
        return labeled
    except Exception as exc:
        LOGGER.error("get_labeled_reviews_for_training failed: %s", exc)
        return []
