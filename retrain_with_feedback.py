"""
Retraining pipeline — merges user-feedback labeled data from Supabase
with the existing synthetic training data and retrains all models.

Run locally:
    python retrain_with_feedback.py

Run by GitHub Actions (see .github/workflows/retrain.yml).
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Ensure the project root is on sys.path when called from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database import get_labeled_reviews_for_training
from src.train import train_dataset, _ensure_directories


def _load_base_dataset(name: str) -> pd.DataFrame:
    path = Path("data") / f"{name}_reviews.csv"
    if not path.exists():
        raise FileNotFoundError(f"Base dataset not found: {path}")
    return pd.read_csv(path)


def _append_feedback_rows(base_df: pd.DataFrame, feedback_rows: list[dict]) -> pd.DataFrame:
    if not feedback_rows:
        LOGGER.info("No feedback rows to append.")
        return base_df

    fb_df = pd.DataFrame(feedback_rows)
    # Align columns — fill missing optional cols with defaults
    for col, default in [
        ("rating", 5),
        ("verified_purchase", 0),
        ("reviewer_review_count", 1),
        ("product_id", "P00000"),
        ("reviewer_id", "RFEEDBACK"),
        ("review_date", "2024-01-01"),
    ]:
        if col not in fb_df.columns:
            fb_df[col] = default

    fb_df = fb_df[base_df.columns]
    combined = pd.concat([base_df, fb_df], ignore_index=True)
    LOGGER.info(
        "Dataset augmented: %d base rows + %d feedback rows = %d total",
        len(base_df), len(fb_df), len(combined),
    )
    return combined


def retrain(datasets: list[str] | None = None) -> None:
    datasets = datasets or ["amazon", "yelp"]

    LOGGER.info("Fetching user-feedback labeled rows from Supabase…")
    feedback_rows = get_labeled_reviews_for_training()
    LOGGER.info("Fetched %d labeled rows with user feedback.", len(feedback_rows))

    _ensure_directories()

    for name in datasets:
        LOGGER.info("=== Retraining: %s ===", name)
        try:
            base_df = _load_base_dataset(name)
        except FileNotFoundError as exc:
            LOGGER.error("%s — skipping.", exc)
            continue

        augmented = _append_feedback_rows(base_df, feedback_rows)

        # Write augmented CSV temporarily so train_dataset() can read it
        tmp_path = Path("data") / f"{name}_reviews.csv"
        augmented.to_csv(tmp_path, index=False)
        LOGGER.info("Wrote augmented dataset to %s", tmp_path)

        try:
            train_dataset(name)
            LOGGER.info("Retrained %s successfully.", name)
        except Exception as exc:
            LOGGER.error("Training failed for %s: %s", name, exc)
            # Restore original
            base_df.to_csv(tmp_path, index=False)
            raise


if __name__ == "__main__":
    retrain()
    LOGGER.info("Retraining complete.")
