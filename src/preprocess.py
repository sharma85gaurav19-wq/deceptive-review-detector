"""Text preprocessing utilities for deceptive review detection."""

import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(s: str) -> str:
    """Normalize and clean review text for TF-IDF processing."""
    if not isinstance(s, str):
        return ""

    text = s.lower()
    # replace all punctuation characters with spaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [token for token in text.split() if token not in ENGLISH_STOP_WORDS and len(token) >= 2]
    return " ".join(tokens)
