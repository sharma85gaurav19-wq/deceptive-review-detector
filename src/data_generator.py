"""Synthetic dataset generator for Amazon and Yelp review data."""

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _sample_date(rng: np.random.Generator) -> str:
    start = datetime(2022, 1, 1)
    end = datetime(2024, 1, 1)
    delta = end - start
    days = int(rng.integers(0, delta.days))
    return (start + timedelta(days=days)).date().isoformat()


def _build_review_text(label: int, rng: np.random.Generator, product_vocab: List[str], domain: str) -> str:
    if label == 0:
        # genuine review patterns
        adjectives = ["good", "solid", "clean", "comfortable", "reliable", "fast"]
        connectors = ["and", "but", "so", "because", "with", "while"]
        templates = [
            "I received the {product} on time and the {feature} is very {adj}.",
            "The {product} comes with great {feature} and the {aspect} makes it worth buying.",
            "Delivery was quick and the {feature} quality is impressive.",
            "I like the {aspect}, especially the {feature} and the overall {service}.",
            "The {product} is useful for family use and the {feature} works as expected.",
        ]
        sentence_count = rng.integers(3, 7)
        sentences = []
        for _ in range(sentence_count):
            template = rng.choice(templates)
            feature = rng.choice(product_vocab)
            product = rng.choice(["item", "product", "service", "place", "food"])
            aspect = rng.choice(["size", "color", "fit", "taste", "ambience", "value"])
            service = rng.choice(["support", "delivery", "packaging", "experience", "staff"])
            adj = rng.choice(adjectives)
            sentence = template.format(product=product, feature=feature, aspect=aspect, service=service, adj=adj)
            if rng.random() < 0.3:
                sentence = sentence.replace("the", "the final")
            if rng.random() < 0.2:
                sentence += " I will buy again."
            sentences.append(sentence)
        body = " ".join(sentences)
        if len(body) < 80:
            body += " " + " ".join(rng.choice(product_vocab, size=5, replace=True))
        return body[:400]

    # deceptive review pattern
    phrases = [
        "amazing", "highly recommend", "absolutely", "best", "perfect", "love it", "must buy",
        "great quality", "value for money", "five stars", "so good", "do not miss", "very happy",
        "10/10", "best ever", "life changing", "buy it now"
    ]
    fragments = [
        "This is the best purchase ever.",
        "I absolutely love it and would buy again.",
        "Highly recommend for family and friends.",
        "Perfect choice if you want amazing value.",
        "Great quality product with fantastic results.",
        "Five stars, very happy with everything.",
        "The best product I have ever used in my entire life.",
        "Everything was absolutely perfect and I cannot imagine life without it.",
        "You must buy this immediately if you want a perfect experience.",
        "Amazing quality, 10/10 would recommend to everyone.",
        "Simply perfect, nothing else comes close.",
    ]
    sentence_count = rng.integers(2, 4)
    sentences = []
    for _ in range(sentence_count):
        sentence = rng.choice(fragments)
        if rng.random() < 0.4:
            sentence = sentence.replace("product", rng.choice(["item", "service", "dish", "business"]))
        if rng.random() < 0.3:
            sentence += " " + rng.choice(phrases).capitalize() + "."
        sentences.append(sentence)
    body = " ".join(sentences)
    body = body.replace("  ", " ").strip()
    if len(body) < 40:
        body += " awesome experience"
    return body[:250]


def _add_deceptive_emphasis(text: str, rng: np.random.Generator) -> str:
    """Add realistic shouty emphasis (caps, exclamation bursts) to deceptive text."""
    style = rng.random()
    if style < 0.2:
        text = text.upper()
    elif style < 0.6:
        # uppercase a handful of hype words, like real fake reviews do
        words = text.split()
        for i in range(len(words)):
            if rng.random() < 0.18 and len(words[i]) > 3:
                words[i] = words[i].upper()
        text = " ".join(words)
    # exclamation bursts: replace some sentence-ending periods and append a burst
    if rng.random() < 0.7:
        text = text.replace(".", "!" * int(rng.integers(1, 4)), int(rng.integers(1, 4)))
    text += "!" * int(rng.integers(0, 5))
    return text


def _draw_reviewer_id(index: int) -> str:
    return f"R{index:06d}"


def _draw_product_id(index: int, kind: str) -> str:
    prefix = "P" if kind == "amazon" else "B"
    return f"{prefix}{index:05d}"


def _generate_dataset(name: str, n_rows: int, product_count: int, rng: np.random.Generator) -> pd.DataFrame:
    product_vocab = [
        "battery", "screen", "delivery", "packaging", "size", "color", "fit", "taste", "service",
        "ambience", "quality", "sound", "texture", "price", "comfort", "style", "performance", "design"
    ]
    rows = []
    labels = np.array([1] * int(math.floor(n_rows * 0.22)) + [0] * int(math.ceil(n_rows * 0.78)))
    rng.shuffle(labels)
    product_ids = [_draw_product_id(i + 1, name) for i in range(product_count)]
    reviewer_pool = [_draw_reviewer_id(i + 1) for i in range(int(n_rows * 0.4))]

    product_ratings: Dict[str, List[int]] = {pid: [] for pid in product_ids}
    for label in labels:
        if label == 0:
            rating = int(rng.choice([3, 4, 4, 5, 5, 5]))
        else:
            rating = int(rng.choice([1, 1, 1, 5, 5, 5, 5]))
        product_ratings[rng.choice(product_ids)].append(rating)

    for idx, label in enumerate(labels):
        product_id = rng.choice(product_ids)
        if label == 0:
            # ~10% of genuine reviewers write over-enthusiastic, deceptive-sounding
            # text; their metadata (verified, history) still marks them genuine.
            # These are only separable through behavioural features.
            mimic = rng.random() < 0.10
            review_text = _build_review_text(1 if mimic else 0, rng, product_vocab, name)
            if mimic and rng.random() < 0.5:
                review_text += "!" * int(rng.integers(1, 3))
            rating = int(rng.choice([3, 4, 4, 5, 5, 5]))
            verified = int(rng.random() < 0.85)
            reviewer_review_count = int(rng.lognormal(2.5, 1.0)) + 1
        else:
            # ~15% of deceptive reviews mimic genuine writing style (paid reviewers
            # copying real reviews); their metadata still betrays them.
            mimic = rng.random() < 0.15
            review_text = _build_review_text(0 if mimic else 1, rng, product_vocab, name)
            if not mimic:
                review_text = _add_deceptive_emphasis(review_text, rng)
            rating = int(rng.choice([1, 1, 1, 5, 5, 5, 5]))
            verified = int(rng.random() < 0.35)
            reviewer_review_count = int(rng.poisson(3)) + 1
        reviewer_id = rng.choice(reviewer_pool)
        date = _sample_date(rng)
        rows.append({
            "review_text": review_text,
            "rating": rating,
            "verified_purchase": verified,
            "reviewer_id": reviewer_id,
            "product_id": product_id,
            "review_date": date,
            "reviewer_review_count": reviewer_review_count,
            "label": int(label),
        })

    return pd.DataFrame(rows)


def generate_synthetic_datasets() -> None:
    """Generate the Amazon and Yelp synthetic review CSV datasets."""
    rng = np.random.default_rng(42)
    amazon = _generate_dataset("amazon", 21000, 800, rng)
    yelp = _generate_dataset("yelp", 38000, 1500, rng)

    Path("data").mkdir(parents=True, exist_ok=True)
    amazon.to_csv(Path("data") / "amazon_reviews.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    yelp.to_csv(Path("data") / "yelp_reviews.csv", index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    generate_synthetic_datasets()
