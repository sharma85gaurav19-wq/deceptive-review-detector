"""Interactive demo for deceptive review prediction."""

import joblib
import pandas as pd
from pathlib import Path
from scipy import sparse

from src.features import compute_behavioral_features


def main() -> None:
    model_file = Path("outputs/models") / "models_amazon.joblib"
    vectorizer_file = Path("outputs/models") / "tfidf_amazon.joblib"
    scaler_file = Path("outputs/models") / "scaler_amazon.joblib"

    if not model_file.exists() or not vectorizer_file.exists() or not scaler_file.exists():
        raise FileNotFoundError("Model artifacts not found. Run: python main.py --train")

    models = joblib.load(model_file)
    hybrid_model = models["hybrid_rf"]
    vectorizer = joblib.load(vectorizer_file)
    scaler = joblib.load(scaler_file)

    print("Deceptive Review Detector Demo")
    print("Enter a review text and press Enter. Type 'exit' to quit.")
    while True:
        review_text = input("\nReview: ").strip()
        if review_text.lower() in {"exit", "quit"}:
            print("Exiting demo.")
            break
        if not review_text:
            continue

        df = pd.DataFrame([{
            "review_text": review_text,
            "rating": 5,
            "verified_purchase": 0,
            "product_id": "P00000",
            "reviewer_review_count": 1,
        }])

        text_vector = vectorizer.transform([review_text])
        behavioral = compute_behavioral_features(df)
        behavior_scaled = scaler.transform(behavioral)
        X_demo = sparse.hstack([text_vector, sparse.csr_matrix(behavior_scaled)], format="csr")

        prob = hybrid_model.predict_proba(X_demo)[0, 1]
        label = "DECEPTIVE" if prob >= 0.5 else "GENUINE"

        print(f"Prediction  : {label}")
        print(f"Confidence  : {prob * 100:.1f}% deceptive")


if __name__ == "__main__":
    main()
