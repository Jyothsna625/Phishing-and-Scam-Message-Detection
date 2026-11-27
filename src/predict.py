"""Inference helper for classifying user-submitted messages."""

import json
from functools import lru_cache
from typing import Any

import joblib

from src.config import ID_TO_LABEL, MODEL_METADATA_PATH, MODEL_PATH, VECTORIZER_PATH
from src.feature_extraction import extract_scam_signals
from src.preprocessing import clean_text


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the saved model, TF-IDF vectorizer, and metadata once."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Trained model artifacts not found. Run `python -m src.train` first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata: dict[str, Any] = {}
    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return model, vectorizer, metadata


def get_model_metadata() -> dict[str, Any]:
    """Return metadata for the currently deployed model."""
    _, _, metadata = load_artifacts()
    return metadata


def predict_message(message: str) -> dict[str, Any]:
    """Classify a single message as legitimate or phishing/scam."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")

    cleaned = clean_text(message)
    if not cleaned:
        raise ValueError("Message contains no analyzable text after preprocessing.")

    model, vectorizer, metadata = load_artifacts()
    features = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(features)[0]
    phishing_prob = float(probabilities[1])
    pred_id = int(model.predict(features)[0])
    label = ID_TO_LABEL[pred_id]
    is_phishing = pred_id == 1

    if phishing_prob >= 0.75:
        risk_level = "High"
    elif phishing_prob >= 0.45:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    confidence = phishing_prob if is_phishing else (1.0 - phishing_prob)

    return {
        "label": label,
        "prediction": "Phishing / Scam" if is_phishing else "Legitimate",
        "is_phishing": is_phishing,
        "confidence": round(confidence * 100, 2),
        "phishing_probability": round(phishing_prob * 100, 2),
        "risk_level": risk_level,
        "signals": extract_scam_signals(message),
        "cleaned_text": cleaned,
        "model_used": metadata.get("best_model", type(model).__name__),
    }
