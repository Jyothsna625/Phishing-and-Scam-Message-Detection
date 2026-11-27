"""Training pipeline for Naive Bayes and Logistic Regression classifiers."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.config import (
    LABEL_TO_ID,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_SIZE,
    VECTORIZER_PATH,
)
from src.evaluate import compute_classification_metrics, save_evaluation_reports
from src.feature_extraction import build_tfidf_vectorizer
from src.preprocessing import preprocess_dataframe


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and validate the raw labeled messages dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    required = {"label", "message"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    valid_labels = set(LABEL_TO_ID.keys())
    unknown = set(df["label"].unique()) - valid_labels
    if unknown:
        raise ValueError(f"Unsupported labels found: {sorted(unknown)}")

    return df


def train_and_evaluate(dataset_path: Path = RAW_DATA_PATH) -> dict:
    """Run preprocessing, train candidate models, evaluate, and save artifacts."""
    raw_df = load_raw_dataset(dataset_path)
    processed_df = preprocess_dataframe(raw_df, text_col="message")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    X = processed_df["cleaned_message"].values
    y = processed_df["label"].map(LABEL_TO_ID).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vectorizer = build_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    trained_models = {}
    evaluation_results = {}

    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        y_prob = model.predict_proba(X_test_vec)[:, 1]
        trained_models[name] = model
        evaluation_results[name] = compute_classification_metrics(y_test, y_pred, y_prob)

    best_model_name = max(
        evaluation_results.keys(),
        key=lambda name: (
            evaluation_results[name]["f1_score"],
            evaluation_results[name]["roc_auc"],
            evaluation_results[name]["accuracy"],
        ),
    )
    best_model = trained_models[best_model_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metadata = {
        "best_model": best_model_name,
        "dataset_size": int(len(processed_df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "class_distribution": processed_df["label"].value_counts().to_dict(),
        "metrics": {
            name: {k: v for k, v in vals.items() if k != "roc_curve"}
            for name, vals in evaluation_results.items()
        },
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_evaluation_reports(evaluation_results, best_model_name)
    return metadata


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(f"Training complete. Best model: {summary['best_model']}")
    print(json.dumps(summary["metrics"], indent=2))
