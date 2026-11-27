"""Configuration constants and file paths for the project."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "messages_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_messages.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_SUMMARY_PATH = REPORTS_DIR / "metrics_summary.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 3000
NGRAM_RANGE = (1, 2)

LABEL_TO_ID = {"legitimate": 0, "phishing": 1}
ID_TO_LABEL = {0: "legitimate", 1: "phishing"}
