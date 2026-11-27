"""NLP text preprocessing utilities for phishing and scam message detection."""

import re
import string
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s\(\)]{7,}\d")
CURRENCY_PATTERN = re.compile(r"[$€£₹]\s*\d+(?:[.,]\d+)?")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Keep negation words because phrases like "never share" or "do not click" matter.
NEGATION_WORDS = {"no", "not", "never", "neither", "nor", "cannot", "dont", "wont"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Normalize and clean a raw message for TF-IDF feature extraction.

    Steps:
    1. Convert text to lowercase.
    2. Replace URLs, email addresses, phone numbers, and currency amounts with
       semantic tokens so structural scam cues are preserved.
    3. Remove unnecessary punctuation and special characters.
    4. Remove common English stopwords (keeping negation terms).
    """
    if not isinstance(text, str):
        return ""

    normalized = text.lower().strip()
    normalized = URL_PATTERN.sub(" urllink ", normalized)
    normalized = EMAIL_PATTERN.sub(" emailaddr ", normalized)
    normalized = CURRENCY_PATTERN.sub(" moneyamount ", normalized)
    normalized = PHONE_PATTERN.sub(" phonenum ", normalized)

    # Remove punctuation and non-alphanumeric characters
    translation_table = str.maketrans("", "", string.punctuation)
    normalized = normalized.translate(translation_table)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    tokens = normalized.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    else:
        tokens = [token for token in tokens if len(token) > 1]

    cleaned = WHITESPACE_PATTERN.sub(" ", " ".join(tokens)).strip()
    return cleaned


def preprocess_series(messages: Iterable[str], remove_stopwords: bool = True) -> list[str]:
    """Clean an iterable of raw message strings."""
    return [clean_text(msg, remove_stopwords=remove_stopwords) for msg in messages]


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
    """Return a copy of the DataFrame with a `cleaned_message` column added."""
    if text_col not in df.columns:
        raise ValueError(f"Expected column '{text_col}' in dataset.")

    processed = df.copy()
    processed[text_col] = processed[text_col].fillna("").astype(str)
    processed["cleaned_message"] = preprocess_series(processed[text_col])
    processed = processed[processed["cleaned_message"].str.len() > 0].reset_index(drop=True)
    return processed
