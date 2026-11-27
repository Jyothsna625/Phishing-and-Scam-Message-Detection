"""Feature extraction and heuristic scam-signal detection."""

import re
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import MAX_FEATURES, NGRAM_RANGE

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

SIGNAL_RULES = [
    (
        "Suspicious Link Detected",
        "high",
        re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE),
    ),
    (
        "Urgency or Threat Language",
        "high",
        re.compile(
            r"\b(urgent|immediately|suspended|blocked|locked|deactivated|final notice|last chance|expires?|within \d+ hours|arrest|warrant|blacklisted)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Credential or Sensitive Info Request",
        "high",
        re.compile(
            r"\b(password|pin|cvv|otp|ssn|aadhaar|pan card|seed phrase|recovery phrase|bank details|credit card|routing number|verify your identity|log\s?in|re-?authenticate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Prize, Reward, or Financial Lure",
        "medium",
        re.compile(
            r"\b(winner|won|prize|gift card|lottery|lucky draw|free|guaranteed|double your|cash|claim|voucher|airdrop|pre-approved)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Fee, Wire, or Payment Demand",
        "medium",
        re.compile(
            r"\b(clearance fee|processing charge|redelivery fee|wire|deposit|unpaid|overdue|settle|billing|payment failed|declined)\b",
            re.IGNORECASE,
        ),
    ),
]


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Create the TF-IDF vectorizer configured for message classification."""
    return TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        min_df=1,
    )


def extract_scam_signals(message: str) -> list[dict[str, str]]:
    """Identify human-readable phishing and scam cues in a raw message."""
    if not isinstance(message, str) or not message.strip():
        return []

    signals: list[dict[str, str]] = []
    for label, severity, pattern in SIGNAL_RULES:
        matches = pattern.findall(message)
        if matches:
            sample = matches[0] if isinstance(matches[0], str) else matches[0][0]
            signals.append(
                {
                    "title": label,
                    "severity": severity,
                    "evidence": sample.strip(),
                }
            )

    alpha_chars = [ch for ch in message if ch.isalpha()]
    if len(alpha_chars) >= 12:
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        if upper_ratio > 0.35:
            signals.append(
                {
                    "title": "Excessive Capitalization",
                    "severity": "low",
                    "evidence": f"{upper_ratio:.0%} uppercase letters",
                }
            )

    if message.count("!") >= 2:
        signals.append(
            {
                "title": "Multiple Exclamation Marks",
                "severity": "low",
                "evidence": f"{message.count('!')} exclamation marks",
            }
        )

    return signals
