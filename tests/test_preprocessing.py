"""Tests for NLP preprocessing and heuristic signal extraction."""

import unittest

from src.feature_extraction import extract_scam_signals
from src.preprocessing import clean_text


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_lowercases_and_normalizes_urls(self):
        raw = "URGENT!! Visit HTTP://SCAM-SITE.COM to claim $500!!!"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, cleaned.lower())
        self.assertIn("urllink", cleaned)
        self.assertIn("moneyamount", cleaned)
        self.assertNotIn("!", cleaned)

    def test_clean_text_preserves_negations(self):
        cleaned = clean_text("Never share your OTP with anyone.")
        self.assertIn("never", cleaned)

    def test_extract_scam_signals_flags_phishing_cues(self):
        msg = "URGENT: Your account is suspended! Verify password at http://fake-login.com"
        signals = extract_scam_signals(msg)
        titles = {s["title"] for s in signals}
        self.assertIn("Suspicious Link Detected", titles)
        self.assertIn("Urgency or Threat Language", titles)
        self.assertIn("Credential or Sensitive Info Request", titles)


if __name__ == "__main__":
    unittest.main()
