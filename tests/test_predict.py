"""Tests for model inference and Flask web endpoints."""

import unittest

from app.main import app
from src.predict import predict_message


class TestPredictAndApp(unittest.TestCase):
    def test_predict_phishing_message(self):
        result = predict_message(
            "URGENT: Your bank account is suspended. Verify your PIN and password at http://bank-verify.biz"
        )
        self.assertTrue(result["is_phishing"])
        self.assertEqual(result["label"], "phishing")
        self.assertGreaterEqual(result["phishing_probability"], 50.0)

    def test_predict_legitimate_message(self):
        result = predict_message(
            "Hi team, please review the updated project timeline before Friday's standup."
        )
        self.assertFalse(result["is_phishing"])
        self.assertEqual(result["label"], "legitimate")

    def test_predict_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            predict_message("   ")

    def test_flask_routes(self):
        client = app.test_client()

        home_resp = client.get("/")
        self.assertEqual(home_resp.status_code, 200)

        health_resp = client.get("/api/health")
        self.assertEqual(health_resp.status_code, 200)

        bad_resp = client.post("/api/predict", json={"message": ""})
        self.assertEqual(bad_resp.status_code, 400)

        ok_resp = client.post(
            "/api/predict",
            json={"message": "Congratulations! You won a $1000 gift card at http://win-prize.biz"},
        )
        self.assertEqual(ok_resp.status_code, 200)
        payload = ok_resp.get_json()
        self.assertIn("prediction", payload)
        self.assertIn("confidence", payload)


if __name__ == "__main__":
    unittest.main()
