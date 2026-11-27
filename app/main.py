"""Flask web server for Phishing and Scam Message Detection."""

from flask import Flask, jsonify, render_template, request

from src.predict import get_model_metadata, predict_message

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render the main message classification interface."""
    try:
        metadata = get_model_metadata()
    except FileNotFoundError:
        metadata = {}
    return render_template("index.html", metadata=metadata)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint reporting loaded model metadata."""
    try:
        metadata = get_model_metadata()
        return jsonify({"status": "ok", "model": metadata.get("best_model", "loaded")}), 200
    except FileNotFoundError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503


@app.route("/api/predict", methods=["POST"])
def predict():
    """Classify a submitted SMS, email, or chat message."""
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please enter a message to analyze."}), 400

    try:
        result = predict_message(message)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
