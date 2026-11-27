# Phishing and Scam Message Detection Using Machine Learning

A machine learning and Natural Language Processing (NLP) web application that classifies SMS, email, and chat messages as **Legitimate** or **Phishing / Scam**, highlights suspicious cues, and provides confidence scores through a Flask web interface.

## Project Structure

```text
Phishing and Scam Message Detection Using Machine Learning/
├── data/
│   ├── raw/
│   │   └── messages_dataset.csv
│   └── processed/
│       └── cleaned_messages.csv
├── notebooks/
│   └── eda_and_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── phishing_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metadata.json
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
├── reports/
│   ├── metrics_summary.json
│   └── figures/
│       ├── confusion_matrix.png
│       ├── model_comparison.png
│       └── roc_curve.png
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

- **NLP Text Preprocessing (`src/preprocessing.py`)**: Lowercasing, URL/email/currency/phone normalization, punctuation & special-character removal, and stopword filtering (preserving negations).
- **Feature Extraction & Signal Explainability (`src/feature_extraction.py`)**: TF-IDF n-gram features (`1, 2`) plus rule-based scam cue detection (suspicious links, urgency threats, credential requests, prize lures, payment demands).
- **Model Training & Evaluation (`src/train.py`, `src/evaluate.py`)**: Compares **Multinomial Naive Bayes** and **Logistic Regression** on a stratified 80/20 split, saves the best model to `models/`, and generates evaluation charts in `reports/figures/`.
- **Interactive Web UI (`app/main.py`)**: Flask application with `/api/predict` endpoint, sample messages, risk badge, confidence breakdown, detected signals, and preprocessed text preview.

## Latest Evaluation Results

Evaluated on a stratified 20% test split (32 messages out of 160):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Bayes | 87.50% | 83.33% | 93.75% | 88.24% | 0.9727 |
| Logistic Regression (Selected) | 96.88% | 94.12% | 100.00% | 96.97% | 1.0000 |

## Setup & Usage

1. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. Train models and generate evaluation reports:
   ```bash
   python -m src.train
   ```

3. Run unit tests:
   ```bash
   python -m unittest discover -s tests -v
   ```

4. Start the web application:
   ```bash
   python -m app.main
   ```
   Then open `http://127.0.0.1:5000` in your browser.
