"""Flask web application for the Naive Bayes spam filter.

Run locally with:  python app.py   (opens http://127.0.0.1:<PORT>)
Run on Render with:  gunicorn app:app --bind 0.0.0.0:$PORT

The server binds to 0.0.0.0 (all network interfaces) and uses the PORT
environment variable, defaulting to 10000, so it works both locally and on
Render without further configuration.

On startup the application runs the FULL training pipeline exactly once:
  1. load spam.csv
  2. clean the dataset
  3. stratified 80/20 train/test split
  4. fit the TF-IDF vectorizer (word + character n-grams) on training data
  5. train the Multinomial Naive Bayes classifier (uniform class priors)
  6. evaluate the model on the held-out test set
  7. keep the trained pipeline in memory
  8. start the Flask server

The model is NEVER retrained on user requests.
"""

import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.indicators import detect_indicators
from src.predict import predict_details
from src.preprocessing import clean_message

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MAX_MESSAGE_LENGTH = 5000

app = Flask(__name__)


def _train_on_startup():
    """Load, clean, split, train and evaluate the model once at startup."""
    from src.evaluate import evaluate_model
    from src.preprocessing import load_and_preprocess_data, print_dataset_stats
    from src.train_model import split_data, train_naive_bayes

    X, y = load_and_preprocess_data(DATA_DIR)
    print_dataset_stats(X, y)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print()
    print("=" * 60)
    print("TRAIN / TEST SPLIT (stratified 80/20)")
    print("=" * 60)
    print(f"Training messages    : {len(X_train)}")
    print(f"Testing messages     : {len(X_test)}")

    print()
    print("Training Multinomial Naive Bayes on TF-IDF features...")
    vectorizer, model = train_naive_bayes(X_train, y_train)
    print("Training complete. Evaluating on the held-out test set...")

    evaluate_model(model, vectorizer, X_test, y_test, results_dir=RESULTS_DIR)

    return vectorizer, model


try:
    vectorizer, model = _train_on_startup()
    MODEL_READY = True
except Exception as exc:  # noqa: BLE001 - do not crash the web app
    vectorizer, model = None, None
    MODEL_READY = False
    print("WARNING: could not train the model: " + str(exc))


def risk_level(prediction, spam_prob):
    """UI risk level derived from the prediction and the spam probability.

    - Prediction = SPAM : >= 80% spam  -> HIGH, >= 60% -> MEDIUM, else LOW
    - Prediction = HAM  : < 30% spam   -> LOW,  30-60% -> MEDIUM, else HIGH
    """
    if prediction == "SPAM":
        if spam_prob >= 80.0:
            return "HIGH"
        if spam_prob >= 60.0:
            return "MEDIUM"
        return "LOW"
    if spam_prob < 30.0:
        return "LOW"
    if spam_prob <= 60.0:
        return "MEDIUM"
    return "HIGH"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return (
            jsonify({"error": "Model not trained. Run 'python train_model.py' first."}),
            500,
        )

    message = request.form.get("message", "").strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return (
            jsonify(
                {"error": f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."}
            ),
            400,
        )

    cleaned = clean_message(message)
    result = predict_details(cleaned, vectorizer, model)
    result["risk_level"] = risk_level(result["prediction"], result["spam_probability"])
    result["indicators"] = detect_indicators(cleaned)
    result["message"] = cleaned
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
