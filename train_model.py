"""Train the Naive Bayes spam classifier and evaluate it.

Run with:  python train_model.py

Loads the Kaggle dataset, preprocesses it, trains a MultinomialNB model on
combined word + character TF-IDF features (uniform class priors so the class
imbalance does not bias the predictions), saves model + vectorizer with joblib
and prints the evaluation metrics (accuracy, precision, recall, F1, confusion
matrix) plus a list of misclassified test messages.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluate import evaluate_model, print_misclassified
from src.preprocessing import load_and_preprocess_data, print_dataset_stats
from src.train_model import (
    compare_models,
    save_model,
    split_data,
    train_naive_bayes,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def run_training_pipeline():
    print("BAYESIAN SPAM FILTERING - NAIVE BAYES")
    print("Dataset source: Kaggle SMS Spam Collection")
    print()

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
    print("Training Naive Bayes classifier on TF-IDF features...")
    vectorizer, model = train_naive_bayes(X_train, y_train)
    print("Training complete.")

    save_model(vectorizer, model)

    compare_models(X_train, y_train, X_test, y_test, vectorizer)

    evaluate_model(model, vectorizer, X_test, y_test, results_dir=RESULTS_DIR)

    print_misclassified(model, vectorizer, X_test, y_test)

    print()
    print("Done.")


if __name__ == "__main__":
    run_training_pipeline()
