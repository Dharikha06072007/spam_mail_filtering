import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _plot_confusion_matrix(cm, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - Naive Bayes Spam Filter")
    plt.tight_layout()
    path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix image: {path}")


def evaluate_model(model, vectorizer, X_test, y_test, results_dir="results"):
    """Evaluate a trained model on the held-out test set.

    Predictions come from model.predict() (the plain argmax decision rule,
    the same rule used by the web application), so the reported metrics match
    what the app actually predicts.
    """
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print()
    print("=" * 60)
    print("EVALUATION ON TEST SET (20% held out, argmax decision rule)")
    print("=" * 60)
    print(f"Accuracy           : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (spam)   : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (spam)      : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score (spam)    : {f1_score(y_test, y_pred):.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, results_dir)
    return y_pred, cm


def print_misclassified(model, vectorizer, X_test, y_test, limit=15):
    """Print test messages the model classified incorrectly, for diagnostics."""
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    proba = model.predict_proba(X_test_vec)
    spam_col = int(np.where(model.classes_ == 1)[0][0])
    mis = np.where(y_pred != y_test)[0]

    print()
    print("=" * 62)
    print(f"MISCLASSIFIED TEST MESSAGES ({len(mis)} total, showing up to {limit})")
    print("=" * 62)
    for i in mis[:limit]:
        actual = "spam" if y_test[i] == 1 else "ham"
        predicted = "spam" if y_pred[i] == 1 else "ham"
        print(
            f"[{actual:4s} -> {predicted:4s}] spam_prob = {proba[i, spam_col] * 100:6.2f}% "
            f"| {X_test.iloc[i]}"
        )
    return mis
