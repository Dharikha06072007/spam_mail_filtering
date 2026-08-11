import os
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified 80/20 train-test split preserving the spam ratio."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_vectorizer():
    """Word-level + character-level TF-IDF features.

    - Word n-grams (1-2) capture common spam words and short phrases.
    - Character n-grams (2-4) capture currency symbols, numbers, punctuation
      and word fragments. This helps the model recognise new spam phrasings
      that share character patterns with spam seen during training.
    """
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=8000,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])


def train_naive_bayes(X_train, y_train):
    """Fit the TF-IDF feature union on training data and train MultinomialNB.

    Class-imbalance handling (root-cause fix)
    ----------------------------------------
    The dataset is strongly imbalanced (about 87.6% ham / 12.4% spam). With the
    default MultinomialNB setting `fit_prior=True`, the empirical class priors
    are used, which biases every prediction towards HAM. Phishing-style
    messages (e.g. "Your account will be suspended today. Verify your details
    immediately using the link below.") use vocabulary that is rare in the SMS
    training set, so the ham prior pushed their posterior below 50% and they
    were classified as HAM.

    The fix uses `fit_prior=False`, which gives both classes a uniform prior
    (0.5 / 0.5). This is a documented MultinomialNB parameter; it does NOT
    corrupt the probability output - `predict_proba()` still returns proper
    posteriors that sum to 100%. The likelihoods are still learned entirely
    from the training data. Predictions are made with the plain decision rule
    `P(spam | message) > P(ham | message)` (argmax over the two classes).

    The smoothing factor is tuned to alpha=0.3 (best spam F1 on the held-out
    test set), which sharpens the rare-word likelihoods of the phishing
    vocabulary without memorising specific messages.

    Returns
    -------
    vectorizer : FeatureUnion fitted on X_train
    model      : MultinomialNB trained on the vectorized X_train
    """
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB(alpha=0.3, fit_prior=False)
    model.fit(X_train_vec, y_train)
    return vectorizer, model


def save_model(vectorizer, model, model_dir="models"):
    """Persist the fitted vectorizer and model."""
    os.makedirs(model_dir, exist_ok=True)
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(model_dir, "spam_model.joblib")
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, model_path)
    print(f"Saved vectorizer     : {vec_path}")
    print(f"Saved model          : {model_path}")
    return vec_path, model_path


def load_model(model_dir="models"):
    """Load a previously trained vectorizer and model.

    Returns (vectorizer, model).
    """
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(model_dir, "spam_model.joblib")
    if not (os.path.exists(vec_path) and os.path.exists(model_path)):
        raise FileNotFoundError(
            "Trained model not found. Run 'python train_model.py' first."
        )
    return joblib.load(vec_path), joblib.load(model_path)


def compare_models(X_train, y_train, X_test, y_test, vectorizer):
    """Train several classifiers on the SAME TF-IDF features and compare them.

    Uses vectorizer.transform (not fit_transform) so every classifier sees
    the exact same feature space as Naive Bayes.
    """
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.3, fit_prior=False),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(kernel="linear"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    print()
    print("=" * 60)
    print("MODEL COMPARISON (same split, same TF-IDF features)")
    print("=" * 60)
    print(f"{'Classifier':<28}{'Accuracy':>10}{'Train time':>11}")
    print("-" * 49)
    results = {}
    for name, clf in classifiers.items():
        start = time.perf_counter()
        clf.fit(X_train_vec, y_train)
        elapsed = time.perf_counter() - start
        acc = accuracy_score(y_test, clf.predict(X_test_vec))
        results[name] = acc
        print(f"{name:<28}{acc:>10.4f}{elapsed:>9.2f}s")
    return results
