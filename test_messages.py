"""Run the official demonstration messages through the trained model.

Usage:  python test_messages.py

Trains the model on the dataset (same pipeline as the web app), then prints
the prediction, probabilities and confidence for every demo message.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_and_preprocess_data
from src.predict import predict_details
from src.train_model import split_data, train_naive_bayes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TEST_MESSAGES = [
    ("HAM", "Hey, are we still meeting for lunch tomorrow at the usual place?"),
    ("HAM", "Reminder: Project submission is due on Friday. Please review the attached file."),
    ("SPAM", "Congratulations! You have won a free iPhone. Click here to claim your prize!"),
    ("SPAM", "URGENT! You have been selected to receive a cash reward. Claim your prize now!"),
    ("SPAM", "FREE MONEY! Claim your $1000 gift card now, limited time offer!"),
    ("SPAM", "Your account will be suspended today. Verify your details immediately using the link below."),
    ("SPAM", "Exclusive offer! Get a FREE bonus and earn money from home. Join now before the offer expires!"),
    ("SPAM", "Dear Customer, your PayPal account has been locked. Click here to verify your account."),
    ("SPAM", "Security Alert: Someone tried to log into your bank account. Confirm your details now."),
    ("HAM", "Hi John, can you send me the project report by tomorrow evening? Thanks."),
]


def main():
    print("BAYESIAN SPAM FILTERING - DEMO MESSAGE TESTS")
    print("=" * 80)

    X, y = load_and_preprocess_data(DATA_DIR)
    X_train, _, y_train, _ = split_data(X, y)
    print("Training model...")
    vectorizer, model = train_naive_bayes(X_train, y_train)

    print()
    print(f"{'Expected':>8} {'Predicted':>9} {'Spam %':>8} {'Ham %':>8} {'Conf %':>8}   Message")
    print("-" * 80)
    passed = 0
    for expected, message in TEST_MESSAGES:
        details = predict_details(message, vectorizer, model)
        ok = details["prediction"] == expected
        passed += ok
        flag = "OK " if ok else "WRONG"
        print(
            f"{expected:>8} {details['prediction']:>9} "
            f"{details['spam_probability']:>7.2f}% {details['ham_probability']:>7.2f}% "
            f"{details['confidence']:>7.2f}%   {flag}  {message[:42]}"
        )

    print("-" * 80)
    print(f"Passed {passed}/{len(TEST_MESSAGES)}")
    return 0 if passed == len(TEST_MESSAGES) else 1


if __name__ == "__main__":
    sys.exit(main())
