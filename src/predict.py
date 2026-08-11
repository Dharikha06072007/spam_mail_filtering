from src.preprocessing import clean_message


def predict_details(text, vectorizer, model):
    """Classify one message with the trained model and return full details.

    Applies the same preprocessing as training, transforms with the fitted
    TF-IDF vectorizer, then reads the actual class probabilities from
    model.predict_proba(). The class labels are taken from model.classes_
    (never assumed by position), and the decision is the plain argmax rule:

        if spam_probability > ham_probability: prediction = "SPAM"
        else:                                  prediction = "HAM"

    The confidence is the probability of the predicted class, so the label and
    the displayed confidence are always consistent.

    Returns a dict with:
        prediction        : "SPAM" or "HAM"
        spam_probability  : P(spam | message) in percent
        ham_probability   : P(ham  | message) in percent
        confidence        : probability of the predicted class in percent
    """
    cleaned = clean_message(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]

    # Map probabilities to class labels through model.classes_ so the indexing
    # is correct regardless of how sklearn orders the classes.
    prob_by_class = dict(zip(model.classes_, proba))
    spam_prob = round(float(prob_by_class[1]) * 100, 2)
    ham_prob = round(float(prob_by_class[0]) * 100, 2)

    if spam_prob > ham_prob:
        prediction = "SPAM"
        confidence = spam_prob
    else:
        prediction = "HAM"
        confidence = ham_prob

    return {
        "prediction": prediction,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
        "confidence": confidence,
    }


def classify_message(text, vectorizer, model):
    """Small wrapper returning (label, confidence_percent) for CLI use."""
    details = predict_details(text, vectorizer, model)
    return details["prediction"], details["confidence"]
