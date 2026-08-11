# Bayesian Spam Filtering for Email Classification Using Naive Bayes

A complete machine-learning project that classifies SMS / email messages as **spam** or **ham** (legitimate) using a **Naive Bayes** classifier. Built on the publicly available **Kaggle SMS Spam Collection** dataset.

## How Naive Bayes Works

Naive Bayes applies **Bayes' theorem** and a "naive" independence assumption: it treats every word in a message as independent evidence for the class. We decide between two competing hypotheses for a given message:

```
P(Spam | Message)  ∝  P(Spam)  ×  P(Message | Spam)
P(Ham  | Message)  ∝  P(Ham)   ×  P(Message | Ham)
```

- `P(Spam)` and `P(Ham)` are the **priors** - the overall fraction of spam/ham in the dataset.
- `P(Message | Spam)` is the **likelihood** - the probability of seeing exactly those words given that the message is spam. With the naive assumption this becomes the product of the per-word probabilities:

```
P(Message | Spam) = P(w1 | Spam) × P(w2 | Spam) × ... × P(wn | Spam)
```

We classify as **spam** when `P(Spam | Message) > P(Ham | Message)` (the plain
argmax rule). The confidence shown in the UI is the probability of the
predicted class, so a `SPAM` result always means the model assigned it more
than 50% spam probability.

### Pipeline

1. **Load** the dataset CSV (auto-detected in the `data/` folder).
2. **Preprocess** - clean text, drop empties, encode labels (`spam = 1`, `ham = 0`).
3. **Split** - stratified 80/20 train/test split (preserves the spam ratio in both sets).
4. **Vectorize** - TF-IDF (Term Frequency - Inverse Document Frequency) converts raw text into numeric features. The vectorizer is fitted **only on the training set** to avoid data leakage.
5. **Train** - `MultinomialNB` (the Naive Bayes variant designed for word counts / TF-IDF features). The dataset is imbalanced (~12.4% spam), so the classifier is trained with **uniform class priors** (`fit_prior=False`): the learned word likelihoods decide the prediction while the ham-heavy prior cannot push every message towards HAM. This is a documented parameter that does not corrupt `predict_proba()` output. Laplace smoothing is set to `alpha=0.3`.
6. **Evaluate** - accuracy, precision, recall, F1-score, classification report, and a confusion matrix image.
7. **Predict** - classify new messages and print the posterior probabilities.

## Project Structure

```
spam mail filtering/
├── data/
│   └── spam.csv                 # Kaggle SMS Spam Collection dataset
├── src/
│   ├── preprocessing.py         # Load, clean and encode the data
│   ├── train_model.py           # Split, TF-IDF, MultinomialNB, model comparison
│   ├── evaluate.py              # Metrics + confusion matrix plot
│   └── predict.py               # Classify messages with the trained model
├── models/                      # Saved model + vectorizer (joblib)
├── templates/
│   └── index.html               # Web app page
├── static/
│   └── style.css                # Web app styles
├── results/
│   └── confusion_matrix.png     # Confusion matrix image
├── train_model.py               # Training + evaluation pipeline (entry point)
├── test_messages.py             # Runs the demo messages through the model
├── app.py                       # Flask web application
├── main.py                      # Alias for train_model.py (kept for compatibility)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (loads the dataset, trains the MultinomialNB classifier,
   saves `models/spam_model.joblib` and `models/tfidf_vectorizer.joblib`,
   and prints the evaluation metrics):

   ```bash
   python train_model.py
   ```

3. Start the web application (trains the model once at startup, then serves):

   ```bash
   python app.py
   ```

4. Open the localhost URL shown by Flask in your browser
   (e.g. <http://127.0.0.1:5000>).

### Run the demonstration messages

```bash
python test_messages.py
```

Trains the model and prints the prediction, probabilities and confidence for
every demo message, with a Passed/Total summary at the end.

## Web Application

Type or paste any email / message into the textarea and click **Check Message**.

- The message is cleaned with the **same preprocessing** used during training.
- It is transformed with the **same fitted TF-IDF vectorizer**.
- The trained **MultinomialNB** model predicts SPAM or HAM with the plain
  argmax rule (`spam_probability > ham_probability`).
- The confidence percentage is the probability of the predicted class from
  `model.predict_proba()` - no fake values and always consistent with the label.

The model is trained **once when `python app.py` starts** (load data, clean,
split, fit TF-IDF, train Naive Bayes, evaluate) and then kept in memory; it is
never retrained per request.

## Dataset

- **Source:** Kaggle SMS Spam Collection (5,574 raw messages).
- **Labels:** `ham` (legitimate) and `spam`.
- After cleaning: **5,157** messages -> **4,515 ham** and **642 spam** (12.45% spam ratio).

## Results

All numbers below are **actual** output from `python train_model.py` on this
dataset (80/20 stratified split, TF-IDF features, `random_state=42`,
MultinomialNB with uniform class priors, `alpha=0.3`).

### Model comparison (same split, same TF-IDF features)

| Classifier                | Accuracy | Train time |
| ------------------------- | -------- | ---------- |
| Support Vector Machine    | 0.9903   | 7.28s      |
| Random Forest             | 0.9893   | 2.94s      |
| **Multinomial Naive Bayes** | 0.9864 | 0.01s      |
| Logistic Regression       | 0.9816   | 0.12s      |

### Naive Bayes evaluation on the test set (1,032 messages)

| Metric           | Value  |
| ---------------- | ------ |
| Accuracy         | 0.9864 |
| Precision (spam) | 0.9385 |
| Recall (spam)    | 0.9531 |
| F1-score (spam)  | 0.9457 |

Classification report:

```
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99       904
        spam       0.94      0.95      0.95       128

    accuracy                           0.99      1032
   macro avg       0.97      0.97      0.97      1032
weighted avg       0.99      0.99      0.99      1032
```

## Class-imbalance handling (root-cause fix)

The dataset contains ~87.6% ham and ~12.4% spam. With the default
`MultinomialNB(fit_prior=True)`, the empirical class priors are used, so the
model starts every prediction heavily biased towards HAM. Phishing-style
messages such as *"Your account will be suspended today. Verify your details
immediately using the link below."* use vocabulary that is rare in the SMS
training set, so the ham prior pushed their spam probability below 50% and they
were classified as HAM.

The fix is to train with `fit_prior=False` (uniform 0.5/0.5 class priors) and
`alpha=0.3` smoothing. The word likelihoods are still learned entirely from the
data and `predict_proba()` still returns proper posteriors that sum to 100%.
The prediction is the plain argmax rule, so a message is only called SPAM when
the model genuinely assigns it more than 50% spam probability. This improves
test accuracy from 0.9845 to 0.9864 and spam recall from 0.8828 to 0.9531
(using the same stratified split).

## Limitations

- **Not perfect:** test accuracy is 0.9864 and spam recall is 0.9531, so a small
  fraction of actual spam (~5%) still slips through and a few ham messages are
  flagged as spam. Very short or vocabulary-unusual messages remain the hardest.
- **SMS vocabulary:** the model is trained on the SMS dataset; real email HTML,
  headers and attachments are not modeled.
- **Demonstration examples are real output:** predictions shown in the UI come
  directly from `model.predict_proba()` and the argmax rule - no hardcoded
  labels.

## Dependencies

- flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
