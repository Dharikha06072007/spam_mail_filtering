import os
import re

import pandas as pd

LABEL_COLUMNS = ["v1", "label", "labels", "class", "category", "type", "target"]
MESSAGE_COLUMNS = ["v2", "message", "messages", "text", "sms", "msg", "content", "body"]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def clean_text(text):
    """Clean a single message exactly like the training pipeline does.

    - converts the text to lowercase
    - removes unnecessary HTML tags (keeps the readable text)
    - normalises URLs to the token ' url ' and emails to ' email ' so that
      the model learns URL/email presence as features instead of over-fitting
      to one specific address
    - trims and collapses whitespace
    - keeps numbers, currency symbols ($), punctuation such as '!' and the
      spam-indicator words themselves, so the character n-gram features and
      the word vocabulary can use them as spam signals
    """
    text = str(text).strip()
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" url ", text)
    text = EMAIL_PATTERN.sub(" email ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


# Backwards-compatible alias.
clean_message = clean_text


def find_csv_in_data_dir(data_dir):
    """Auto-detect a CSV file inside the data directory."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Create a 'data' folder and place the dataset CSV inside it."
        )
    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found inside '{data_dir}'. "
            "Please add the spam dataset CSV (e.g. spam.csv) there."
        )
    return os.path.join(data_dir, csv_files[0])


def _resolve_columns(df):
    """Locate the label and message columns automatically."""
    label_col = None
    message_col = None

    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    for name in LABEL_COLUMNS:
        if name in lower_cols:
            label_col = lower_cols[name]
            break
    for name in MESSAGE_COLUMNS:
        if name in lower_cols:
            message_col = lower_cols[name]
            break

    if label_col is None:
        for col in df.columns:
            values = df[col].dropna().astype(str).str.strip().str.lower()
            if len(values) > 0 and values.isin(["spam", "ham"]).all():
                label_col = col
                break

    if message_col is None and label_col is not None:
        candidates = [c for c in df.columns if c != label_col and df[c].dtype == object]
        if candidates:
            message_col = max(
                candidates, key=lambda c: df[c].astype(str).str.len().mean()
            )

    if label_col is None:
        raise ValueError(
            "Could not detect the label column. Expected labels such as "
            "'spam'/'ham' or columns named v1 / label / class."
        )
    if message_col is None:
        raise ValueError(
            "Could not detect the message/text column. Expected columns named "
            "v2 / message / text / sms / msg."
        )
    return label_col, message_col


def load_and_preprocess_data(data_dir="data"):
    """Load, validate and preprocess the SMS spam dataset.

    Returns
    -------
    X : pd.Series -> cleaned message texts
    y : np.ndarray -> binary labels (0 = ham, 1 = spam)
    """
    csv_path = find_csv_in_data_dir(data_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin-1", dtype=str)

    label_col, message_col = _resolve_columns(df)

    y = df[label_col].astype(str).str.strip().str.lower()
    if not y.isin(["spam", "ham"]).all():
        raise ValueError(
            "The label column contains values other than 'spam'/'ham'. "
            "Cannot encode targets."
        )
    y = y.map({"spam": 1, "ham": 0}).to_numpy(dtype=int)

    X = df[message_col].map(clean_message)

    empty_count = int((X.str.len() == 0).sum())
    keep = X.str.len() > 0
    X, y = X[keep].reset_index(drop=True), y[keep]

    dup_mask = X.duplicated()
    dup_count = int(dup_mask.sum())
    X, y = X[~dup_mask].reset_index(drop=True), y[~dup_mask]

    print(f"Loaded dataset      : {os.path.basename(csv_path)}")
    print(f"Columns detected    : label='{label_col}', message='{message_col}'")
    print(f"Empty messages      : {empty_count}")
    print(f"Duplicates removed  : {dup_count}")
    print(f"Final samples       : {len(y)}")
    return X, y


def print_dataset_stats(X, y):
    """Print basic statistics about the loaded dataset."""
    total = len(y)
    spam = int(y.sum())
    ham = total - spam
    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total messages          : {total}")
    print(f"Ham (legitimate)        : {ham}")
    print(f"Spam                    : {spam}")
    print(f"Spam ratio              : {spam / total:.2%}")
    print(f"Average message length  : {X.str.len().mean():.1f} characters")
