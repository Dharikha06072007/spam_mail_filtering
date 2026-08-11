"""Keyword-based text indicators for the web UI (educational feature only).

These indicators are DISPLAYED to help a user understand why a message looks
suspicious. They are computed independently from the model and NEVER change the
actual Naive Bayes prediction.
"""

INDICATOR_CATEGORIES = {
    "Promotional language": [
        "free", "win", "won", "winner", "winning", "prize", "gift", "claim",
        "offer", "cash", "bonus", "discount", "deal", "congratulations",
        "congrats", "click here", "order now", "buy now", "best price",
    ],
    "Prize / reward terms": [
        "prize", "reward", "gift card", "giftcard", "lottery", "jackpot",
        "winner", "lucky", "win", "won",
    ],
    "Urgent language": [
        "urgent", "immediately", "act now", "asap", "hurry", "limited time",
        "expires", "expiry", "deadline", "last chance", "right now", "now",
    ],
    "Suspicious links / web terms": [
        "http", "www", "bit.ly", "tinyurl", ".com", ".net", ".click", "click here",
        "link", "subscribe", "verify", "confirm",
    ],
    "Financial / bank terms": [
        "bank", "account", "password", "credit", "debit", "wire", "transfer",
        "paypal", "invoice", "loan", "cash", "atm", "money", "card", "refund",
    ],
}


def detect_indicators(text, limit=6):
    """Find indicator keywords in a cleaned message.

    Returns a list of dicts: {"category": str, "keywords": [str, ...]}.
    Only categories with at least one keyword match are included.
    """
    lowered = (text or "").lower()
    found = []
    for category, keywords in INDICATOR_CATEGORIES.items():
        matches = [kw for kw in keywords if kw in lowered]
        if matches:
            found.append({"category": category, "keywords": matches})
    return found[:limit]
