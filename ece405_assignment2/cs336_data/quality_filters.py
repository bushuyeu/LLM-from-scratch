import os

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def gopher_quality_filter(text: str) -> bool:
    """Return True if the document passes all Gopher heuristic quality filters."""
    words = nltk.word_tokenize(text)

    if not words:
        return False

    # Word count bounds: 50–100,000
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    num_words = len(alpha_words)
    if num_words < 50 or num_words > 100_000:
        return False

    # Mean word length: 3–10 characters
    mean_word_len = sum(len(w) for w in alpha_words) / num_words
    if mean_word_len < 3 or mean_word_len > 10:
        return False

    # Ellipsis lines: at most 30%
    lines = text.split("\n")
    if lines:
        ellipsis_count = sum(1 for line in lines if line.rstrip().endswith("...") or line.rstrip().endswith("\u2026"))
        if ellipsis_count / len(lines) > 0.3:
            return False

    # Alphabetic words: at least 80%
    alpha_ratio = len(alpha_words) / len(words)
    if alpha_ratio < 0.8:
        return False

    return True


# --- Quality Classifier (fastText) ---

_quality_model = None

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_QUALITY_MODEL_PATH = os.environ.get(
    "QUALITY_MODEL_PATH",
    os.path.join(_ASSETS_DIR, "quality_classifier.bin"),
)


def set_quality_model_path(path: str):
    global _QUALITY_MODEL_PATH, _quality_model
    _QUALITY_MODEL_PATH = path
    _quality_model = None


def _get_quality_model():
    global _quality_model
    if _quality_model is None:
        import fasttext
        fasttext.FastText.eprint = lambda x: None
        _quality_model = fasttext.load_model(_QUALITY_MODEL_PATH)
    return _quality_model


def classify_quality(text: str) -> tuple[str, float]:
    """Classify text as 'wiki' (high quality) or 'cc' (low quality)."""
    model = _get_quality_model()
    text_clean = text.replace("\n", " ").strip()
    predictions = model.predict(text_clean, k=2)
    labels = predictions[0]
    scores = predictions[1]

    # Map fastText labels to expected labels
    label_map = {"__label__wiki": "wiki", "__label__cc": "cc"}

    top_label = label_map.get(labels[0], labels[0])
    top_score = float(scores[0])
    return top_label, top_score
