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
        ellipsis_count = sum(1 for line in lines if line.rstrip().endswith("..."))
        if ellipsis_count / len(lines) > 0.3:
            return False

    # Alphabetic words: at least 80%
    alpha_ratio = len(alpha_words) / len(words)
    if alpha_ratio < 0.8:
        return False

    return True
