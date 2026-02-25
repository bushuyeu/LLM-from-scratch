import os

import fasttext


_model = None

# Default path; override with set_lid_model_path() or LID_MODEL_PATH env var
_model_path = os.environ.get("LID_MODEL_PATH", "lid.176.bin")


def set_lid_model_path(path: str):
    """Set the path to the fastText language ID model."""
    global _model_path, _model
    _model_path = path
    _model = None  # force reload


def _get_model():
    """Lazy-load the fastText language identification model."""
    global _model
    if _model is None:
        # Suppress fastText's warning about loading model with the old format
        fasttext.FastText.eprint = lambda x: None
        _model = fasttext.load_model(_model_path)
    return _model


def identify_language(text: str) -> tuple[str, float]:
    """Identify the main language of a Unicode string.

    Returns a (language_code, confidence_score) tuple.
    The language code is a two-letter ISO 639-1 code (e.g., "en", "zh").
    The confidence score is between 0 and 1.
    """
    model = _get_model()
    # fastText expects single-line input
    text_clean = text.replace("\n", " ").strip()
    predictions = model.predict(text_clean, k=1)
    # predictions is ([labels], [scores]) e.g. (['__label__en'], [0.99])
    label = predictions[0][0].replace("__label__", "")
    score = float(predictions[1][0])
    return label, score
