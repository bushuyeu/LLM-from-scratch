import os

import fasttext

_nsfw_model = None
_toxic_model = None

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

_NSFW_MODEL_PATH = os.environ.get(
    "NSFW_MODEL_PATH",
    os.path.join(_ASSETS_DIR, "dolma_fasttext_nsfw_jigsaw_model.bin"),
)
_TOXIC_MODEL_PATH = os.environ.get(
    "TOXIC_MODEL_PATH",
    os.path.join(_ASSETS_DIR, "dolma_fasttext_hatespeech_jigsaw_model.bin"),
)


def set_nsfw_model_path(path: str):
    global _NSFW_MODEL_PATH, _nsfw_model
    _NSFW_MODEL_PATH = path
    _nsfw_model = None


def set_toxic_model_path(path: str):
    global _TOXIC_MODEL_PATH, _toxic_model
    _TOXIC_MODEL_PATH = path
    _toxic_model = None


def _get_nsfw_model():
    global _nsfw_model
    if _nsfw_model is None:
        fasttext.FastText.eprint = lambda x: None
        _nsfw_model = fasttext.load_model(_NSFW_MODEL_PATH)
    return _nsfw_model


def _get_toxic_model():
    global _toxic_model
    if _toxic_model is None:
        fasttext.FastText.eprint = lambda x: None
        _toxic_model = fasttext.load_model(_TOXIC_MODEL_PATH)
    return _toxic_model


def classify_nsfw(text: str) -> tuple[str, float]:
    model = _get_nsfw_model()
    text_clean = text.replace("\n", " ").strip()
    predictions = model.predict(text_clean, k=2)
    labels = predictions[0]
    scores = predictions[1]

    # Dolma labels: __label__nsfw, __label__non-nsfw
    # Find the nsfw score
    nsfw_score = 0.0
    for label, score in zip(labels, scores):
        if label == "__label__nsfw":
            nsfw_score = float(score)
            break

    if nsfw_score >= 0.5:
        return "nsfw", nsfw_score
    else:
        return "non-nsfw", 1.0 - nsfw_score


def classify_toxic_speech(text: str) -> tuple[str, float]:
    model = _get_toxic_model()
    text_clean = text.replace("\n", " ").strip()
    predictions = model.predict(text_clean, k=2)
    labels = predictions[0]
    scores = predictions[1]

    # Dolma labels: __label__hatespeech, __label__non-hatespeech
    # Map to toxic/non-toxic for test compatibility
    toxic_score = 0.0
    for label, score in zip(labels, scores):
        if label == "__label__hatespeech":
            toxic_score = float(score)
            break

    if toxic_score >= 0.5:
        return "toxic", toxic_score
    else:
        return "non-toxic", 1.0 - toxic_score
