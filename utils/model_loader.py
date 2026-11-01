
"""
utils/model_loader.py — Model Loading Utilities
Fake News Detection System | Ayush Deval

Main Flask service loads only:
    1. Logistic Regression
    2. LSTM

DistilBERT is deployed separately as a remote API.
"""

import os
import pickle
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _path(*parts: str) -> str:
    return os.path.join(MODELS_DIR, *parts)


# ──────────────────────────────────────────────
# LOGISTIC REGRESSION
# ──────────────────────────────────────────────

def load_lr_model() -> dict:
    logger.info("Loading Logistic Regression model...")

    with open(_path("lr_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)

    with open(_path("tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    logger.info("LR model loaded successfully.")

    return {
        "lr_model": lr_model,
        "tfidf": tfidf
    }


# ──────────────────────────────────────────────
# LSTM
# ──────────────────────────────────────────────

def load_lstm_model() -> dict:
    logger.info("Loading LSTM model...")

    from keras.models import load_model

    lstm_model = load_model(
        _path("lstm_model.h5"),
        compile=False
    )

    with open(_path("tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    logger.info("LSTM model loaded successfully.")

    return {
        "lstm_model": lstm_model,
        "lstm_tokenizer": tokenizer
    }


# ──────────────────────────────────────────────
# MAIN LOADER
# ──────────────────────────────────────────────

def load_all_models() -> dict:
    """
    Load only LR and LSTM into the main Flask service.

    DistilBERT is handled by the separate Render API.
    """

    cache = {}

    # Load Logistic Regression
    try:
        cache.update(load_lr_model())
    except Exception:
        logger.exception("Failed to load Logistic Regression model.")

    # Load LSTM
    try:
        cache.update(load_lstm_model())
    except Exception:
        logger.exception("Failed to load LSTM model.")

    logger.info(
        "Main service models loaded: %s",
        list(cache.keys())
    )

    return cache
