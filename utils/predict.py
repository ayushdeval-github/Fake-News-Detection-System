
"""
utils/predict.py — Model Inference Functions
Fake News Detection System

LR and LSTM run locally.
BERT runs through the separate BERT API service.
"""

import os
import logging
import requests

from utils.preprocess import clean_text, clean_text_for_bert

logger = logging.getLogger(__name__)

MAX_LSTM_LEN = 300


# ──────────────────────────────────────────────
# LOGISTIC REGRESSION
# ──────────────────────────────────────────────

def predict_lr(text: str, cache: dict) -> tuple[int, float]:

    lr_model = cache["lr_model"]
    tfidf = cache["tfidf"]

    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    pred = int(lr_model.predict(vec)[0])
    proba = lr_model.predict_proba(vec)[0]
    confidence = float(proba[pred])

    logger.info(
        "LR prediction: %s confidence: %.4f",
        "Fake" if pred else "Real",
        confidence
    )

    return pred, confidence


# ──────────────────────────────────────────────
# LSTM
# ──────────────────────────────────────────────

def predict_lstm(text: str, cache: dict) -> tuple[int, float]:

    from keras.preprocessing.sequence import pad_sequences

    lstm_model = cache["lstm_model"]
    tokenizer = cache["lstm_tokenizer"]

    cleaned = clean_text(text)
    sequences = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LSTM_LEN,
        padding="post"
    )

    prob_fake = float(
        lstm_model.predict(padded, verbose=0)[0][0]
    )

    pred = 1 if prob_fake >= 0.5 else 0
    confidence = prob_fake if pred == 1 else (1.0 - prob_fake)

    logger.info(
        "LSTM prediction: %s confidence: %.4f",
        "Fake" if pred else "Real",
        confidence
    )

    return pred, confidence


# ──────────────────────────────────────────────
# BERT — REMOTE API
# ──────────────────────────────────────────────

def predict_bert(text: str, cache: dict) -> tuple[int, float]:
    """
    Sends text to the separate BERT API service.

    Required environment variables:
        BERT_API_URL
        BERT_API_KEY

    Expected API response:
        {
            "success": true,
            "label": 0,
            "prediction": "Real News",
            "confidence": 0.9553
        }

    Labels:
        0 = Real
        1 = Fake
    """

    api_url = os.getenv("BERT_API_URL", "").strip()
    api_key = os.getenv("BERT_API_KEY", "").strip()

    if not api_url:
        raise RuntimeError("BERT_API_URL environment variable is missing.")

    if not api_key:
        raise RuntimeError("BERT_API_KEY environment variable is missing.")

    # Accept either the base URL or the complete /predict URL.
    api_url = api_url.rstrip("/")

    if not api_url.endswith("/predict"):
        api_url += "/predict"

    cleaned = clean_text_for_bert(text)

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "text": cleaned
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(
                result.get("error", "BERT API returned an unsuccessful response.")
            )

        pred = int(result["label"])
        confidence = float(result["confidence"])

        if pred not in (0, 1):
            raise ValueError("BERT API returned an invalid prediction label.")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("BERT API returned an invalid confidence value.")

        logger.info(
            "BERT API prediction: %s confidence: %.4f",
            "Fake" if pred else "Real",
            confidence
        )

        return pred, confidence

    except requests.exceptions.Timeout as exc:
        logger.exception("BERT API request timed out.")
        raise RuntimeError("BERT API request timed out. Please try again.") from exc

    except requests.exceptions.ConnectionError as exc:
        logger.exception("Could not connect to BERT API.")
        raise RuntimeError("Could not connect to the BERT API service.") from exc

    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        logger.exception("BERT API returned an HTTP error.")
        raise RuntimeError(
            f"BERT API request failed with HTTP status {status_code}."
        ) from exc

    except (ValueError, KeyError) as exc:
        logger.exception("Invalid response received from BERT API.")
        raise RuntimeError("BERT API returned an invalid response.") from exc


# ──────────────────────────────────────────────
# ROUTER — CALLED BY app.py
# ──────────────────────────────────────────────

def run_prediction(
    text: str,
    model_choice: str,
    cache: dict
) -> tuple[int, float]:

    if model_choice == "lr":
        return predict_lr(text, cache)

    elif model_choice == "lstm":
        return predict_lstm(text, cache)

    elif model_choice == "bert":
        return predict_bert(text, cache)

    else:
        raise ValueError(f"Unknown model: {model_choice}")
