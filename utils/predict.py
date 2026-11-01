
"""
utils/predict.py — Model Inference Functions
Fake News Detection System

LR and LSTM run locally.
BERT runs through the separate BERT API service.
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

from utils.preprocess import clean_text, clean_text_for_bert

logger = logging.getLogger(__name__)

MAX_LSTM_LEN = 300
BERT_CONNECT_TIMEOUT = 5
BERT_READ_TIMEOUT = 45


# --------------------------------------------------
# LOGISTIC REGRESSION
# --------------------------------------------------

def predict_lr(text: str, cache: dict) -> tuple[int, float]:
    lr_model = cache["lr_model"]
    tfidf = cache["tfidf"]

    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    pred = int(lr_model.predict(vec)[0])
    probabilities = lr_model.predict_proba(vec)[0]
    confidence = float(probabilities[pred])

    logger.info("LR prediction: label=%s confidence=%.4f", pred, confidence)

    return pred, confidence


# --------------------------------------------------
# LSTM
# --------------------------------------------------

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

    prob_fake = float(lstm_model.predict(padded, verbose=0)[0][0])

    pred = 1 if prob_fake >= 0.5 else 0
    confidence = prob_fake if pred == 1 else 1.0 - prob_fake

    logger.info("LSTM prediction: label=%s confidence=%.4f", pred, confidence)

    return pred, confidence


# --------------------------------------------------
# BERT — REMOTE API
# --------------------------------------------------

def predict_bert(text: str, cache: dict) -> tuple[int, float]:
    api_url = os.getenv("BERT_API_URL", "").strip().rstrip("/")
    api_key = os.getenv("BERT_API_KEY", "").strip()

    if not api_url:
        raise RuntimeError("BERT_API_URL is missing.")

    if not api_key:
        raise RuntimeError("BERT_API_KEY is missing.")

    if not api_url.endswith("/predict"):
        api_url += "/predict"

    cleaned = clean_text_for_bert(text)

    if not cleaned:
        raise ValueError("No text available for BERT prediction.")

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    payload = {"text": cleaned}

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=(BERT_CONNECT_TIMEOUT, BERT_READ_TIMEOUT)
        )

        response.raise_for_status()

        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError("BERT API returned invalid JSON.") from exc

        if not isinstance(result, dict):
            raise RuntimeError("BERT API response must be a JSON object.")

        # Accept an omitted success field, but reject an explicit false.
        if result.get("success") is False:
            raise RuntimeError(
                result.get("error", "BERT API reported an unsuccessful prediction.")
            )

        if "label" not in result or "confidence" not in result:
            raise RuntimeError(
                "BERT API response must contain 'label' and 'confidence'."
            )

        pred = int(result["label"])
        confidence = float(result["confidence"])

        if pred not in (0, 1):
            raise ValueError("BERT API returned an invalid label.")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("BERT API returned invalid confidence.")

        logger.info(
            "BERT prediction successful: label=%s confidence=%.4f",
            pred,
            confidence
        )

        return pred, confidence

    except requests.exceptions.Timeout as exc:
        logger.error("BERT API timed out: %s", exc)
        raise RuntimeError(
            "BERT API timed out. The service may be starting or overloaded."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        logger.error("BERT API connection failed: %s", exc)
        raise RuntimeError(
            "Could not connect to the BERT API service."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        logger.error("BERT API HTTP error: %s", status)
        raise RuntimeError(
            f"BERT API returned HTTP {status}."
        ) from exc


# --------------------------------------------------
# PREDICTION ROUTER
# --------------------------------------------------

def run_prediction(
    text: str,
    model_choice: str,
    cache: dict
) -> tuple[int, float]:

    model_choice = model_choice.strip().lower()

    if model_choice == "lr":
        return predict_lr(text, cache)

    if model_choice == "lstm":
        return predict_lstm(text, cache)

    if model_choice == "bert":
        return predict_bert(text, cache)

    raise ValueError(f"Unknown model: {model_choice}")
