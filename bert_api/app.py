import os
import re
import logging
import hmac
from pathlib import Path
from dotenv import load_dotenv
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
load_dotenv()
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from root .env
load_dotenv(BASE_DIR / ".env")

MODEL_DIR = Path(__file__).resolve().parent / "models" / "bert_model_fp16"

API_KEY = os.getenv("BERT_API_KEY")

MAX_LENGTH = 512
MAX_CHARS = 10000

# --------------------------------------------------
# TEXT PREPROCESSING
# Same minimal cleaning used by your existing BERT
# --------------------------------------------------

def clean_text_for_bert(text: str, max_chars: int = 512) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# --------------------------------------------------
# LOAD DISTILBERT MODEL ONCE
# --------------------------------------------------

logger.info("Loading DistilBERT model...")

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(
        f"DistilBERT model directory not found: {MODEL_DIR}"
    )

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

model.to("cpu")
model.eval()

logger.info("DistilBERT model loaded successfully!")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "Fake News DistilBERT API"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def authenticate_request():
    if not API_KEY:
        return False

    provided_key = request.headers.get("X-API-Key", "")

    return hmac.compare_digest(provided_key, API_KEY)


# --------------------------------------------------
# DISTILBERT PREDICTION API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if not authenticate_request():
        return jsonify({
            "error": "Unauthorized. Invalid or missing API key."
        }), 401

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON request."}), 400

        text = data.get("text", "")

        if not isinstance(text, str):
            return jsonify({"error": "Text must be a string."}), 400

        if not text.strip():
            return jsonify({"error": "Input text is empty."}), 400

        if len(text) < 10:
            return jsonify({"error": "Text must contain at least 10 characters."}), 400

        if len(text) > MAX_CHARS:
            return jsonify({"error": "Text exceeds 10,000 characters."}), 400

        cleaned_text = clean_text_for_bert(text)

        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits.float(), dim=-1)[0]

        prediction = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[prediction].item())

        # Training label convention:
        # 0 = REAL
        # 1 = FAKE

        result = "Fake News" if prediction == 1 else "Real News"

        logger.info(
            "Prediction: %s | Confidence: %.4f",
            result,
            confidence
        )

        return jsonify({
            "success": True,
            "prediction": result,
            "label": prediction,
            "confidence": round(confidence, 4)
        }), 200

    except Exception as exc:
        logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": "Prediction failed."
        }), 500


# --------------------------------------------------
# LOCAL DEVELOPMENT
# --------------------------------------------------

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    try:
        app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")