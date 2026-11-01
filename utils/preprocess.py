
"""
utils/preprocess.py — Text Cleaning & Preprocessing Pipeline
Fake News Detection System
"""

import re
import string
import logging
from pathlib import Path

import nltk

logger = logging.getLogger(__name__)

# --------------------------------------------------
# NLTK DATA CONFIGURATION
# --------------------------------------------------

NLTK_DATA_DIR = Path.home() / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))


def ensure_nltk_resource(package, resource):
    """Check and download an NLTK resource when missing."""

    try:
        nltk.data.find(resource)
        return

    except LookupError:
        logger.info("Downloading NLTK resource: %s", package)

        success = nltk.download(
            package,
            download_dir=str(NLTK_DATA_DIR),
            quiet=True
        )

        if not success:
            raise RuntimeError(
                f"Failed to download NLTK resource: {package}"
            )

        try:
            nltk.data.find(resource)

        except LookupError as exc:
            raise RuntimeError(
                f"NLTK resource unavailable: {package}"
            ) from exc


# Required resources
ensure_nltk_resource("stopwords", "corpora/stopwords")
ensure_nltk_resource("wordnet", "corpora/wordnet.zip")


# --------------------------------------------------
# NLTK IMPORTS AND INITIALIZATION
# --------------------------------------------------

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))

STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


# --------------------------------------------------
# TEXT CLEANING — LR AND LSTM
# --------------------------------------------------

def clean_text(text: str, use_stemming: bool = False) -> str:
    """
    Clean news text for Logistic Regression and LSTM.

    Default: WordNet lemmatization.
    Optional: Porter stemming.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = text.split()

    # Remove stopwords and short tokens
    tokens = [
        token for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    # Apply stemming or lemmatization
    if use_stemming:
        tokens = [STEMMER.stem(token) for token in tokens]
    else:
        tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    return " ".join(tokens)


# --------------------------------------------------
# TEXT CLEANING — BERT
# --------------------------------------------------

def clean_text_for_bert(text: str, max_chars: int = 512) -> str:
    """
    Minimal text cleaning for BERT.
    Preserve original words and sentence structure.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_chars]
