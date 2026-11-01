
"""
utils/groq_client.py — Groq API Integration
Fake News Detection System
"""
import os
import re
import logging
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

logger = logging.getLogger(__name__)

GROQ_MODEL_PRIMARY = "openai/gpt-oss-120b"
GROQ_MODEL_FALLBACK = "openai/gpt-oss-20b"

MAX_TOKENS = 400
GROQ_TIMEOUT = 25.0


SYSTEM_PROMPT = """
You are a fake news analysis assistant.

Analyze the supplied news text carefully.
Do not invent facts or claim independent verification.

Return ONLY valid JSON in this exact structure:

{
    "verdict": "Fake",
    "explanation": "A concise explanation in 2-3 sentences.",
    "confidence": "High"
}

Rules:
- verdict must be exactly "Fake" or "Real".
- confidence must be exactly "High", "Medium", or "Low".
- explanation must contain 2-3 concise sentences.
- Do not include Markdown or text outside the JSON.
"""


# --------------------------------------------------
# GROQ ANALYSIS
# --------------------------------------------------

def analyze_with_groq(text: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        logger.error("GROQ_API_KEY is missing.")
        return _groq_unavailable("GROQ_API_KEY is missing.")

    if not api_key.startswith("gsk_"):
        logger.warning("GROQ_API_KEY does not have the expected prefix.")

    for model in (GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK):
        result = _call_groq(api_key, model, text)

        if result.get("groq_used"):
            logger.info("Groq succeeded using model: %s", model)
            return result

        logger.warning(
            "Groq model %s failed: %s",
            model,
            result.get("error", "Unknown error")
        )

    return _groq_unavailable(
        "Both configured Groq models failed. Check the API key, model access, "
        "rate limits, and Render logs."
    )


# --------------------------------------------------
# SINGLE GROQ REQUEST
# --------------------------------------------------

def _call_groq(api_key: str, model: str, text: str) -> dict:
    """Make a Groq request and parse its JSON response."""

    try:
        client = Groq(
            api_key=api_key,
            timeout=GROQ_TIMEOUT,
            max_retries=0
        )

        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=500,
            temperature=0.1,
            reasoning_effort="low",
            include_reasoning=False,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze the following news and return the "
                        "required JSON response:\n\n" + text
                    )
                }
            ]
        )

        if not response.choices:
            raise RuntimeError("Groq returned no response choices.")

        message = response.choices[0].message
        raw = message.content

        if not raw or not raw.strip():
            logger.error(
                "Empty Groq content. Finish reason: %s",
                response.choices[0].finish_reason
            )
            raise RuntimeError("Groq returned empty response content.")

        logger.info("Groq response received from %s.", model)

        import json
        data = json.loads(raw)

        verdict = str(data.get("verdict", "")).strip().capitalize()
        explanation = str(data.get("explanation", "")).strip()
        confidence = str(data.get("confidence", "Medium")).strip().capitalize()

        if verdict not in ("Fake", "Real"):
            raise ValueError("Invalid verdict returned by Groq.")

        if not explanation:
            raise ValueError("Groq returned an empty explanation.")

        if confidence not in ("High", "Medium", "Low"):
            confidence = "Medium"

        return {
            "verdict": verdict,
            "explanation": explanation,
            "confidence": confidence,
            "groq_used": True,
            "raw": raw[:500]
        }

    except Exception as exc:
        logger.exception("Groq request failed using model %s.", model)
        return _groq_unavailable(str(exc))


# --------------------------------------------------
# RESPONSE PARSER
# --------------------------------------------------

def _parse_groq_response(raw: str) -> dict:
    verdict = "Unknown"
    explanation = ""
    confidence = "Medium"

    # Prefer the explicit verdict tag.
    verdict_match = re.search(
        r"\[VERDICT:\s*(FAKE|REAL)\s*\]",
        raw,
        re.IGNORECASE
    )

    if verdict_match:
        verdict = verdict_match.group(1).capitalize()

    lines = raw.strip().splitlines()

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("explanation:"):
            explanation = stripped.split(":", 1)[1].strip()
            break

    if not explanation:
        explanation = raw[:300].strip()

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("confidence:"):
            value = stripped.split(":", 1)[1].strip().lower()

            if "high" in value:
                confidence = "High"
            elif "low" in value:
                confidence = "Low"
            else:
                confidence = "Medium"

            break

    return {
        "verdict": verdict,
        "explanation": explanation,
        "confidence": confidence
    }


# --------------------------------------------------
# UNAVAILABLE FALLBACK
# --------------------------------------------------

def _groq_unavailable(error: str = "") -> dict:
    return {
        "verdict": "Unavailable",
        "explanation": (
            "Groq AI is unavailable. The result will use the ML model only."
        ),
        "confidence": "N/A",
        "groq_used": False,
        "error": error
    }


# --------------------------------------------------
# COMBINE ML + GROQ RESULTS
# --------------------------------------------------

def combined_verdict(
    ml_prediction: int,
    ml_confidence: float,
    groq_result: dict
) -> dict:

    ml_label = "Fake" if ml_prediction == 1 else "Real"
    ml_percentage = round(ml_confidence * 100)

    # If Groq failed or returned an unrecognized verdict, trust ML only.
    if (
        not groq_result.get("groq_used")
        or groq_result.get("verdict") not in ("Fake", "Real")
    ):
        return {
            "final_verdict": ml_label,
            "final_confidence": ml_percentage,
            "ml_verdict": ml_label,
            "ml_confidence": ml_percentage,
            "groq_verdict": "Unavailable",
            "groq_explanation": groq_result.get(
                "explanation",
                "Groq analysis is unavailable."
            ),
            "agreement": "ML Model only (Groq unavailable)",
            "verdict_source": "ML Model only"
        }

    groq_label = groq_result["verdict"]
    groq_confidence = groq_result.get("confidence", "Medium")

    if ml_label == groq_label:
        final_confidence = round(min(ml_confidence + 0.05, 0.99) * 100)

        return {
            "final_verdict": ml_label,
            "final_confidence": final_confidence,
            "ml_verdict": ml_label,
            "ml_confidence": ml_percentage,
            "groq_verdict": groq_label,
            "groq_explanation": groq_result.get("explanation", ""),
            "groq_confidence": groq_confidence,
            "agreement": "Both models agree",
            "verdict_source": "ML + Groq (agreed)"
        }

    # Disagreement: retain ML as the final prediction rather than
    # automatically treating an LLM response as verified ground truth.
    return {
        "final_verdict": ml_label,
        "final_confidence": ml_percentage,
        "ml_verdict": ml_label,
        "ml_confidence": ml_percentage,
        "groq_verdict": groq_label,
        "groq_explanation": groq_result.get("explanation", ""),
        "groq_confidence": groq_confidence,
        "agreement": "Models disagree",
        "verdict_source": "ML Model (Groq disagreed)"
    }


# --------------------------------------------------
# LOCAL GROQ CONNECTION TEST
# --------------------------------------------------

def test_groq():
    print("Testing Groq API connection...")
    print("API key found:", bool(os.getenv("GROQ_API_KEY")))
    print("Primary model:", GROQ_MODEL_PRIMARY)

    result = analyze_with_groq(
        "The Government of India announced a new education policy."
    )

    print("Groq used:", result.get("groq_used"))
    print("Verdict:", result.get("verdict"))
    print("Explanation:", result.get("explanation"))
    print("Error:", result.get("error"))


if __name__ == "__main__":
    test_groq()
