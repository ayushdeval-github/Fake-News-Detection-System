"""
app.py — Main Gradio Application Entry Point
Fake News Detection System | Ayush Deval | Branch: ayush-deval | 2026-27

Architecture:
  User input → ML Model (LR/LSTM/BERT) → Groq LLaMA 3 70B → Combined verdict
"""

import gradio as gr
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Load ML Models globally so they only load once when the Space starts
from utils.model_loader import load_all_models
models_cache = load_all_models()

# ADD THIS LINE: Diagnostic print to check actual dictionary keys
print("AVAILABLE MODELS IN CACHE:", models_cache.keys()) 

logger.info("All ML models loaded into memory.")

def analyze_news(text, model_choice):
    """
    Core prediction function. 
    Takes inputs from the Gradio UI and returns a dictionary for the UI to display.
    """
    start_time = time.time()
    text = text.strip()
    model_choice = model_choice.lower()

    # Input Validation
    if not text or len(text) < 10:
        return {"error": "Text must be at least 10 characters long."}
    if model_choice not in ("lr", "lstm", "bert"):
        return {"error": "Invalid model choice."}

    # Step 1: ML Model Inference
    try:
        from utils.predict import run_prediction
        ml_prediction, ml_confidence = run_prediction(text, model_choice, models_cache)
    except Exception as exc:
        logger.error("ML inference error: %s", exc, exc_info=True)
        return {"error": "ML model inference failed.", "detail": str(exc)}

    # Step 2: Groq LLaMA 3 70B Analysis
    try:
        from utils.groq_client import analyze_with_groq
        groq_result = analyze_with_groq(text)
    except Exception as exc:
        logger.error("Groq error: %s", exc)
        groq_result = {
            "verdict": "Unavailable", 
            "explanation": f"Groq API Error: {str(exc)}",
            "confidence": "N/A", 
            "groq_used": False
        }

    # Step 3: Combine Results
    try:
        from utils.groq_client import combined_verdict
        result = combined_verdict(ml_prediction, ml_confidence, groq_result)
        result["model_used"] = model_choice.upper()
        result["time_ms"] = round((time.time() - start_time) * 1000, 2)
        return result
    except Exception as exc:
        logger.error("Combiner error: %s", exc)
        return {"error": "Failed to generate final verdict.", "detail": str(exc)}

# ---------------------------------------------------------
# Build the Gradio Web Interface
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📰 Fake News Detection System")
    gr.Markdown("**Architecture:** User input → ML Model (LR/LSTM/BERT) → Groq LLaMA 3 70B → Combined verdict")
    
    with gr.Row():
        # Left side: User Inputs
        with gr.Column():
            news_input = gr.Textbox(
                lines=8, 
                label="News Article Text", 
                placeholder="Paste the news article here (minimum 10 characters)..."
            )
            model_selector = gr.Dropdown(
                choices=["lr", "lstm", "bert"], 
                value="lr", 
                label="Select Machine Learning Backend"
            )
            submit_btn = gr.Button("Analyze News", variant="primary")
            
        # Right side: Results Output
        with gr.Column():
            output_json = gr.JSON(label="Detection Results & AI Explanation")

    # Connect the button to the function
    submit_btn.click(
        fn=analyze_news,
        inputs=[news_input, model_selector],
        outputs=output_json
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()