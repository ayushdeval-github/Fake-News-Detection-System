"""
app.py — Main Application Entry Point
Fake News Detection System | Ayush Deval | 2026-27

Architecture:
    User input → ML Model (LR/LSTM/BERT) → Groq LLaMA 3 70B → Combined verdict

Deployment:
    HuggingFace Spaces (Gradio SDK)
    → Set GROQ_API_KEY in HF Spaces → Settings → Repository Secrets
"""

import time
import logging
import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ── LOGGING ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ── LOAD ML MODELS AT STARTUP ────────────────────────────────
from utils.model_loader import load_all_models
models_cache = load_all_models()
logger.info("All ML models loaded into memory.")


# ── CORE PREDICTION FUNCTION ─────────────────────────────────

def predict(text: str, model_choice: str) -> tuple:
    """
    Main prediction function called by Gradio UI.

    Args:
        text:         News text from user
        model_choice: "Logistic Regression" | "LSTM" | "BERT"

    Returns:
        Tuple of (final_verdict, confidence, ml_result, groq_explanation, agreement)
    """
    # Map Gradio dropdown labels to model keys
    model_map = {
        "Logistic Regression": "lr",
        "LSTM":                "lstm",
        "BERT":                "bert",
    }
    model_key = model_map.get(model_choice, "lr")

    # Validate input
    if not text or len(text.strip()) < 10:
        return "⚠️ Error", 0, "Text too short", "Please enter at least 10 characters.", ""

    start_time = time.time()

    # Step 1: ML Model inference
    try:
        from utils.predict import run_prediction
        ml_prediction, ml_confidence = run_prediction(text.strip(), model_key, models_cache)
        ml_label = "🔴 FAKE" if ml_prediction == 1 else "🟢 REAL"
        ml_result = f"{ml_label}  ({round(ml_confidence * 100)}% confidence)"
    except Exception as exc:
        logger.error("ML inference error: %s", exc, exc_info=True)
        return "⚠️ Error", 0, f"ML Error: {str(exc)}", "ML model failed.", ""

    # Step 2: Groq LLaMA 3 70B
    try:
        from utils.groq_client import analyze_with_groq
        groq_result = analyze_with_groq(text.strip())
    except Exception as exc:
        logger.error("Groq error: %s", exc)
        groq_result = {
            "verdict":     "Unavailable",
            "explanation": str(exc),
            "confidence":  "N/A",
            "groq_used":   False
        }

    # Step 3: Combine both verdicts
    from utils.groq_client import combined_verdict
    result     = combined_verdict(ml_prediction, ml_confidence, groq_result)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    # Format outputs for Gradio
    final      = result.get("final_verdict", "Unknown")
    confidence = result.get("final_confidence", 0)
    agreement  = result.get("agreement", "")
    explanation = groq_result.get("explanation", "Groq unavailable.")
    groq_conf  = groq_result.get("confidence", "N/A")

    verdict_label = f"🔴 FAKE NEWS" if final == "Fake" else f"🟢 REAL NEWS"
    groq_text     = f'"{explanation}"\n\nGroq Confidence: {groq_conf} | Model: LLaMA 3 70B | Time: {elapsed_ms}ms'

    return verdict_label, confidence, ml_result, groq_text, agreement


# ── GRADIO UI ────────────────────────────────────────────────

def build_gradio_app():
    """
    Build the Gradio interface.
    WHY GRADIO:
        HuggingFace Spaces runs Gradio natively — no extra config needed.
        Gradio provides a clean UI with zero frontend code required.
        Perfect for ML demos and college project presentations.
    """

    with gr.Blocks(
        title="TruthLens — Fake News Detector",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        .title-text { text-align: center; margin-bottom: 10px; }
        .verdict-box { font-size: 1.4em !important; font-weight: bold !important; text-align: center; }
        footer { display: none !important; }
        """
    ) as demo:

        # ── Header ──────────────────────────────
        gr.Markdown("""
        <div class="title-text">
        <h1>🔍 TruthLens — Fake News Detection System</h1>
        <p>Powered by <b>LR + LSTM + BERT</b> + <b>Groq LLaMA 3 70B</b></p>
        <p><i>Fake News Detection System | Ayush Deval | NLP Major Project 2026-27</i></p>
        </div>
        """)

        gr.Markdown("---")

        # ── Input Section ────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="📰 Paste News Text Here",
                    placeholder="Paste any news headline or article here...\n\nExample: 'Aryan is the Prime Minister of India'",
                    lines=6,
                    max_lines=15,
                )
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    label="🤖 Select ML Model",
                    choices=["Logistic Regression", "LSTM", "BERT"],
                    value="BERT",
                    info="BERT = highest accuracy"
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze News",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("""
                **Model Info:**
                - 🟢 LR — fastest (~50ms)
                - 🟡 LSTM — balanced (~200ms)
                - 🔵 BERT — best accuracy (~2s)
                """)

        gr.Markdown("---")

        # ── Output Section ───────────────────────
        gr.Markdown("### 📊 Analysis Results")

        with gr.Row():
            verdict_output = gr.Textbox(
                label="🏁 Final Verdict",
                interactive=False,
                elem_classes=["verdict-box"],
            )
            confidence_output = gr.Number(
                label="📈 Confidence Score (%)",
                interactive=False,
            )
            agreement_output = gr.Textbox(
                label="🤝 Model Agreement",
                interactive=False,
            )

        with gr.Row():
            ml_output = gr.Textbox(
                label="🧠 ML Model Result",
                interactive=False,
            )

        groq_output = gr.Textbox(
            label="🤖 Groq LLaMA 3 70B — AI Explanation",
            interactive=False,
            lines=4,
        )

        # ── Examples ────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 💡 Try These Examples")

        gr.Examples(
            examples=[
                ["Aryan is the Prime Minister of India", "BERT"],
                ["Narendra Modi announced new education policy today", "BERT"],
                ["SHOCKING: Scientists prove Earth is flat, NASA confirms!!!", "LSTM"],
                ["According to Reuters, RBI increased repo rate by 0.25%", "Logistic Regression"],
                ["Doctors HATE this one trick to cure diabetes at home", "BERT"],
                ["Lok Sabha passed the new budget bill with 342 votes", "LSTM"],
            ],
            inputs=[text_input, model_dropdown],
            label="Click any example to try it"
        )

        # ── Footer ───────────────────────────────
        gr.Markdown("""
        ---
        <div style="text-align:center; color:gray; font-size:0.85em;">
        Built by <b>Ayush Deval</b> · Fake News Detection System · NLP Major Project 2026-27<br>
        Dataset: ISOT Fake News Dataset · Models: LR + LSTM + DistilBERT · AI: Groq LLaMA 3 70B
        </div>
        """)

        # ── Event Handler ────────────────────────
        analyze_btn.click(
            fn=predict,
            inputs=[text_input, model_dropdown],
            outputs=[
                verdict_output,
                confidence_output,
                ml_output,
                groq_output,
                agreement_output,
            ],
        )

        # Also trigger on Enter key
        text_input.submit(
            fn=predict,
            inputs=[text_input, model_dropdown],
            outputs=[
                verdict_output,
                confidence_output,
                ml_output,
                groq_output,
                agreement_output,
            ],
        )

    return demo


# ── ENTRY POINT ──────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),  # HF Spaces uses port 7860
        share=False,
    )