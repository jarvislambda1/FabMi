#!/usr/bin/env python
"""FabMi - Semiconductor Root Cause Analysis Assistant."""

import os
import json
import time
from pathlib import Path
import gradio as gr
from openai import OpenAI

# Configuration - can be overridden via environment variables
API_URL = os.environ.get("FABMI_API_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("FABMI_MODEL_NAME", "gpt-3.5-turbo")
MAX_TOKENS = int(os.environ.get("FABMI_MAX_TOKENS", "1500"))
TEMPERATURE = float(os.environ.get("FABMI_TEMPERATURE", "0.1"))

# Load examples from same directory as this script
SCRIPT_DIR = Path(__file__).parent
EXAMPLES_FILE = SCRIPT_DIR / "examples.json"

with open(EXAMPLES_FILE) as f:
    EXAMPLES = json.load(f)

EXAMPLE_LIST = [[ex['input']] for ex in EXAMPLES]


def get_client():
    return OpenAI(base_url=API_URL, api_key="not-needed")


def analyze_defect(defect_description: str) -> tuple[str, str]:
    """Call the model API and return (response, latency)."""
    if not defect_description.strip():
        return "Please enter a defect description.", ""

    instruction = "Analyze the following semiconductor wafer defect and provide root cause analysis."
    prompt = f"{instruction}\n\n{defect_description}"

    client = get_client()
    start = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior semiconductor process engineer. Provide detailed root cause analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        latency = time.time() - start
        result = response.choices[0].message.content
        return result, f"{latency:.2f}s"
    except Exception as e:
        return f"Error: {str(e)}", ""


# Custom CSS for semiconductor/fab theme
custom_css = """
/* Main theme colors - semiconductor blue/silver */
.gradio-container {
    background: linear-gradient(135deg, #0a1628 0%, #1a2744 50%, #0d2137 100%) !important;
    min-height: 100vh;
}

/* Header styling */
.header-title {
    text-align: center;
    padding: 20px;
    margin-bottom: 10px;
}

.header-title h1 {
    color: #00d4ff !important;
    font-size: 2.5em !important;
    font-weight: 700 !important;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    margin-bottom: 5px !important;
}

.header-title p {
    color: #a0b0c0 !important;
    font-size: 1.1em;
}

/* All text should be light colored */
.gradio-container, .gradio-container * {
    color: #e0e8f0 !important;
}

/* Labels - cyan color */
label, .gr-box label, span.svelte-1gfkn6j, .gr-input-label, .label-wrap span {
    color: #00d4ff !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* Specific label styling for textbox labels */
.svelte-1f354aw, .svelte-1gfkn6j, span[data-testid="block-label"] {
    color: #00d4ff !important;
}

/* Section headers */
h4, .gr-markdown h4 {
    color: #00d4ff !important;
    font-weight: 600 !important;
}

h3, .gr-markdown h3 {
    color: #00d4ff !important;
}

/* Italic text */
em, i {
    color: #a0b8d0 !important;
}

/* Input/Output textboxes */
textarea {
    background: #0d1926 !important;
    border: 1px solid #3a5a7a !important;
    color: #e0e8f0 !important;
    border-radius: 8px !important;
}

textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.2) !important;
}

textarea::placeholder {
    color: #6080a0 !important;
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg, #0066cc 0%, #00d4ff 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 30px !important;
    border-radius: 8px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4) !important;
}

/* Force examples to be 1 per row */
.examples-table {
    max-width: 100% !important;
}

.examples-table > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
}

.examples-table .gr-samples-table {
    display: flex !important;
    flex-direction: column !important;
}

.examples-table button, .examples-table .gr-sample-textbox {
    width: 100% !important;
    max-width: 100% !important;
    display: block !important;
    background: #0d1926 !important;
    border: 1px solid #3a5a7a !important;
    color: #c0d0e0 !important;
    text-align: left !important;
    padding: 12px 16px !important;
    margin: 4px 0 !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 50px !important;
}

.examples-table button:hover {
    background: #1a3050 !important;
    border-color: #00d4ff !important;
    color: #ffffff !important;
}

/* Performance indicator */
.performance-box input {
    background: #0d1926 !important;
    border: 1px solid #2a6a4a !important;
    color: #00ff88 !important;
    text-align: center;
    font-family: monospace;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
}

.footer p {
    color: #6080a0 !important;
    font-size: 0.9em;
}

/* Block backgrounds */
.gr-block, .gr-box, .gr-panel {
    background: transparent !important;
}

/* Form elements */
.gr-form {
    background: transparent !important;
}

/* Input wrapper */
.gr-input, .gr-text-input {
    background: #0d1926 !important;
}

/* Fix label containers - remove white backgrounds */
.label-wrap, .container > label, div[class*="label"], .block label {
    background: transparent !important;
    color: #00d4ff !important;
}

/* Target all span elements that might be labels */
.block span:first-child, .form span, .wrap span {
    color: #00d4ff !important;
    background: transparent !important;
}

/* Override any white/light backgrounds on input containers */
.wrap, .container, .block {
    background: transparent !important;
}

/* Ensure textarea containers are dark */
.gr-textbox, .textbox {
    background: transparent !important;
}

.gr-textbox > div, .textbox > div {
    background: transparent !important;
}

/* Fix wrap for example gallery */
.gr-examples {
    display: block !important;
}

.gr-examples > div > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}
"""

# Build Gradio UI
with gr.Blocks(
    title="FabMi - Semiconductor RCA",
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )
) as demo:

    # Header
    gr.HTML("""
    <div class="header-title">
        <h1>FabMi</h1>
        <p>AI-Powered Semiconductor Root Cause Analysis</p>
    </div>
    """)

    with gr.Row():
        # Input Column
        with gr.Column():
            gr.Markdown("#### Defect Input")
            input_text = gr.Textbox(
                label="Defect Description",
                placeholder="Enter wafer defect characteristics, measurements, and observations...",
                lines=10,
                show_label=True
            )
            analyze_btn = gr.Button(
                "Analyze Root Cause",
                variant="primary",
                size="lg"
            )

        # Output Column
        with gr.Column():
            gr.Markdown("#### Analysis Results")
            output_text = gr.Textbox(
                label="Root Cause Analysis",
                lines=10,
                show_copy_button=True,
                show_label=True
            )
            latency_text = gr.Textbox(
                label="Response Time",
                interactive=False,
                max_lines=1,
                elem_classes="performance-box"
            )

    # Examples Section
    gr.HTML('<h3 style="color: #00d4ff; margin-top: 20px;">Sample Defect Descriptions</h3>')
    gr.HTML('<p style="color: #a0b8d0; font-style: italic; margin-bottom: 10px;">Click any example below to load it:</p>')

    with gr.Column(elem_classes="examples-table"):
        gr.Examples(
            examples=EXAMPLE_LIST,
            inputs=[input_text],
            label="",
            examples_per_page=9
        )

    # Footer
    gr.HTML("""
    <div class="footer">
        <p>Powered by Fine-tuned ERNIE 4.5 | Built for Semiconductor Manufacturing</p>
    </div>
    """)

    # Event handler
    analyze_btn.click(
        fn=analyze_defect,
        inputs=[input_text],
        outputs=[output_text, latency_text]
    )

if __name__ == "__main__":
    port = int(os.environ.get("FABMI_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
