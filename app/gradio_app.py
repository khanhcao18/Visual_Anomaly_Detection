from __future__ import annotations

from pathlib import Path

import gradio as gr

from predict import predict_image

CHECKPOINT = "models/autoencoder.pt"
THRESHOLD = "models/autoencoder_threshold.json"
HEATMAP = "reports/figures/gradio_heatmap.png"


def infer(image_path: str) -> tuple[str, str]:
    if not Path(CHECKPOINT).exists() or not Path(THRESHOLD).exists():
        return "Train the autoencoder first with `python train.py`.", ""
    result = predict_image(image_path, CHECKPOINT, THRESHOLD, HEATMAP)
    summary = (
        f"Prediction: {result['prediction']}\n"
        f"Anomaly score: {result['anomaly_score']:.6f}\n"
        f"Threshold: {result['threshold']:.6f}\n"
        f"Confidence: {result['confidence']}"
    )
    return summary, result["heatmap_path"] or ""


demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="filepath", label="Upload product/object image"),
    outputs=[gr.Textbox(label="Result"), gr.Image(label="Original | Reconstruction | Error heatmap")],
    title="Visual Anomaly Detection",
)


if __name__ == "__main__":
    demo.launch()
