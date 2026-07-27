from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.preprocessing import image_transform, load_rgb_image
from src.models.autoencoder import ConvAutoencoder, reconstruction_errors
from src.utils import load_json, resolve_device
from src.visualization.heatmap import save_reconstruction_panel


def load_model(checkpoint_path: str, device: str) -> ConvAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ConvAutoencoder(latent_channels=checkpoint.get("latent_channels", 128)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_image(image_path: str, checkpoint_path: str, threshold_path: str, output_heatmap: str | None = None) -> dict:
    threshold_payload = load_json(threshold_path)
    device = resolve_device("auto")
    model = load_model(checkpoint_path, device)
    transform = image_transform(threshold_payload.get("image_size", 128))
    image = transform(load_rgb_image(image_path)).unsqueeze(0).to(device)
    reconstruction = model(image)
    score = float(reconstruction_errors(image, reconstruction).item())
    threshold = float(threshold_payload["threshold"])
    ratio = score / max(threshold, 1e-12)
    confidence = "High" if ratio >= 1.5 or ratio <= 0.67 else "Medium"
    prediction = "Defective" if score >= threshold else "Normal"
    heatmap_path = None
    if output_heatmap:
        heatmap_path = str(save_reconstruction_panel(image[0].cpu(), reconstruction[0].cpu(), output_heatmap))
    return {
        "prediction": prediction,
        "anomaly_score": score,
        "threshold": threshold,
        "confidence": confidence,
        "heatmap_path": heatmap_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether one image is anomalous.")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="models/autoencoder.pt")
    parser.add_argument("--threshold", default="models/autoencoder_threshold.json")
    parser.add_argument("--heatmap", default="reports/figures/prediction_heatmap.png")
    args = parser.parse_args()

    result = predict_image(args.image, args.checkpoint, args.threshold, args.heatmap)
    print(f"Prediction: {result['prediction']}")
    print(f"Anomaly score: {result['anomaly_score']:.6f}")
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"Confidence: {result['confidence']}")
    if result["heatmap_path"]:
        print(f"Heatmap: {Path(result['heatmap_path']).resolve()}")


if __name__ == "__main__":
    main()
