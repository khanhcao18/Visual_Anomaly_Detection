from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LabeledAnomalyDataset
from src.evaluation.metrics import classification_metrics
from src.evaluation.threshold import best_f1_threshold, threshold_sweep
from src.models.autoencoder import ConvAutoencoder, reconstruction_errors
from src.utils import load_config, load_json, resolve_device


@torch.no_grad()
def score_dataset(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scores: list[float] = []
    labels: list[int] = []
    paths: list[str] = []
    model.eval()
    for batch, y, batch_paths in loader:
        batch = batch.to(device)
        scores.extend(reconstruction_errors(batch, model(batch)).cpu().tolist())
        labels.extend(y.tolist())
        paths.extend(batch_paths)
    return np.array(scores), np.array(labels), paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection metrics.")
    parser.add_argument("--config", default="configs/autoencoder.yaml")
    parser.add_argument("--use-best-f1-threshold", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    threshold_payload = load_json(cfg["outputs"]["threshold"])
    checkpoint = torch.load(cfg["outputs"]["checkpoint"], map_location=device)

    model = ConvAutoencoder(latent_channels=checkpoint.get("latent_channels", 128)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = LabeledAnomalyDataset(
        cfg["data"]["test_normal_dir"],
        cfg["data"]["test_abnormal_dir"],
        image_size=cfg["data"]["image_size"],
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    scores, labels, paths = score_dataset(model, loader, device)
    threshold = best_f1_threshold(labels, scores) if args.use_best_f1_threshold else threshold_payload["threshold"]
    metrics = classification_metrics(labels, scores, threshold)
    sweep_points = np.quantile(scores, [0.25, 0.5, 0.75, 0.9, 0.95]).tolist()
    sweep = threshold_sweep(labels, scores, sweep_points)

    results_path = Path(cfg["outputs"]["results_csv"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "score", "prediction"])
        writer.writeheader()
        for path, label, score in zip(paths, labels, scores):
            writer.writerow({"path": path, "label": int(label), "score": float(score), "prediction": int(score >= threshold)})

    print(f"threshold: {threshold:.6f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1: {metrics['f1']:.4f}")
    print(f"roc_auc: {metrics['roc_auc']:.4f}")
    print(f"pr_auc: {metrics['pr_auc']:.4f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")
    print("threshold sweep:")
    for row in sweep:
        print(f"{row['threshold']:.6f},{row['precision']:.4f},{row['recall']:.4f},{row['f1']:.4f}")


if __name__ == "__main__":
    main()
