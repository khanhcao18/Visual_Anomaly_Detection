from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ImageFolderDataset, LabeledAnomalyDataset
from src.evaluation.metrics import classification_metrics
from src.models.feature_extractor import build_resnet_feature_extractor, extract_features
from src.utils import load_config, resolve_device, set_seed


def make_model(model_type: str, contamination: float) -> object:
    if model_type == "one_class_svm":
        return Pipeline([("scaler", StandardScaler()), ("model", OneClassSVM(gamma="scale", nu=contamination))])
    if model_type == "pca":
        return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95))])
    return IsolationForest(contamination=contamination, random_state=42)


def anomaly_scores(model: object, features: np.ndarray) -> np.ndarray:
    if isinstance(model, Pipeline) and "pca" in model.named_steps:
        transformed = model.transform(features)
        reconstructed = model.named_steps["scaler"].inverse_transform(model.named_steps["pca"].inverse_transform(transformed))
        return np.mean((features - reconstructed) ** 2, axis=1)
    if hasattr(model, "decision_function"):
        return -model.decision_function(features)
    return -model.score_samples(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pretrained-feature anomaly baseline.")
    parser.add_argument("--config", default="configs/resnet_isolation_forest.yaml")
    parser.add_argument("--model-type", choices=["isolation_forest", "one_class_svm", "pca"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "auto"))
    model_type = args.model_type or cfg["model"].get("type", "isolation_forest")
    contamination = cfg["model"].get("contamination", 0.05)

    train_ds = ImageFolderDataset(cfg["data"]["train_normal_dir"], image_size=cfg["data"]["image_size"])
    test_ds = LabeledAnomalyDataset(
        cfg["data"]["test_normal_dir"],
        cfg["data"]["test_abnormal_dir"],
        image_size=cfg["data"]["image_size"],
    )
    extractor = build_resnet_feature_extractor(device=device)
    train_features, _, _ = extract_features(extractor, DataLoader(train_ds, batch_size=32), device=device)
    test_features, labels, _ = extract_features(extractor, DataLoader(test_ds, batch_size=32), device=device)

    model = make_model(model_type, contamination)
    model.fit(train_features)
    scores = anomaly_scores(model, test_features)
    threshold = float(np.percentile(anomaly_scores(model, train_features), 95))
    metrics = classification_metrics(labels, scores, threshold)

    output = Path(cfg["outputs"]["model"])
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "model_type": model_type}, output)

    print(f"model: {model_type}")
    print(f"checkpoint: {output}")
    print(f"threshold: {threshold:.6f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1: {metrics['f1']:.4f}")
    print(f"roc_auc: {metrics['roc_auc']:.4f}")
    print(f"pr_auc: {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
