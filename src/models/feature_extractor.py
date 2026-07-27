from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models


def build_resnet_feature_extractor(device: str = "cpu") -> torch.nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_features(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    for batch, y, batch_paths in loader:
        output = model(batch.to(device)).cpu().numpy()
        features.append(output)
        labels.append(y.numpy())
        paths.extend(batch_paths)
    return np.vstack(features), np.concatenate(labels), paths
