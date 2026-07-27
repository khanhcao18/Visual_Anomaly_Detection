from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return array


def save_reconstruction_panel(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_np = tensor_to_image(original)
    reconstruction_np = tensor_to_image(reconstruction)
    error = np.mean((original_np - reconstruction_np) ** 2, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[1].imshow(reconstruction_np)
    axes[1].set_title("Reconstruction")
    axes[2].imshow(original_np)
    axes[2].imshow(error, cmap="inferno", alpha=0.68)
    axes[2].set_title("Error heatmap")
    for axis in axes:
        axis.axis("off")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
