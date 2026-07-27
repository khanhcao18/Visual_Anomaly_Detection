from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.data.dataset import ImageFolderDataset
from src.data.preprocessing import ensure_dirs
from src.evaluation.threshold import percentile_threshold
from src.models.autoencoder import reconstruction_errors
from src.training.trainer import train_autoencoder
from src.utils import load_config, resolve_device, save_json, set_seed


@torch.no_grad()
def score_training_normals(model, dataset, batch_size: int, device: str) -> list[float]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    model.eval()
    for batch, _, _ in loader:
        batch = batch.to(device)
        scores.extend(reconstruction_errors(batch, model(batch)).cpu().tolist())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a convolutional autoencoder on normal images.")
    parser.add_argument("--config", default="configs/autoencoder.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "auto"))
    ensure_dirs(["models", "reports", "reports/figures"])

    dataset = ImageFolderDataset(
        cfg["data"]["train_normal_dir"],
        label=0,
        image_size=cfg["data"]["image_size"],
        augment=True,
    )
    model, history = train_autoencoder(
        dataset=dataset,
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        device=device,
        checkpoint_path=cfg["outputs"]["checkpoint"],
        latent_channels=cfg["training"]["latent_channels"],
        validation_split=cfg["training"]["validation_split"],
        patience=cfg["training"]["patience"],
    )

    scores = score_training_normals(model, dataset, cfg["training"]["batch_size"], device)
    threshold = percentile_threshold(scores, cfg["threshold"].get("percentile", 95))
    save_json(
        {
            "threshold": threshold,
            "method": cfg["threshold"].get("method", "percentile"),
            "percentile": cfg["threshold"].get("percentile", 95),
            "image_size": cfg["data"]["image_size"],
            "checkpoint": cfg["outputs"]["checkpoint"],
        },
        cfg["outputs"]["threshold"],
    )

    history_path = Path(cfg["outputs"]["history_csv"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for idx, (train_loss, val_loss) in enumerate(zip(history.train_loss, history.val_loss), start=1):
            writer.writerow({"epoch": idx, "train_loss": train_loss, "val_loss": val_loss})

    print(f"trained autoencoder on {len(dataset)} normal images")
    print(f"checkpoint: {cfg['outputs']['checkpoint']}")
    print(f"threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
