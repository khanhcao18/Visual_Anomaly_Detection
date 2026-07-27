from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.models.autoencoder import ConvAutoencoder
from src.training.losses import mse_reconstruction_loss


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def train_autoencoder(
    dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    checkpoint_path: str | Path,
    latent_channels: int = 128,
    validation_split: float = 0.2,
    patience: int = 8,
) -> tuple[ConvAutoencoder, TrainHistory]:
    if len(dataset) == 0:
        raise ValueError("No training images found. Add normal images to the configured train directory.")

    val_size = max(1, int(len(dataset) * validation_split)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size]) if val_size else (dataset, None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    model = ConvAutoencoder(latent_channels=latent_channels).to(device)
    criterion = mse_reconstruction_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainHistory()
    best_loss = float("inf")
    stale_epochs = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for _epoch in range(epochs):
        model.train()
        train_total = 0.0
        for batch, _, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch.size(0)
        history.train_loss.append(train_total / len(train_ds))

        model.eval()
        if val_loader:
            val_total = 0.0
            with torch.no_grad():
                for batch, _, _ in val_loader:
                    batch = batch.to(device)
                    val_total += criterion(model(batch), batch).item() * batch.size(0)
            current = val_total / len(val_ds)
        else:
            current = history.train_loss[-1]
        history.val_loss.append(current)

        if current < best_loss:
            best_loss = current
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "latent_channels": latent_channels}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model, history
