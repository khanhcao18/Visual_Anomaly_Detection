from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: str | Path) -> list[Path]:
    """Return image paths under a folder in deterministic order."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_transform(image_size: int, augment: bool = False) -> transforms.Compose:
    steps: list = [
        transforms.Resize((image_size, image_size)),
    ]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04),
            ]
        )
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
