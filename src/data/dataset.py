from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.preprocessing import image_transform, list_images, load_rgb_image


class ImageFolderDataset(Dataset):
    """Simple image dataset with one label per folder."""

    def __init__(
        self,
        root: str | Path,
        label: int = 0,
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.paths = list_images(self.root)
        self.label = label
        self.transform = image_transform(image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path = self.paths[index]
        return self.transform(load_rgb_image(path)), self.label, str(path)


class LabeledAnomalyDataset(Dataset):
    """Dataset built from normal and abnormal directories."""

    def __init__(self, normal_dir: str | Path, abnormal_dir: str | Path, image_size: int = 128) -> None:
        self.samples: list[tuple[Path, int]] = [(p, 0) for p in list_images(normal_dir)]
        self.samples.extend((p, 1) for p in list_images(abnormal_dir))
        self.samples.sort(key=lambda item: str(item[0]))
        self.transform = image_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        return self.transform(load_rgb_image(path)), label, str(path)
