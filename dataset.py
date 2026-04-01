"""
Dataset loader for paired semantic-label / photo datasets.

Supports:
    - Facades (auto-download)
    - Cityscapes
    - Any custom dataset with side-by-side paired images

Each image file is expected to be a horizontally concatenated pair:
    [image_A | image_B]
Where A and B are determined by the `direction` config (AtoB or BtoA).
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ── Download URLs for built-in datasets ────────────────────────────────
DATASET_URLS = {
    "facades": "https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.tar.gz",
}


class PairedImageDataset(Dataset):
    """
    Dataset of paired images (label map ↔ photo).

    Expects images to be side-by-side in a single file:
        [left_image | right_image]

    With direction='AtoB': left = input (label), right = target (photo).
    With direction='BtoA': reversed.

    Args:
        root_dir:   Directory containing the paired images.
        image_size: Target spatial resolution (square).
        direction:  'AtoB' or 'BtoA'.
        split:      'train', 'val', or 'test' (subfolder name).
        augment:    Apply random augmentations during training.
    """

    def __init__(self, root_dir, image_size=256, direction="AtoB",
                 split="train", augment=True):
        super().__init__()
        self.image_size = image_size
        self.direction = direction
        self.augment = augment and (split == "train")

        # Find image files
        self.root = Path(root_dir)
        search_dir = self.root / split if (self.root / split).exists() else self.root

        self.image_paths = sorted([
            p for p in search_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {search_dir}. "
                f"Expected paired images (.jpg/.png) in this directory."
            )

        print(f"[Dataset] Found {len(self.image_paths)} paired images in '{search_dir}'")

        # Transforms (applied after splitting the pair)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def _random_augment(self, img_a, img_b):
        """Apply synchronized random augmentations to both images."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            img_a = img_a.flip(-1)  # Flip width dimension
            img_b = img_b.flip(-1)

        return img_a, img_b

    def __getitem__(self, idx):
        """
        Returns:
            label_map: Tensor [C, H, W] in [-1, 1].
            real_image: Tensor [C, H, W] in [-1, 1].
        """
        # Load the paired image
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Split into two halves (left = A, right = B)
        w, h = img.size
        mid = w // 2

        img_a = img.crop((0, 0, mid, h))       # Left half
        img_b = img.crop((mid, 0, w, h))        # Right half

        # Apply transforms
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        # Apply augmentations
        if self.augment:
            img_a, img_b = self._random_augment(img_a, img_b)

        # Assign based on direction
        if self.direction == "AtoB":
            label_map, real_image = img_a, img_b
        else:
            label_map, real_image = img_b, img_a

        return label_map, real_image


def create_dataloaders(cfg):
    """
    Create train and validation DataLoaders.

    Args:
        cfg: Config object with dataset_dir, image_size, batch_size, etc.

    Returns:
        train_loader, val_loader (val_loader may be None if no val split).
    """
    train_dataset = PairedImageDataset(
        root_dir=cfg.dataset_dir,
        image_size=cfg.image_size,
        direction=cfg.direction,
        split="train",
        augment=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Try to create validation loader
    val_loader = None
    val_dir = Path(cfg.dataset_dir) / "val"
    if val_dir.exists() and any(val_dir.iterdir()):
        val_dataset = PairedImageDataset(
            root_dir=cfg.dataset_dir,
            image_size=cfg.image_size,
            direction=cfg.direction,
            split="val",
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def download_facades(target_dir="./data/facades"):
    """
    Download and prepare the CMP Facades dataset.

    Creates a paired-image format compatible with PairedImageDataset.
    """
    target_dir = Path(target_dir)

    if (target_dir / "train").exists():
        print(f"[Download] Facades already exists at {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"

    tar_path = target_dir / "facades.tar.gz"
    print(f"[Download] Downloading Facades dataset from {url}...")

    urllib.request.urlretrieve(url, tar_path)
    print("[Download] Extracting...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(target_dir.parent)

    tar_path.unlink()
    print(f"[Download] Facades dataset ready at {target_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=str, choices=["facades"],
                        help="Download a built-in dataset")
    args = parser.parse_args()

    if args.download == "facades":
        download_facades()
    else:
        print("Usage: python dataset.py --download facades")
