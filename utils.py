"""
Utility functions: visualization, checkpointing, logging, metrics.
"""

import csv
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] for display."""
    return (tensor + 1.0) / 2.0


def save_sample_grid(label_maps, generated, real_images, path, max_samples=4):
    """
    Save a grid of [label | generated | real] triplets.

    Args:
        label_maps: Tensor [B, C, H, W] in [-1, 1].
        generated:  Tensor [B, C, H, W] in [-1, 1].
        real_images: Tensor [B, C, H, W] in [-1, 1].
        path:       Output file path.
        max_samples: Max number of rows to display.
    """
    n = min(max_samples, label_maps.size(0))

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.unsqueeze(0) if hasattr(axes, "unsqueeze") else [axes]

    titles = ["Semantic Label", "Generated", "Ground Truth"]

    for i in range(n):
        images = [
            denormalize(label_maps[i]).cpu().permute(1, 2, 0).numpy().clip(0, 1),
            denormalize(generated[i]).cpu().permute(1, 2, 0).numpy().clip(0, 1),
            denormalize(real_images[i]).cpu().permute(1, 2, 0).numpy().clip(0, 1),
        ]
        for j in range(3):
            ax = axes[i][j] if n > 1 else axes[j]
            ax.imshow(images[j])
            if i == 0:
                ax.set_title(titles[j], fontsize=14)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(generator, discriminator, opt_g, opt_d, epoch, path):
    """Save a training checkpoint."""
    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
    }, path)


def load_checkpoint(path, generator, discriminator=None, opt_g=None, opt_d=None):
    """
    Load a training checkpoint.

    Returns:
        epoch: The epoch the checkpoint was saved at.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])

    if discriminator is not None and "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    if opt_g is not None and "optimizer_g_state_dict" in checkpoint:
        opt_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    if opt_d is not None and "optimizer_d_state_dict" in checkpoint:
        opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    return checkpoint.get("epoch", 0)


class LossLogger:
    """Simple CSV logger for training losses."""

    def __init__(self, log_dir):
        self.log_path = Path(log_dir) / "losses.csv"
        self.entries = []
        self._header_written = False

    def log(self, epoch, step, **losses):
        """Log a set of loss values."""
        entry = {"epoch": epoch, "step": step, **losses}
        self.entries.append(entry)

    def flush(self):
        """Write all buffered entries to CSV."""
        if not self.entries:
            return

        fieldnames = list(self.entries[0].keys())
        mode = "a" if self._header_written else "w"

        with open(self.log_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(self.entries)

        self.entries.clear()

    def plot_losses(self, output_path=None):
        """Generate a loss curve plot from the CSV log."""
        if not self.log_path.exists():
            return

        epochs, d_losses, g_losses, l1_losses = [], [], [], []

        with open(self.log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                d_losses.append(float(row.get("d_loss", 0)))
                g_losses.append(float(row.get("g_loss_adv", 0)))
                l1_losses.append(float(row.get("g_loss_l1", 0)))

        if not epochs:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, d_losses, label="D Loss", alpha=0.7)
        ax1.plot(epochs, g_losses, label="G Loss (Adv)", alpha=0.7)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Adversarial Losses")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, l1_losses, label="G Loss (L1)", color="green", alpha=0.7)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.set_title("Reconstruction Loss (L1)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = output_path or (self.log_path.parent / "loss_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
