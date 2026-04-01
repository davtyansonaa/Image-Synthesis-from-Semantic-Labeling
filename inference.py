"""
Inference script for Image Synthesis from Semantic Labeling.

Loads a trained generator and produces synthesized images from semantic label maps.

Usage:
    # Single image
    python inference.py --input label.png --checkpoint outputs/checkpoints/latest.pth

    # Directory of images
    python inference.py --input_dir ./test_labels/ --checkpoint outputs/checkpoints/latest.pth
"""

from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config
from models import UNetGenerator
from utils import denormalize


def load_generator(checkpoint_path, cfg, device):
    """Load a trained generator from checkpoint."""
    generator = UNetGenerator(
        input_nc=cfg.input_nc,
        output_nc=cfg.output_nc,
        ngf=cfg.ngf,
        depth=cfg.unet_depth,
        dropout=0.0,  # No dropout at inference
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    print(f"Loaded generator from {checkpoint_path}")
    return generator


def preprocess_label_map(image_path, image_size=256):
    """Load and preprocess a semantic label map."""
    img = Image.open(image_path).convert("RGB")

    # Check if this is a paired image (side-by-side) — use left half
    w, h = img.size
    if w > h * 1.5:
        img = img.crop((0, 0, w // 2, h))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return transform(img).unsqueeze(0)  # Add batch dimension


def save_result(label_tensor, generated_tensor, output_path):
    """Save a side-by-side comparison of label map and generated image."""
    label_img = denormalize(label_tensor[0]).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    gen_img = denormalize(generated_tensor[0]).cpu().permute(1, 2, 0).numpy().clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(label_img)
    axes[0].set_title("Semantic Label Map")
    axes[0].axis("off")

    axes[1].imshow(gen_img)
    axes[1].set_title("Generated Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also save just the generated image
    gen_only_path = output_path.parent / f"{output_path.stem}_generated{output_path.suffix}"
    gen_pil = Image.fromarray((gen_img * 255).astype("uint8"))
    gen_pil.save(gen_only_path)


def run_inference(cfg):
    """Run inference on single image or directory."""
    device = torch.device(cfg.device)

    if cfg.checkpoint is None:
        # Default to latest checkpoint
        cfg.checkpoint = str(cfg.checkpoint_dir / "latest.pth")

    if not Path(cfg.checkpoint).exists():
        print(f"Error: Checkpoint not found at {cfg.checkpoint}")
        print("Train a model first with: python train.py")
        return

    generator = load_generator(cfg.checkpoint, cfg, device)

    # Collect input paths
    input_paths = []
    if cfg.input:
        input_paths.append(Path(cfg.input))
    elif cfg.input_dir:
        input_dir = Path(cfg.input_dir)
        input_paths = sorted([
            p for p in input_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ])
    else:
        print("Error: Provide --input or --input_dir")
        return

    if not input_paths:
        print("No input images found.")
        return

    print(f"\nGenerating {len(input_paths)} images...")
    output_dir = cfg.inference_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, img_path in enumerate(input_paths):
            label_tensor = preprocess_label_map(img_path, cfg.image_size).to(device)
            generated = generator(label_tensor)

            output_path = output_dir / f"result_{i:04d}.png"
            save_result(label_tensor, generated, output_path)
            print(f"  [{i+1}/{len(input_paths)}] {img_path.name} → {output_path.name}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    cfg = get_config()
    run_inference(cfg)
