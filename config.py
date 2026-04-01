"""
Configuration for Image Synthesis from Semantic Labeling.
All hyperparameters and paths are defined here.
"""

import argparse
from pathlib import Path


def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Image Synthesis from Semantic Labeling using Conditional GAN"
    )

    # ── Paths ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset_dir", type=str, default="./data/facades",
        help="Root directory of the paired dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory for checkpoints, samples, logs"
    )

    # ── Data ───────────────────────────────────────────────────────────
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--input_nc", type=int, default=3,
                        help="Number of channels in semantic label input")
    parser.add_argument("--output_nc", type=int, default=3,
                        help="Number of channels in generated output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--direction", type=str, default="AtoB",
                        choices=["AtoB", "BtoA"],
                        help="AtoB = label->photo, BtoA = photo->label")

    # ── Generator ──────────────────────────────────────────────────────
    parser.add_argument("--ngf", type=int, default=64,
                        help="Base number of generator filters")
    parser.add_argument("--unet_depth", type=int, default=8,
                        help="Number of downsampling blocks in U-Net")
    parser.add_argument("--gen_dropout", type=float, default=0.5,
                        help="Dropout rate in generator decoder (first 3 blocks)")

    # ── Discriminator ──────────────────────────────────────────────────
    parser.add_argument("--ndf", type=int, default=64,
                        help="Base number of discriminator filters")
    parser.add_argument("--n_layers_d", type=int, default=3,
                        help="Number of conv layers in PatchGAN discriminator")

    # ── Training ───────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_g", type=float, default=2e-4,
                        help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-4,
                        help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--lambda_l1", type=float, default=100.0,
                        help="Weight for L1 reconstruction loss")
    parser.add_argument("--lambda_fm", type=float, default=10.0,
                        help="Weight for feature matching loss")
    parser.add_argument("--use_feature_matching", action="store_true",
                        help="Enable feature matching loss")
    parser.add_argument("--n_critic", type=int, default=1,
                        help="Discriminator updates per generator update")

    # ── Logging & Checkpointing ────────────────────────────────────────
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="Save visual samples every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # ── Inference ──────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to generator checkpoint for inference")
    parser.add_argument("--input", type=str, default=None,
                        help="Single label map image for inference")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of label maps for batch inference")

    # ── Device ─────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to train/infer on")

    cfg = parser.parse_args(args if args is not None else [])

    # Resolve device
    if cfg.device == "auto":
        import torch
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directories
    cfg.checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
    cfg.sample_dir = Path(cfg.output_dir) / "samples"
    cfg.log_dir = Path(cfg.output_dir) / "logs"
    cfg.inference_dir = Path(cfg.output_dir) / "inference"

    for d in [cfg.checkpoint_dir, cfg.sample_dir, cfg.log_dir, cfg.inference_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return cfg
