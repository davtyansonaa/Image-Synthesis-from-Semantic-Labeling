"""
Training script for Image Synthesis from Semantic Labeling.

Trains a U-Net Generator and PatchGAN Discriminator using:
    - Adversarial loss (vanilla GAN or LSGAN)
    - L1 reconstruction loss
    - Optional feature matching loss

Usage:
    python train.py
    python train.py --dataset_dir ./data/facades --epochs 200 --batch_size 4
"""

import sys
import time
from pathlib import Path

import torch

from config import get_config
from dataset import PairedImageDataset, create_dataloaders
from models import (
    UNetGenerator,
    PatchGANDiscriminator,
    GANLoss,
    L1ReconstructionLoss,
    FeatureMatchingLoss,
)
from utils import (
    save_sample_grid,
    save_checkpoint,
    load_checkpoint,
    LossLogger,
)


def train(cfg):
    """Main training function."""
    device = torch.device(cfg.device)
    print(f"\n{'='*60}")
    print(f"  Image Synthesis from Semantic Labeling — Training")
    print(f"{'='*60}")
    print(f"  Device:     {device}")
    print(f"  Dataset:    {cfg.dataset_dir}")
    print(f"  Image size: {cfg.image_size}×{cfg.image_size}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Epochs:     {cfg.epochs}")
    print(f"  Lambda L1:  {cfg.lambda_l1}")
    print(f"{'='*60}\n")

    # ── Data ───────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(cfg)

    # ── Models ─────────────────────────────────────────────────────────
    generator = UNetGenerator(
        input_nc=cfg.input_nc,
        output_nc=cfg.output_nc,
        ngf=cfg.ngf,
        depth=cfg.unet_depth,
        dropout=cfg.gen_dropout,
    ).to(device)

    discriminator = PatchGANDiscriminator(
        input_nc=cfg.input_nc,
        output_nc=cfg.output_nc,
        ndf=cfg.ndf,
        n_layers=cfg.n_layers_d,
    ).to(device)

    print(f"Generator params:     {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # ── Losses ─────────────────────────────────────────────────────────
    criterion_gan = GANLoss(mode="vanilla").to(device)
    criterion_l1 = L1ReconstructionLoss(weight=cfg.lambda_l1)
    criterion_fm = FeatureMatchingLoss(weight=cfg.lambda_fm) if cfg.use_feature_matching else None

    # ── Optimizers ─────────────────────────────────────────────────────
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2)
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2)
    )

    # ── Resume from checkpoint ─────────────────────────────────────────
    start_epoch = 0
    if cfg.resume:
        start_epoch = load_checkpoint(
            cfg.resume, generator, discriminator, opt_g, opt_d
        )
        print(f"Resumed from epoch {start_epoch}")

    # ── Logger ─────────────────────────────────────────────────────────
    logger = LossLogger(cfg.log_dir)

    # ── Keep a fixed batch for consistent visual progress tracking ─────
    fixed_batch = next(iter(train_loader))
    fixed_labels = fixed_batch[0].to(device)
    fixed_reals = fixed_batch[1].to(device)

    # ── Training Loop ──────────────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, cfg.epochs):
        generator.train()
        discriminator.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (label_maps, real_images) in enumerate(train_loader):
            label_maps = label_maps.to(device)
            real_images = real_images.to(device)

            # ── (1) Update Discriminator ───────────────────────────────
            for _ in range(cfg.n_critic):
                opt_d.zero_grad()

                # Generate fake images
                with torch.no_grad():
                    fake_images = generator(label_maps)

                # Discriminator on real pairs
                feats_real, pred_real = discriminator(label_maps, real_images)
                loss_d_real = criterion_gan(pred_real, is_real=True)

                # Discriminator on fake pairs
                feats_fake, pred_fake = discriminator(label_maps, fake_images)
                loss_d_fake = criterion_gan(pred_fake, is_real=False)

                # Total discriminator loss
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                opt_d.step()

            # ── (2) Update Generator ───────────────────────────────────
            opt_g.zero_grad()

            fake_images = generator(label_maps)

            # Adversarial loss: fool the discriminator
            feats_fake, pred_fake = discriminator(label_maps, fake_images)
            loss_g_adv = criterion_gan(pred_fake, is_real=True)

            # L1 reconstruction loss
            loss_g_l1 = criterion_l1(fake_images, real_images)

            # Feature matching loss (optional)
            loss_g_fm = torch.tensor(0.0, device=device)
            if criterion_fm is not None:
                feats_real, _ = discriminator(label_maps, real_images)
                loss_g_fm = criterion_fm(feats_real, feats_fake)

            # Total generator loss
            loss_g = loss_g_adv + loss_g_l1 + loss_g_fm
            loss_g.backward()
            opt_g.step()

            # ── Logging ────────────────────────────────────────────────
            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()
            global_step += 1

            logger.log(
                epoch=epoch, step=global_step,
                d_loss=loss_d.item(),
                g_loss_adv=loss_g_adv.item(),
                g_loss_l1=loss_g_l1.item(),
                g_loss_fm=loss_g_fm.item(),
                g_loss_total=loss_g.item(),
            )

        # ── Epoch Summary ──────────────────────────────────────────────
        n_batches = len(train_loader)
        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
            f"D: {epoch_d_loss/n_batches:.4f}  "
            f"G: {epoch_g_loss/n_batches:.4f}  "
            f"({elapsed:.1f}s)"
        )

        logger.flush()

        # ── Save visual samples ────────────────────────────────────────
        if (epoch + 1) % cfg.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                fixed_fakes = generator(fixed_labels)
            save_sample_grid(
                fixed_labels, fixed_fakes, fixed_reals,
                path=cfg.sample_dir / f"epoch_{epoch+1:04d}.png",
            )
            generator.train()

        # ── Save checkpoint ────────────────────────────────────────────
        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint(
                generator, discriminator, opt_g, opt_d, epoch + 1,
                path=cfg.checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pth",
            )

    # ── Final save ─────────────────────────────────────────────────────
    save_checkpoint(
        generator, discriminator, opt_g, opt_d, cfg.epochs,
        path=cfg.checkpoint_dir / "latest.pth",
    )
    logger.plot_losses()
    print(f"\nTraining complete. Outputs saved to {cfg.output_dir}")


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
