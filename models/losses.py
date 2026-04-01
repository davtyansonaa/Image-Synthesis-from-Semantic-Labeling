"""
Loss functions for the Conditional GAN.

All losses are implemented from scratch using basic PyTorch operations.

Losses:
    1. GANLoss:              Adversarial loss (vanilla BCE or LSGAN)
    2. L1ReconstructionLoss: Pixel-wise L1 distance to ground truth
    3. FeatureMatchingLoss:  L1 distance between real/fake discriminator features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """
    Adversarial loss for GAN training.

    Supports two modes:
        - 'vanilla': Binary Cross-Entropy with logits (original GAN)
        - 'lsgan':   Least-Squares GAN (MSE loss, more stable)

    The loss automatically creates target tensors of the correct shape,
    filled with the appropriate real/fake label value.

    Args:
        mode:       'vanilla' or 'lsgan'.
        real_label: Target value for real samples (default 1.0).
        fake_label: Target value for fake samples (default 0.0).
    """

    def __init__(self, mode="vanilla", real_label=1.0, fake_label=0.0):
        super().__init__()
        assert mode in ("vanilla", "lsgan"), f"Unknown GAN loss mode: {mode}"
        self.mode = mode
        # Register as buffers so they move to the correct device automatically
        self.register_buffer("real_val", torch.tensor(real_label))
        self.register_buffer("fake_val", torch.tensor(fake_label))

    def _make_target(self, prediction, is_real):
        """Create a target tensor matching the prediction shape."""
        value = self.real_val if is_real else self.fake_val
        return value.expand_as(prediction)

    def forward(self, prediction, is_real):
        """
        Compute adversarial loss.

        Args:
            prediction: Discriminator output [B, 1, H', W'].
            is_real:    True if computing loss against real targets.

        Returns:
            Scalar loss value.
        """
        target = self._make_target(prediction, is_real)

        if self.mode == "vanilla":
            loss = F.binary_cross_entropy_with_logits(prediction, target)
        else:  # lsgan
            loss = F.mse_loss(prediction, target)

        return loss


class L1ReconstructionLoss(nn.Module):
    """
    Pixel-wise L1 reconstruction loss.

    Encourages the generator to produce images that are close to the
    ground truth at the pixel level. This stabilizes training and ensures
    the output is structurally similar to the target.

    Args:
        weight: Scaling factor (lambda_L1), typically 100.0.
    """

    def __init__(self, weight=100.0):
        super().__init__()
        self.weight = weight

    def forward(self, generated, target):
        """
        Args:
            generated: Generator output [B, C, H, W].
            target:    Ground truth image [B, C, H, W].

        Returns:
            Weighted L1 loss (scalar).
        """
        loss = torch.mean(torch.abs(generated - target))
        return self.weight * loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss.

    Computes the L1 distance between intermediate discriminator features
    extracted from real and generated images. This encourages the generator
    to produce images whose internal discriminator statistics match those
    of real images, leading to more stable training.

    Args:
        weight: Scaling factor (lambda_FM), typically 10.0.
    """

    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight

    def forward(self, features_real, features_fake):
        """
        Args:
            features_real: List of feature maps from D(label, real_image).
            features_fake: List of feature maps from D(label, fake_image).

        Returns:
            Weighted feature matching loss (scalar).
        """
        assert len(features_real) == len(features_fake), \
            "Feature lists must have the same length"

        loss = 0.0
        for feat_real, feat_fake in zip(features_real, features_fake):
            # Detach real features — we don't want gradients flowing back
            # through the discriminator when optimizing the generator
            loss += torch.mean(torch.abs(feat_real.detach() - feat_fake))

        # Average over the number of feature layers
        loss = loss / len(features_real)
        return self.weight * loss


def test_losses():
    """Quick sanity check for all losses."""
    # GAN Loss
    gan_loss = GANLoss(mode="vanilla")
    pred = torch.randn(2, 1, 30, 30)
    loss_real = gan_loss(pred, is_real=True)
    loss_fake = gan_loss(pred, is_real=False)
    print(f"GAN loss (real): {loss_real.item():.4f}")
    print(f"GAN loss (fake): {loss_fake.item():.4f}")

    # L1 Loss
    l1_loss = L1ReconstructionLoss(weight=100.0)
    gen = torch.randn(2, 3, 256, 256)
    tgt = torch.randn(2, 3, 256, 256)
    loss_l1 = l1_loss(gen, tgt)
    print(f"L1 loss: {loss_l1.item():.4f}")

    # Feature Matching Loss
    fm_loss = FeatureMatchingLoss(weight=10.0)
    feats_r = [torch.randn(2, 64, 128, 128), torch.randn(2, 128, 64, 64)]
    feats_f = [torch.randn(2, 64, 128, 128), torch.randn(2, 128, 64, 64)]
    loss_fm = fm_loss(feats_r, feats_f)
    print(f"Feature matching loss: {loss_fm.item():.4f}")

    print("✓ All loss tests passed")


if __name__ == "__main__":
    test_losses()
