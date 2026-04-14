"""
PatchGAN Discriminator for Conditional Image Synthesis.

Classifies overlapping 70×70 patches of the image as real or fake.
Conditioned on both the semantic label map and the image (real or generated).

Architecture:
    [label_map | image] → Conv blocks → Spatial real/fake map

Written from scratch — no pre-built discriminator modules used.
"""

import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """Conv → [BN] → LeakyReLU block for the discriminator."""

    def __init__(self, in_channels, out_channels, stride=2, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, padding=1,
                bias=not use_batchnorm
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for conditional GAN.

    Takes the concatenation of a semantic label map and an image (real or fake)
    and outputs a 2D map of real/fake predictions. Each spatial location in the
    output corresponds to a 70×70 receptive field in the input (for n_layers=3).

    Args:
        input_nc:  Channels in the label map.
        output_nc: Channels in the image.
        ndf:       Base number of discriminator filters.
        n_layers:  Number of intermediate conv layers (controls receptive field).

    Returns:
        features: List of intermediate feature maps (for feature matching loss).
        output:   Final prediction map [B, 1, H', W'].
    """

    def __init__(self, input_nc=3, output_nc=3, ndf=64, n_layers=3):
        super().__init__()

        self.n_layers = n_layers

        # Input: concatenated label map + image
        total_input_nc = input_nc + output_nc

        # ── Build the discriminator as a sequence of blocks ────────────
        self.blocks = nn.ModuleList()

        # First layer: no batchnorm
        self.blocks.append(
            DiscriminatorBlock(total_input_nc, ndf, stride=2, use_batchnorm=False)
        )

        # Intermediate layers: progressively increase filters
        in_ch = ndf
        for i in range(1, n_layers):
            out_ch = ndf * min(2 ** i, 8)
            self.blocks.append(
                DiscriminatorBlock(in_ch, out_ch, stride=2, use_batchnorm=True)
            )
            in_ch = out_ch

        # Penultimate layer: stride=1 to keep spatial size
        out_ch = ndf * min(2 ** n_layers, 8)
        self.blocks.append(
            DiscriminatorBlock(in_ch, out_ch, stride=1, use_batchnorm=True)
        )
        in_ch = out_ch

        # Final layer: 1-channel prediction map (no batchnorm, no activation)
        self.final = nn.Conv2d(
            in_ch, 1, kernel_size=4, stride=1, padding=1
        )

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Initialize weights with N(0, 0.02)."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, label_map, image):
        """
        Args:
            label_map: Semantic label tensor [B, input_nc, H, W].
            image:     Image tensor (real or fake) [B, output_nc, H, W].

        Returns:
            features: List of intermediate feature maps (for feature matching).
            prediction: Real/fake prediction map [B, 1, H', W'].
        """
        # Concatenate label map and image along channel dimension
        x = torch.cat([label_map, image], dim=1)

        # Collect intermediate features for feature matching loss
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        # Final prediction
        prediction = self.final(x)

        return features, prediction


def test_discriminator():
    """Quick sanity check."""
    disc = PatchGANDiscriminator(input_nc=3, output_nc=3, ndf=64, n_layers=3)

    label_map = torch.randn(2, 3, 256, 256)
    image = torch.randn(2, 3, 256, 256)

    features, pred = disc(label_map, image)
    print(f"Discriminator input: label {label_map.shape}, image {image.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Feature maps: {[f.shape for f in features]}")
    total_params = sum(p.numel() for p in disc.parameters())
    print(f"Total parameters: {total_params:,}")
    print("✓ Discriminator test passed")


if __name__ == "__main__":
    test_discriminator()
