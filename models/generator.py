"""
U-Net Generator for Image Synthesis from Semantic Labels.

Architecture:
    Encoder (downsampling) → Bottleneck → Decoder (upsampling) with skip connections.
    Each encoder block: Conv2d → BatchNorm → LeakyReLU
    Each decoder block: ConvTranspose2d → BatchNorm → (Dropout) → ReLU + skip concat

Written from scratch — no pre-built generator modules used.
"""

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """Single downsampling block: Conv → [BN] → LeakyReLU."""

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=not use_batchnorm
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Single upsampling block: ConvTranspose → BN → [Dropout] → ReLU."""

    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        """
        Args:
            x: Feature map from previous decoder layer.
            skip: Corresponding encoder feature map (skip connection).
        """
        x = self.block(x)
        # Concatenate along channel dimension
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net Generator for conditional image synthesis.

    Converts a semantic label map into a photorealistic image.
    The architecture follows a symmetric encoder-decoder with skip connections
    at every spatial resolution level.

    Args:
        input_nc:  Number of input channels (label map).
        output_nc: Number of output channels (synthesized image).
        ngf:       Base number of filters (doubled at each encoder level).
        depth:     Number of downsampling steps (default 8 for 256×256 input).
        dropout:   Dropout probability in the first 3 decoder blocks.
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, depth=8, dropout=0.5):
        super().__init__()
        assert depth >= 4, "U-Net depth must be >= 4"

        self.depth = depth

        # ── Build Encoder ──────────────────────────────────────────────
        # Layer channel progression: input_nc → ngf → ngf*2 → ngf*4 → ngf*8 → ...
        self.encoders = nn.ModuleList()
        encoder_channels = []

        in_ch = input_nc
        for i in range(depth):
            if i == 0:
                out_ch = ngf
                use_bn = False  # First encoder: no batchnorm
            elif i <= 3:
                out_ch = ngf * min(2 ** i, 8)
                use_bn = True
            else:
                out_ch = ngf * 8  # Cap at ngf*8
                use_bn = True

            self.encoders.append(EncoderBlock(in_ch, out_ch, use_batchnorm=use_bn))
            encoder_channels.append(out_ch)
            in_ch = out_ch

        # ── Build Decoder ──────────────────────────────────────────────
        # Mirror of encoder, with skip connections doubling the channel count
        self.decoders = nn.ModuleList()

        in_ch = encoder_channels[-1]  # Bottleneck output
        for i in range(depth - 1):
            # Determine output channels (mirror encoder)
            enc_idx = depth - 2 - i  # Index of the encoder to skip-connect with
            skip_ch = encoder_channels[enc_idx]

            # First 3 decoder blocks use dropout
            use_drop = (i < 3) and (dropout > 0)

            if i == 0:
                out_ch = skip_ch  # No concatenation input for first decoder
                self.decoders.append(DecoderBlock(in_ch, out_ch, use_dropout=use_drop))
            else:
                # Input channels = previous decoder out + skip channels (from concat)
                out_ch = skip_ch
                self.decoders.append(
                    DecoderBlock(in_ch, out_ch, use_dropout=use_drop)
                )
            # After concatenation: out_ch + skip_ch
            in_ch = out_ch + skip_ch

        # ── Final output layer ─────────────────────────────────────────
        # Upsample from the last concatenated features to the output image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, output_nc,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Initialize weights with N(0, 0.02) as in the original pix2pix."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        Args:
            x: Semantic label map tensor [B, input_nc, H, W].

        Returns:
            Synthesized image tensor [B, output_nc, H, W] in range [-1, 1].
        """
        # ── Encoder pass (save activations for skip connections) ───────
        encoder_features = []
        h = x
        for encoder in self.encoders:
            h = encoder(h)
            encoder_features.append(h)

        # ── Decoder pass (with skip connections) ───────────────────────
        # Start from bottleneck (last encoder output)
        h = encoder_features[-1]
        for i, decoder in enumerate(self.decoders):
            skip = encoder_features[self.depth - 2 - i]
            h = decoder(h, skip)

        # ── Final output ───────────────────────────────────────────────
        return self.final(h)


def test_generator():
    """Quick sanity check."""
    gen = UNetGenerator(input_nc=3, output_nc=3, ngf=64, depth=8)
    x = torch.randn(2, 3, 256, 256)
    out = gen(x)
    print(f"Generator: {x.shape} → {out.shape}")
    total_params = sum(p.numel() for p in gen.parameters())
    print(f"Total parameters: {total_params:,}")
    assert out.shape == (2, 3, 256, 256), f"Unexpected output shape: {out.shape}"
    print("✓ Generator test passed")


if __name__ == "__main__":
    test_generator()
