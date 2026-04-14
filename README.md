# Image Synthesis from Semantic Labeling

A from-scratch implementation of a **Conditional GAN** for generating photorealistic images
from semantic label maps. 

## Architecture

```
Semantic Label Map ──► U-Net Generator ──► Synthesized Image
                            ▲
                            │ Adversarial + L1 Loss
                            ▼
                      PatchGAN Discriminator ◄── Real Image
```

### Generator (U-Net)
- Encoder-decoder with skip connections
- 8-block depth: progressively downsamples then upsamples
- Skip connections preserve spatial detail from input labels

### Discriminator (PatchGAN)
- Classifies overlapping 70×70 patches as real/fake
- Conditions on both the input label map and the output/target image
- Outputs a spatial map of real/fake probabilities

### Losses
- **Adversarial loss**: Binary cross-entropy (vanilla GAN)
- **L1 reconstruction loss**: Pixel-level similarity to ground truth
- **Feature matching loss** (optional): Matches intermediate discriminator features

## Project Structure

```
image-synthesis-semantic/
├── README.md
├── requirements.txt
├── config.py              # All hyperparameters & paths
├── dataset.py             # Dataset loader (Cityscapes / Facades / custom)
├── models/
│   ├── __init__.py
│   ├── generator.py       # U-Net Generator (from scratch)
│   ├── discriminator.py   # PatchGAN Discriminator (from scratch)
│   └── losses.py          # GAN losses (from scratch)
├── train.py               # Training loop
├── inference.py           # Generate images from label maps
└── utils.py               # Visualization, checkpointing, metrics
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

The project supports **paired datasets** where each sample is a side-by-side image
containing the semantic label map and the corresponding real photo.

### Option A: Facades Dataset (small, good for testing)
```bash
python dataset.py --download facades
```

### Option B: Cityscapes
1. Download from https://www.cityscapes-dataset.com/
2. Set paths in `config.py`

### Option C: Custom Dataset
Place paired images (label|photo side-by-side) in a folder and point `config.py` to it.

## Training

```bash
# Train with default config (Facades dataset)
python train.py

# Train with custom settings
python train.py --dataset_dir ./data/facades \
                --epochs 200 \
                --batch_size 4 \
                --image_size 256 \
                --lr 0.0002 \
                --lambda_l1 100.0
```

Training saves checkpoints and sample visualizations to `./outputs/`.

## Inference

```bash
# Generate from a single label map
python inference.py --input label_map.png --checkpoint outputs/checkpoints/latest.pth

# Generate from a directory of label maps
python inference.py --input_dir ./test_labels/ --checkpoint outputs/checkpoints/latest.pth
```

## Hyperparameters (config.py)

| Parameter       | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `image_size`    | 256     | Input/output resolution              |
| `batch_size`    | 4       | Training batch size                  |
| `epochs`        | 200     | Number of training epochs            |
| `lr_g`          | 0.0002  | Generator learning rate              |
| `lr_d`          | 0.0002  | Discriminator learning rate          |
| `beta1`         | 0.5     | Adam beta1                           |
| `lambda_l1`     | 100.0   | Weight for L1 reconstruction loss    |
| `lambda_fm`     | 10.0    | Weight for feature matching loss     |
| `n_critic`      | 1       | Discriminator updates per G update   |

## Results

After training, `outputs/` will contain:
- `checkpoints/` — Model weights at intervals
- `samples/` — Side-by-side visualizations (label → generated → real)
- `logs/` — Loss curves (CSV)
