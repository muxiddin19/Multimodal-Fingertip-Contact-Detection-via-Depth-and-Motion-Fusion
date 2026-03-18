# Fine-tuning Depth Anything V2 for Close-Range Metric Depth

This guide describes how to fine-tune [Depth Anything V2 ViT-S](https://github.com/DepthAnything/Depth-Anything-V2) on our D405 hand-surface depth dataset for sub-4mm depth accuracy at 25-45cm range.

## Overview

The base Depth Anything V2 model produces excellent relative depth but lacks metric accuracy at close range (7-50cm). Our fine-tuning adapts the model to the Intel RealSense D405's depth characteristics, reducing MAE from 12.8mm to 3.2mm.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | `depth_anything_v2_vits.pth` (relative) | Better than metric-hypersim pretrained for close-range domain shift |
| Loss function | SiLog | Scale-invariant; handles varying absolute depth across angles |
| max_depth | 0.5m | Matches D405 sensor range (7-50cm). DPT head output = sigmoid x max_depth |
| LR schedule | Cosine annealing | Paper standard; smooth decay prevents late-training instability |
| Head LR | 10x encoder LR | Faster adaptation of randomly-initialized depth head |

## Prerequisites

```bash
# Clone Depth Anything V2 (or use our bundled version)
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

# Install dependencies
pip install torch torchvision timm tensorboard opencv-python numpy

# Download base weights
wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

## Dataset Preparation

### Option A: Use Our Pre-built Splits

Download the D405 dataset and splits from the dataset release. Split files are at `depth_finetuning/splits/custom_d405/`.

### Option B: Prepare From Raw Recordings

```bash
# From raw D405 recordings with rgb/ and depth/ directories:
python data/prepare_dataset.py \
    --data-dir /path/to/d405/recordings \
    --output-dir depth_finetuning/splits/custom_d405 \
    --filter \
    --min-depth-coverage 0.80 \
    --train-ratio 0.8 \
    --val-ratio 0.05 \
    --seed 42
```

### Split File Format

Each line in `train.txt` / `val.txt` / `test.txt`:

```
/absolute/path/to/rgb/000000.png /absolute/path/to/depth/000000.png
```

- RGB: BGR uint8 PNG, 640x480
- Depth: uint16 PNG in **millimeters** (value 350 = 0.350m)

### Split Strategy

All frames from a given participant go to **one split only** (stratified by participant ID). This prevents identity leakage from hand shape, skin tone, and typing style.

```
Train: 40,209 samples (11 participants)
Val:    2,557 samples (2 participants)
Test:   9,021 samples (2 participants)
```

## Training

### Configuration

| Parameter | Value | Paper Reference |
|-----------|-------|----------------|
| Encoder | ViT-S | Section 4.1 |
| Optimizer | AdamW (betas=0.9/0.999, wd=0.01) | Section 4.1 |
| Learning rate | 5e-6 (encoder), 5e-5 (head) | Supp. Section 6 |
| LR schedule | Cosine annealing | Section 4.1 |
| Effective batch size | 16 (bs=4 x grad_accum=4) | Section 4.1 |
| Epochs | 80 (~200K optimizer steps) | Section 4.1 |
| Image size | 518 x 518 | DA2 default |
| Depth range | 0.001m - 0.5m | Section 3.1 |
| Loss | SiLog | Section 4.1 |
| Augmentation | Random h-flip, color jitter | Section 4.2 |
| Normalization | ImageNet mean/std | DA2 standard |

### Launch Training

```bash
cd depth_finetuning
bash configs/train_d405.sh
```

Or manually:

```bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --master_port=20597 \
    train.py \
    --epochs 80 \
    --encoder vits \
    --bs 4 \
    --lr 0.000005 \
    --grad-accum 4 \
    --dataset custom \
    --split-dir splits/custom_d405 \
    --img-size 518 \
    --min-depth 0.001 \
    --max-depth 0.5 \
    --pretrained-from ../checkpoints/depth_anything_v2_vits.pth \
    --save-path exp/my_run
```

### Monitoring

```bash
tensorboard --logdir depth_finetuning/exp/
```

Key metrics to watch:
- `eval/d1`: delta1 accuracy (target: >0.99)
- `eval/abs_rel`: absolute relative error (target: <0.01)
- `eval/rmse`: root mean square error in meters (target: <0.005)
- `train/loss`: SiLog loss (should decrease smoothly)

### Checkpoints

The training script saves:
- `latest.pth` — every epoch (resume training)
- `best.pth` — best d1 accuracy on validation set
- `epoch_N.pth` — every 5 epochs (for comparison)

Each checkpoint contains `model`, `optimizer`, `epoch`, and `previous_best` metrics.

## Evaluation

### Depth Metrics

```bash
python evaluation/evaluate_depth.py \
    --checkpoint checkpoints/depth_anything_v2_vits_d405_finetuned.pth \
    --split-file depth_finetuning/splits/custom_d405/test.txt \
    --max-depth 0.5
```

### Expected Results

| Metric | Pre-trained | Fine-tuned |
|--------|-------------|------------|
| MAE (mm) | 12.8 | **3.2** |
| delta1 (%) | 38.2 | **96.2** |
| RMSE (mm) | 15.4 | **5.1** |
| abs_rel | 0.042 | **0.008** |
| SiLog | 0.312 | **0.057** |

## Inference

### Critical: max_depth Must Match

The DPT head computes `depth = sigmoid(features) x max_depth`. If you train with `max_depth=0.5` but load with `max_depth=20.0` (the default), predicted depths will be 40x too large.

```python
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(
    encoder='vits',
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=0.5  # MUST match training config
)

# Load fine-tuned weights (strip DDP 'module.' prefix if needed)
state_dict = torch.load('best.pth', map_location='cpu')['model']
cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(cleaned)
```

## Common Issues

**Q: Validation loss is much higher than training loss.**
A: Check that val.txt paths are correct and depth PNGs are uint16 millimeters (not raw sensor units).

**Q: Predicted depth values are 20-50x too large.**
A: `max_depth` mismatch between training and inference. Must be 0.5 at both stages.

**Q: Model predicts all zeros or constant depth.**
A: Learning rate too high. Try reducing from 5e-6 to 1e-6. Also verify ImageNet normalization is applied to input images.

**Q: Training crashes with CUDA OOM.**
A: Reduce `bs` from 4 to 2 and increase `grad_accum` from 4 to 8 (keeps effective batch size at 16).
