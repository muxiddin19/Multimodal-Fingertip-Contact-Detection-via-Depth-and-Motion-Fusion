# Depth Anything V2 Fine-tuning for Close-Range Metric Depth

Fine-tuning code for [Depth Anything V2 ViT-S](https://github.com/DepthAnything/Depth-Anything-V2) on our D405 hand-surface depth dataset, reducing MAE from 12.8mm to 3.2mm at 25-45cm range.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script with DDP, SiLog loss, cosine LR, gradient accumulation |
| `dataset/custom_depth_dataset.py` | RGB-depth pair loader with ImageNet normalization and augmentation |
| `configs/train_d405.sh` | Training launcher with paper-matched hyperparameters |
| `splits/custom_d405/` | Train/val/test split files (participant-stratified, 53,300 frames) |
| `FINETUNING.md` | Detailed training configuration, monitoring, evaluation, and troubleshooting |

## Prerequisites

```bash
# Clone Depth Anything V2 (needed for model architecture)
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

# Download base weights
wget -P ../checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

## Training Configuration (Paper Section 4.1)

| Parameter | Value |
|-----------|-------|
| Base model | `depth_anything_v2_vits.pth` (relative) |
| Optimizer | AdamW (lr=5e-6, betas=0.9/0.999, wd=0.01) |
| Head LR | 10x encoder LR (5e-5) |
| LR schedule | Cosine annealing |
| Effective batch size | 16 (bs=4 x grad_accum=4) |
| Epochs | 80 (~200K optimizer steps) |
| Loss | SiLog (Scale-Invariant Logarithmic) |
| Depth range | 0.001m - 0.5m |
| Image size | 518 x 518 |

## Training

```bash
bash configs/train_d405.sh
```

Training takes ~30 hours on a single RTX 4090.

## Critical: max_depth Must Match at Inference

The DPT head computes `depth = sigmoid(features) x max_depth`. If trained with `max_depth=0.5`, inference MUST also use `max_depth=0.5`.

## Expected Results

| Metric | Pre-trained | Fine-tuned |
|--------|-------------|------------|
| MAE (mm) | 12.8 | **3.2** |
| delta1 (%) | 38.2 | **96.2** |
| RMSE (mm) | 15.4 | **5.1** |

See `FINETUNING.md` for full documentation.
