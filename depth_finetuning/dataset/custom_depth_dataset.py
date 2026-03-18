"""
Custom Depth Dataset Loader for PNG RGB + Depth pairs
Designed for VR Keyboard typing data (CVPR 2026)

Fixes applied:
  1. Added ImageNet normalization (matching DA2 Hypersim/KITTI loaders)
  2. Fixed valid_mask: create BEFORE clipping, use >= / <= for boundaries
  3. Removed unsqueeze(0) on depth/valid_mask to match other DA2 loaders
  4. Added file-existence filtering to skip missing samples
  5. Added random horizontal flip augmentation for training
  6. Added color jitter augmentation for training
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import random


# ImageNet normalization (same as DA2 Hypersim/KITTI loaders)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CustomDepthDataset(Dataset):
    """
    Dataset for custom RGB + Depth PNG pairs.

    Expected format in filelist (train.txt/val.txt):
        /path/to/rgb/000000.png /path/to/depth/000000.png [focal_length]

    Depth PNGs are uint16 in millimeters (RealSense D405 output).
    Output tensors match DA2 convention: depth (H, W), valid_mask (H, W).
    """

    def __init__(self, filelist_path, mode='train',
                 img_size=518, min_depth=0.001, max_depth=0.4):
        self.mode = mode
        self.img_size = img_size
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Load file list, filtering out missing files
        self.samples = []
        skipped = 0
        with open(filelist_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    rgb_path = parts[0]
                    depth_path = parts[1]
                    focal = float(parts[2]) if len(parts) > 2 else 518.8579

                    # Skip missing files instead of crashing during training
                    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                        skipped += 1
                        continue

                    self.samples.append({
                        'rgb': rgb_path,
                        'depth': depth_path,
                        'focal': focal
                    })

        print(f"[CustomDepthDataset] Loaded {len(self.samples)} {mode} samples"
              f" (skipped {skipped} missing)")
        print(f"[CustomDepthDataset] Depth range: [{min_depth}, {max_depth}]m")
        print(f"[CustomDepthDataset] Image size: {img_size}x{img_size}")

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {filelist_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # ---- Load RGB ----
        rgb_path = sample_info['rgb']
        image = cv2.imread(rgb_path)
        if image is None:
            raise ValueError(f"Cannot load RGB image: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---- Load depth (uint16 PNG, values in millimeters) ----
        depth_path = sample_info['depth']
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Cannot load depth image: {depth_path}")

        # Convert to meters
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # mm -> m
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0 * self.max_depth
        else:
            depth = depth.astype(np.float32)

        # Handle multi-channel depth (some PNGs may load as 3-channel)
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]

        # ---- Create valid mask BEFORE clipping ----
        valid_mask = (depth >= self.min_depth) & (depth <= self.max_depth)

        # ---- Resize ----
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)
        valid_mask = cv2.resize(valid_mask.astype(np.uint8),
                                (self.img_size, self.img_size),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

        # ---- Normalize RGB: /255 then ImageNet mean/std ----
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # ---- Training augmentations ----
        if self.mode == 'train':
            # Random horizontal flip
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                depth = np.flip(depth, axis=1).copy()
                valid_mask = np.flip(valid_mask, axis=1).copy()

            # Color jitter (brightness / contrast only, no spatial distortion)
            if random.random() < 0.5:
                brightness = random.uniform(0.9, 1.1)
                contrast = random.uniform(0.9, 1.1)
                # Apply in normalized space
                image = image * contrast + (brightness - 1.0)

        # ---- Convert to tensors (DA2 convention: no channel dim on depth) ----
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        depth = torch.from_numpy(np.ascontiguousarray(depth)).float()              # (H, W)
        valid_mask = torch.from_numpy(np.ascontiguousarray(valid_mask)).float()     # (H, W)

        return {
            'image': image,          # (3, H, W), ImageNet-normalized
            'depth': depth,           # (H, W), meters
            'valid_mask': valid_mask,  # (H, W), 0/1
            'filename': os.path.basename(rgb_path)
        }


def test_dataset():
    """Test function to verify dataset loading and normalization."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python custom_depth_dataset.py <train.txt> [max_depth]")
        sys.exit(1)

    filelist = sys.argv[1]
    max_depth = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4

    print(f"Testing dataset with: {filelist}, max_depth={max_depth}")

    dataset = CustomDepthDataset(
        filelist_path=filelist,
        mode='train',
        img_size=518,
        min_depth=0.001,
        max_depth=max_depth
    )

    print(f"\nDataset length: {len(dataset)}")

    # Test first sample
    print("\nLoading first sample...")
    sample = dataset[0]

    print(f"Image shape: {sample['image'].shape}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  (Expected ~[-2, 3] for ImageNet-normalized)")
    print(f"Depth shape: {sample['depth'].shape}")
    print(f"Depth range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]m")
    print(f"Valid pixels: {sample['valid_mask'].sum().item():.0f} / {sample['valid_mask'].numel()}")
    print(f"Valid ratio: {sample['valid_mask'].mean().item()*100:.1f}%")
    print(f"Filename: {sample['filename']}")

    # Verify tensor shapes match DA2 convention
    assert sample['depth'].dim() == 2, f"Depth should be 2D, got {sample['depth'].dim()}D"
    assert sample['valid_mask'].dim() == 2, f"Mask should be 2D, got {sample['valid_mask'].dim()}D"
    print("\nShape check passed (2D depth/mask, matching DA2 convention)")

    # Test a few more samples
    print("\nTesting 10 random samples...")
    for i in random.sample(range(len(dataset)), min(10, len(dataset))):
        s = dataset[i]
        valid_pct = s['valid_mask'].mean().item() * 100
        print(f"  Sample {i}: depth [{s['depth'].min():.3f}, {s['depth'].max():.3f}]m, "
              f"valid: {valid_pct:.1f}%")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_dataset()
