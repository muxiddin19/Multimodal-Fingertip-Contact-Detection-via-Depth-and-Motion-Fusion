#!/bin/bash

# ============================================================================
# Training Script for Depth Anything V2 - ViT-S
# NEW D405 Multi-User Multi-Angle Dataset (15 participants, 4 angles)
# ============================================================================
#
# Dataset: /nas/Dataset/custom_depth_data/d405/
#   Train: 40,209 samples (11 participants: P01,P03,P04,P06,P07,P09-P12,P17,P18)
#   Val:   2,557  samples (2 participants: P13, P16)
#   Test:  9,021  samples (2 participants: P05, P08)
#   Angles: 30°, 45°, 60°, 90° (all in train+test; val=90° only)
#
# Paper config (Section 4.1, Supp Section 6):
#   "AdamW optimizer, lr=1e-5, cosine annealing, 200K steps, batch size 16"
#
# Calculation:
#   40,209 samples / physical_bs=4 = 10,052 iters/epoch
#   With grad_accum=4: 10,052/4 = 2,513 optimizer steps/epoch
#   Target 200K steps: 200,000 / 2,513 = 80 epochs
#   (previous run: 19 epochs x old_dataset ≈ 50K steps — undertrained)
#
# ============================================================================

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/..

# ============================================================================
# CUDA SETUP
# ============================================================================
export CUDA_VISIBLE_DEVICES=1

echo "============================================================================"
echo "GPU Configuration"
echo "============================================================================"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

if ! nvidia-smi -i $CUDA_VISIBLE_DEVICES &> /dev/null; then
    echo "ERROR: GPU $CUDA_VISIBLE_DEVICES is not accessible!"
    echo "Available GPUs:"
    nvidia-smi --list-gpus
    exit 1
fi

echo "GPU Info:"
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "============================================================================"

# ============================================================================
# Memory Optimization Settings
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export OMP_NUM_THREADS=1

# ============================================================================
# Training Configuration
# ============================================================================
now=$(date +"%Y%m%d_%H%M%S")

# --- Core hyperparameters (Paper Section 4.1) ---
epoch=80              # 80 epochs × 2,513 steps/epoch ≈ 201K optimizer steps
bs=4                  # Physical batch size (fits 24GB GPU)
grad_accum=4          # Effective batch = 4 × 4 = 16 (Paper: batch size 16)
gpus=1
lr=0.000005           # 5e-6: conservative LR for fine-tuning on real D405 data
                      # (1e-5 may be too aggressive for this close-range domain)
encoder=vits
dataset=custom
img_size=518
min_depth=0.001
max_depth=0.5         # D405 full range: 0-50cm. Uses 100% of valid depth pixels.
                      # (was 0.4 — masked 17% of valid pixels unnecessarily)

# --- Split directory (new multi-user multi-angle dataset) ---
split_dir=dataset/splits/custom_d405

# --- Pretrained weights ---
pretrained_from=../checkpoints/depth_anything_v2_vits.pth

# --- Output ---
save_path=exp/d405_multiuser_${now}_${max_depth}_${encoder}

# Create save directory
mkdir -p $save_path

echo ""
echo "============================================================================"
echo "Training Configuration"
echo "============================================================================"
echo "Encoder:        $encoder"
echo "Dataset:        $dataset (multi-user, multi-angle D405)"
echo "Split dir:      $split_dir"
echo "Epochs:         $epoch"
echo "Batch size:     $bs (effective: $((bs * grad_accum)) with grad_accum=$grad_accum)"
echo "Learning rate:  $lr"
echo "Image size:     ${img_size}x${img_size}"
echo "Depth range:    ${min_depth}m - ${max_depth}m"
echo "Pretrained:     $pretrained_from"
echo "Save path:      $save_path"
echo "============================================================================"
echo ""

# ============================================================================
# Verify dataset split files exist
# ============================================================================
if [ ! -f "${split_dir}/train.txt" ]; then
    echo "ERROR: ${split_dir}/train.txt not found"
    echo "Available dataset splits:"
    find dataset/splits -name "*.txt" 2>/dev/null || echo "  No splits directory found"
    exit 1
fi

if [ ! -f "${split_dir}/val.txt" ]; then
    echo "ERROR: ${split_dir}/val.txt not found"
    exit 1
fi

echo "Dataset splits:"
echo "  Train: $(wc -l < ${split_dir}/train.txt) samples"
echo "  Val:   $(wc -l < ${split_dir}/val.txt) samples"
if [ -f "${split_dir}/test.txt" ]; then
    echo "  Test:  $(wc -l < ${split_dir}/test.txt) samples (for post-training eval)"
fi

# ============================================================================
# Verify Pretrained Weights
# ============================================================================
if [ ! -f "$pretrained_from" ]; then
    echo "WARNING: Pretrained weights not found at: $pretrained_from"
    echo ""
    echo "Available pretrained weights:"
    ls -lh ../checkpoints/*.pth 2>/dev/null || echo "  No .pth files found in ../checkpoints/"
    echo ""
    read -p "Continue without pretrained weights? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================================================
# Launch Training
# ============================================================================
echo "Starting training at $(date)"
echo "Estimated: ~80 epochs × ~10,052 iters/epoch = ~804K iterations (~201K optimizer steps)"
echo "============================================================================"
echo ""

torchrun \
    --nproc_per_node=$gpus \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20597 \
    train1.py \
        --epochs $epoch \
        --encoder $encoder \
        --bs $bs \
        --lr $lr \
        --grad-accum $grad_accum \
        --save-path $save_path \
        --dataset $dataset \
        --split-dir $split_dir \
        --img-size $img_size \
        --min-depth $min_depth \
        --max-depth $max_depth \
        --pretrained-from $pretrained_from \
        --port 20597 \
    2>&1 | tee -a $save_path/$now.log

# ============================================================================
# Training Complete
# ============================================================================
echo ""
echo "============================================================================"
echo "Training completed at $(date)"
echo "Logs saved to: $save_path/$now.log"
echo "Checkpoints saved to: $save_path/"
echo ""
echo "Next steps:"
echo "  1. Check best.pth vs latest.pth metrics in the log"
echo "  2. Run test evaluation:"
echo "     python evaluate_touch_detection.py --checkpoint $save_path/best.pth --split-dir $split_dir"
echo "  3. Copy best.pth to Windows PC for live testing"
echo "============================================================================"
