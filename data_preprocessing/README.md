# Data Preprocessing

Tools for preparing recorded D405 data into train/val/test splits for Depth Anything V2 fine-tuning.

## Files

| Script | Purpose |
|--------|---------|
| `prepare_dataset.py` | Create participant-stratified train/val/test splits with quality filtering |

## Usage

```bash
python prepare_dataset.py \
    --data-dir /path/to/d405/recordings \
    --output-dir ../depth_finetuning/splits/custom_d405 \
    --filter \
    --min-depth-coverage 0.80 \
    --train-ratio 0.8 \
    --val-ratio 0.05 \
    --seed 42
```

## Split Strategy

All frames from a given participant go to **one split only** (stratified by participant ID) to prevent identity leakage from hand shape, skin tone, and typing style.

| Split | Samples | Participants |
|-------|---------|-------------|
| Train | 42,640 | P01, P03, P04, P06, P07, P09–P13, P18 + supplementary |
| Val | 5,330 | P05, P16 + supplementary |
| Test | 5,330 | P08, P17 + supplementary |

## Split File Format

Each line in `train.txt` / `val.txt` / `test.txt`:

```
/absolute/path/to/rgb/000000.png /absolute/path/to/depth/000000.png
```

- RGB: BGR uint8 PNG, 640x480
- Depth: uint16 PNG in **millimeters** (value 350 = 0.350m)

## Quality Filtering

When `--filter` is enabled, frames are excluded if:
- Valid depth coverage < 80% in the hand-table ROI
- Motion blur detected (Laplacian variance < 100)
