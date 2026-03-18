# D405 Hand-Surface Depth Dataset

Multi-user, multi-angle RGB-depth dataset captured with Intel RealSense D405 for close-range hand-surface interaction research.

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total frames | 53,300 RGB-depth pairs |
| Participants | 15 users (P01-P18) |
| Camera angles | 30°, 45°, 60°, 90° |
| Resolution | 640 x 480 |
| Depth sensor | Intel RealSense D405 |
| Depth format | uint16 PNG (millimeters) |
| Depth accuracy | < 0.5mm at 35cm |
| Annotations | 21 hand landmarks + 5 fingertip depths per frame |
| Labels | Per-fingertip contact/hover state |

## Download

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-orange.svg)](https://huggingface.co/datasets/muxiddin19/d405-hand-surface-depth)

**HuggingFace**: [muxiddin19/d405-hand-surface-depth](https://huggingface.co/datasets/muxiddin19/d405-hand-surface-depth)

## Data Structure

```
d405-hand-surface-depth/
  P01/
    P01_angle90_white_desk_typing_074155/
      rgb/000000.png          # BGR uint8, 640x480
      depth/000000.png        # uint16 millimeters, 640x480
      annotations/000000.json # MediaPipe 21 landmarks + fingertip depths
      labels/000000.json      # Contact/hover state per fingertip
      metadata.json           # Camera intrinsics, calibration, session info
    P01_angle45_white_desk_typing_094354/
      ...
  P02/
    ...
```

## File Formats

| File | Format | Details |
|------|--------|---------|
| `rgb/*.png` | BGR uint8, 640x480 | Standard OpenCV format |
| `depth/*.png` | uint16, 640x480 | Values in millimeters (0 = invalid, range 70-500) |
| `annotations/*.json` | JSON | Per-frame: hand count, landmarks, fingertip depths |
| `metadata.json` | JSON | Camera intrinsics, filters, session info, calibration |

## Train/Val/Test Splits

Participant-stratified (no identity leakage):

| Split | Samples | Participants |
|-------|---------|-------------|
| Train | 42,640 | P01, P03, P04, P06, P07, P09–P13, P18 + supplementary |
| Val | 5,330 | P05, P16 + supplementary |
| Test | 5,330 | P08, P17 + supplementary |

Split files are at `../depth_finetuning/splits/custom_d405/`.

## Recording Your Own Data

See `../custom_data_recording/` for the recording tools and protocol.
