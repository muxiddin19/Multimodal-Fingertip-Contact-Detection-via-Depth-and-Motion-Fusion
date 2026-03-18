# D405 Dataset Recording Guide
## For Depth Anything V2 Fine-tuning & Fingertip Contact Detection

---

## Table of Contents

1. [Overview](#1-overview)
2. [Hardware Setup](#2-hardware-setup)
3. [Software Dependencies](#3-software-dependencies)
4. [Recording Protocol](#4-recording-protocol)
5. [Angles, Surfaces, and Actions](#5-angles-surfaces-and-actions)
6. [Contact & Hover Labels](#6-contact--hover-labels)
7. [Step-by-Step Recording Workflow](#7-step-by-step-recording-workflow)
8. [Data Format & Directory Structure](#8-data-format--directory-structure)
9. [Quality Assurance](#9-quality-assurance)
10. [Preparing Data for Training](#10-preparing-data-for-training)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Overview

### What We Record

Each recording session captures **synchronized RGB + metric depth** from an Intel RealSense D405, plus **per-frame hand annotations** (21 MediaPipe landmarks, 5 fingertip depths, handedness). The data is used for:

- **Depth model fine-tuning**: RGB-depth pairs train Depth Anything V2 for close-range metric depth
- **Contact detection**: Fingertip depth relative to the calibrated surface plane determines contact vs. hover state

### Current Dataset Status

| Source | Users | Angles | Depth GT | Frames |
|--------|-------|--------|----------|--------|
| D405 existing (45°) | 1 | 45° | Yes | 42,632 |
| D405 overhead (90°) | 1 (you) | 90° | Yes | 231 |
| iPhone videos | 15 | 30°/45°/60° | **No** | Raw video |

### Target Dataset (Paper Section 3.2)

```
15 users x 3+ angles x 2+ surfaces x 2 sessions x ~900 frames/session
= ~160,000+ high-quality RGB-depth pairs
After quality filtering (~60% retention): ~100,000 pairs
```

---

## 2. Hardware Setup

### Required Equipment

| Item | Specification | Notes |
|------|--------------|-------|
| Intel RealSense D405 | Firmware latest | Optimal range: 7-50cm |
| Tripod/clamp | Adjustable angle | Must hold D405 steady |
| Printed keyboard layout | Standard QWERTY, A4 paper | Taped to desk surface |
| LED lighting | 5000K, 800+ lux, diffuse | Avoid direct glare on surface |
| USB 3.0 cable | Type-C, <2m | Longer cables cause frame drops |
| Monitor/laptop | For live preview | Running record.py |

### Camera Mounting Positions

```
         [90° overhead]
              |
              |  35cm
              |
    ──────────┼──────────  ← desk surface
   /          |          \
  / 60°     45°      30° \
 /            |            \
[cam]       [cam]        [cam]

Distance from surface center:
  90°: 30-40cm directly above
  45°: 35-45cm at 45° angle
  30°: 40-50cm at shallow angle
  60°: 30-40cm at steep angle
```

### D405 Camera Settings (Automatic)

The recording script (`record.py`) configures these automatically:

| Parameter | Value |
|-----------|-------|
| Resolution | 640 x 480 |
| FPS | 30 |
| Depth scale | 0.0001 m/unit |
| Min depth | 0.07m (7cm) |
| Max depth | 0.50m (50cm) |
| Visual preset | Short-range (preset 4) |
| Spatial filter | magnitude=2, alpha=0.5, delta=20 |
| Temporal filter | alpha=0.4, delta=20 |
| Hole filling | Mode 1 (farthest from around) |

---

## 3. Software Dependencies

```bash
# From the vrkeyb environment
pip install pyrealsense2 opencv-python numpy mediapipe
```

Verify camera connection:

```bash
#lINUX:

rs-enumerate-devices  # Should show "Intel RealSense D405"
```
#Windows:

python -c "import pyrealsense2 as rs; ctx = rs.context(); print(f'Devices: {len(ctx.devices)}'); [print(f'  {d.get_info(rs.camera_info.name)}') for d in ctx.devices]"
---

## 4. Recording Protocol

### Per-Participant Session Plan

Each participant should complete the following grid:

| Angle | Surface | Action | Duration | Target Frames |
|-------|---------|--------|----------|---------------|
| 90° | white_desk | typing | 60s | ~1,800 |
| 90° | white_desk | hovering | 30s | ~900 |
| 90° | white_desk | transition | 30s | ~900 |
| 45° | white_desk | typing | 60s | ~1,800 |
| 45° | white_desk | hovering | 30s | ~900 |
| 45° | white_desk | transition | 30s | ~900 |
| 45° | wood | typing | 60s | ~1,800 |
| 30° | white_desk | typing | 60s | ~1,800 |
| 60° | white_desk | typing | 60s | ~1,800 |

**Minimum per participant**: 3 sessions (~5,400 frames, ~3 minutes recording)
**Full protocol per participant**: 9 sessions (~12,600 frames, ~7 minutes recording)

### Participant Instructions (From Paper Section 3.2)

Read these to each participant before recording:

1. **Vary typing speeds**: Start with deliberate "hunt-and-peck," then switch to rapid touch typing
2. **Alternate hands**: Some periods single-hand, some periods bimanual
3. **Incorporate hovering**: Pause fingers 1-3cm above keys between keystrokes
4. **Intentional contact**: Press keys firmly and deliberately
5. **Varied postures**: Change wrist angles and finger curvature periodically
6. **All five fingers**: Make sure thumb, index, middle, ring, AND pinky all make contact
7. **False positives**: Occasionally pass fingers through the contact zone quickly without intending to type

### Behavioral Targets (Paper Section 3.2)

- **Contact ratio**: ~48-55% of frames should have at least one fingertip in contact
- **Hover ratio**: ~45-52% of frames should have all fingertips hovering
- **Both hands visible**: In >=80% of frames for bimanual sessions

---

## 5. Angles, Surfaces, and Actions

### Camera Angles

| Angle | CLI Value | Description | Advantages |
|-------|-----------|-------------|------------|
| 90° | `--angle 90` | Directly overhead (top-down) | Best depth accuracy (Z = depth directly), no occlusion between fingers |
| 45° | `--angle 45` | Standard diagonal view | Matches existing 42K dataset, balanced view |
| 30° | `--angle 30` | Shallow/frontal view | Realistic HMD-like perspective, more finger occlusion |
| 60° | `--angle 60` | Steep diagonal view | Between overhead and diagonal, good depth gradients |

### Typing Surfaces

| Surface | CLI Value | Description | Why It Matters |
|---------|-----------|-------------|---------------|
| White desk | `--surface white_desk` | Plain white laminate | High contrast, clean depth, baseline surface |
| Wood grain | `--surface wood` | Natural wood texture | Tests robustness to textured surfaces |
| Dark surface | `--surface dark` | Black/dark desk | IR absorption challenges, tests depth robustness |
| Semi-reflective | `--surface semi_reflective` | Glossy laminate | Specular reflections confuse depth sensors |

### Actions

| Action | CLI Value | Description | Contact Ratio Target |
|--------|-----------|-------------|---------------------|
| Typing | `--action typing` | Normal keyboard typing | ~50% contact, ~50% hover |
| Hovering | `--action hovering` | Hands hovering 1-5cm above surface, no contact | ~0% contact, ~100% hover |
| Transition | `--action transition` | Deliberate slow approach/retract cycles | ~30% contact, ~70% hover |

The `transition` action is critical for training the contact detector's hysteresis thresholds (the boundary between 4.5mm contact entry and 6.0mm exit).

---

## 6. Contact & Hover Labels

### Yes, Automatic Contact/Hover Labels Are Generated

The recording pipeline generates **automatic contact/hover labels** for every fingertip in every frame using the following method:

### How It Works

#### Step 1: Surface Calibration (Run Once Per Angle + Surface)

```bash
python record.py --calibrate --angle 90 --surface white_desk
```

#After calibration, verify the variance dropped:

type data\calibrations\calib_angle90_white_desk.json

(vr2) PS D:\VoiceAI\cvpr2026\data1> type data\calibrations\calib_angle90_white_desk.json
{
  "plane_coefficients": [
    -4.280820926167822e-07,
    -7.524870652842287e-05,
    0.9999999971687245,
    -0.226439751753146
  ],
  "inlier_ratio": 0.9959,
  "avg_surface_depth_m": 0.2446366250514984,
  "num_frames_used": 30
}
(vr2) PS D:\VoiceAI\cvpr2026\data1> 

This fits a **RANSAC plane** to 30 empty-desk depth frames:

```
Plane equation: ax + by + cz + d = 0
avg_surface_depth_m: 0.363m  (example for 90° white_desk)
```

The calibration is saved to `data/calibrations/calib_angle{N}_{surface}.json`.

#### Step 2: Per-Frame Fingertip Depth Measurement

During recording, MediaPipe detects all 21 hand landmarks. For each of the 5 fingertips (thumb=4, index=8, middle=12, ring=16, pinky=20), the depth is sampled using a **fringe-safe median** method:

- If the fingertip is inside the eroded hand mask: 3x3 median patch
- If the fingertip is on the hand edge (fringe): 5x5 median patch with validity filtering

This produces per-frame annotations like:

```json
{
  "frame_id": 10,
  "num_hands": 2,
  "hands": [
    {
      "handedness": "Right",
      "fingertip_depths_m": {
        "thumb": 0.337, "index": 0.331, "middle": 0.337,
        "ring": 0.342, "pinky": 0.338
      },
      "fingertip_pixels": {
        "thumb": [277, 325], "index": [276, 249], ...
      },
      "landmarks_px": [[160, 446], [214, 432], ...]
    }
  ]
}
```

#### Step 3: Contact Label Derivation (Post-Processing)

Contact labels are computed by comparing each fingertip's depth to the calibrated surface plane:

```
distance_to_surface = fingertip_depth - surface_depth_at(x, y)
```

Then the **velocity-gated hysteresis state machine** (Paper Table 6) applies:

| Condition | Label | Meaning |
|-----------|-------|---------|
| `distance_to_surface <= 4.5mm` | **CONTACT** (1) | Fingertip touching surface |
| `distance_to_surface >= 6.0mm` (from contact state) | **HOVER** (0) | Fingertip lifted |
| `4.5mm < distance < 6.0mm` | **Previous state** | Hysteresis prevents flicker |

### Post-Recording Label Generation Script

After recording, run this to compute contact/hover labels from the raw annotations + calibration:

```bash
python generate_contact_labels.py \
    --session-dir data/P01/P01_angle90_white_desk_typing_152436 \
    --calib-file data/calibrations/calib_angle90_white_desk.json \
    --contact-threshold 0.0045 \
    --exit-threshold 0.006
```

This will create a `labels/` directory with per-frame contact labels:

```json
{
  "frame_id": 10,
  "hands": [
    {
      "handedness": "Right",
      "fingertips": {
        "thumb":  {"depth_m": 0.337, "surface_depth_m": 0.363, "distance_mm": 26.0, "state": "hover"},
        "index":  {"depth_m": 0.331, "surface_depth_m": 0.363, "distance_mm": 32.0, "state": "hover"},
        "middle": {"depth_m": 0.362, "surface_depth_m": 0.363, "distance_mm": 1.0,  "state": "contact"},
        "ring":   {"depth_m": 0.342, "surface_depth_m": 0.363, "distance_mm": 21.0, "state": "hover"},
        "pinky":  {"depth_m": 0.338, "surface_depth_m": 0.363, "distance_mm": 25.0, "state": "hover"}
      }
    }
  ]
}
```

### Label Quality Notes

- **Automatic labels are derived from depth**, not manually annotated. Accuracy depends on depth quality.
- At 90° (overhead), depth = Z directly, so labels are most accurate.
- At 30°/45°/60°, the surface plane is tilted relative to the camera, so distance computation uses the full plane equation.
- The D405's depth accuracy at 25-45cm is <0.5mm, so fingertip-to-surface distances are reliable at the 4.5mm threshold.
- **Edge case**: Fast finger motion can cause motion blur in depth, leading to momentary label noise. The 450ms cooldown and temporal hysteresis mitigate this.

---

## 7. Step-by-Step Recording Workflow

### First Time Setup

```bash
cd ~/vrkeyb/data
```

### Step 1: Mount Camera at Desired Angle

Secure the D405 on a tripod. Measure the distance from the camera to the desk surface center (should be 25-45cm).

### Step 2: Calibrate Surface (Required Once Per Angle + Surface)

Remove all objects from the desk. Only the bare surface should be visible.

```bash
# Calibrate for each angle+surface combination you plan to use
python record.py --calibrate --angle 90 --surface white_desk
python record.py --calibrate --angle 45 --surface white_desk
python record.py --calibrate --angle 45 --surface wood
python record.py --calibrate --angle 30 --surface white_desk
python record.py --calibrate --angle 60 --surface white_desk
```

A live preview window opens. Press **SPACE** to collect 30 calibration frames. The script fits a RANSAC plane and saves to `data/calibrations/`.

Verify the output:
```
Plane fit: 7704/10000 inliers (77.0%)
avg_surface_depth_m: 0.363m
```

If inlier ratio < 50%, the surface is too uneven or the camera sees too many non-surface objects. Clear the desk and retry.



### Step 3: Record Sessions

Place the printed keyboard layout on the desk. Have the participant sit normally.

```bash
# Basic recording command
python record.py --participant P01 --angle 90 --surface white_desk --action typing

# Full example with all parameters
python record.py \
    --participant P02 \
    --angle 45 \
    --surface wood \
    --action transition \
    --hand both
```

**Live preview window controls:**

| Key | Action |
|-----|--------|
| **SPACE** | Start/stop recording |
| **C** | Quick re-calibrate (30 frames, no need to remove hands) |
| **Q** | Quit and save metadata |

**What you see in the preview:**
- Left panel: RGB with hand skeleton overlay, fingertip depth labels (in mm)
- Right panel: Colorized depth map (TURBO colormap) with fingertip markers
- Status bar: frame count, valid depth %, mean depth, hand count
- Red "REC" indicator when recording is active

### Step 4: Verify Recording Quality

After each session, check the output:

```bash
# Check frame count and file sizes
ls data/P01/P01_angle90_white_desk_typing_*/rgb/ | wc -l
ls data/P01/P01_angle90_white_desk_typing_*/depth/ | wc -l
ls data/P01/P01_angle90_white_desk_typing_*/annotations/ | wc -l

#After it finishes, verify:

python -c "import os; d='data/P01'; sessions=[s for s in os.listdir(d) if os.path.isdir(os.path.join(d,s))]; [print(f'{s}: {len(os.listdir(os.path.join(d,s,\"rgb\")))} frames') for s in sessions]"


(Get-ChildItem data\P01\*\rgb\*.png).Count

(Get-ChildItem data\P01\*\rgb\*.png -Recurse).Count


# Check depth range (should be 200-500mm for close-range)
python3 -c "
import cv2, numpy as np
d = cv2.imread('data/P01/P01_angle90_white_desk_typing_152436/depth/000010.png', cv2.IMREAD_UNCHANGED)
valid = d[d > 0]
print(f'Depth range: {valid.min()}-{valid.max()} mm, valid: {len(valid)/d.size:.0%}')
"

# Check annotation sample
python3 -m json.tool data/P01/P01_angle90_white_desk_typing_*/annotations/000010.json
```

### Step 5: Repeat for All Combinations

Use the batch recording approach:

```bash
# Full protocol for one participant (all angles on white_desk)
for angle in 90 45 30 60; do
    for action in typing hovering transition; do
        echo "=== P01 angle=$angle action=$action ==="
        python record.py --participant P01 --angle $angle --surface white_desk --action $action
    done
done
```

Or generate a full batch script:

```bash
python record.py --gen-script
# Creates record_all.sh with all participant x angle x surface x action combinations
```

---

## 8. Data Format & Directory Structure

### Output Hierarchy

```
data/
├── calibrations/
│   ├── calib_angle90_white_desk.json      # RANSAC plane fit
│   ├── calib_angle45_white_desk.json
│   ├── calib_angle30_white_desk.json
│   └── calib_angle45_wood.json
│
├── P01/
│   ├── P01_angle90_white_desk_typing_152436/
│   │   ├── rgb/                            # BGR uint8 PNG, 640x480
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── depth/                          # uint16 PNG, millimeters
│   │   │   ├── 000000.png                  # (value 240 = 0.240m)
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── depth_m/                        # float32 .npy, meters (backup)
│   │   │   ├── 000000.npy
│   │   │   └── ...
│   │   ├── annotations/                    # Per-frame hand tracking JSON
│   │   │   ├── 000000.json
│   │   │   └── ...
│   │   └── metadata.json                   # Session metadata + calibration
│   │
│   ├── P01_angle45_white_desk_typing_160012/
│   │   └── ...
│   └── P01_angle90_white_desk_hovering_161530/
│       └── ...
│
├── P02/
│   └── ...
└── ...
```

### File Formats

| File | Format | Details |
|------|--------|---------|
| `rgb/*.png` | BGR uint8, 640x480 | Standard OpenCV format |
| `depth/*.png` | uint16, 640x480 | Values in millimeters. 0 = invalid. Range 70-500. |
| `depth_m/*.npy` | float32, 640x480 | Values in meters. Backup for analysis. |
| `annotations/*.json` | JSON | Per-frame: hand count, landmarks, fingertip depths |
| `metadata.json` | JSON | Camera intrinsics, filters, session info, calibration |

### Depth Format Compatibility

The `depth/*.png` files are **directly compatible** with `CustomDepthDataset` in the DAv2 training pipeline:

```python
# CustomDepthDataset reads depth like this:
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
depth_m = depth.astype(np.float32) / 1000.0            # convert mm -> meters
```

### Annotation Format

Each `annotations/XXXXXX.json` contains:

```json
{
  "frame_id": 17,
  "num_hands": 2,
  "hands": [
    {
      "handedness": "Right",
      "fingertip_depths_m": {
        "thumb": 0.352,
        "index": 0.348,
        "middle": 0.324,
        "ring": 0.350,
        "pinky": 0.357
      },
      "fingertip_pixels": {
        "thumb": [294, 326],
        "index": [303, 241],
        "middle": [338, 218],
        "ring": [368, 224],
        "pinky": [402, 248]
      },
      "landmarks_px": [
        [180, 426], [231, 418], [268, 387], ...
      ]
    },
    {
      "handedness": "Left",
      ...
    }
  ]
}
```

### Metadata Format

```json
{
  "camera": "Intel RealSense D405",
  "resolution": [640, 480],
  "fps": 30,
  "depth_scale_m": 0.0001,
  "depth_range_m": [0.07, 0.5],
  "intrinsics": {
    "fx": 391.76, "fy": 390.61,
    "cx": 322.81, "cy": 239.81,
    "width": 640, "height": 480
  },
  "filters": {
    "spatial_magnitude": 2,
    "spatial_alpha": 0.5,
    "spatial_delta": 20,
    "temporal_alpha": 0.4,
    "temporal_delta": 20,
    "hole_fill_mode": 1
  },
  "session": {
    "participant": "P01",
    "angle_degrees": 90,
    "surface": "white_desk",
    "action": "typing",
    "hand": "both",
    "total_frames": 231,
    "timestamp": "2026-03-13 15:31:51"
  },
  "surface_calibration": {
    "plane_coefficients": [-1.24e-05, 1.63e-05, 1.0, -0.3647],
    "inlier_ratio": 0.7704,
    "avg_surface_depth_m": 0.363,
    "num_frames_used": 30
  }
}
```

---

## 9. Quality Assurance

### Automatic Quality Checks During Recording

The live preview shows real-time quality indicators:

| Indicator | Good | Bad | Action |
|-----------|------|-----|--------|
| Valid depth % | >30% (green) | <30% (red) | Adjust camera angle/distance |
| Mean depth | 200-450mm | <150mm or >500mm | Move camera closer/farther |
| Hand count | 1-2 | 0 | Ensure hands are in frame |
| Frame drops | 0 | >0 | Use shorter USB cable, reduce background load |


### From other Claude source for making tarining data from recording one:
(vr2) PS D:\voiceai\cvpr2026\data1> python record.py --export --data-root data --export-dir data\dav2_finetune
### Post-Recording Quality Filtering

Use `prepare_custom_dataset.py` for quality filtering before training:

```bash
cd ~/vrkeyb/Depth-Anything-V2/metric_depth

python prepare_custom_dataset.py \
    --data-dir ~/vrkeyb/data/data \
    --output-dir dataset/splits/custom_v2 \
    --filter \
    --min-depth-coverage 0.80 \
    --blur-threshold 100.0
```

This filters out frames with:
- <80% valid depth coverage in the hand-table ROI
- Motion blur (Laplacian variance < 100)
- Optionally: no MediaPipe hand detection (add `--check-hands` flag)

### Quality Targets

| Metric | Target | Paper Reference |
|--------|--------|----------------|
| Valid depth coverage | >80% in ROI | Section 3.3 |
| Laplacian variance | >100 | Section 3.3 |
| Hand detection rate | >95% | Section 3.3 |
| Contact ratio (typing) | 48-55% | Section 3.2 |
| Both hands visible | >80% (bimanual) | Section 3.2 |

---

## 10. Preparing Data for Training

### Method A: Direct Integration (Recommended)

The new recorded data uses the exact same format as `CustomDepthDataset`. To add it to training:

```bash
cd ~/vrkeyb/Depth-Anything-V2/metric_depth

# Generate split files from new data
python prepare_custom_dataset.py \
    --data-dir ~/vrkeyb/data/data \
    --output-dir dataset/splits/custom_v2 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42
```

This creates `train.txt`, `val.txt`, `test.txt` with lines like:

```
/path/to/rgb/000000.png /path/to/depth/000000.png
/path/to/rgb/000001.png /path/to/depth/000001.png
```

### Method B: Merge with Existing Dataset

To combine the new 90° data with the existing 42,632 pairs:

```bash
# Concatenate split files
cat dataset/splits/custom/train.txt dataset/splits/custom_v2/train.txt > dataset/splits/custom_merged/train.txt
cat dataset/splits/custom/val.txt dataset/splits/custom_v2/val.txt > dataset/splits/custom_merged/val.txt
```

Then update `dist_train1.sh` split path or change `train1.py` to point to `custom_merged`.

### Split Strategy (Paper Section 3.3)

**Critical rule**: All frames from a given participant go to ONE split only.

```
Participants: P01, P02, P03, ..., P15
├── Train: P01-P12 (80%)    ← 12 participants
├── Val:   P13-P14 (10%)    ← 1-2 participants
└── Test:  P14-P15 (10%)    ← 1-2 participants

NEVER mix frames from the same participant across splits.
This prevents identity leakage (hand shape, skin tone, typing style).
```

---

## 11. Troubleshooting

### Depth appears black in saved PNGs

**Fixed.** The old `save_frame` saved raw sensor units (0-4447 out of uint16 65535). Now saves millimeters (0-444), which is correctly visible and compatible with training.

If you have old recordings with `depth_u16/` directory, convert them:

```bash
python convert_existing_depth.py data/P01/P01_angle90_white_desk_typing_152436
```

### No depth data (all zeros)

- Check USB cable: D405 requires USB 3.0. USB 2.0 drops depth entirely.
- Check distance: D405 minimum is 7cm. If camera is <7cm from surface, depth is invalid.
- Check lighting: Extremely bright direct light can saturate the IR sensor.

### MediaPipe fails to detect hands

- Ensure both hands are fully in frame (not cut off at edges)
- Increase lighting (MediaPipe needs adequate RGB quality)
- Hands should be 15-40cm from camera for optimal detection
- At 90° overhead, hand pose can be ambiguous. MediaPipe's `min_detection_confidence=0.7` may reject some frames.

### Low depth coverage (<30%)

- Camera too close: Move to 25-45cm range
- Reflective surface: Switch to matte/white surface
- D405 viewing angle too steep: At extreme angles, IR pattern can't resolve depth

### Frame drops during recording

- Close other applications (especially browsers, video players)
- Use USB 3.0 port directly (not through a hub)
- Reduce FPS from 30 to 15 if needed: edit `FPS = 15` in record.py

### Calibration inlier ratio too low (<50%)

- Surface must be flat and uniform (remove objects, papers, keyboards)
- Camera must see mostly the desk surface
- Try adjusting camera position so the desk fills more of the frame

---

## Quick Reference: Common Recording Commands

```bash
# Surface calibration (run first, empty desk)
python record.py --calibrate --angle 90 --surface white_desk

# Standard recording session
python record.py --participant P01 --angle 90 --surface white_desk --action typing

# All angles for one participant
for angle in 30 45 60 90; do
    python record.py --calibrate --angle $angle --surface white_desk
    python record.py --participant P01 --angle $angle --surface white_desk --action typing
done

# Full protocol with multiple surfaces
for surface in white_desk wood dark; do
    python record.py --calibrate --angle 45 --surface $surface
    python record.py --participant P02 --angle 45 --surface $surface --action typing
done

# Generate batch script for all combinations
python record.py --gen-script

# Convert old depth format to millimeters
python convert_existing_depth.py data/P01/P01_angle90_white_desk_typing_152436

# Prepare data for DAv2 training
cd ~/vrkeyb/Depth-Anything-V2/metric_depth
python prepare_custom_dataset.py --data-dir ~/vrkeyb/data/data --output-dir dataset/splits/custom_v2 --filter
```
