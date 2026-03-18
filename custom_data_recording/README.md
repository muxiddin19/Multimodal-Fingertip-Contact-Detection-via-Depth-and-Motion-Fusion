# Custom Data Recording

Tools for recording RGB-depth data with the Intel RealSense D405 camera and generating contact/hover labels for training.

## Files

| Script | Purpose |
|--------|---------|
| `record.py` | D405 RGB+depth recording with live hand tracking overlay |
| `generate_contact_labels.py` | Generate per-fingertip contact/hover labels from recordings |
| `convert_existing_depth.py` | Convert old raw-sensor-unit depth PNGs to millimeters |
| `RECORDING_GUIDE.md` | Full recording protocol (hardware setup, angles, surfaces, troubleshooting) |

## Quick Start

```bash
# 1. Calibrate surface plane (empty desk, no hands)
python record.py --calibrate --angle 90 --surface white_desk

# 2. Record a session
python record.py --participant P01 --angle 90 --surface white_desk --action typing

# 3. Generate contact/hover labels
python generate_contact_labels.py \
    --session-dir data/P01/P01_angle90_white_desk_typing_152436 \
    --calib-file data/calibrations/calib_angle90_white_desk.json
```

## Camera Angles

| Angle | Description |
|-------|-------------|
| 90° | Directly overhead (top-down) — best depth accuracy |
| 60° | Steep diagonal — good depth gradients |
| 45° | Standard diagonal — balanced view |
| 30° | Shallow/frontal — realistic HMD-like perspective |

## Contact Labels

Labels are derived automatically from depth using a velocity-gated hysteresis state machine:
- **Contact**: fingertip-to-surface distance <= 4.5mm
- **Hover**: distance >= 6.0mm (from contact state)
- **Hysteresis zone** (4.5-6.0mm): retains previous state to prevent flicker

See `RECORDING_GUIDE.md` for the full protocol.
