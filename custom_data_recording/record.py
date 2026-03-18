#!/usr/bin/env python3
"""
D405 RGB+Depth Recording Pipeline for Depth Anything V2 Fine-tuning
====================================================================
Records paired RGB + depth frames from Intel RealSense D405.
Outputs clean, filtered depth maps compatible with DAv2 training format.

Usage:
    python record.py --participant P01 --angle 30 --surface white_desk
    python record.py --participant P01 --angle 60 --surface wood --action typing
    python record.py --calibrate --angle 30 --surface white_desk
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import time
import argparse
from pathlib import Path
# Add at top of record.py, after existing imports
# Add at top of record.py, after existing imports
import mediapipe as mp


# ============================================================
# Hand Tracking + Clean Fingertip Depth
# ============================================================
class HandDepthTracker:
    """MediaPipe hand tracking with fringe-safe depth sampling."""

    FINGERTIP_IDS = {
        "thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20
    }

    def __init__(self, max_hands=2, min_detection_confidence=0.7):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb, depth_m):
        """
        Detect hands and sample clean fingertip depths.

        Returns:
            hands_data: list of dicts per detected hand, each containing:
                - landmarks_px: all 21 landmarks in pixel coords
                - fingertip_depths: {name: depth_m} for 5 fingertips
                - fingertip_pixels: {name: (x, y)} for 5 fingertips
                - handedness: 'Left' or 'Right'
        """
        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_input)

        if not result.multi_hand_landmarks:
            return []

        h, w = depth_m.shape[:2]
        hands_data = []

        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Handedness
            handedness = "Unknown"
            if result.multi_handedness and i < len(result.multi_handedness):
                handedness = result.multi_handedness[i].classification[0].label

            # All landmarks in pixel coords
            landmarks_px = []
            for lm in hand_landmarks.landmark:
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                landmarks_px.append((px, py))

            # Build eroded hand mask
            mask = self._make_eroded_mask(landmarks_px, h, w)

            # Sample fingertip depths
            tips = {}
            tips_px = {}
            for name, idx in self.FINGERTIP_IDS.items():
                px, py = landmarks_px[idx]
                depth = self._sample_depth(depth_m, mask, px, py)
                tips[name] = depth
                tips_px[name] = (px, py)

            hands_data.append({
                "landmarks_px": landmarks_px,
                "fingertip_depths": tips,
                "fingertip_pixels": tips_px,
                "handedness": handedness,
            })

        return hands_data

    def _make_eroded_mask(self, landmarks_px, h, w):
        """Convex hull mask eroded by 3px to avoid fringe."""
        points = np.array(landmarks_px)
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.erode(mask, kernel, iterations=1)

    def _sample_depth(self, depth_m, eroded_mask, px, py):
        """
        Sample depth at (px, py) with fringe avoidance.
        If on fringe (outside eroded mask), use 5x5 median.
        If inside eroded mask, use 3x3 median.
        """
        h, w = depth_m.shape
        px = int(np.clip(px, 3, w - 4))
        py = int(np.clip(py, 3, h - 4))

        if eroded_mask[py, px] == 0:
            # On fringe — use wider patch
            patch = depth_m[py - 2:py + 3, px - 2:px + 3]
        else:
            # Safe interior — use tight patch
            patch = depth_m[py - 1:py + 2, px - 1:px + 2]

        valid = patch[(patch > 0.07) & (patch < 0.50)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def close(self):
        self.hands.close()

# ============================================================
# D405 Configuration
# ============================================================
RESOLUTION = (640, 480)
FPS = 30
# D405 optimal range: 7cm - 50cm. Hands at 15-40cm.
MIN_DEPTH_M = 0.07
MAX_DEPTH_M = 0.50
# Post-processing filter params (tuned for close-range hand capture)
SPATIAL_MAGNITUDE = 2
SPATIAL_SMOOTH_ALPHA = 0.5
SPATIAL_SMOOTH_DELTA = 20
TEMPORAL_SMOOTH_ALPHA = 0.4
TEMPORAL_SMOOTH_DELTA = 20
HOLE_FILL_MODE = 1  # farthest from around


# ============================================================
# Depth Filtering Pipeline
# ============================================================
class DepthFilter:
    """Multi-stage depth filtering for clean training data."""

    def __init__(self):
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 1)  # no decimation

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, SPATIAL_MAGNITUDE)
        self.spatial.set_option(rs.option.filter_smooth_alpha, SPATIAL_SMOOTH_ALPHA)
        self.spatial.set_option(rs.option.filter_smooth_delta, SPATIAL_SMOOTH_DELTA)
        self.spatial.set_option(rs.option.holes_fill, HOLE_FILL_MODE)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, TEMPORAL_SMOOTH_ALPHA)
        self.temporal.set_option(rs.option.filter_smooth_delta, TEMPORAL_SMOOTH_DELTA)

        self.hole_fill = rs.hole_filling_filter()
        self.hole_fill.set_option(rs.option.holes_fill, HOLE_FILL_MODE)

        self.threshold = rs.threshold_filter()
        self.threshold.set_option(rs.option.min_distance, MIN_DEPTH_M)
        self.threshold.set_option(rs.option.max_distance, MAX_DEPTH_M)

    def apply(self, depth_frame):
        """Apply full filter chain: threshold -> spatial -> temporal -> hole fill."""
        frame = depth_frame
        frame = self.threshold.process(frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.hole_fill.process(frame)
        return frame


# ============================================================
# Surface Calibration (RANSAC plane fitting)
# ============================================================
class SurfaceCalibrator:
    """RANSAC-based surface plane calibration from empty desk frames."""

    def __init__(self, num_frames=30):
        self.num_frames = num_frames
        self.depth_frames = []

    def collect_frame(self, depth_m):
        """Collect a depth frame for calibration."""
        self.depth_frames.append(depth_m.copy())
        return len(self.depth_frames) >= self.num_frames

    def fit_plane(self):
        """Fit plane to averaged depth using RANSAC."""
        if len(self.depth_frames) < self.num_frames:
            raise ValueError(f"Need {self.num_frames} frames, got {len(self.depth_frames)}")

        avg_depth = np.mean(self.depth_frames, axis=0)
        h, w = avg_depth.shape

        # Sample valid points
        ys, xs = np.where(avg_depth > 0)
        if len(ys) < 100:
            raise ValueError("Not enough valid depth points for plane fitting")

        # Subsample for speed
        idx = np.random.choice(len(ys), min(10000, len(ys)), replace=False)
        points = np.column_stack([xs[idx], ys[idx], avg_depth[ys[idx], xs[idx]]])

        # RANSAC plane fitting
        best_plane = None
        best_inliers = 0
        inlier_threshold = 0.008  # 2mm

        for _ in range(1000):
            sample_idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[sample_idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            d = -np.dot(normal, p1)

            distances = np.abs(points @ normal + d)
            n_inliers = np.sum(distances < inlier_threshold)

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_plane = np.append(normal, d)

        if best_plane is None:
            raise ValueError("RANSAC failed to fit plane")

        inlier_ratio = best_inliers / len(points)
        print(f"Plane fit: {best_inliers}/{len(points)} inliers ({inlier_ratio:.1%})")
        print(f"Plane equation: {best_plane[0]:.4f}x + {best_plane[1]:.4f}y + "
              f"{best_plane[2]:.4f}z + {best_plane[3]:.4f} = 0")

        return {
            "plane_coefficients": best_plane.tolist(),
            "inlier_ratio": float(inlier_ratio),
            "avg_surface_depth_m": float(np.median(avg_depth[avg_depth > 0])),
            "num_frames_used": len(self.depth_frames)
        }


# ============================================================
# D405 Recorder
# ============================================================
class D405Recorder:
    """Main recording class for D405 RGB+Depth capture."""

    def __init__(self, output_root="data"):
        self.output_root = Path(output_root)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.depth_filter = DepthFilter()
        self.depth_scale = None
        self.intrinsics = None
        self.hand_tracker = HandDepthTracker()
        

# In D405Recorder.__init__, add:
    #self.hand_tracker = HandDepthTracker() 

    # Add new save method to D405Recorder:
    def save_hand_data(self, session_dir, idx, hands_data):
        """Save per-frame hand tracking + fingertip depth data."""
        annotations_dir = session_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)

        frame_annotation = {
            "frame_id": idx,
            "num_hands": len(hands_data),
            "hands": []
        }

        for hand in hands_data:
            hand_entry = {
                "handedness": hand["handedness"],
                "fingertip_depths_m": hand["fingertip_depths"],
                "fingertip_pixels": {
                    k: list(v) for k, v in hand["fingertip_pixels"].items()
                },
                "landmarks_px": hand["landmarks_px"],
            }
            frame_annotation["hands"].append(hand_entry)

        with open(annotations_dir / f"{idx:06d}.json", "w") as f:
            json.dump(frame_annotation, f)

    def start_camera(self):
        """Initialize and start D405 pipeline."""
        self.config.enable_stream(
            rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS
        )
        self.config.enable_stream(
            rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS
        )

        profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale} m/unit ({self.depth_scale * 1000:.2f} mm/unit)")

        # Set D405 short-range preset for close-up hand recording
        try:
            depth_sensor.set_option(rs.option.visual_preset, 4)
            print("Set short-range visual preset")
        except Exception as e:
            print(f"Could not set visual preset: {e}")

        # Get intrinsics (needed for DAv2 training)
        color_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        print(f"Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}, "
              f"cx={self.intrinsics.ppx:.1f}, cy={self.intrinsics.ppy:.1f}")

        # Warm up (auto-exposure stabilization)
        print("Warming up camera (2s)...")
        for _ in range(60):
            self.pipeline.wait_for_frames()
        print("Camera ready")

    def stop_camera(self):
        self.pipeline.stop()

    def get_frames(self):
        """Get aligned, filtered RGB + depth frames."""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        # Apply depth filters
        filtered_depth = self.depth_filter.apply(depth_frame)

        rgb = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(filtered_depth.get_data())  # uint16
        depth_m = depth_raw.astype(np.float32) * self.depth_scale  # float32 meters

        # Zero out invalid regions
        depth_m[depth_m < MIN_DEPTH_M] = 0
        depth_m[depth_m > MAX_DEPTH_M] = 0

        return rgb, depth_raw, depth_m

    def make_session_dir(self, participant, angle, surface, action):
        """Create structured output directory."""
        session_name = f"{participant}_angle{angle}_{surface}_{action}_{time.strftime('%H%M%S')}"
        session_dir = self.output_root / participant / session_name
        (session_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (session_dir / "depth").mkdir(parents=True, exist_ok=True)
        (session_dir / "depth_m").mkdir(parents=True, exist_ok=True)
        return session_dir

    def save_frame(self, session_dir, idx, rgb, depth_raw, depth_m):
        """Save one frame triplet."""
        fname = f"{idx:06d}"
        # RGB as PNG (lossless)
        cv2.imwrite(str(session_dir / "rgb" / f"{fname}.png"), rgb)
        # Depth as 16-bit PNG in millimeters (compatible with CustomDepthDataset)
        # Raw sensor units are too small (0-4447 out of 65535) → appears black
        # Converting to mm gives range 0-444 which is correct for close-range
        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        cv2.imwrite(str(session_dir / "depth" / f"{fname}.png"), depth_mm)
        # Depth as float32 .npy in meters (backup / for analysis)
        np.save(str(session_dir / "depth_m" / f"{fname}.npy"), depth_m)

    def save_metadata(self, session_dir, args, frame_count, calibration=None):
        """Save session metadata."""
        meta = {
            "camera": "Intel RealSense D405",
            "resolution": list(RESOLUTION),
            "fps": FPS,
            "depth_scale_m": self.depth_scale,
            "depth_range_m": [MIN_DEPTH_M, MAX_DEPTH_M],
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.ppx,
                "cy": self.intrinsics.ppy,
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
            },
            "filters": {
                "spatial_magnitude": SPATIAL_MAGNITUDE,
                "spatial_alpha": SPATIAL_SMOOTH_ALPHA,
                "spatial_delta": SPATIAL_SMOOTH_DELTA,
                "temporal_alpha": TEMPORAL_SMOOTH_ALPHA,
                "temporal_delta": TEMPORAL_SMOOTH_DELTA,
                "hole_fill_mode": HOLE_FILL_MODE,
            },
            "session": {
                "participant": args.participant,
                "angle_degrees": args.angle,
                "surface": args.surface,
                "action": args.action,
                "hand": args.hand,
                "total_frames": frame_count,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        if calibration:
            meta["surface_calibration"] = calibration

        with open(session_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def compute_depth_quality(self, depth_m):
        """Compute depth quality metrics for display."""
        valid = depth_m > 0
        valid_ratio = np.sum(valid) / depth_m.size
        if np.sum(valid) == 0:
            return 0.0, 0.0, 0.0
        mean_d = np.mean(depth_m[valid]) * 1000  # mm
        std_d = np.std(depth_m[valid]) * 1000  # mm
        return valid_ratio, mean_d, std_d

    # Replace make_display method in D405Recorder:
    def make_display(self, rgb, depth_m, frame_idx, recording, quality_info,
                     hands_data=None):
        """Build visualization with hand overlay."""
        depth_vis = np.zeros_like(rgb)
        valid = depth_m > 0
        if np.any(valid):
            depth_normalized = np.clip(depth_m / MAX_DEPTH_M, 0, 1)
            depth_u8 = (depth_normalized * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            depth_vis[~valid] = 0

        # Draw hand tracking on both panels
        rgb_draw = rgb.copy()
        depth_draw = depth_vis.copy()

        if hands_data:
            for hand in hands_data:
                # Draw fingertip circles + depth labels
                for name, (px, py) in hand["fingertip_pixels"].items():
                    d = hand["fingertip_depths"][name]
                    color = (0, 255, 0) if d > 0 else (0, 0, 255)
                    # RGB panel
                    cv2.circle(rgb_draw, (px, py), 5, color, -1)
                    if d > 0:
                        label = f"{d * 1000:.1f}"
                        cv2.putText(rgb_draw, label, (px + 8, py - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                    # Depth panel
                    cv2.circle(depth_draw, (px, py), 5, (255, 255, 255), -1)

                # Draw skeleton connections
                connections = [
                    (0,1),(1,2),(2,3),(3,4),      # thumb
                    (0,5),(5,6),(6,7),(7,8),      # index
                    (5,9),(9,10),(10,11),(11,12),  # middle
                    (9,13),(13,14),(14,15),(15,16),# ring
                    (13,17),(17,18),(18,19),(19,20),# pinky
                    (0,17),
                ]
                lm = hand["landmarks_px"]
                for a, b in connections:
                    cv2.line(rgb_draw, lm[a], lm[b], (200, 200, 200), 1)

        display = np.hstack([rgb_draw, depth_draw])

        # Info overlay
        valid_ratio, mean_d, std_d = quality_info
        color = (0, 255, 0) if valid_ratio > 0.3 else (0, 0, 255)
        cv2.putText(display, f"Frame: {frame_idx}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Valid: {valid_ratio:.0%}", (10, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"Depth: {mean_d:.0f}+/-{std_d:.0f}mm", (10, 75),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        n_hands = len(hands_data) if hands_data else 0
        cv2.putText(display, f"Hands: {n_hands}", (10, 100),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if recording:
            cv2.circle(display, (display.shape[1] - 30, 25), 12, (0, 0, 255), -1)
            cv2.putText(display, "REC", (display.shape[1] - 75, 32),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, "[SPACE]=rec [C]=calibrate [Q]=quit",
                     (10, display.shape[0] - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display

    # ----------------------------------------------------------
    # Calibration mode
    # ----------------------------------------------------------
    def run_calibration(self, args):
        """Record empty desk frames and fit surface plane."""
        print("\n=== SURFACE CALIBRATION ===")
        print("Remove all objects from the desk. Press [SPACE] to start.\n")

        self.start_camera()
        calibrator = SurfaceCalibrator(num_frames=30)
        collecting = False
        calib_result = None

        try:
            while True:
                rgb, depth_raw, depth_m = self.get_frames()
                if rgb is None:
                    continue

                quality = self.compute_depth_quality(depth_m)

                # Display
                display = self.make_display(rgb, depth_m, len(calibrator.depth_frames),
                                             collecting, quality)
                if collecting:
                    cv2.putText(display, f"Collecting: {len(calibrator.depth_frames)}/30",
                                 (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("D405 Calibration", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and not collecting:
                    collecting = True
                    print("Collecting calibration frames...")
                elif key == ord('q'):
                    break

                if collecting:
                    done = calibrator.collect_frame(depth_m)
                    if done:
                        print("Fitting plane...")
                        calib_result = calibrator.fit_plane()
                        # Save calibration
                        calib_dir = self.output_root / "calibrations"
                        calib_dir.mkdir(parents=True, exist_ok=True)
                        calib_file = calib_dir / f"calib_angle{args.angle}_{args.surface}.json"
                        with open(calib_file, "w") as f:
                            json.dump(calib_result, f, indent=2)
                        print(f"Calibration saved to {calib_file}")
                        break

        finally:
            cv2.destroyAllWindows()
            self.stop_camera()

        return calib_result

    # ----------------------------------------------------------
    # Recording mode
    # ----------------------------------------------------------
    def run_recording(self, args):
        """Main recording loop."""
        # Load calibration if exists
        calib_file = (self.output_root / "calibrations" /
                      f"calib_angle{args.angle}_{args.surface}.json")
        calibration = None
        if calib_file.exists():
            with open(calib_file) as f:
                calibration = json.load(f)
            print(f"Loaded surface calibration: avg depth = "
                  f"{calibration['avg_surface_depth_m']*1000:.1f}mm")
        else:
            print(f"WARNING: No calibration for angle={args.angle}, surface={args.surface}")
            print(f"Run: python record.py --calibrate --angle {args.angle} "
                  f"--surface {args.surface}")

        session_dir = self.make_session_dir(
            args.participant, args.angle, args.surface, args.action
        )
        print(f"\n=== RECORDING SESSION ===")
        print(f"Participant: {args.participant}")
        print(f"Angle: {args.angle}°  Surface: {args.surface}")
        print(f"Action: {args.action}  Hand: {args.hand}")
        print(f"Output: {session_dir}")
        print(f"Press [SPACE] to start/stop recording, [Q] to quit\n")

        self.start_camera()
        frame_idx = 0
        recording = False
        dropped = 0

        try:
            while True:
                rgb, depth_raw, depth_m = self.get_frames()
                if rgb is None:
                    continue

                quality = self.compute_depth_quality(depth_m)
                valid_ratio = quality[0]

                # Try hand tracking for display + saving
                hands_data = None
                if hasattr(self, 'hand_tracker'):
                    hands_data = self.hand_tracker.process(rgb, depth_m)

                # Display with hand overlay
                display = self.make_display(rgb, depth_m, frame_idx, recording, quality,
                                            hands_data=hands_data)
                cv2.imshow("D405 Recording", display)

                # Save if recording
                if recording:
                    if valid_ratio > 0.1:
                        self.save_frame(session_dir, frame_idx, rgb, depth_raw, depth_m)

                        if hands_data:
                            self.save_hand_data(session_dir, frame_idx, hands_data)

                        frame_idx += 1
                    else:
                        dropped += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    recording = not recording
                    state = "STARTED" if recording else "PAUSED"
                    print(f"Recording {state} | Frames: {frame_idx} | Dropped: {dropped}")
                elif key == ord('c'):
                    # Quick recalibrate without stopping
                    print("Quick calibration - hold still...")
                    calibrator = SurfaceCalibrator(num_frames=30)
                    for _ in range(30):
                        _, _, dm = self.get_frames()
                        if dm is not None:
                            calibrator.collect_frame(dm)
                    calibration = calibrator.fit_plane()
                    calib_dir = self.output_root / "calibrations"
                    calib_dir.mkdir(parents=True, exist_ok=True)
                    with open(calib_file, "w") as f:
                        json.dump(calibration, f, indent=2)
                    print("Calibration updated")
                elif key == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            self.stop_camera()
            self.save_metadata(session_dir, args, frame_idx, calibration)
            print(f"\nSession complete: {frame_idx} frames saved, {dropped} dropped")
            print(f"Output: {session_dir}")

# Add to record.py or as a separate post-processing step
# Erode hand mask by 3px before sampling depth at fingertip landmarks

def get_clean_fingertip_depth(depth_m, hand_landmarks, frame_shape):
    """
    Sample fingertip depth with 3px erosion to avoid edge fringe.
    
    hand_landmarks: MediaPipe hand landmarks (normalized)
    Returns: dict of {finger_name: depth_in_meters}
    """
    h, w = frame_shape[:2]
    
    # Create hand mask from convex hull of landmarks
    points = np.array([(int(lm.x * w), int(lm.y * h)) 
                       for lm in hand_landmarks.landmark])
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Erode 3px to avoid fringe
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    
    # Fingertip landmark indices: thumb=4, index=8, middle=12, ring=16, pinky=20
    tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    depths = {}
    
    for name, idx in tips.items():
        lm = hand_landmarks.landmark[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        px = np.clip(px, 3, w - 4)
        py = np.clip(py, 3, h - 4)
        
        # Only sample if inside eroded mask
        if mask_eroded[py, px] == 0:
            # Landmark on fringe — shift inward using 5x5 patch median
            patch = depth_m[py-2:py+3, px-2:px+3]
            valid = patch[(patch > 0.07) & (patch < 0.50)]
            depths[name] = float(np.median(valid)) if len(valid) > 0 else 0.0
        else:
            # Safe region — use 3x3 median for sub-pixel robustness
            patch = depth_m[py-1:py+2, px-1:px+2]
            valid = patch[(patch > 0.07) & (patch < 0.50)]
            depths[name] = float(np.median(valid)) if len(valid) > 0 else 0.0
    
    return depths
# ============================================================
# DAv2 Export Utility
# ============================================================
def export_for_dav2(data_root, output_dir, max_depth_m=0.5):
    """
    Convert recorded data to Depth Anything V2 fine-tuning format.

    DAv2 expects:
      - RGB images: {split}/rgb/{id}.png
      - Depth maps: {split}/depth/{id}.npy  (float32, meters)
      - Filelist: {split}/filelist.txt (one relative path per line)

    Depth maps are normalized to [0, max_depth] range.
    Invalid pixels (depth=0) are stored as 0 (masked in loss).
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    (output_dir / "train" / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "depth").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "depth").mkdir(parents=True, exist_ok=True)

    all_sessions = sorted(data_root.glob("P*/P*_angle*"))
    print(f"Found {len(all_sessions)} sessions")

    train_list = []
    val_list = []
    global_idx = 0
    total_frames = 0

    for session in all_sessions:
        rgb_dir = session / "rgb"
        depth_dir = session / "depth_m"

        if not rgb_dir.exists() or not depth_dir.exists():
            continue

        rgb_files = sorted(rgb_dir.glob("*.png"))
        meta_file = session / "metadata.json"
        participant = "unknown"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                participant = meta.get("session", {}).get("participant", "unknown")

        for rgb_path in rgb_files:
            stem = rgb_path.stem
            depth_path = depth_dir / f"{stem}.npy"

            if not depth_path.exists():
                continue

            depth = np.load(depth_path)

            # Quality gate: skip if >50% invalid
            valid_ratio = np.sum(depth > 0) / depth.size
            if valid_ratio < 0.5:
                continue

            # Normalize depth to max_depth range
            depth_clean = np.clip(depth, 0, max_depth_m)

            # Assign to train/val (hold out last 2 participants for val)
            split = "val" if participant in ["P14", "P15"] else "train"
            out_name = f"{global_idx:07d}"

            # Copy RGB
            rgb = cv2.imread(str(rgb_path))
            cv2.imwrite(str(output_dir / split / "rgb" / f"{out_name}.png"), rgb)

            # Save normalized depth
            np.save(str(output_dir / split / "depth" / f"{out_name}.npy"), depth_clean)

            if split == "train":
                train_list.append(out_name)
            else:
                val_list.append(out_name)

            global_idx += 1
            total_frames += 1

    # Write filelists
    with open(output_dir / "train" / "filelist.txt", "w") as f:
        f.write("\n".join(train_list))
    with open(output_dir / "val" / "filelist.txt", "w") as f:
        f.write("\n".join(val_list))

    print(f"\nExport complete:")
    print(f"  Train: {len(train_list)} frames")
    print(f"  Val:   {len(val_list)} frames")
    print(f"  Total: {total_frames} frames")
    print(f"  Output: {output_dir}")


# ============================================================
# Batch Recording Script Generator
# ============================================================
def generate_recording_script(output_path="record_all.sh"):
    """Generate shell script for full recording protocol."""

    participants = [f"P{i:02d}" for i in range(1, 16)]
    angles = [30, 45, 60]
    surfaces = ["white_desk", "wood", "dark", "semi_reflective"]
    actions = ["typing", "hovering", "transition"]

    lines = [
        "#!/bin/bash",
        "# Full recording protocol: 15 participants x 3 angles x 4 surfaces x 3 actions",
        f"# Total sessions: {len(participants) * len(angles) * len(surfaces) * len(actions)}",
        f"# Target: ~500 frames per session at 30fps ≈ 17 seconds recording each",
        "",
        "set -e",
        "",
        "# Step 1: Calibrate each angle+surface combo (do once)",
        "echo '=== CALIBRATION PHASE ==='",
    ]

    for angle in angles:
        for surface in surfaces:
            lines.append(
                f"echo 'Calibrate: angle={angle}, surface={surface}'"
            )
            lines.append(
                f"python record.py --calibrate --angle {angle} --surface {surface}"
            )
            lines.append("")

    lines.extend([
        "# Step 2: Record all sessions",
        "echo '=== RECORDING PHASE ==='",
    ])

    for p in participants:
        lines.append(f"\necho '--- Participant {p} ---'")
        for angle in angles:
            for surface in surfaces:
                for action in actions:
                    lines.append(
                        f"python record.py --participant {p} --angle {angle} "
                        f"--surface {surface} --action {action}"
                    )

    lines.extend([
        "",
        "# Step 3: Export for DAv2 fine-tuning",
        "echo '=== EXPORT PHASE ==='",
        "python record.py --export --data-root data --export-dir data/dav2_finetune",
        "",
        "echo 'All done!'",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(output_path, 0o755)
    print(f"Recording script saved to {output_path}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="D405 RGB+Depth Recorder for DAv2 Fine-tuning"
    )

    # Modes
    parser.add_argument("--calibrate", action="store_true",
                        help="Run surface calibration mode")
    parser.add_argument("--export", action="store_true",
                        help="Export recorded data to DAv2 format")
    parser.add_argument("--gen-script", action="store_true",
                        help="Generate batch recording shell script")

    # Session params
    parser.add_argument("--participant", type=str, default="P01",
                        help="Participant ID (e.g. P01)")
    parser.add_argument("--angle", type=int, default=30,
                        choices=[30, 45, 60, 90],
                        help="Camera angle in degrees")
    parser.add_argument("--surface", type=str, default="white_desk",
                        choices=["white_desk", "wood", "dark", "semi_reflective"],
                        help="Surface type")
    parser.add_argument("--action", type=str, default="typing",
                        choices=["typing", "hovering", "transition"],
                        help="Hand action type")
    parser.add_argument("--hand", type=str, default="both",
                        choices=["left", "right", "both"],
                        help="Which hand(s)")

    # Paths
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--export-dir", type=str, default="data/dav2_finetune",
                        help="DAv2 export output directory")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.gen_script:
        generate_recording_script()
        return

    if args.export:
        export_for_dav2(args.data_root, args.export_dir)
        return

    recorder = D405Recorder(output_root=args.data_root)

    if args.calibrate:
        recorder.run_calibration(args)
    else:
        recorder.run_recording(args)


if __name__ == "__main__":
    main()
