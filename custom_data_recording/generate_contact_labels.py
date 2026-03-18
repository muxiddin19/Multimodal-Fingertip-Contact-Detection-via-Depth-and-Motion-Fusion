"""
Generate contact/hover labels from recorded annotations + surface calibration.

Uses the velocity-gated hysteresis state machine (Paper Table 6):
  - Contact entry threshold: 4.5mm (fingertip within 4.5mm of surface)
  - Contact exit threshold:  6.0mm (fingertip must lift 6.0mm to exit contact)
  - Cooldown: 450ms (~14 frames @ 30fps)

Usage:
  python generate_contact_labels.py \
      --session-dir data/P01/P01_angle90_white_desk_typing_152436

  # Custom thresholds
  python generate_contact_labels.py \
      --session-dir data/P01/P01_angle90_white_desk_typing_152436 \
      --contact-threshold 0.0045 \
      --exit-threshold 0.006

  # Process all sessions for a participant
  python generate_contact_labels.py --data-root data/P01

  # Process everything
  python generate_contact_labels.py --data-root data
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


class HysteresisContactDetector:
    """Per-fingertip contact state machine with hysteresis and cooldown."""

    def __init__(self, contact_threshold_m=0.0045, exit_threshold_m=0.006,
                 cooldown_frames=14):
        self.contact_threshold = contact_threshold_m
        self.exit_threshold = exit_threshold_m
        self.cooldown_frames = cooldown_frames

        # Per-finger state: {(hand, finger): {"state": str, "cooldown": int}}
        self.states = {}

    def _key(self, handedness, finger):
        return (handedness, finger)

    def update(self, handedness, finger, distance_to_surface_m):
        """
        Update contact state for one fingertip.

        Args:
            handedness: "Left" or "Right"
            finger: "thumb", "index", "middle", "ring", "pinky"
            distance_to_surface_m: depth distance from fingertip to surface plane (meters)

        Returns:
            state: "contact" or "hover"
        """
        key = self._key(handedness, finger)

        if key not in self.states:
            self.states[key] = {"state": "hover", "cooldown": 0}

        s = self.states[key]

        # Decrease cooldown
        if s["cooldown"] > 0:
            s["cooldown"] -= 1

        if distance_to_surface_m <= 0:
            # Invalid depth — keep previous state
            return s["state"]

        if s["state"] == "hover":
            if distance_to_surface_m <= self.contact_threshold:
                s["state"] = "contact"
                s["cooldown"] = self.cooldown_frames
        elif s["state"] == "contact":
            if distance_to_surface_m >= self.exit_threshold and s["cooldown"] == 0:
                s["state"] = "hover"

        return s["state"]

    def reset(self):
        self.states.clear()


def compute_surface_depth_at_pixel(plane_coefficients, px, py):
    """
    Compute the expected surface depth at pixel (px, py) using the calibrated plane.

    Plane equation: a*x + b*y + c*z + d = 0
    Solve for z: z = -(a*x + b*y + d) / c
    """
    a, b, c, d = plane_coefficients
    if abs(c) < 1e-10:
        return 0.0
    z = -(a * px + b * py + d) / c
    return z


def process_session(session_dir, calib_data, contact_threshold, exit_threshold,
                    cooldown_frames):
    """Generate contact labels for one recording session."""
    session_dir = Path(session_dir)
    annotations_dir = session_dir / "annotations"
    labels_dir = session_dir / "labels"

    if not annotations_dir.exists():
        print(f"  SKIP: No annotations/ in {session_dir.name}")
        return None

    labels_dir.mkdir(exist_ok=True)

    plane_coeff = calib_data["plane_coefficients"]
    avg_surface_depth = calib_data["avg_surface_depth_m"]

    detector = HysteresisContactDetector(
        contact_threshold_m=contact_threshold,
        exit_threshold_m=exit_threshold,
        cooldown_frames=cooldown_frames,
    )

    annotation_files = sorted(annotations_dir.glob("*.json"))
    stats = defaultdict(int)
    frame_labels = []

    for ann_file in annotation_files:
        with open(ann_file) as f:
            ann = json.load(f)

        frame_label = {
            "frame_id": ann["frame_id"],
            "num_hands": ann["num_hands"],
            "hands": [],
        }

        for hand in ann.get("hands", []):
            handedness = hand.get("handedness", "Unknown")
            fingertips_label = {}

            for finger, depth_m in hand.get("fingertip_depths_m", {}).items():
                # Get fingertip pixel position
                fpx = hand.get("fingertip_pixels", {}).get(finger, [0, 0])
                px, py = fpx[0], fpx[1]

                # Compute surface depth at this pixel using plane equation
                surface_depth = compute_surface_depth_at_pixel(plane_coeff, px, py)

                # Fallback to average surface depth if plane gives bad values
                if surface_depth <= 0 or surface_depth > 1.0:
                    surface_depth = avg_surface_depth

                # Distance: positive means above surface (hovering)
                # For overhead (90°) camera: surface is farther from camera,
                # so fingertip_depth < surface_depth means finger is between camera and surface
                distance_m = surface_depth - depth_m

                # Clamp negative distances (finger below surface = definitely contact)
                distance_m = max(distance_m, 0.0)

                state = detector.update(handedness, finger, distance_m)
                distance_mm = distance_m * 1000.0

                fingertips_label[finger] = {
                    "depth_m": round(depth_m, 4),
                    "surface_depth_m": round(surface_depth, 4),
                    "distance_mm": round(distance_mm, 1),
                    "state": state,
                }

                stats[f"{state}"] += 1

            frame_label["hands"].append({
                "handedness": handedness,
                "fingertips": fingertips_label,
            })

        # Save label
        with open(labels_dir / f"{ann['frame_id']:06d}.json", "w") as f:
            json.dump(frame_label, f, indent=2)

        frame_labels.append(frame_label)

    # Summary
    total = stats["contact"] + stats["hover"]
    contact_ratio = stats["contact"] / max(total, 1)

    return {
        "session": session_dir.name,
        "frames": len(annotation_files),
        "total_fingertip_samples": total,
        "contact_count": stats["contact"],
        "hover_count": stats["hover"],
        "contact_ratio": contact_ratio,
    }


def find_calibration(session_dir, calibrations_dir):
    """Find matching calibration file for a session."""
    session_dir = Path(session_dir)
    meta_file = session_dir / "metadata.json"

    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

        # Check if calibration is embedded in metadata
        if "surface_calibration" in meta:
            return meta["surface_calibration"]

        # Try to find calibration file by angle + surface
        session_info = meta.get("session", {})
        angle = session_info.get("angle_degrees")
        surface = session_info.get("surface")

        if angle and surface:
            calib_file = calibrations_dir / f"calib_angle{angle}_{surface}.json"
            if calib_file.exists():
                with open(calib_file) as f:
                    return json.load(f)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate contact/hover labels from annotations + calibration"
    )
    parser.add_argument("--session-dir", type=str, default=None,
                        help="Single session directory to process")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root data directory (processes all sessions)")
    parser.add_argument("--contact-threshold", type=float, default=0.0045,
                        help="Contact entry threshold in meters (default: 4.5mm)")
    parser.add_argument("--exit-threshold", type=float, default=0.006,
                        help="Contact exit threshold in meters (default: 6.0mm)")
    parser.add_argument("--cooldown-ms", type=float, default=450,
                        help="Cooldown period in milliseconds (default: 450ms)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS for cooldown frame conversion")
    args = parser.parse_args()

    cooldown_frames = int(args.cooldown_ms / 1000.0 * args.fps)

    print("=" * 70)
    print("Contact/Hover Label Generation")
    print("=" * 70)
    print(f"Contact threshold: {args.contact_threshold * 1000:.1f}mm")
    print(f"Exit threshold:    {args.exit_threshold * 1000:.1f}mm")
    print(f"Cooldown:          {args.cooldown_ms:.0f}ms ({cooldown_frames} frames @ {args.fps}fps)")
    print()

    if args.session_dir:
        sessions = [Path(args.session_dir)]
        calibrations_dir = sessions[0].parent.parent / "calibrations"
    elif args.data_root:
        data_root = Path(args.data_root)
        sessions = sorted(data_root.glob("P*/P*_angle*"))
        if not sessions:
            sessions = sorted(data_root.glob("P*_angle*"))
        calibrations_dir = data_root / "calibrations"
    else:
        print("ERROR: Provide --session-dir or --data-root")
        return

    print(f"Found {len(sessions)} session(s)")
    print(f"Calibrations dir: {calibrations_dir}")
    print()

    all_results = []

    for session in sessions:
        calib = find_calibration(session, calibrations_dir)
        if calib is None:
            print(f"  SKIP: No calibration for {session.name}")
            continue

        print(f"Processing: {session.name}")
        result = process_session(
            session, calib,
            args.contact_threshold, args.exit_threshold, cooldown_frames
        )

        if result:
            all_results.append(result)
            print(f"  {result['frames']} frames, "
                  f"contact={result['contact_count']}, hover={result['hover_count']}, "
                  f"ratio={result['contact_ratio']:.1%}")

    if all_results:
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        total_frames = sum(r["frames"] for r in all_results)
        total_contact = sum(r["contact_count"] for r in all_results)
        total_hover = sum(r["hover_count"] for r in all_results)
        total_samples = total_contact + total_hover
        print(f"Sessions processed: {len(all_results)}")
        print(f"Total frames:       {total_frames}")
        print(f"Fingertip samples:  {total_samples}")
        print(f"Contact:            {total_contact} ({total_contact / max(total_samples, 1):.1%})")
        print(f"Hover:              {total_hover} ({total_hover / max(total_samples, 1):.1%})")
        print()
        print("Labels written to labels/ subdirectory of each session.")


if __name__ == "__main__":
    main()
