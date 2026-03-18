"""
Convert existing D405 recordings from raw sensor units to millimeters.

The old save_frame stored depth_raw (raw sensor units, depth_scale=0.0001)
as depth_u16 PNGs. CustomDepthDataset expects uint16 in millimeters.

This script reads the depth_m .npy files (float32 meters) and creates
proper uint16 mm PNGs in a new 'depth' directory.

Usage:
  python convert_existing_depth.py /path/to/session_dir
  python convert_existing_depth.py data/data/P01/P01_angle90_white_desk_typing_152436
"""

import sys
import numpy as np
import cv2
from pathlib import Path


def convert_session(session_dir):
    session_dir = Path(session_dir)
    depth_m_dir = session_dir / "depth_m"
    depth_out_dir = session_dir / "depth"

    if not depth_m_dir.exists():
        print(f"ERROR: {depth_m_dir} not found")
        return

    depth_out_dir.mkdir(exist_ok=True)

    npy_files = sorted(depth_m_dir.glob("*.npy"))
    print(f"Converting {len(npy_files)} frames: {session_dir}")

    for npy_path in npy_files:
        depth_m = np.load(npy_path)  # float32, meters
        depth_mm = (depth_m * 1000.0).astype(np.uint16)

        out_path = depth_out_dir / f"{npy_path.stem}.png"
        cv2.imwrite(str(out_path), depth_mm)

    print(f"  Done. Wrote {len(npy_files)} PNGs to {depth_out_dir}")

    # Quick sanity check on first frame
    first_npy = np.load(str(npy_files[0]))
    first_png = cv2.imread(str(depth_out_dir / f"{npy_files[0].stem}.png"), cv2.IMREAD_UNCHANGED)
    print(f"  Sanity check (frame 0):")
    print(f"    .npy range: {first_npy[first_npy > 0].min():.4f} - {first_npy.max():.4f} m")
    print(f"    .png range: {first_png[first_png > 0].min()} - {first_png.max()} mm")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: convert all sessions under data/data/
        data_root = Path("data/data")
        sessions = sorted(data_root.glob("P*/P*_angle*"))
        if not sessions:
            print("Usage: python convert_existing_depth.py <session_dir>")
            sys.exit(1)
        for s in sessions:
            convert_session(s)
    else:
        convert_session(sys.argv[1])
