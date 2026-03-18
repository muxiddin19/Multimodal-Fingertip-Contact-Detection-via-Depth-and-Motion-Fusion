"""
Prepare custom dataset from D405 recordings for DA2 fine-tuning.

Takes a directory of RGB + depth PNGs from Intel RealSense D405 and:
  1. Validates RGB-depth pairs exist and are aligned
  2. Runs quality filtering (depth coverage, hand detection, blur)
  3. Generates train/val/test splits (stratified by participant ID)
  4. Writes split files in the format expected by CustomDepthDataset

Usage:
  python prepare_custom_dataset.py \
    --data-dir /path/to/recordings \
    --output-dir dataset/splits/custom_v2 \
    --participant-id P01

Directory structure expected:
  data_dir/
    P01/
      rgb/000000.png, 000001.png, ...
      depth/000000.png, 000001.png, ...
    P02/
      rgb/...
      depth/...
    ...

Or flat structure (single participant):
  data_dir/
    rgb/000000.png, ...
    depth/000000.png, ...
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def check_depth_quality(depth_path, min_valid_ratio=0.80, roi_fraction=0.6):
    """Check if depth frame has sufficient valid coverage in the ROI."""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return False, "cannot_load"

    h, w = depth.shape[:2]
    # ROI: center 60% of the image (where hands typically are)
    margin_h = int(h * (1 - roi_fraction) / 2)
    margin_w = int(w * (1 - roi_fraction) / 2)
    roi = depth[margin_h:h-margin_h, margin_w:w-margin_w]

    valid = (roi > 0).sum()
    total = roi.size
    ratio = valid / total

    if ratio < min_valid_ratio:
        return False, f"low_coverage_{ratio:.2f}"
    return True, "ok"


def check_blur(rgb_path, threshold=100.0):
    """Check if image is too blurry (Laplacian variance)."""
    img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "cannot_load"
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    if variance < threshold:
        return False, f"blurry_{variance:.1f}"
    return True, "ok"


def check_hands_detected(rgb_path):
    """Check if MediaPipe can detect hands with all 21 landmarks."""
    try:
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        img = cv2.imread(rgb_path)
        if img is None:
            return False, "cannot_load"
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        hands.close()
        if results.multi_hand_landmarks:
            return True, f"hands_{len(results.multi_hand_landmarks)}"
        return False, "no_hands"
    except ImportError:
        # MediaPipe not available, skip this check
        return True, "skipped"


# def discover_participants(data_dir):
#     """Discover participant directories or detect flat structure."""
#     participants = {}
#     data_path = Path(data_dir)

#     # Check for participant subdirectories (P01, P02, ...)
#     for subdir in sorted(data_path.iterdir()):
#         if subdir.is_dir() and (subdir / 'rgb').exists() and (subdir / 'depth').exists():
#             pid = subdir.name
#             rgb_files = sorted([f for f in (subdir / 'rgb').iterdir() if f.suffix == '.png'])
#             depth_files = sorted([f for f in (subdir / 'depth').iterdir() if f.suffix == '.png'])
#             if rgb_files and depth_files:
#                 participants[pid] = {'rgb_dir': subdir / 'rgb', 'depth_dir': subdir / 'depth'}

#     # Flat structure
#     if not participants and (data_path / 'rgb').exists():
#         participants['P00'] = {'rgb_dir': data_path / 'rgb', 'depth_dir': data_path / 'depth'}

#     return participants

# def discover_participants(data_dir):
#     participants = {}
#     data_path = Path(data_dir)

#     for subdir in sorted(data_path.iterdir()):
#         if not subdir.is_dir():
#             continue
#         # Direct structure: full_data/P01/rgb/
#         if (subdir / 'rgb').exists() and (subdir / 'depth').exists():
#             pid = subdir.name
#             rgb_files = sorted([f for f in (subdir / 'rgb').iterdir() if f.suffix == '.png'])
#             depth_files = sorted([f for f in (subdir / 'depth').iterdir() if f.suffix == '.png'])
#             if rgb_files and depth_files:
#                 participants[pid] = {'rgb_dir': subdir / 'rgb', 'depth_dir': subdir / 'depth'}
#         else:
#             # One extra level: full_data/P01/P01_session.../rgb/
#             for session_dir in sorted(subdir.iterdir()):
#                 if session_dir.is_dir() and (session_dir / 'rgb').exists() and (session_dir / 'depth').exists():
#                     pid = session_dir.name  # or subdir.name if you want P01 as key
#                     rgb_files = sorted([f for f in (session_dir / 'rgb').iterdir() if f.suffix == '.png'])
#                     depth_files = sorted([f for f in (session_dir / 'depth').iterdir() if f.suffix == '.png'])
#                     if rgb_files and depth_files:
#                         participants[pid] = {'rgb_dir': session_dir / 'rgb', 'depth_dir': session_dir / 'depth'}

#     if not participants and (data_path / 'rgb').exists():
#         participants['P00'] = {'rgb_dir': data_path / 'rgb', 'depth_dir': data_path / 'depth'}

#     return participants
def discover_participants(data_dir):
    participants = {}
    data_path = Path(data_dir)

    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir():
            continue
        if (subdir / 'rgb').exists() and (subdir / 'depth').exists():
            pid = subdir.name
            participants[pid] = {'rgb_dir': subdir / 'rgb', 'depth_dir': subdir / 'depth'}
        else:
            for session_dir in sorted(subdir.iterdir()):
                if session_dir.is_dir() and (session_dir / 'rgb').exists() and (session_dir / 'depth').exists():
                    pid = subdir.name  # group by participant, not session
                    if pid not in participants:
                        participants[pid] = {'sessions': []}
                    participants[pid]['sessions'].append(session_dir)

    return participants
def find_pairs(rgb_dir, depth_dir):
    """Find matching RGB-depth file pairs."""
    rgb_files = {f.stem: f for f in Path(rgb_dir).iterdir() if f.suffix == '.png'}
    depth_files = {f.stem: f for f in Path(depth_dir).iterdir() if f.suffix == '.png'}

    common = sorted(set(rgb_files.keys()) & set(depth_files.keys()))
    pairs = [(str(rgb_files[k]), str(depth_files[k])) for k in common]
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Root directory of recordings')
    parser.add_argument('--output-dir', default='dataset/splits/custom_v2',
                        help='Output directory for split files')
    parser.add_argument('--filter', action='store_true',
                        help='Run quality filtering (slower)')
    parser.add_argument('--min-depth-coverage', type=float, default=0.80)
    parser.add_argument('--blur-threshold', type=float, default=100.0)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("Dataset Preparation for Depth Anything V2 Fine-tuning")
    print("=" * 70)

    # Discover data
    participants = discover_participants(args.data_dir)
    if not participants:
        print(f"ERROR: No valid data found in {args.data_dir}")
        print("Expected: data_dir/P01/rgb/*.png + data_dir/P01/depth/*.png")
        print("      or: data_dir/rgb/*.png + data_dir/depth/*.png")
        return

    print(f"\nFound {len(participants)} participant(s):")

    all_pairs = {}  # pid -> [(rgb, depth), ...]
    # for pid, dirs in sorted(participants.items()):
    #     pairs = find_pairs(dirs['rgb_dir'], dirs['depth_dir'])
    #     all_pairs[pid] = pairs
    for pid, info in sorted(participants.items()):
        pairs = []
        sessions = info.get('sessions', [])
        if not sessions and 'rgb_dir' in info:
            pairs = find_pairs(info['rgb_dir'], info['depth_dir'])
        else:
            for session_dir in sessions:
                pairs.extend(find_pairs(session_dir / 'rgb', session_dir / 'depth'))
        all_pairs[pid] = pairs
        print(f"  {pid}: {len(pairs)} RGB-depth pairs")

        total = sum(len(p) for p in all_pairs.values())
        print(f"\nTotal pairs: {total}")

    # Quality filtering
    if args.filter:
        print("\nRunning quality filtering...")
        filtered_pairs = {}
        stats = defaultdict(int)

        for pid, pairs in all_pairs.items():
            good = []
            for rgb_path, depth_path in pairs:
                # Depth coverage check
                ok, reason = check_depth_quality(depth_path, args.min_depth_coverage)
                if not ok:
                    stats[reason] += 1
                    continue

                # Blur check
                ok, reason = check_blur(rgb_path, args.blur_threshold)
                if not ok:
                    stats[reason] += 1
                    continue

                good.append((rgb_path, depth_path))

            filtered_pairs[pid] = good
            print(f"  {pid}: {len(pairs)} -> {len(good)} ({len(good)/max(len(pairs),1)*100:.0f}% kept)")

        all_pairs = filtered_pairs
        total_after = sum(len(p) for p in all_pairs.values())
        print(f"\nAfter filtering: {total_after}/{total} pairs kept ({total_after/total*100:.0f}%)")
        print(f"Rejection reasons: {dict(stats)}")

    # Generate splits (stratified by participant ID)
    pids = sorted(all_pairs.keys())
    np.random.shuffle(pids)

    n = len(pids)
    n_train = max(1, int(n * args.train_ratio))
    n_val = max(1, int(n * args.val_ratio))

    if n <= 2:
        # Too few participants for proper stratification
        # Split within the participant's frames instead
        print(f"\nOnly {n} participant(s) — splitting frames within participants")
        all_frames = []
        for pid in pids:
            for pair in all_pairs[pid]:
                all_frames.append(pair)
        np.random.shuffle(all_frames)

        n_frames = len(all_frames)
        n_train_f = int(n_frames * args.train_ratio)
        n_val_f = int(n_frames * args.val_ratio)

        train_pairs = all_frames[:n_train_f]
        val_pairs = all_frames[n_train_f:n_train_f + n_val_f]
        test_pairs = all_frames[n_train_f + n_val_f:]
    else:
        train_pids = pids[:n_train]
        val_pids = pids[n_train:n_train + n_val]
        test_pids = pids[n_train + n_val:]

        print(f"\nSplit by participant ID (Paper Section 3.3):")
        print(f"  Train: {train_pids}")
        print(f"  Val:   {val_pids}")
        print(f"  Test:  {test_pids}")

        train_pairs = [p for pid in train_pids for p in all_pairs[pid]]
        val_pairs = [p for pid in val_pids for p in all_pairs[pid]]
        test_pairs = [p for pid in test_pids for p in all_pairs[pid]]

    # Write split files
    os.makedirs(args.output_dir, exist_ok=True)

    for name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        path = os.path.join(args.output_dir, f'{name}.txt')
        with open(path, 'w') as f:
            for rgb, depth in pairs:
                f.write(f'{rgb} {depth}\n')
        print(f"  {name}.txt: {len(pairs)} pairs")

    print(f"\nSplit files written to {args.output_dir}/")
    print(f"  Train: {len(train_pairs)} ({len(train_pairs)/total*100:.0f}%)")
    print(f"  Val:   {len(val_pairs)} ({len(val_pairs)/total*100:.0f}%)")
    print(f"  Test:  {len(test_pairs)} ({len(test_pairs)/total*100:.0f}%)")
    print("Done!")


if __name__ == '__main__':
    main()
