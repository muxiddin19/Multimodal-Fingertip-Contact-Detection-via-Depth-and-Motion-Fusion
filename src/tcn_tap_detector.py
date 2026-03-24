"""
TCN Tap Detector for VR Keyboard
==================================
Temporal Convolutional Network that maps hand pose sequences
directly to per-fingertip contact probabilities.

Trained on the CVPR 2026 dataset (P01-P18, 102K+ labeled frames)
with per-fingertip contact/hover annotations.

Architecture:
- Input: 47 features per frame (21 landmarks × 2 coords + 5 fingertip depths)
- 3-layer 1D TCN with causal convolutions (64 channels)
- Output: 5 per-fingertip contact probabilities

References:
- Decoding Surface Touch Typing (Meta, UIST 2020): 73 WPM
- StegoType (Meta, UIST 2024): 75 WPM
"""

import os
import json
import glob
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ======================== Model Architecture ========================

class CausalConv1d(nn.Module):
    """Causal convolution: only looks at past frames."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Single TCN block with residual connection."""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.dropout(F.relu(self.bn2(self.conv2(out))))
        return out + x


class TCNModel(nn.Module):
    """
    TCN for per-fingertip contact detection.

    Input:  (batch, 47, seq_len)
    Output: (batch, 5, seq_len) — 5 fingertip contact logits
    """
    def __init__(self, input_dim=47, hidden_dim=64, num_layers=3,
                 kernel_size=5, num_fingers=5):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, num_fingers, 1)

    def forward(self, x):
        out = F.relu(self.input_proj(x))
        for block in self.blocks:
            out = block(out)
        return self.output_proj(out)


# ======================== Dataset ========================

FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']


def load_frame_features(ann_path: str) -> Optional[np.ndarray]:
    """
    Extract 47-dim feature vector from an annotation JSON file.
    Features: 21 landmarks × 2 (x,y normalized) + 5 fingertip depths
    Returns None if no hand detected.
    """
    with open(ann_path, 'r') as f:
        ann = json.load(f)

    if ann.get('num_hands', 0) == 0 or not ann.get('hands'):
        return None

    hand = ann['hands'][0]  # use first hand
    landmarks = hand.get('landmarks_px', [])
    if len(landmarks) != 21:
        return None

    # Normalize landmarks to 0-1 range (assuming 640x480)
    features = []
    for lm in landmarks:
        features.append(lm[0] / 640.0)
        features.append(lm[1] / 480.0)

    # Add fingertip depths
    depths = hand.get('fingertip_depths_m', {})
    for finger in FINGER_NAMES:
        features.append(depths.get(finger, 0.3))

    return np.array(features, dtype=np.float32)  # (47,)


def load_frame_labels(label_path: str) -> Optional[np.ndarray]:
    """
    Extract 5-dim binary label vector from a label JSON file.
    Returns None if no hand detected.
    """
    with open(label_path, 'r') as f:
        lab = json.load(f)

    if lab.get('num_hands', 0) == 0 or not lab.get('hands'):
        return None

    hand = lab['hands'][0]
    fingertips = hand.get('fingertips', {})

    labels = []
    for finger in FINGER_NAMES:
        ft = fingertips.get(finger, {})
        state = ft.get('state', 'hover')
        labels.append(1.0 if state == 'contact' else 0.0)

    return np.array(labels, dtype=np.float32)  # (5,)


class TapDataset(Dataset):
    """
    Dataset of sliding windows from the CVPR 2026 typing sessions.

    Each sample is a (features, labels) pair:
    - features: (47, window_size) tensor
    - labels: (5, window_size) tensor (per-fingertip contact)
    """
    def __init__(self, data_root: str, window_size: int = 30,
                 stride: int = 5, participants: Optional[List[str]] = None):
        self.window_size = window_size
        self.samples = []  # list of (features_array, labels_array) per session

        # Find all sessions
        sessions = []
        for p_dir in sorted(glob.glob(os.path.join(data_root, 'P*'))):
            p_name = os.path.basename(p_dir)
            if participants and p_name not in participants:
                continue
            for sess_dir in sorted(glob.glob(os.path.join(p_dir, f'{p_name}_*'))):
                ann_dir = os.path.join(sess_dir, 'annotations')
                lab_dir = os.path.join(sess_dir, 'labels')
                if os.path.isdir(ann_dir) and os.path.isdir(lab_dir):
                    sessions.append((ann_dir, lab_dir))

        print(f"[TCN Dataset] Found {len(sessions)} sessions")

        # Load all sessions
        self._windows = []
        for ann_dir, lab_dir in sessions:
            frames = sorted(glob.glob(os.path.join(ann_dir, '*.json')))
            sess_features = []
            sess_labels = []

            for frame_path in frames:
                frame_id = os.path.splitext(os.path.basename(frame_path))[0]
                label_path = os.path.join(lab_dir, f'{frame_id}.json')

                if not os.path.exists(label_path):
                    continue

                feat = load_frame_features(frame_path)
                lab = load_frame_labels(label_path)

                if feat is not None and lab is not None:
                    sess_features.append(feat)
                    sess_labels.append(lab)

            if len(sess_features) < window_size:
                continue

            sess_features = np.array(sess_features)  # (N, 47)
            sess_labels = np.array(sess_labels)      # (N, 5)

            # Create sliding windows
            for start in range(0, len(sess_features) - window_size, stride):
                end = start + window_size
                self._windows.append((
                    sess_features[start:end].T,  # (47, W)
                    sess_labels[start:end].T      # (5, W)
                ))

        print(f"[TCN Dataset] Created {len(self._windows)} training windows")

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        feat, lab = self._windows[idx]
        return torch.from_numpy(feat), torch.from_numpy(lab)


# ======================== Detector ========================

class TCNTapDetector:
    """
    Real-time tap detector using a small TCN trained on the CVPR 2026 dataset.
    """

    INPUT_DIM = 47  # 21 landmarks × 2 + 5 depths
    WINDOW_SIZE = 30

    def __init__(self, model_path: Optional[str] = None, device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = TCNModel(input_dim=self.INPUT_DIM).to(self.device)
        self._buffer = deque(maxlen=self.WINDOW_SIZE)
        self._is_trained = False

        if model_path and os.path.isfile(model_path):
            self.load(model_path)

    def extract_features_from_mediapipe(self, hand_landmarks, image_shape,
                                         depth_map=None) -> np.ndarray:
        """Extract 47-dim features from live MediaPipe landmarks."""
        h, w = image_shape
        features = []

        # 21 landmarks × 2 coords (normalized)
        for lm in hand_landmarks.landmark:
            features.append(lm.x)  # already 0-1
            features.append(lm.y)

        # 5 fingertip depths (from depth map if available)
        tip_ids = [4, 8, 12, 16, 20]
        for tip_id in tip_ids:
            if depth_map is not None:
                lm = hand_landmarks.landmark[tip_id]
                px = int(lm.x * w)
                py = int(lm.y * h)
                px = max(0, min(px, w - 1))
                py = max(0, min(py, h - 1))
                features.append(float(depth_map[py, px]))
            else:
                features.append(hand_landmarks.landmark[tip_id].z)

        return np.array(features, dtype=np.float32)

    def predict_frame(self, features: np.ndarray) -> np.ndarray:
        """
        Predict per-fingertip contact probabilities.

        Returns: array of 5 probabilities [thumb, index, middle, ring, pinky]
        """
        self._buffer.append(features)

        if not self._is_trained or len(self._buffer) < 5:
            return np.zeros(5)

        buf = np.array(list(self._buffer), dtype=np.float32)
        x = torch.from_numpy(buf.T).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits[0, :, -1]).cpu().numpy()

        return probs  # (5,) — one probability per fingertip

    def train(self, data_root: str, epochs: int = 30, lr: float = 0.001,
              batch_size: int = 64, save_path: str = 'src/tcn_tap_model.pth',
              val_split: float = 0.15):
        """
        Train the TCN on the CVPR 2026 dataset.

        Args:
            data_root: Path to full_data/ directory
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            save_path: Where to save the trained model
            val_split: Fraction of data for validation
        """
        print(f"\n{'='*60}")
        print(f"  TCN TAP DETECTOR TRAINING")
        print(f"{'='*60}")

        # Load dataset
        dataset = TapDataset(data_root, window_size=self.WINDOW_SIZE, stride=5)
        if len(dataset) == 0:
            print("[ERROR] No training data found!")
            return

        # Split train/val
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=0)

        print(f"  Train samples: {n_train}")
        print(f"  Val samples: {n_val}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        # Compute class weights (contact is rarer than hover)
        all_labels = np.array([dataset[i][1].numpy() for i in range(len(dataset))])
        pos_count = all_labels.sum(axis=(0, 2))  # per finger
        neg_count = all_labels.shape[0] * all_labels.shape[2] - pos_count
        pos_weight = torch.from_numpy(neg_count / (pos_count + 1)).float().to(self.device)
        print(f"  Pos weights per finger: {pos_weight.cpu().numpy().round(2)}")

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(1))

        best_val_f1 = 0.0
        self.model.train()

        for epoch in range(epochs):
            # Train
            train_loss = 0
            for feat, lab in train_loader:
                feat = feat.to(self.device)
                lab = lab.to(self.device)

                logits = self.model(feat)
                loss = criterion(logits, lab)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            # Validate
            self.model.eval()
            val_tp = np.zeros(5)
            val_fp = np.zeros(5)
            val_fn = np.zeros(5)

            with torch.no_grad():
                for feat, lab in val_loader:
                    feat = feat.to(self.device)
                    lab = lab.to(self.device)

                    logits = self.model(feat)
                    preds = (torch.sigmoid(logits) > 0.5).float()

                    for f in range(5):
                        val_tp[f] += ((preds[:, f] == 1) & (lab[:, f] == 1)).sum().item()
                        val_fp[f] += ((preds[:, f] == 1) & (lab[:, f] == 0)).sum().item()
                        val_fn[f] += ((preds[:, f] == 0) & (lab[:, f] == 1)).sum().item()

            self.model.train()

            # Per-finger F1
            precision = val_tp / (val_tp + val_fp + 1e-8)
            recall = val_tp / (val_tp + val_fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            mean_f1 = f1.mean()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                finger_f1 = ' '.join(f'{FINGER_NAMES[i]}:{f1[i]:.3f}' for i in range(5))
                print(f"  Epoch {epoch+1:3d}/{epochs} | loss={train_loss:.4f} | "
                      f"val_F1={mean_f1:.3f} | {finger_f1}")

            if mean_f1 > best_val_f1:
                best_val_f1 = mean_f1
                self.save(save_path)

        print(f"\n  Best val F1: {best_val_f1:.3f}")
        print(f"  Model saved to: {save_path}")
        self._is_trained = True

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': True,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._is_trained = checkpoint.get('is_trained', True)
        self.model.eval()
        print(f"[TCN] Loaded model from {path}")

    def reset(self):
        self._buffer.clear()


# ======================== CLI Training ========================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train TCN Tap Detector')
    parser.add_argument('--data', type=str, default=r'D:\VoiceAI\CVPR2026\data1\full_data',
                        help='Path to full_data directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', type=str, default='src/tcn_tap_model.pth')
    args = parser.parse_args()

    detector = TCNTapDetector()
    detector.train(args.data, epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, save_path=args.save)
