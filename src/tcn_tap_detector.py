"""
TCN Tap Detector for VR Keyboard
==================================
Temporal Convolutional Network that maps hand pose sequences
directly to tap probabilities, replacing threshold-based detection.

Architecture:
- Input: 63 features per frame (21 MediaPipe landmarks × 3 coords)
- 3-layer 1D TCN with causal convolutions
- Output: per-frame tap probability (binary classification)
- Real-time inference at 30fps

References:
- Decoding Surface Touch Typing (Meta, UIST 2020): 73 WPM
- StegoType (Meta, UIST 2024): 75 WPM
- TouchInsight (ETH/Meta, UIST 2024): 37 WPM

Training:
    detector = TCNTapDetector()
    detector.train_from_sessions('training_data/')
    detector.save('tcn_tap_model.pth')

Inference:
    detector = TCNTapDetector.load('tcn_tap_model.pth')
    # Each frame:
    tap_prob = detector.predict_frame(landmarks_63d)
"""

import os
import time
import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal convolution: only looks at past frames, not future."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN block with residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return out + residual


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for tap detection.

    Input: (batch, 63, seq_len) — 21 landmarks × 3 coords
    Output: (batch, 1, seq_len) — per-frame tap probability
    """

    def __init__(self, input_dim: int = 63, hidden_dim: int = 64, num_layers: int = 3, kernel_size: int = 5):
        super().__init__()

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        # TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim, seq_len)
        Returns:
            (batch, 1, seq_len) — logits (apply sigmoid for probability)
        """
        out = F.relu(self.input_proj(x))
        for block in self.blocks:
            out = block(out)
        return self.output_proj(out)


class TCNTapDetector:
    """
    Real-time tap detector using a small TCN.

    Maintains a sliding window of recent frames and runs inference
    on each new frame to produce a tap probability.
    """

    # MediaPipe landmark count
    NUM_LANDMARKS = 21
    INPUT_DIM = NUM_LANDMARKS * 3  # x, y, z per landmark
    WINDOW_SIZE = 30  # ~1 second at 30fps

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.model = TCNModel(
            input_dim=self.INPUT_DIM,
            hidden_dim=64,
            num_layers=3,
            kernel_size=5
        ).to(device)

        # Sliding window buffer
        self._buffer = deque(maxlen=self.WINDOW_SIZE)
        self._is_trained = False

        # Try to load pre-trained model
        if model_path and os.path.isfile(model_path):
            self.load(model_path)

    def extract_features(self, hand_landmarks, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract 63-dim feature vector from MediaPipe hand landmarks.

        Args:
            hand_landmarks: MediaPipe hand landmarks object
            image_shape: (height, width) for normalization

        Returns:
            numpy array of shape (63,) — normalized landmark positions
        """
        h, w = image_shape
        features = np.zeros(self.INPUT_DIM, dtype=np.float32)

        for i, landmark in enumerate(hand_landmarks.landmark):
            features[i * 3] = landmark.x  # already normalized 0-1
            features[i * 3 + 1] = landmark.y
            features[i * 3 + 2] = landmark.z  # relative depth

        return features

    def update_buffer(self, features: np.ndarray):
        """Add a frame's features to the sliding window."""
        self._buffer.append(features)

    def predict_frame(self, features: Optional[np.ndarray] = None) -> float:
        """
        Predict tap probability for the current frame.

        Args:
            features: 63-dim feature vector (if None, uses last buffered frame)

        Returns:
            Tap probability (0-1). Returns 0.0 if model not trained or buffer too small.
        """
        if not self._is_trained:
            return 0.0

        if features is not None:
            self.update_buffer(features)

        if len(self._buffer) < 5:  # need at least 5 frames
            return 0.0

        # Prepare input tensor: (1, 63, seq_len)
        buf = np.array(list(self._buffer), dtype=np.float32)  # (seq_len, 63)
        x = torch.from_numpy(buf.T).unsqueeze(0).to(self.device)  # (1, 63, seq_len)

        with torch.no_grad():
            logits = self.model(x)  # (1, 1, seq_len)
            prob = torch.sigmoid(logits[0, 0, -1]).item()  # last frame probability

        return prob

    def train_from_csv(
        self,
        csv_path: str,
        landmarks_dir: str,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        save_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Train the TCN from collected data.

        Args:
            csv_path: Path to depth_velocity_log.csv (with label column)
            landmarks_dir: Directory containing per-frame landmark files
            epochs: Training epochs
            lr: Learning rate
            batch_size: Batch size
            save_path: Where to save the trained model
            verbose: Print training progress
        """
        # This is a scaffold — full implementation requires:
        # 1. Collecting landmark data during typing sessions
        # 2. Saving per-frame landmarks alongside the CSV labels
        # 3. Creating sliding window training samples
        # 4. Training with binary cross-entropy loss

        if verbose:
            print("[TCN] Training scaffold - collecting data format:")
            print(f"  CSV: {csv_path}")
            print(f"  Landmarks dir: {landmarks_dir}")
            print("  To collect training data, run with --collect-landmarks flag")
            print("  Training will be available once sufficient data is collected")

    def save(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': self._is_trained,
        }, path)
        print(f"[TCN] Model saved to {path}")

    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._is_trained = checkpoint.get('is_trained', True)
        self.model.eval()
        print(f"[TCN] Model loaded from {path}")

    def reset(self):
        """Clear the frame buffer (e.g., between sessions)."""
        self._buffer.clear()


def collect_landmarks_for_training(hand_landmarks, frame_idx: int, label: int,
                                    output_dir: str, image_shape: Tuple[int, int]):
    """
    Utility to save landmark data during typing sessions for later training.

    Call this every frame during a labeled typing session.
    Creates a .npy file per frame with the 63-dim feature vector + label.
    """
    os.makedirs(output_dir, exist_ok=True)

    detector = TCNTapDetector()
    features = detector.extract_features(hand_landmarks, image_shape)

    data = np.concatenate([features, [float(label)]])  # 64-dim: 63 features + 1 label
    np.save(os.path.join(output_dir, f"frame_{frame_idx:06d}.npy"), data)
