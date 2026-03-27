"""
Fine-tune the TCN Tap Detector on live camera data.

Loads the pre-trained TCN (trained on CVPR 2026 dataset) and fine-tunes
it on collected live session data to close the domain gap.
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.tcn_tap_detector import TCNModel


class LiveSessionDataset(Dataset):
    """Load sliding windows from live-collected session data."""

    def __init__(self, data_dir: str, window_size: int = 30, stride: int = 3):
        self._windows = []
        sessions = sorted(glob.glob(os.path.join(data_dir, 'session_*')))

        total_frames = 0
        total_contacts = 0

        for sess_dir in sessions:
            lm_path = os.path.join(sess_dir, 'landmarks.json')
            if not os.path.exists(lm_path):
                continue

            with open(lm_path) as f:
                data = json.load(f)

            if len(data) < window_size:
                continue

            # Extract features and labels
            features = []
            labels = []
            for frame in data:
                feat = frame['features']
                if len(feat) != 47:
                    continue
                features.append(feat)
                # Binary label: contact for index finger (the primary typing finger)
                labels.append(float(frame['label']))

            if len(features) < window_size:
                continue

            features = np.array(features, dtype=np.float32)  # (N, 47)
            labels = np.array(labels, dtype=np.float32)       # (N,)

            total_frames += len(features)
            total_contacts += int(labels.sum())

            # Create sliding windows
            for start in range(0, len(features) - window_size, stride):
                end = start + window_size
                feat_window = features[start:end].T  # (47, W)
                # Only label index finger (channel 1), others are 0
                lab_window = np.zeros((5, window_size), dtype=np.float32)
                lab_window[1] = labels[start:end]  # index finger only
                self._windows.append((feat_window, lab_window))

        print(f"  Loaded: {total_frames} frames, {total_contacts} contacts")
        print(f"  Windows: {len(self._windows)}")

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        feat, lab = self._windows[idx]
        return torch.from_numpy(feat), torch.from_numpy(lab)


def finetune(
    pretrained_path: str = 'src/tcn_tap_model.pth',
    data_dir: str = 'tcn_training_data',
    epochs: int = 50,
    lr: float = 0.0003,  # lower LR for fine-tuning
    batch_size: int = 32,
    save_path: str = 'src/tcn_tap_model_finetuned.pth',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"  TCN FINE-TUNING ON LIVE CAMERA DATA")
    print(f"{'='*60}")

    # Load pre-trained model
    # Auto-detect hidden dim from checkpoint, default 512
    hidden_dim = 512
    if os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        hidden_dim = ckpt['model_state_dict']['input_proj.weight'].shape[0]
        print(f"  Detected hidden_dim={hidden_dim} from checkpoint")
    model = TCNModel(input_dim=47, hidden_dim=hidden_dim, num_layers=3).to(device)
    if os.path.isfile(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded pre-trained model from {pretrained_path}")
    else:
        print(f"  No pre-trained model found, training from scratch")

    # Load live data
    print(f"\n  Loading live session data from {data_dir}/")
    dataset = LiveSessionDataset(data_dir, window_size=30, stride=3)

    if len(dataset) == 0:
        print("[ERROR] No training data!")
        return

    # Split
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {n_train}, Val: {n_val}")
    print(f"  Device: {device}")

    # Compute class weights
    all_labels = []
    for i in range(len(dataset)):
        all_labels.append(dataset[i][1].numpy())
    all_labels = np.array(all_labels)
    pos_count = all_labels.sum()
    neg_count = all_labels.size - pos_count
    pos_weight_val = neg_count / (pos_count + 1)
    pos_weight = torch.tensor([pos_weight_val] * 5, dtype=torch.float32).to(device)
    print(f"  Pos weight: {pos_weight_val:.1f} (contact ratio: {pos_count/all_labels.size*100:.1f}%)")
    print(f"{'='*60}\n")

    # Fine-tune
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(1))

    best_val_f1 = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for feat, lab in train_loader:
            feat = feat.to(device)
            lab = lab.to(device)
            logits = model(feat)
            loss = criterion(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        # Validate
        model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for feat, lab in val_loader:
                feat = feat.to(device)
                lab = lab.to(device)
                preds = (torch.sigmoid(model(feat)) > 0.5).float()
                # Use index finger (channel 1) for metrics
                tp += ((preds[:, 1] == 1) & (lab[:, 1] == 1)).sum().item()
                fp += ((preds[:, 1] == 1) & (lab[:, 1] == 0)).sum().item()
                fn += ((preds[:, 1] == 0) & (lab[:, 1] == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={train_loss:.4f} | "
                  f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'is_trained': True,
            }, save_path)

    print(f"\n  Best val F1 (index finger): {best_val_f1:.3f}")
    print(f"  Model saved to: {save_path}")
    print(f"\n  To use: python keyboard.py --tcn {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default='src/tcn_tap_model.pth')
    parser.add_argument('--data', default='tcn_training_data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save', default='src/tcn_tap_model_finetuned.pth')
    args = parser.parse_args()

    finetune(args.pretrained, args.data, args.epochs, args.lr, save_path=args.save)
