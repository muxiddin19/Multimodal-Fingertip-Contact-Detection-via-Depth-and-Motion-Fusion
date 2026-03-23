"""
Gaussian Touch Model for VR Keyboard
=====================================
Models fingertip position as a bivariate Gaussian to compute P(key | touch).
Handles tracking uncertainty (~1cm) by weighting nearby keys probabilistically
instead of hard nearest-center selection.

Reference: TouchInsight (UIST 2024)
"""

import numpy as np
from typing import Dict, List, Tuple


class GaussianTouchModel:
    """
    Bivariate Gaussian touch probability model.

    For each detected contact at (x, y), computes the probability that
    each key was the intended target, based on distance from key center
    weighted by a Gaussian kernel.
    """

    def __init__(self, keys: List[Dict], sigma_x: float = 20.0, sigma_y: float = 15.0):
        """
        Args:
            keys: List of key dicts with 'name', 'corners', 'center'
            sigma_x: Horizontal uncertainty in pixels (default 20)
            sigma_y: Vertical uncertainty in pixels (default 15)
        """
        self.keys = keys
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        # Pre-compute key centers as arrays for vectorized computation
        self.key_names = [k['name'] for k in keys]
        self.centers = np.array([k['center'] for k in keys], dtype=np.float64)  # (N, 2)

        # Pre-compute inverse variance
        self._inv_sx2 = 1.0 / (sigma_x * sigma_x)
        self._inv_sy2 = 1.0 / (sigma_y * sigma_y)

        # Cutoff distance (3 sigma) for early pruning
        self._cutoff_x = 3.0 * sigma_x
        self._cutoff_y = 3.0 * sigma_y

    def compute_key_probabilities(self, touch_x: float, touch_y: float) -> Dict[str, float]:
        """
        Compute P(key | touch) for all keys using bivariate Gaussian.

        Returns dict mapping key name -> probability (sums to ~1.0).
        """
        # Vectorized distance computation
        dx = self.centers[:, 0] - touch_x
        dy = self.centers[:, 1] - touch_y

        # Log probabilities (unnormalized)
        log_probs = -0.5 * (dx * dx * self._inv_sx2 + dy * dy * self._inv_sy2)

        # Log-sum-exp normalization for numerical stability
        max_log = np.max(log_probs)
        exp_probs = np.exp(log_probs - max_log)
        total = np.sum(exp_probs)

        if total > 0:
            probs = exp_probs / total
        else:
            probs = np.zeros(len(self.keys))

        return {name: float(p) for name, p in zip(self.key_names, probs)}

    def get_top_k(self, touch_x: float, touch_y: float, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k keys by probability, sorted descending."""
        probs = self.compute_key_probabilities(touch_x, touch_y)
        sorted_keys = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_keys[:k]
