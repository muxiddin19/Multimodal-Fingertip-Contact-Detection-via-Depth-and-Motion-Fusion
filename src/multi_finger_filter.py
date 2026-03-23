"""
Multi-Finger Tap Filter
========================
Combines three strategies to distinguish intentional taps from
sympathetic finger motion in camera-based hand tracking:

1. Per-finger velocity thresholds (ATK, UIST 2015)
   - Scaled by finger independence index from neuroscience literature
2. Temporal deduplication (TouchInsight, UIST 2024)
   - If multiple fingers trigger within 100ms, keep highest velocity
3. MCP-to-TIP curl ratio (joint angle analysis)
   - Intentional tap = rapid decrease in TIP-to-MCP distance

References:
- ATK: Enabling Ten-Finger Freehand Typing (UIST 2015)
- TouchInsight (UIST 2024)
- Finger independence indices (bioRxiv 2025)
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


# MediaPipe landmark IDs
# TIP: 4(thumb), 8(index), 12(middle), 16(ring), 20(pinky)
# MCP: 1(thumb), 5(index), 9(middle), 13(ring), 17(pinky)
# PIP: 3(thumb), 6(index), 10(middle), 14(ring), 18(pinky)

FINGER_TIP_IDS = {4: 'thumb', 8: 'index', 12: 'middle', 16: 'ring', 20: 'pinky'}
FINGER_MCP_IDS = {4: 1, 8: 5, 12: 9, 16: 13, 20: 17}
FINGER_PIP_IDS = {4: 3, 8: 6, 12: 10, 16: 14, 20: 18}

# Finger independence indices (from neuroscience literature)
# Higher = more independent = less sympathetic motion
FINGER_INDEPENDENCE = {
    4: 0.90,   # thumb (very independent)
    8: 0.812,  # index (most independent finger)
    12: 0.530, # middle (moderate)
    16: 0.479, # ring (least independent)
    20: 0.606, # pinky
}

# Per-finger velocity threshold multipliers (inverse of independence)
# Index is baseline (1.0x), others scaled up
_INDEX_INDEPENDENCE = FINGER_INDEPENDENCE[8]
FINGER_VELOCITY_MULTIPLIER = {
    tip_id: _INDEX_INDEPENDENCE / max(indep, 0.3)
    for tip_id, indep in FINGER_INDEPENDENCE.items()
}
# Result: index=1.0, middle=1.53, ring=1.70, pinky=1.34, thumb=0.90


class MultiFingerFilter:
    """
    Filters multi-finger tap candidates to reject sympathetic motion.

    Usage:
        filter = MultiFingerFilter()

        # Each frame, for each finger that passes velocity detection:
        filter.update_curl(hand_idx, fingertip_id, hand_landmarks, image_shape)

        # When a contact is detected:
        if filter.should_accept(hand_idx, fingertip_id, velocity, timestamp):
            # Accept the tap
    """

    def __init__(
        self,
        temporal_window_ms: float = 100.0,
        curl_ratio_threshold: float = 0.15,
        curl_history_size: int = 5,
        base_velocity_threshold: float = 8.0,
        base_min_peak_velocity: float = 15.0,
    ):
        """
        Args:
            temporal_window_ms: Window for temporal deduplication (ms)
            curl_ratio_threshold: Min curl ratio change for valid tap
            curl_history_size: Frames of curl history to track
            base_velocity_threshold: Baseline approach velocity (for index finger)
            base_min_peak_velocity: Baseline peak velocity (for index finger)
        """
        self.temporal_window_s = temporal_window_ms / 1000.0
        self.curl_ratio_threshold = curl_ratio_threshold
        self.curl_history_size = curl_history_size
        self.base_velocity_threshold = base_velocity_threshold
        self.base_min_peak_velocity = base_min_peak_velocity

        # Track pending taps for temporal deduplication
        # Key: hand_idx, Value: list of (finger_id, velocity, timestamp, x, y)
        self._pending_taps: Dict[int, List] = {}
        self._last_accepted_time: Dict[int, float] = {}  # per hand

        # Curl ratio history per finger
        # Key: "h{hand_idx}_f{fingertip_id}"
        self._curl_history: Dict[str, deque] = {}

    def get_velocity_threshold(self, fingertip_id: int) -> float:
        """Get per-finger velocity threshold (approach)."""
        mult = FINGER_VELOCITY_MULTIPLIER.get(fingertip_id, 1.5)
        return self.base_velocity_threshold * mult

    def get_min_peak_velocity(self, fingertip_id: int) -> float:
        """Get per-finger minimum peak velocity."""
        mult = FINGER_VELOCITY_MULTIPLIER.get(fingertip_id, 1.5)
        return self.base_min_peak_velocity * mult

    def compute_curl_ratio(self, hand_landmarks, fingertip_id: int, image_shape: Tuple[int, int]) -> float:
        """
        Compute curl ratio: distance(TIP, MCP) / distance(MCP, WRIST).
        Lower ratio = more curled finger = tapping.
        """
        h, w = image_shape
        mcp_id = FINGER_MCP_IDS.get(fingertip_id)
        if mcp_id is None:
            return 1.0

        tip = hand_landmarks.landmark[fingertip_id]
        mcp = hand_landmarks.landmark[mcp_id]
        wrist = hand_landmarks.landmark[0]

        # Convert to pixel coords
        tip_pos = np.array([tip.x * w, tip.y * h])
        mcp_pos = np.array([mcp.x * w, mcp.y * h])
        wrist_pos = np.array([wrist.x * w, wrist.y * h])

        tip_mcp_dist = np.linalg.norm(tip_pos - mcp_pos)
        mcp_wrist_dist = np.linalg.norm(mcp_pos - wrist_pos)

        if mcp_wrist_dist < 1.0:
            return 1.0

        return tip_mcp_dist / mcp_wrist_dist

    def update_curl(self, hand_idx: int, fingertip_id: int, hand_landmarks, image_shape: Tuple[int, int]):
        """Update curl ratio history for a finger. Call every frame."""
        finger_key = f"h{hand_idx}_f{fingertip_id}"
        if finger_key not in self._curl_history:
            self._curl_history[finger_key] = deque(maxlen=self.curl_history_size)

        curl = self.compute_curl_ratio(hand_landmarks, fingertip_id, image_shape)
        self._curl_history[finger_key].append(curl)

    def get_curl_change(self, hand_idx: int, fingertip_id: int) -> float:
        """
        Get recent curl ratio change (positive = finger is curling/tapping).
        Returns the max decrease in curl ratio over the history window.
        """
        finger_key = f"h{hand_idx}_f{fingertip_id}"
        history = self._curl_history.get(finger_key)
        if not history or len(history) < 2:
            return 0.0

        vals = list(history)
        # Max curl - current curl = how much the finger has curled
        max_curl = max(vals[:-1])  # peak before current
        current_curl = vals[-1]
        return max_curl - current_curl  # positive = curling down

    def check_curl(self, hand_idx: int, fingertip_id: int) -> bool:
        """Check if finger shows intentional curl (not sympathetic motion)."""
        change = self.get_curl_change(hand_idx, fingertip_id)
        return change >= self.curl_ratio_threshold

    def register_tap_candidate(
        self, hand_idx: int, fingertip_id: int,
        velocity: float, timestamp: float, x: int, y: int
    ):
        """Register a tap candidate for temporal deduplication."""
        if hand_idx not in self._pending_taps:
            self._pending_taps[hand_idx] = []
        self._pending_taps[hand_idx].append((fingertip_id, abs(velocity), timestamp, x, y))

    def resolve_pending_taps(self, hand_idx: int, current_time: float) -> Optional[int]:
        """
        Resolve pending taps for a hand. If multiple fingers triggered
        within the temporal window, return only the one with highest velocity.
        Returns fingertip_id of the winner, or None.
        """
        pending = self._pending_taps.get(hand_idx, [])
        if not pending:
            return None

        # Filter to taps within the temporal window
        recent = [(fid, vel, t, x, y) for fid, vel, t, x, y in pending
                  if current_time - t < self.temporal_window_s]

        if not recent:
            self._pending_taps[hand_idx] = []
            return None

        # If the oldest tap is old enough (window expired), resolve
        oldest_time = min(t for _, _, t, _, _ in recent)
        if current_time - oldest_time < self.temporal_window_s:
            return None  # Still waiting for more candidates

        # Pick the finger with highest velocity
        winner = max(recent, key=lambda x: x[1])
        self._pending_taps[hand_idx] = []
        self._last_accepted_time[hand_idx] = current_time
        return winner[0]  # fingertip_id

    def should_accept(
        self,
        hand_idx: int,
        fingertip_id: int,
        peak_velocity: float,
        timestamp: float,
        debug: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if a tap from this finger should be accepted.
        Combines per-finger threshold + curl check.

        Returns: (accept, reason)
        """
        finger_name = FINGER_TIP_IDS.get(fingertip_id, f"f{fingertip_id}")

        # 1. Per-finger peak velocity threshold
        min_peak = self.get_min_peak_velocity(fingertip_id)
        if peak_velocity < min_peak:
            return False, f"{finger_name}: peak_vel={peak_velocity:.0f} < {min_peak:.0f}"

        # 2. Curl ratio check (skip for index finger — most reliable)
        if fingertip_id != 8:  # Not index finger
            curl_ok = self.check_curl(hand_idx, fingertip_id)
            if not curl_ok:
                curl_change = self.get_curl_change(hand_idx, fingertip_id)
                return False, f"{finger_name}: curl_change={curl_change:.3f} < {self.curl_ratio_threshold}"

        return True, f"{finger_name}: accepted"
