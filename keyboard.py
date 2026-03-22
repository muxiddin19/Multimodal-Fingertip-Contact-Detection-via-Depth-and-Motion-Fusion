"""
VR Keyboard with AI Depth Estimation - CVPR 2026 Implementation (Version 1)
============================================================================
Real-Time Multimodal Fingertip Contact Detection via Depth and Motion Fusion
for Vision-Based Human-Computer Interaction

This implementation matches the methodology described in the CVPR 2026 paper:
- Velocity-based tap detection (PRIMARY) with state machine
- Threshold hysteresis mechanism (4.5mm entry, 6.0mm exit)
- Multi-modal contact detection fusion (depth + motion)
- Cooldown mechanism (450ms / ~15 frames @ 30fps)
- Fine-tuned Depth Anything V2 model

Paper Claims:
- Contact Detection Accuracy: 94.2% (DepthAnythingV2-ft)
- MAE: 3.2mm (after fine-tuning, from 12.8mm pre-trained)
- WPM: 45.6 (best configuration)
- CER: 3.1%
- F1-Score: 94.4%
- False Positive Rate: 4.2%

Author: Mukhiddin Toshpulatov
Institution: KAIST SpaceTop Research Center
"""

import sys
import os
import numpy
from pathlib import Path
# Add depth model path
DEPTH_MODEL_PATH = r"D:\Codes\vscode\Depth_Anything_V2_main\metric_depth"
if os.path.exists(DEPTH_MODEL_PATH):
    sys.path.insert(0, DEPTH_MODEL_PATH)

import cv2
import numpy as np
import json
import time
import mediapipe as mp
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import torch
import torch.nn.functional as F
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# Try to import depth model
# try:
#     # Try local version first (with our custom code)
#     from depth_model_manager import DepthEstimator
#     DEPTH_MODEL_AVAILABLE = True
#     print("[INFO] Using LOCAL depth_model_manager.py")
# except ImportError:
#     try:
#         # Fallback to installed version
#         from src.depth_model_manager import DepthEstimator
#         DEPTH_MODEL_AVAILABLE = True
#         print("[INFO] Using INSTALLED depth_model_manager from src/")
try:
    # Try src version first (the GOOD one with vitl fix)
    from src.depth_model_manager1 import DepthEstimator
    DEPTH_MODEL_AVAILABLE = True
    print("[INFO] Using DEPTH_MODEL_MANAGER from src/")
except ImportError:
    try:
        # Fallback to local version
        from depth_model_manager import DepthEstimator
        DEPTH_MODEL_AVAILABLE = True
        print("[INFO] Using LOCAL depth_model_manager.py")
    except ImportError:
        print("WARNING: DepthEstimator not available. Using mock depth.")
        DEPTH_MODEL_AVAILABLE = False

# Try to import pynput for real keyboard simulation
try:
    from pynput.keyboard import Controller, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    print("WARNING: pynput not installed. Install with: pip install pynput")
    PYNPUT_AVAILABLE = False
    Controller = None
    Key = None


class TapState(Enum):
    """Tap detection state machine states."""
    IDLE = "idle"
    APPROACHING = "approaching"
    CONTACT = "contact"
    RETRACTING = "retracting"


@dataclass
class ContactMetrics:
    """Real-time contact detection metrics for evaluation."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_taps: int = 0
    total_frames: int = 0

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def false_positive_rate(self) -> float:
        if self.false_positives + self.true_negatives == 0:
            return 0.0
        return self.false_positives / (self.false_positives + self.true_negatives)


@dataclass
class TypingMetrics:
    """Typing performance metrics."""
    total_characters: int = 0
    correct_characters: int = 0
    total_words: int = 0
    start_time: float = field(default_factory=time.time)
    errors: int = 0

    @property
    def wpm(self) -> float:
        """Words per minute (using standard 5 chars = 1 word)."""
        elapsed = time.time() - self.start_time
        if elapsed == 0 or elapsed < 1.0:  # ← Avoid division by tiny numbers
            return 0.0
        # Standard WPM: (characters / 5) / (time in minutes)
        words = self.total_characters / 5.0  # ← Changed
        minutes = elapsed / 60.0
        return words / minutes  # ← Changed

    @property
    def cer(self) -> float:
        """Character error rate."""
        if self.total_characters == 0:
            return 0.0
        return self.errors / self.total_characters


class VelocityBasedContactDetector:
    """
    Advanced multi-modal contact detection with velocity-based tap detection.

    This implements the paper's methodology:
    1. Velocity-based tap detection (PRIMARY)
    2. Threshold hysteresis (4.5mm entry, 6.0mm exit)
    3. Depth fusion
    4. Temporal consistency
    5. Cooldown mechanism
    """

    def __init__(
        self,
        history_size: int = 5,
        contact_entry_threshold_cm: float = .45, #0.45,
        contact_exit_threshold_cm: float = 0.6, #.6,
        velocity_threshold_approach: float = 8.0,
        velocity_threshold_stop: float = 6.0,
        velocity_drop_ratio: float = 0.5,
        min_peak_velocity: float = 15.0,
        velocity_threshold_retract: float = -8.0,
        required_contact_frames: int = 1,
        cooldown_frames: int =4, #8  # ← Increase from 15
        confidence_threshold: float = 0.40  # ← Increase from 0.55
    ):
        self.history_size = history_size

        # Threshold hysteresis (paper Section 5)
        self.contact_entry_threshold_cm = contact_entry_threshold_cm
        self.contact_exit_threshold_cm = contact_exit_threshold_cm

        # Velocity parameters
        self.velocity_threshold_approach = velocity_threshold_approach
        self.velocity_threshold_stop = velocity_threshold_stop
        self.velocity_drop_ratio = velocity_drop_ratio
        self.min_peak_velocity = min_peak_velocity
        self.velocity_threshold_retract = velocity_threshold_retract

        # Temporal parameters
        self.required_contact_frames = required_contact_frames
        self.cooldown_frames = cooldown_frames
        self.confidence_threshold = confidence_threshold

        # Per-finger tracking
        self.position_history: Dict[str, deque] = {}
        self.velocity_history: Dict[str, deque] = {}
        self.depth_history: Dict[str, deque] = {}
        self.brightness_history: Dict[str, deque] = {}

        # State tracking
        self.tap_state: Dict[str, TapState] = {}
        self.peak_velocity: Dict[str, float] = {}
        self.contact_frames: Dict[str, int] = {}
        self.cooldown_counter: Dict[str, int] = {}
        self.in_contact: Dict[str, bool] = {}  # For hysteresis

        # ← ADD THESE NEW LINES:
        self.tap_triggered: Dict[str, bool] = {}  # Prevent multiple triggers per tap
        self.was_in_contact: Dict[str, bool] = {}  # Debouncing
        
        # Metrics
        self.metrics = ContactMetrics()

        self.tap_triggered: Dict[str, bool] = {}
        self.last_tap_time: Dict[str, float] = {}
    # def get_smoothed_depth(self, finger_id: str, current_depth: float) -> float:
    #     """
    #     Apply temporal smoothing to depth readings.
    #     Reduces jitter from depth estimation noise.
    #     """
    #     if finger_id not in self.depth_history:
    #         return current_depth
        
    #     history = list(self.depth_history[finger_id])
    #     if len(history) == 0:
    #         return current_depth
        
    #     # Exponential moving average (alpha=0.3 for smoothing)
    #     smoothed = current_depth * 0.3 + np.mean(history) * 0.7
        
    #     return smoothed
    def get_smoothed_depth(self, finger_id: str) -> float:
        """
        Get temporally smoothed depth value.
        Uses exponential moving average to reduce jitter.
        """
        if finger_id not in self.depth_history or len(self.depth_history[finger_id]) == 0:
            return 0.0
        
        history = list(self.depth_history[finger_id])
        
        if len(history) == 1:
            return history[0]
        
        # Exponential moving average (alpha=0.4 for balance)
        # Recent values weighted more, but still smooth
        alpha = 0.4
        smoothed = history[-1]  # Start with most recent
        
        for i in range(len(history) - 2, -1, -1):
            smoothed = alpha * history[i] + (1 - alpha) * smoothed
        
        return smoothed
    def _init_finger(self, finger_id: str):
        """Initialize tracking for a new finger."""
        if finger_id not in self.position_history:
            self.position_history[finger_id] = deque(maxlen=self.history_size)
            self.velocity_history[finger_id] = deque(maxlen=self.history_size)
            self.depth_history[finger_id] = deque(maxlen=self.history_size)
            self.brightness_history[finger_id] = deque(maxlen=self.history_size)
            self.tap_state[finger_id] = TapState.IDLE
            self.peak_velocity[finger_id] = 0.0
            self.contact_frames[finger_id] = 0
            self.cooldown_counter[finger_id] = 0
            self.in_contact[finger_id] = False

    def update_history(
        self,
        finger_id: str,
        x: int,
        y: int,
        depth: float,
        brightness: float,
        timestamp: float
    ):
        """Update tracking history for a finger."""
        self._init_finger(finger_id)

        self.position_history[finger_id].append((x, y, timestamp))
        self.depth_history[finger_id].append(depth)
        self.brightness_history[finger_id].append(brightness)

        # Calculate velocity from last two positions
        if len(self.position_history[finger_id]) >= 2:
            pos_curr = self.position_history[finger_id][-1]
            pos_prev = self.position_history[finger_id][-2]

            dt = pos_curr[2] - pos_prev[2]
            # if dt > 0:
            #     vx = (pos_curr[0] - pos_prev[0]) / dt
            #     vy = (pos_curr[1] - pos_prev[1]) / dt  # Positive = moving down
            #     self.velocity_history[finger_id].append((vx, vy))
            if dt > 0:
                vx = (pos_curr[0] - pos_prev[0]) / dt
                vy = (pos_curr[1] - pos_prev[1]) / dt

                # CLAMP velocity to filter noise spikes
                MAX_VELOCITY = 300.0  # px/s - realistic finger tap limit
                vx = max(-MAX_VELOCITY, min(MAX_VELOCITY, vx))
                vy = max(-MAX_VELOCITY, min(MAX_VELOCITY, vy))

                self.velocity_history[finger_id].append((vx, vy))
            else:
                self.velocity_history[finger_id].append((0, 0))

    def get_velocity_profile(self, finger_id: str) -> Tuple[float, bool, bool, bool]:
        """
        Analyze velocity to detect typing motion pattern.

        Returns:
            (current_vy, is_approaching, is_stopping, is_retracting)
        """
        if finger_id not in self.velocity_history or len(self.velocity_history[finger_id]) < 3:
            return (0, False, False, False)

        velocities = list(self.velocity_history[finger_id])
        vx_curr, vy_curr = velocities[-1]
        vx_prev, vy_prev = velocities[-2] if len(velocities) >= 2 else (0, 0)

        # APPROACHING: Moving down fast with sustained velocity
        is_approaching = (
            vy_curr > self.velocity_threshold_approach and
            vy_curr >= vy_prev * 0.6 # 0.85
        )

        # STOPPING: Velocity drops significantly (indicates contact)
        velocity_drop = (
            vy_prev > self.velocity_threshold_approach and
            vy_curr < self.velocity_threshold_stop and
            vy_curr < vy_prev * self.velocity_drop_ratio
        )
        is_stopping = velocity_drop

        # RETRACTING: Moving up
        is_retracting = vy_curr < self.velocity_threshold_retract

        return (vy_curr, is_approaching, is_stopping, is_retracting)

    def update_tap_state(
        self,
        finger_id: str,
        vy: float,
        is_approaching: bool,
        is_stopping: bool,
        is_retracting: bool
    ) -> Optional[str]:
        """
        Update tap state machine.

        State transitions:
        IDLE -> APPROACHING -> CONTACT -> RETRACTING -> IDLE

        Returns 'contact' when a contact event is detected.
        """
        current_state = self.tap_state.get(finger_id, TapState.IDLE)

        if current_state == TapState.IDLE:
            if is_approaching:
                self.tap_state[finger_id] = TapState.APPROACHING
                self.peak_velocity[finger_id] = vy

        elif current_state == TapState.APPROACHING:
            # Track peak velocity
            if vy > self.peak_velocity[finger_id]:
                self.peak_velocity[finger_id] = vy

            # Transition to contact if:
            # 1. Velocity is stopping
            # 2. Peak velocity was high enough (deliberate tap)
            if is_stopping and self.peak_velocity[finger_id] >= self.min_peak_velocity:
                self.tap_state[finger_id] = TapState.CONTACT
                return 'contact'

            # Reset if starts retracting without contact
            if is_retracting:
                self.tap_state[finger_id] = TapState.IDLE
                self.peak_velocity[finger_id] = 0

        elif current_state == TapState.CONTACT:
            # Transition to retracting
            if is_retracting or vy < -5:
                self.tap_state[finger_id] = TapState.RETRACTING

        elif current_state == TapState.RETRACTING:
            # Return to idle when motion stops
            if abs(vy) < 25 and not is_approaching and not is_retracting:
                self.tap_state[finger_id] = TapState.IDLE
                self.peak_velocity[finger_id] = 0

        return current_state.value

    def check_hysteresis(self, finger_id: str, depth_distance_cm: float) -> bool:
        """
        Apply threshold hysteresis for stable contact detection.

        Paper methodology:
        - Contact entry: depth < 4.5mm
        - Contact exit: depth > 6.0mm
        - In between: maintain previous state
        """
        self._init_finger(finger_id)

        if depth_distance_cm < self.contact_entry_threshold_cm:
            self.in_contact[finger_id] = True
        elif depth_distance_cm > self.contact_exit_threshold_cm:
            self.in_contact[finger_id] = False
        # Else: maintain previous state (hysteresis)

        return self.in_contact[finger_id]

    def get_temporal_stability(self, finger_id: str) -> float:
        """
        Check if fingertip is stable over recent frames.
        Returns stability score (0-1, higher = more stable).
        """
        if finger_id not in self.position_history:
            return 0.0

        positions = list(self.position_history[finger_id])
        if len(positions) < 3:
            return 0.0

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        x_var = np.var(xs)
        y_var = np.var(ys)
        total_var = x_var + y_var

        # Lower variance = more stable = higher score
        stability = 1.0 - min(total_var / 100.0, 1.0)
        return stability

    def get_brightness(self, frame: np.ndarray, x: int, y: int, radius: int = 8) -> float:
        """Get average brightness around fingertip."""
        try:
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(frame.shape[1], x + radius), min(frame.shape[0], y + radius)

            region = frame[y1:y2, x1:x2]
            if region.shape[0] < 3 or region.shape[1] < 3:
                return 0.0

            if len(region.shape) == 3:
                region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                region_gray = region

            return float(np.mean(region_gray))
        except Exception:
            return 0.0

    def check_contact(
        self,
        finger_id: str,
        frame: np.ndarray,
        x: int,
        y: int,
        depth_corrected: float,
        surface_depth: float,
        timestamp: float,
        debug: bool = False
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Multi-modal contact detection with velocity-based tap detection.

        Returns:
            (is_contact, confidence_score, debug_info)
        """
        debug_info = {}

        # Calculate distance from surface
        # smoothed_depth = self.get_smoothed_depth(finger_id, depth_corrected)
        smoothed_depth = self.get_smoothed_depth(finger_id)  # ← Use history
        distance_from_surface = surface_depth - smoothed_depth  # ← Use smoothed
        distance_cm = distance_from_surface * 100
        # distance_from_surface = surface_depth - depth_corrected
        # distance_cm = distance_from_surface * 100

        debug_info['depth_cm'] = distance_cm

        # 1. DEPTH CHECK with hysteresis
        depth_ok = self.check_hysteresis(finger_id, distance_cm)
        debug_info['depth_ok'] = depth_ok
        debug_info['in_contact_hysteresis'] = self.in_contact.get(finger_id, False)

        # 2. BRIGHTNESS
        brightness = self.get_brightness(frame, x, y)
        self.update_history(finger_id, x, y, depth_corrected, brightness, timestamp)

        # 3. VELOCITY ANALYSIS
        vy, is_approaching, is_stopping, is_retracting = self.get_velocity_profile(finger_id)
        tap_event = self.update_tap_state(finger_id, vy, is_approaching, is_stopping, is_retracting)

        velocity_contact = (tap_event == 'contact')

        debug_info['velocity_y'] = vy
        debug_info['is_approaching'] = is_approaching
        debug_info['is_stopping'] = is_stopping
        debug_info['is_retracting'] = is_retracting
        debug_info['tap_state'] = self.tap_state.get(finger_id, TapState.IDLE).value
        debug_info['velocity_contact'] = velocity_contact
        debug_info['peak_velocity'] = self.peak_velocity.get(finger_id, 0)

        # 4. TEMPORAL STABILITY
        stability = self.get_temporal_stability(finger_id)
        is_stable = stability > 0.6
        debug_info['stability'] = stability
        debug_info['is_stable'] = is_stable

        # 5. BRIGHTNESS CHANGE
        brightness_changed = False
        if finger_id in self.brightness_history and len(self.brightness_history[finger_id]) >= 3:
            brightness_history = list(self.brightness_history[finger_id])
            brightness_change = abs(brightness - np.mean(brightness_history[:-1]))
            brightness_changed = brightness_change > 5
        debug_info['brightness'] = brightness
        debug_info['brightness_changed'] = brightness_changed

        # 6. CHECK COOLDOWN
        if self.cooldown_counter.get(finger_id, 0) > 0:
            self.cooldown_counter[finger_id] -= 1
            debug_info['in_cooldown'] = True
            return (False, 0.0, debug_info)
        debug_info['in_cooldown'] = False

        # In VelocityBasedContactDetector.check_contact() method
        # Replace the "DECISION LOGIC" section (around line 380):

        # DECISION LOGIC - Velocity-based (PRIMARY) + MANDATORY Depth Check
        confidence = 0.0

        # if velocity_contact:
        #     # ← ADD MANDATORY DEPTH CHECK:
        #     if depth_ok:  # Finger MUST be near surface
        #         confidence += 0.6  # Strong signal: velocity + depth
        #     elif distance_cm < 1.0:  # Reasonably close (within 1cm)
        #         confidence += 0.4  # Moderate signal
        #     else:
        #         # ← CHANGED: Reject if too far from surface
        #         confidence = 0.0  # Not a valid tap
        #         debug_info['rejected_too_far'] = True
        if velocity_contact:
            # Prevent double triggers
            if self.tap_triggered.get(finger_id, False):
                confidence = 0.0
                velocity_contact = False
                debug_info['already_triggered'] = True
            else:
                # First trigger
                self.tap_triggered[finger_id] = True
                
                # Stricter depth check
                if depth_ok and (-0.5 < distance_cm < 0.8):
                    confidence += 0.6
                elif depth_ok:
                    confidence += 0.3
                else:
                    confidence = 0.0
                    debug_info['rejected_depth'] = True
        elif depth_ok:
            # Fallback: depth-only (needs supporting signals)
            confidence += 0.2
            if is_stable:
                confidence += 0.15
            if brightness_changed:
                confidence += 0.1

        # Reset trigger on retract
        if debug_info.get('tap_state') == 'retracting':
            self.tap_triggered[finger_id] = False

        # Frame-based confirmation
        is_contact = False
        if confidence >= self.confidence_threshold:
            self.contact_frames[finger_id] = self.contact_frames.get(finger_id, 0) + 1

            if self.contact_frames[finger_id] >= self.required_contact_frames:
                is_contact = True
                # Apply cooldown
                self.cooldown_counter[finger_id] = self.cooldown_frames
        else:
            self.contact_frames[finger_id] = 0

        debug_info['confidence'] = confidence
        debug_info['contact_frames'] = self.contact_frames.get(finger_id, 0)

        if debug and (is_contact or velocity_contact):
            print(f"\n[CONTACT - {finger_id}]")
            print(f"  Depth: {distance_cm:.2f}cm (hysteresis: {depth_ok})")
            print(f"  Velocity: {vy:.1f}px/s - State: {debug_info['tap_state']}")
            print(f"  Peak velocity: {debug_info['peak_velocity']:.1f}px/s")
            print(f"  Confidence: {confidence:.2f}")

        return (is_contact, confidence, debug_info)


class VRKeyboardCVPR2026:
    """
    VR Keyboard implementation matching CVPR 2026 paper methodology.

    Features:
    - Fine-tuned Depth Anything V2 for depth estimation
    - Velocity-based tap detection with state machine
    - Threshold hysteresis for stable contact detection
    - Multi-modal fusion (depth + motion)
    - Real-time performance metrics
    """

    SPECIAL_KEYS = {
        'backspace': 'BACKSPACE', 'Backspace': 'BACKSPACE', 'back': 'BACKSPACE',
        'delete': 'DELETE', 'Delete': 'DELETE', 'del': 'DELETE',
        'space': 'SPACE', 'Space': 'SPACE', 'SPACE': 'SPACE',
        'enter': 'ENTER', 'Enter': 'ENTER', 'return': 'ENTER',
        'shift': 'SHIFT', 'Shift': 'SHIFT',
        'tab': 'TAB', 'Tab': 'TAB',
        'caps': 'CAPS', 'Caps': 'CAPS',
        'ctrl': 'CTRL', 'Ctrl': 'CTRL',
        'alt': 'ALT', 'Alt': 'ALT',
        'win': 'WIN', 'Win': 'WIN',
        'esc': 'ESC', 'Esc': 'ESC',
        'B.Spa': 'BACKSPACE', 'B.spa': 'BACKSPACE',
    }

    def __init__(
        self,
        annotation_file: str = 'keyboard_annotations.json',
        depth_checkpoint: str = None,
        threshold_cm: float = 0.8,
        camera_id: int = 0,
        use_real_keyboard: bool = True,
        track_all_fingers: bool = False,  # Index only by default
        debug_mode: bool = False
    ):
        print("=" * 70)
        print("  VR KEYBOARD - CVPR 2026 IMPLEMENTATION (V1)")
        print("  Real-Time Multimodal Fingertip Contact Detection")
        print("=" * 70)
        self._log_data = []
        # Initialize depth estimator
        if DEPTH_MODEL_AVAILABLE:
            print("\n[LOADING] Depth model ...")
            # Use the working model
            #checkpoint = r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2_s\latest32.pth"
            # checkpoint = r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2_vits.pth"
            # checkpoint = r"D:\Codes\vscode\Pretrained_weights\latest_dav2_20260313.pth"
            #checkpoint = r"D:\Codes\vscode\Pretrained_weights\dav2\20260318best.pth"
            checkpoint = r"D:\Codes\vscode\Pretrained_weights\dav2\20260318latest.pth"


            # In vr_keyboard_cvpr2026_v6.py, line ~738:
            # checkpoint = r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2_metric_hypersim_vits.pth"
            self.depth_estimator = DepthEstimator(
                model_type='depth_anything_v2',
                custom_checkpoint=checkpoint
            )
            # checkpoint = depth_checkpoint or r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2\latest20.pth"
            # NEW (BEST MODEL):
            # checkpoint = depth_checkpoint or r"D:\Codes\vscode\Pretrained_weights\custom_unet\best_model_epoch_1.pth"
            # Or for speed:
            # checkpoint = depth_checkpoint or r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2_vits.pth"

            # # self.depth_estimator = DepthEstimator(custom_checkpoint=checkpoint)
            # self.depth_estimator = DepthEstimator(model_path=checkpoint)
            # Old version:
            # Use specified checkpoint or default
            # self.depth_checkpoint_path = depth_checkpoint
            # if self.depth_checkpoint_path is None:
            #     self.depth_checkpoint_path = r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2\latest20.pth"
            # Use specified checkpoint or default
            #Changed here
            self.depth_checkpoint_path = depth_checkpoint
            # if self.depth_checkpoint_path is None:
                # ← Point to ViT-S checkpoint instead
                # self.depth_checkpoint_path = r"D:/Codes/vscode/Pretrained_weights/depth_anything_v2_metric_hypersim_vits.pth"
            if self.depth_checkpoint_path is None:
                # Use the real-world trained ViT-S model (NOT hypersim)
                #self.depth_checkpoint_path = r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2_s\latest32.pth"
                # In vr_keyboard_cvpr2026_v6.py, line ~738:
                #checkpoint = r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2_metric_hypersim_vits.pth"
                # checkpoint = r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2_vits.pth"
                # depth_anything_v2_metric_hypersim_vits.pth
            # self.depth_estimator = DepthEstimator(model_path=self.depth_checkpoint_path)
            # OLD:
            # self.depth_estimator = DepthEstimator(model_path=self.depth_checkpoint_path)

            # NEW:
                self.depth_estimator = DepthEstimator(custom_checkpoint=self.depth_checkpoint_path)
            else:
                print("\n[WARNING] Depth model not available - using mock depth")
                self.depth_estimator = None

        # Initialize contact detector
        print("[LOADING] Velocity-based contact detector...")
        self.contact_detector = VelocityBasedContactDetector(
            contact_entry_threshold_cm=0.45,# 4.5mm
            contact_exit_threshold_cm=.6, # 6.0mm
            required_contact_frames=1,
            cooldown_frames=4 #8             # ~240ms @ 30fps
        )

        # Initialize MediaPipe hands
        print("[LOADING] Hand tracking (MediaPipe)...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5, #0.4,
            min_tracking_confidence=0.5, #0.5,
            model_complexity=0  # Use higher complexity for better accuracy
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Load keyboard layout
        print("[LOADING] Keyboard layout...")
        self.keys = self._load_keyboard_annotation(annotation_file)

        # Initialize camera
        print("[LOADING] Camera...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ADD THESE for better exposure/detection:
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)      # Enable autofocus
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)   # Neutral brightness

        # Keyboard controller
        self.use_real_keyboard = use_real_keyboard and PYNPUT_AVAILABLE
        if self.use_real_keyboard:
            self.keyboard_controller = Controller()
            print("[ENABLED] Real keyboard simulation")
        else:
            self.keyboard_controller = None
            print("[DISABLED] Real keyboard simulation")

        # Calibration parameters
        self.depth_scale_factor = None
        self.actual_distance_m = None
        self.typing_threshold_m = threshold_cm / 100.0
        self.keyboard_surface_depth = None
        self.is_calibrated = False

        # State
        self.typed_text = ""
        self.last_keys_pressed = {}
        self.shift_active = False
        self.caps_lock = False

        # Depth frame skip for performance (reduced for speed)
        self.depth_frame_skip = 2
        self.depth_frame_counter = 0
        self.cached_depth_map = None

        # Finger tracking configuration
        self.track_all_fingers = track_all_fingers
        if track_all_fingers:
            self.fingertip_landmarks = [
                self.mp_hands.HandLandmark.THUMB_TIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP,
            ]
        else:
            self.fingertip_landmarks = [
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            ]

        # Performance tracking
        self.fps = 0
        self.frame_times = []
        self.debug_mode = debug_mode
        self.last_pressed_key_visual = None
        self.last_pressed_key_frames = 0

        # Metrics
        # self.contact_metrics = ContactMetrics()
        self.typing_metrics = TypingMetrics()

        self._print_config()
    @property
    def current_model_name(self):
        return os.path.basename(self.depth_checkpoint_path) if self.depth_checkpoint_path else "Unknown"

    # In vr_keyboard_cvpr2026_v6.py, add auto-calibration:

    # def auto_calibrate_keyboard(self, depth_map):
    #     """
    #     Automatically calibrate using keyboard plane detection.
    #     Fits a plane to multiple keyboard points for robust scale estimation.
    #     """
    #     import cv2
    #     import numpy as np
        
    #     # Sample 9 points across keyboard (corners + center + edges)
    #     h, w = depth_map.shape
    #     sample_points = [
    #         (w//4, h//4), (w//2, h//4), (3*w//4, h//4),      # Top row
    #         (w//4, h//2), (w//2, h//2), (3*w//4, h//2),      # Middle row
    #         (w//4, 3*h//4), (w//2, 3*h//4), (3*w//4, 3*h//4) # Bottom row
    #     ]
        
    #     depths = [depth_map[y, x] for x, y in sample_points]
    #     median_depth = np.median(depths)
        
    #     # Assume keyboard is ~37cm away (adjust to your setup)
    #     KNOWN_KEYBOARD_DISTANCE = 0.37  # meters
    #     self.depth_scale_factor = KNOWN_KEYBOARD_DISTANCE / median_depth
    #     self.keyboard_surface_depth = KNOWN_KEYBOARD_DISTANCE
    #     self.is_calibrated = True
        
    #     print(f"\n[AUTO-CALIBRATED]")
    #     print(f"  Median raw depth: {median_depth:.3f}m")
    #     print(f"  Scale factor: {self.depth_scale_factor:.4f}")
    #     print(f"  Surface depth: {KNOWN_KEYBOARD_DISTANCE*100:.1f}cm\n")

    # In vr_keyboard_cvpr2026_v6.py, add this method to VRKeyboardCVPR2026 class
    # Add it right after the calibrate_keyboard_surface() method (around line 850)

    def auto_calibrate_keyboard(self, depth_map: np.ndarray, known_distance_cm: float = 37.0):
        """
        Automatically calibrate using multiple points across the keyboard plane.
        
        Args:
            depth_map: Current depth map from depth estimator
            known_distance_cm: Actual distance to keyboard in cm (default: 37cm)
        """
        print("\n" + "=" * 70)
        print("[AUTO-CALIBRATION] Measuring keyboard plane...")
        print("=" * 70)
        
        h, w = depth_map.shape
        
        # Sample 9 points in a 3x3 grid across the image
        # This covers center + edges for robust plane fitting
        sample_points = [
            (w//4, h//3), (w//2, h//3), (3*w//4, h//3),      # Top row
            (w//4, h//2), (w//2, h//2), (3*w//4, h//2),      # Middle row
            (w//4, 2*h//3), (w//2, 2*h//3), (3*w//4, 2*h//3) # Bottom row
        ]
        
        # Collect depth readings from all sample points
        depths = []
        for x, y in sample_points:
            depth_value = self._get_depth_at_point(depth_map, x, y)
            depths.append(depth_value)
            print(f"  Point ({x:3d}, {y:3d}): {depth_value:.3f}m")
        
        # ← ADD OUTLIER FILTERING:
        # Remove outliers (anything > 2 std devs from median)
        median_depth_raw = np.median(depths)
        std_depth_raw = np.std(depths)
        depths_filtered = [d for d in depths if abs(d - median_depth_raw) < 2 * std_depth_raw]

        if len(depths_filtered) < 5:
            print("[ERROR] Too many outliers in calibration!")
            return

        median_depth = np.median(depths_filtered)
        std_depth = np.std(depths_filtered)

        print(f"\n  Statistics (after filtering):")
        print(f"    Median depth: {median_depth:.3f}m")
        print(f"    Std deviation: {std_depth:.3f}m")
        print(f"    Outliers removed: {len(depths) - len(depths_filtered)}")
        # Use median (robust to outliers from hands/objects)
        #median_depth = np.median(depths)
        #std_depth = np.std(depths)
        
        #print(f"\n  Statistics:")
        #print(f"    Median depth: {median_depth:.3f}m")
        #print(f"    Std deviation: {std_depth:.3f}m")
        
        # Calculate scale factor
        self.actual_distance_m = known_distance_cm / 100.0
        self.depth_scale_factor = self.actual_distance_m / median_depth if median_depth > 0 else 1.0
        self.keyboard_surface_depth = self.actual_distance_m
        self.is_calibrated = True
        
        print(f"\n[SUCCESS] Auto-calibrated!")
        print(f"  Scale factor: {self.depth_scale_factor:.4f}")
        print(f"  Surface depth: {self.keyboard_surface_depth * 100:.1f}cm")
        print("=" * 70 + "\n")
        print("[READY] Start typing!")
        
        # Reset metrics
        self.typing_metrics = TypingMetrics()
    def _print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("  CONFIGURATION (CVPR 2026 Paper Settings)")
        print("=" * 70)
        print(f"  Threshold Hysteresis:")
        print(f"    - Contact entry: {self.contact_detector.contact_entry_threshold_cm * 10:.1f}mm")
        print(f"    - Contact exit: {self.contact_detector.contact_exit_threshold_cm * 10:.1f}mm")
        print(f"  Velocity Detection:")
        print(f"    - Approach threshold: {self.contact_detector.velocity_threshold_approach} px/s")
        print(f"    - Min peak velocity: {self.contact_detector.min_peak_velocity} px/s")
        print(f"  Cooldown: {self.contact_detector.cooldown_frames} frames")
        print(f"  Confidence threshold: {self.contact_detector.confidence_threshold}")
        print(f"  Tracking: {'All fingers' if self.track_all_fingers else 'Index finger only'}")
        print("=" * 70)
        # ← UPDATE THIS LINE:
        print("\n[CALIBRATION] Press 'A' for auto-calibration OR 'C' after clicking surface")
        print("[CONTROLS] A=Auto-Cal | C=Manual-Cal | D=Debug | M=Metrics | Q=Quit")
        print("=" * 70 + "\n")

    def _load_keyboard_annotation(self, filename: str) -> List[Dict]:
        """Load keyboard annotation from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            keys = []
            if isinstance(data, list):
                for key_obj in data:
                    key_name = key_obj.get('key', 'UNKNOWN')
                    points = key_obj.get('points', [])
                    if len(points) == 4:
                        corners = [[p['x'], p['y']] for p in points]
                        keys.append({
                            'name': key_name,
                            'corners': np.array(corners, dtype=np.float32),
                            'center': np.mean(corners, axis=0).astype(int)
                        })

            print(f"   [SUCCESS] Loaded {len(keys)} keys")
            return keys
        except FileNotFoundError:
            print(f"[ERROR] {filename} not found")
            return []

    def calibrate_keyboard_surface(self, depth_map: np.ndarray, point: Tuple[int, int]):
        """Calibrate keyboard surface depth."""
        x, y = point
        raw_depth = self._get_depth_at_point(depth_map, x, y)

        print("\n" + "=" * 70)
        print("[CALIBRATION] Depth Measurement")
        print("=" * 70)
        print(f"  Raw model depth: {raw_depth:.3f}m")
        print("\n[ACTION] Enter actual distance to keyboard in cm:")

        try:
            user_input = input("Distance (cm): ").strip()
            actual_cm = float(user_input)
            self.actual_distance_m = actual_cm / 100.0
            self.depth_scale_factor = self.actual_distance_m / raw_depth if raw_depth > 0 else 1.0
            self.keyboard_surface_depth = self.actual_distance_m
            self.is_calibrated = True

            print("\n[SUCCESS] Calibrated!")
            print(f"  Scale factor: {self.depth_scale_factor:.4f}")
            print(f"  Surface depth: {self.keyboard_surface_depth * 100:.1f}cm")
            print("=" * 70 + "\n")
            print("[READY] Start typing!")

            # Reset metrics
            self.typing_metrics = TypingMetrics()
        except ValueError:
            print("[ERROR] Invalid input!")

    def _get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """Get depth value at specific point."""
        h, w = depth_map.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return float(depth_map[y, x])

    def _correct_depth(self, raw_depth: float) -> float:
        """Apply depth correction using calibration."""
        if self.depth_scale_factor is None:
            return raw_depth
        return raw_depth * self.depth_scale_factor

    def _estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth from frame."""
        if self.depth_estimator is None:
            # Mock depth map
            return np.full(frame.shape[:2], 0.35, dtype=np.float32)

        # Skip frames for performance
        self.depth_frame_counter += 1
        if self.depth_frame_counter % self.depth_frame_skip == 0 or self.cached_depth_map is None:
            self.cached_depth_map = self.depth_estimator.estimate_depth(frame)

        return self.cached_depth_map

    def _get_fingertip_depth(
        self,
        depth_map: np.ndarray,
        hand_landmarks,
        fingertip_id: int,
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int, float, float]:
        """Get fingertip position and depth."""
        fingertip = hand_landmarks.landmark[fingertip_id]
        h, w = image_shape
        x = int(fingertip.x * w)
        y = int(fingertip.y * h)
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        raw_depth = self._get_depth_at_point(depth_map, x, y)
        corrected_depth = self._correct_depth(raw_depth)
        return (x, y, corrected_depth, raw_depth)

    def _is_fingertip_on_key(self, fingertip_pos: Tuple[int, int], key: Dict) -> bool:
        """Check if fingertip is over a key."""
        x, y = fingertip_pos[:2]
        point = np.array([x, y], dtype=np.float32)
        result = cv2.pointPolygonTest(key['corners'], tuple(point), False)
        return result >= 0

    def _normalize_key_name(self, key_name: str) -> str:
        """Normalize key names."""
        return self.SPECIAL_KEYS.get(key_name, key_name)

    def _simulate_keypress(self, key_name: str):
        """Simulate keyboard press."""
        if not self.use_real_keyboard or self.keyboard_controller is None:
            return
        try:
            normalized = self._normalize_key_name(key_name)
            if normalized == 'BACKSPACE':
                self.keyboard_controller.press(Key.backspace)
                self.keyboard_controller.release(Key.backspace)
            elif normalized == 'SPACE':
                self.keyboard_controller.press(Key.space)
                self.keyboard_controller.release(Key.space)
            elif normalized == 'ENTER':
                self.keyboard_controller.press(Key.enter)
                self.keyboard_controller.release(Key.enter)
            elif normalized not in ['SHIFT', 'CAPS', 'CTRL', 'ALT', 'WIN', 'ESC', 'TAB', 'DELETE']:
                char = key_name.lower() if not (self.shift_active or self.caps_lock) else key_name.upper()
                if len(char) == 1:
                    self.keyboard_controller.type(char)
        except Exception as e:
            print(f"[ERROR] Keypress failed: {e}")

    def _handle_key_press(self, key_name: str) -> bool:
        """Handle key press event."""
        normalized = self._normalize_key_name(key_name)

        if normalized == 'BACKSPACE':
            if len(self.typed_text) > 0:
                self.typed_text = self.typed_text[:-1]
            self._simulate_keypress(key_name)
            print(f"[BACKSPACE]")
            self.contact_detector.metrics.total_taps += 1
            self.contact_detector.metrics.true_positives += 1
            return True
            
        elif normalized == 'DELETE':
            self.typed_text = ""
            print(f"[DELETE - Cleared]")
            self.contact_detector.metrics.total_taps += 1
            self.contact_detector.metrics.true_positives += 1
            return True
            
        elif normalized == 'CAPS':
            self.caps_lock = not self.caps_lock
            print(f"[CAPS LOCK] {'ON' if self.caps_lock else 'OFF'}")
            # Don't count CAPS as a tap
            return True
            
        elif normalized == 'SPACE':
            self.typed_text += ' '
            self.typing_metrics.total_words += 1
            self._simulate_keypress(key_name)
            print(f"[SPACE]")
            self.contact_detector.metrics.total_taps += 1
            self.contact_detector.metrics.true_positives += 1
            return True
            
        elif normalized == 'ENTER':
            self.typed_text += '\n'
            self._simulate_keypress(key_name)
            print(f"[ENTER]")
            self.contact_detector.metrics.total_taps += 1
            self.contact_detector.metrics.true_positives += 1
            return True
            
        elif normalized == 'SHIFT':
            self.shift_active = not self.shift_active
            print(f"[SHIFT] {'ON' if self.shift_active else 'OFF'}")
            # Don't count SHIFT as a tap
            return True
            
        elif normalized in ['CTRL', 'ALT', 'WIN', 'ESC', 'TAB']:
            # Ignore modifier keys to prevent accidents
            print(f"[{normalized}] - IGNORED (modifier key)")
            self.contact_detector.metrics.false_positives += 1
            return False
            
        else:
            # Regular character
            char = key_name
            if len(char) == 1:
                if self.shift_active or self.caps_lock:
                    char = char.upper()
                    if self.shift_active:
                        self.shift_active = False
                else:
                    char = char.lower()
            self.typed_text += char
            self.typing_metrics.total_characters += 1
            self.typing_metrics.correct_characters += 1
            self._simulate_keypress(key_name)
            print(f"[Key: {char}]")
            self.contact_detector.metrics.total_taps += 1
            self.contact_detector.metrics.true_positives += 1
            return True

    def _check_key_press(
        self,
        finger_id: str,
        frame: np.ndarray,
        x: int,
        y: int,
        depth_corrected: float,
        timestamp: float
    ) -> Tuple[Optional[str], float, Dict]:
        
        """Check for key press using contact detector."""
        if not self.is_calibrated or self.keyboard_surface_depth is None:
            return (None, 0.0, {})
        
        # ← ADD SANITY CHECK:
        distance_cm = (self.keyboard_surface_depth - depth_corrected) * 100
        if abs(distance_cm) > 5.0:  # More than 5cm from surface
            return (None, 0.0, {'rejected': 'too_far', 'distance_cm': distance_cm})

        is_contact, confidence, debug_info = self.contact_detector.check_contact(
            finger_id, frame, x, y, depth_corrected,
            self.keyboard_surface_depth, timestamp, debug=self.debug_mode
        )
        if is_contact:
            # FINGER PAD OFFSET: contact happens at PAD (5-10mm below tip)
            contact_x = x
            contact_y = y + 12  # Move contact point down

            # Find BEST key by closest center
            best_key = None
            best_distance = float('inf')

            for key in self.keys:
                center_x, center_y = key['center']
                distance = np.sqrt((contact_x - center_x)**2 + (contact_y - center_y)**2)

                max_distance = 50  # Default max distance
                key_name = key.get('name', '')

                # Stricter threshold for dangerous edge keys (backspace protection)
                # if key_name in ['backspace', 'Backspace', 'B.Spa', 'delete', 'Delete', 'del', 'esc', 'Esc', 'ESC']:
                #     max_distance = 5  # VERY strict for dangerous keys - must be almost exactly on key
                # if key_name in ['backspace', 'Backspace', 'B.Spa', 'delete', 'Delete']:
                #     max_distance = 30  # Much stricter for backspace

                if distance < max_distance and distance < best_distance:
                    best_distance = distance
                    best_key = key

            if best_key:
                if self.debug_mode:
                    print(f"[KEY SELECTED] {best_key['name']} (dist: {best_distance:.1f}px)")
                return (best_key['name'], confidence, debug_info)

        return (None, confidence, debug_info)
        # if is_contact:
        #     # ← ADD FINGER PAD OFFSET:
        #     # Fingertip TIP is detected, but contact happens at PAD (5-10mm below)
        #     # In image space, this is approximately 15-20 pixels down
        #     # FINGER_PAD_OFFSET_Y = 18  # pixels (tune this for your camera/hand size)
        #     contact_x = x
        #     contact_y = y + 18  # Move contact point down
        #     # contact_y = y + FINGER_PAD_OFFSET_Y  # Move contact point down
        #     # Find ALL keys within 40px radius
        # # Find BEST key by closest center (not just any key within polygon)
        # best_key = None
        # best_distance = float('inf')

        # for key in self.keys:
        #     center_x, center_y = key['center']
        #     distance = np.sqrt((contact_x - center_x)**2 + (contact_y - center_y)**2)

        #     # Must be reasonably close (within 50px of center)
        #     if distance < 50 and distance < best_distance:
        #         best_distance = distance
        #         best_key = key

        # if best_key:
        #     if self.debug_mode:
        #         print(f"[KEY SELECTED] {best_key['name']} (dist: {best_distance:.1f}px)")
        #     return (best_key['name'], confidence, debug_info)

        # return (None, confidence, debug_info)
        #     nearby_keys = []
        #     for key in self.keys:
        #         center_x, center_y = key['center']
        #         distance = np.sqrt((contact_x - center_x)**2 + (contact_y - center_y)**2)
        #         if distance < 60:  # Within 60px
        #             nearby_keys.append((key['name'], distance))
            
        #     if nearby_keys:
        #         print(f"[NEARBY KEYS] {nearby_keys}")
        #     # Draw both points on frame for calibration
        #     cv2.circle(frame, (x, y), 8, (255, 0, 255), 2)  # Magenta = TIP
        #     cv2.circle(frame, (contact_x, contact_y), 8, (0, 255, 255), -1)  # Cyan = CONTACT POINT
        #     cv2.line(frame, (x, y), (contact_x, contact_y), (255, 255, 0), 2)  # Yellow line

        #     # Now check which key is at the CONTACT point (not tip point)
        #     for key in self.keys:
        #         if self._is_fingertip_on_key((contact_x, contact_y), key):
        #             return (key['name'], confidence, debug_info)

        # return (None, confidence, debug_info)


            #for key in self.keys:
             #   if self._is_fingertip_on_key((x, y), key):
              #      return (key['name'], confidence, debug_info)

        #return (None, confidence, debug_info)

    def _draw_keyboard(self, frame: np.ndarray):
        """Draw keyboard overlay."""
        overlay = frame.copy()
        for key in self.keys:
            corners = key['corners'].astype(int)
            key_name = self._normalize_key_name(key['name'])

            # Color based on key type
            if key_name in ['SPACE', 'ENTER', 'BACKSPACE']:
                fill_color = (50, 70, 50)
                outline_color = (100, 150, 100)
            else:
                fill_color = (50, 50, 50)
                outline_color = (100, 100, 100)

            # Highlight pressed key
            if self.last_pressed_key_visual == key['name'] and self.last_pressed_key_frames > 0:
                fill_color = (0, 200, 0)
                outline_color = (0, 255, 0)

            cv2.polylines(frame, [corners], True, outline_color, 2)
            cv2.fillPoly(overlay, [corners], fill_color)

            # Key label
            center = key['center']
            label = key['name'].upper()
            if key_name == 'BACKSPACE':
                label = 'BKSP'

            cv2.putText(frame, label, tuple(center - [10, -5]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def _draw_ui(self, frame: np.ndarray):
        """Draw UI elements."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Status bar
        status = "Calibrated" if self.is_calibrated else "Not Calibrated - Press C"
        color = (0, 255, 0) if self.is_calibrated else (0, 165, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Bottom panel
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Legend
        legend_y = h - 65
        cv2.arrowedLine(frame, (10, legend_y), (10, legend_y + 15), (0, 165, 255), 2, tipLength=0.3)
        cv2.putText(frame, "Approaching", (20, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.circle(frame, (125, legend_y + 8), 7, (0, 255, 0), -1)
        cv2.putText(frame, "Contact!", (140, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        cv2.arrowedLine(frame, (230, legend_y + 15), (230, legend_y), (255, 100, 100), 2, tipLength=0.3)
        cv2.putText(frame, "Retracting", (240, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Typed text
        cv2.putText(frame, "Typed:", (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        display_text = self.typed_text[-50:] if len(self.typed_text) > 50 else self.typed_text
        cv2.putText(frame, display_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Metrics (if debug mode)
        if self.debug_mode:
            metrics_y = 60
            wpm = self.typing_metrics.wpm
            cv2.putText(frame, f"WPM: {wpm:.1f}", (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def print_metrics(self):
        """Print current performance metrics."""
        print("\n" + "=" * 70)
        print("  PERFORMANCE METRICS")
        print("=" * 70)
        print(f"  Typing Metrics:")
        print(f"    - WPM: {self.typing_metrics.wpm:.1f}")
        print(f"    - CER: {self.typing_metrics.cer * 100:.1f}%")  # ← ADD THIS
        print(f"    - Total characters: {self.typing_metrics.total_characters}")
        print(f"    - Total words: {self.typing_metrics.total_characters / 5.0:.1f}")  # ← Changed
        print(f"\n  Contact Detection Metrics:")
        print(f"    - Total taps: {self.contact_detector.metrics.total_taps}")  # ← Already correct
        print(f"    - Total frames: {self.contact_detector.metrics.total_frames}")  # ← Already correct
        print(f"    - Accuracy: {self.contact_detector.metrics.accuracy * 100:.1f}%")  # ← ADD THIS
        print(f"    - F1-Score: {self.contact_detector.metrics.f1_score * 100:.1f}%")  # ← ADD THIS
        print("=" * 70 + "\n")
        import pandas as pd
        if self._log_data:
            pd.DataFrame(self._log_data).to_csv('depth_velocity_log.csv', index=False)
            print(f"[SAVED] {len(self._log_data)} samples to depth_velocity_log.csv")

    def run(self):
        """Main loop."""
        cv2.namedWindow('VR Keyboard')

        calibration_point = [None]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                calibration_point[0] = (x, y)
                print(f"[CALIBRATION] Point selected: ({x}, {y})")

        cv2.setMouseCallback('VR Keyboard', mouse_callback)

        print("\n[STARTED] VR Keyboard running...\n")

        # ← ADD AUTOMATIC CALIBRATION ON STARTUP:
        print("[INFO] Running auto-calibration on startup...")
        time.sleep(1.0)  # Wait for camera to stabilize

        # Capture first frame
        ret, frame = self.cap.read()
        if ret:
            depth_map = self._estimate_depth(frame)
            self.auto_calibrate_keyboard(depth_map, known_distance_cm=37.0)
        else:
            print("[WARNING] Could not capture frame for auto-calibration")

        try:
            while True:
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Depth estimation
                depth_map = self._estimate_depth(frame)

                # Hand tracking
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                  # BOOST: If no hands detected, try with slight preprocessing
                if not results.multi_hand_landmarks:
                    # Increase contrast slightly to help detection
                    frame_boosted = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                    frame_rgb_boosted = cv2.cvtColor(frame_boosted, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(frame_rgb_boosted)

                if self.last_pressed_key_frames > 0:
                    self.last_pressed_key_frames -= 1
                if results.multi_hand_landmarks:
                    self.contact_detector.metrics.total_frames += 1
                    
                    # Sort hands left-to-right for consistent labeling
                    hands_with_x = []
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        wrist_x = hand_landmarks.landmark[0].x
                        hands_with_x.append((wrist_x, hand_landmarks))
                    
                    hands_with_x.sort(key=lambda h: h[0])
                    
                    # Process sorted hands - SINGLE loop only
                    for hand_idx, (wrist_x, hand_landmarks) in enumerate(hands_with_x):
                        # Draw hand skeleton
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 255, 0), thickness=3, circle_radius=4),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(255, 255, 255), thickness=2)
                        )

                        # Process fingertips
                        for fingertip_id in self.fingertip_landmarks:
                            x, y, depth_corrected, _ = self._get_fingertip_depth(
                                depth_map, hand_landmarks, fingertip_id, frame.shape[:2])

                            finger_id = f"h{hand_idx}_f{fingertip_id}"
                            # ... rest of existing code unchanged
                # Right after hand tracking (around line 1161)
                # if results.multi_hand_landmarks:
                #     self.contact_detector.metrics.total_frames += 1  # ← ADD THIS
                #     for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                #         # ... rest of code
                # # if results.multi_hand_landmarks:
                #     # for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                #         # Draw hand skeleton
                #         self.mp_draw.draw_landmarks(
                #             frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                #             landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                #                 color=(0, 255, 0), thickness=3, circle_radius=4),
                #             connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                #                 color=(255, 255, 255), thickness=2)
                #         )

                #         # Process fingertips
                #         for fingertip_id in self.fingertip_landmarks:
                #             x, y, depth_corrected, _ = self._get_fingertip_depth(
                #                 depth_map, hand_landmarks, fingertip_id, frame.shape[:2])

                #             finger_id = f"h{hand_idx}_f{fingertip_id}"

                            # Check for key press
                            pressed_key, confidence, debug_info = self._check_key_press(
                                finger_id, frame, x, y, depth_corrected, start_time)
                            
                            # ADD THIS BLOCK:
                            if hasattr(self, '_log_data') and 'depth_cm' in debug_info and 'velocity_y' in debug_info:
                                self._log_data.append({
                                    'depth_mm': debug_info['depth_cm'] * 10,
                                    'velocity': abs(debug_info['velocity_y']),
                                    'label': 1 if pressed_key else 0
                                })

                            # Draw velocity indicator
                            if 'velocity_y' in debug_info:
                                vy = debug_info['velocity_y']
                                tap_state = debug_info.get('tap_state', 'idle')

                                # Color based on state
                                if tap_state == 'approaching':
                                    vel_color = (0, 165, 255)
                                elif tap_state == 'contact':
                                    vel_color = (0, 255, 0)
                                elif tap_state == 'retracting':
                                    vel_color = (255, 100, 100)
                                else:
                                    vel_color = (128, 128, 128)

                                # Draw velocity arrow
                                if abs(vy) > 5:
                                    arrow_len = int(min(abs(vy) / 3, 30))
                                    if vy > 0:
                                        cv2.arrowedLine(frame, (x, y), (x, y + arrow_len), vel_color, 2, tipLength=0.3)
                                    else:
                                        cv2.arrowedLine(frame, (x, y), (x, y - arrow_len), vel_color, 2, tipLength=0.3)

                            # Draw fingertip
                            if pressed_key:
                                cv2.circle(frame, (x, y), 14, (0, 255, 0), -1)
                                cv2.circle(frame, (x, y), 18, (255, 255, 255), 3)
                            else:
                                cv2.circle(frame, (x, y), 10, (100, 100, 255), -1)
                                cv2.circle(frame, (x, y), 14, (150, 150, 150), 2)

                            # Draw confidence
                            if confidence > 0.2:
                                cv2.putText(frame, f"{confidence:.2f}", (x + 20, y - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                            # Handle keypress
                            if self.is_calibrated and pressed_key:
                                if pressed_key != self.last_keys_pressed.get(finger_id):
                                    self._handle_key_press(pressed_key)
                                    self.last_keys_pressed[finger_id] = pressed_key
                                    self.last_pressed_key_visual = pressed_key
                                    self.last_pressed_key_frames = 10
                            elif not pressed_key:
                                self.last_keys_pressed[finger_id] = None
                  # Draw hand detection status
                if results.multi_hand_landmarks:
                    num_hands = len(results.multi_hand_landmarks)
                else:
                    num_hands = 0
                hand_color = (0, 255, 0) if num_hands == 2 else (0, 165, 255) if num_hands == 1 else (0, 0, 255)
                cv2.putText(frame, f"Hands: {num_hands}/2", (frame.shape[1] - 120, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)
                # Draw UI
                self._draw_keyboard(frame)
                self._draw_ui(frame)
                cv2.imshow('VR Keyboard', frame)

                # FPS calculation
                elapsed = time.time() - start_time
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                self.fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0

                # Controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c'):
                    if calibration_point[0] and depth_map is not None:
                        self.calibrate_keyboard_surface(depth_map, calibration_point[0])
                        calibration_point[0] = None
                    else:
                        print("[WARNING] Click on keyboard surface first!")
                # ← ADD THIS NEW BLOCK:
                elif key == ord('a'):  # Auto-calibrate
                    if depth_map is not None:
                        # Prompt for actual distance (optional - can hardcode 37cm)
                        print("\n[AUTO-CALIBRATION] Enter keyboard distance in cm (press Enter for 37cm):")
                        try:
                            user_input = input("Distance (cm) [37]: ").strip()
                            distance_cm = float(user_input) if user_input else 37.0
                            self.auto_calibrate_keyboard(depth_map, distance_cm)
                        except ValueError:
                            print("[ERROR] Invalid input, using default 37cm")
                            self.auto_calibrate_keyboard(depth_map, 37.0)
                    else:
                        print("[WARNING] No depth map available!")

                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"[DEBUG] {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('m'):
                    self.print_metrics()
                # elif key == ord('d'):
                #     self.debug_mode = not self.debug_mode
                #     print(f"[DEBUG] {'ON' if self.debug_mode else 'OFF'}")
                # elif key == ord('m'):
                #     self.print_metrics()

        finally:
            print("\n[SHUTDOWN]")
            self.print_metrics()
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='VR Keyboard - CVPR 2026 (V1)')
    parser.add_argument('--annotation', default='keyboard_annotations.json', help='Keyboard annotation file')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--all-fingers', action='store_true', help='Track all fingers (default: index only)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-keyboard', action='store_true', help='Disable real keyboard simulation')

    args = parser.parse_args()

    try:
        keyboard = VRKeyboardCVPR2026(
            annotation_file=args.annotation,
            camera_id=args.camera,
            track_all_fingers=args.all_fingers,
            debug_mode=args.debug,
            use_real_keyboard=not args.no_keyboard
        )
        keyboard.run()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User stopped the program")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
