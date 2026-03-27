"""
VR Keyboard with AI Depth Estimation - CVPR 2026 Implementation (Version 1)
============================================================================
Real-Time Multimodal Fingertip Contact Detection via Depth and Motion Fusion
for Vision-Based Human-Computer Interaction

This implementation matches the methodology described in the CVPR 2026 paper:
- Velocity-based tap detection (PRIMARY) with state machine
- Threshold hysteresis mechanism (4.5mm entry, 6.0mm exit)
- Multi-modal contact detection fusion (depth + motion)
- One Euro Filter for depth smoothing
- Cooldown mechanism

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
from pathlib import Path

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

from src.one_euro_filter import OneEuroFilter

try:
    from src.depth_model_manager1 import DepthEstimator
    DEPTH_MODEL_AVAILABLE = True
    print("[INFO] Using DEPTH_MODEL_MANAGER from src/")
except ImportError:
    try:
        from depth_model_manager import DepthEstimator
        DEPTH_MODEL_AVAILABLE = True
        print("[INFO] Using LOCAL depth_model_manager.py")
    except ImportError:
        print("WARNING: DepthEstimator not available. Using mock depth.")
        DEPTH_MODEL_AVAILABLE = False

try:
    from pynput.keyboard import Controller, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    print("WARNING: pynput not installed. Install with: pip install pynput")
    PYNPUT_AVAILABLE = False
    Controller = None
    Key = None

try:
    from src.word_predictor import WordPredictor
    WORD_PREDICTOR_AVAILABLE = True
except ImportError:
    WORD_PREDICTOR_AVAILABLE = False


class TapState(Enum):
    IDLE = "idle"
    APPROACHING = "approaching"
    CONTACT = "contact"
    RETRACTING = "retracting"


@dataclass
class ContactMetrics:
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_taps: int = 0
    total_frames: int = 0

    @property
    def precision(self):
        return self.true_positives / max(self.true_positives + self.false_positives, 1)

    @property
    def recall(self):
        return self.true_positives / max(self.true_positives + self.false_negatives, 1)

    @property
    def f1_score(self):
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-8)

    @property
    def accuracy(self):
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / max(total, 1)


@dataclass
class TypingMetrics:
    total_characters: int = 0
    correct_characters: int = 0
    total_words: int = 0
    start_time: float = field(default_factory=time.time)
    errors: int = 0

    @property
    def wpm(self):
        elapsed = time.time() - self.start_time
        if elapsed < 1.0:
            return 0.0
        return (self.total_characters / 5.0) / (elapsed / 60.0)

    @property
    def cer(self):
        return self.errors / max(self.total_characters, 1)


class VelocityBasedContactDetector:
    """
    Multi-modal contact detection: velocity state machine + depth hysteresis.
    One Euro Filter for depth smoothing.
    """

    def __init__(self, history_size=5, contact_entry_threshold_cm=0.45,
                 contact_exit_threshold_cm=0.6, velocity_threshold_approach=8.0,
                 velocity_threshold_stop=6.0, velocity_drop_ratio=0.5,
                 min_peak_velocity=15.0, velocity_threshold_retract=-8.0,
                 required_contact_frames=1, cooldown_frames=8,
                 confidence_threshold=0.50):
        self.history_size = history_size
        self.contact_entry_threshold_cm = contact_entry_threshold_cm
        self.contact_exit_threshold_cm = contact_exit_threshold_cm
        self.velocity_threshold_approach = velocity_threshold_approach
        self.velocity_threshold_stop = velocity_threshold_stop
        self.velocity_drop_ratio = velocity_drop_ratio
        self.min_peak_velocity = min_peak_velocity
        self.velocity_threshold_retract = velocity_threshold_retract
        self.required_contact_frames = required_contact_frames
        self.cooldown_frames = cooldown_frames
        self.confidence_threshold = confidence_threshold

        self.position_history: Dict[str, deque] = {}
        self.velocity_history: Dict[str, deque] = {}
        self.depth_history: Dict[str, deque] = {}
        self.brightness_history: Dict[str, deque] = {}
        self.tap_state: Dict[str, TapState] = {}
        self.peak_velocity: Dict[str, float] = {}
        self.contact_frames: Dict[str, int] = {}
        self.cooldown_counter: Dict[str, int] = {}
        self.in_contact: Dict[str, bool] = {}
        self.tap_triggered: Dict[str, bool] = {}
        self.depth_filters: Dict[str, OneEuroFilter] = {}
        self.metrics = ContactMetrics()

    def _init_finger(self, finger_id):
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

    def get_smoothed_depth(self, finger_id, depth, timestamp):
        if finger_id not in self.depth_filters:
            self.depth_filters[finger_id] = OneEuroFilter(
                t0=timestamp, x0=depth, min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
            return depth
        return self.depth_filters[finger_id](timestamp, depth)

    def update_history(self, finger_id, x, y, depth, brightness, timestamp):
        self._init_finger(finger_id)
        self.position_history[finger_id].append((x, y, timestamp))
        self.depth_history[finger_id].append(depth)
        self.brightness_history[finger_id].append(brightness)

        if len(self.position_history[finger_id]) >= 2:
            pos_curr = self.position_history[finger_id][-1]
            pos_prev = self.position_history[finger_id][-2]
            dt = pos_curr[2] - pos_prev[2]
            if dt > 0:
                vx = max(-300, min(300, (pos_curr[0] - pos_prev[0]) / dt))
                vy = max(-300, min(300, (pos_curr[1] - pos_prev[1]) / dt))
                self.velocity_history[finger_id].append((vx, vy))
            else:
                self.velocity_history[finger_id].append((0, 0))

    def get_velocity_profile(self, finger_id):
        if finger_id not in self.velocity_history or len(self.velocity_history[finger_id]) < 3:
            return (0, False, False, False)
        velocities = list(self.velocity_history[finger_id])
        _, vy_curr = velocities[-1]
        _, vy_prev = velocities[-2] if len(velocities) >= 2 else (0, 0)

        is_approaching = vy_curr > self.velocity_threshold_approach and vy_curr >= vy_prev * 0.6
        is_stopping = (vy_prev > self.velocity_threshold_approach and
                       vy_curr < self.velocity_threshold_stop and
                       vy_curr < vy_prev * self.velocity_drop_ratio)
        is_retracting = vy_curr < self.velocity_threshold_retract
        return (vy_curr, is_approaching, is_stopping, is_retracting)

    def update_tap_state(self, finger_id, vy, is_approaching, is_stopping, is_retracting):
        state = self.tap_state.get(finger_id, TapState.IDLE)
        if state == TapState.IDLE:
            if is_approaching:
                self.tap_state[finger_id] = TapState.APPROACHING
                self.peak_velocity[finger_id] = vy
        elif state == TapState.APPROACHING:
            if vy > self.peak_velocity[finger_id]:
                self.peak_velocity[finger_id] = vy
            if is_stopping and self.peak_velocity[finger_id] >= self.min_peak_velocity:
                self.tap_state[finger_id] = TapState.CONTACT
                return 'contact'
            if is_retracting:
                self.tap_state[finger_id] = TapState.IDLE
                self.peak_velocity[finger_id] = 0
        elif state == TapState.CONTACT:
            if is_retracting or vy < -5:
                self.tap_state[finger_id] = TapState.RETRACTING
        elif state == TapState.RETRACTING:
            if abs(vy) < 25 and not is_approaching and not is_retracting:
                self.tap_state[finger_id] = TapState.IDLE
                self.peak_velocity[finger_id] = 0
        return state.value

    def check_hysteresis(self, finger_id, distance_cm):
        self._init_finger(finger_id)
        if distance_cm < self.contact_entry_threshold_cm:
            self.in_contact[finger_id] = True
        elif distance_cm > self.contact_exit_threshold_cm:
            self.in_contact[finger_id] = False
        return self.in_contact[finger_id]

    def get_brightness(self, frame, x, y, radius=8):
        try:
            x1, y1 = max(0, x-radius), max(0, y-radius)
            x2, y2 = min(frame.shape[1], x+radius), min(frame.shape[0], y+radius)
            region = frame[y1:y2, x1:x2]
            if region.shape[0] < 3 or region.shape[1] < 3:
                return 0.0
            if len(region.shape) == 3:
                region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            return float(np.mean(region))
        except Exception:
            return 0.0

    def check_contact(self, finger_id, frame, x, y, depth_corrected,
                      surface_depth, timestamp, debug=False):
        debug_info = {}
        smoothed = self.get_smoothed_depth(finger_id, depth_corrected, timestamp)
        distance_cm = (surface_depth - smoothed) * 100
        debug_info['depth_cm'] = distance_cm

        depth_ok = self.check_hysteresis(finger_id, distance_cm)
        debug_info['depth_ok'] = depth_ok

        brightness = self.get_brightness(frame, x, y)
        self.update_history(finger_id, x, y, depth_corrected, brightness, timestamp)

        vy, is_approaching, is_stopping, is_retracting = self.get_velocity_profile(finger_id)
        tap_event = self.update_tap_state(finger_id, vy, is_approaching, is_stopping, is_retracting)
        velocity_contact = (tap_event == 'contact')

        debug_info['velocity_y'] = vy
        debug_info['tap_state'] = self.tap_state.get(finger_id, TapState.IDLE).value
        debug_info['velocity_contact'] = velocity_contact
        debug_info['peak_velocity'] = self.peak_velocity.get(finger_id, 0)

        if self.cooldown_counter.get(finger_id, 0) > 0:
            self.cooldown_counter[finger_id] -= 1
            return (False, 0.0, debug_info)

        confidence = 0.0
        if velocity_contact:
            if self.tap_triggered.get(finger_id, False):
                confidence = 0.0
                velocity_contact = False
            else:
                self.tap_triggered[finger_id] = True
                if depth_ok and (-0.5 < distance_cm < 0.8):
                    confidence += 0.6
                elif depth_ok:
                    confidence += 0.3
                else:
                    confidence = 0.0
        elif depth_ok:
            confidence += 0.2

        if debug_info.get('tap_state') == 'retracting':
            self.tap_triggered[finger_id] = False

        is_contact = False
        if confidence >= self.confidence_threshold:
            self.contact_frames[finger_id] = self.contact_frames.get(finger_id, 0) + 1
            if self.contact_frames[finger_id] >= self.required_contact_frames:
                is_contact = True
                self.cooldown_counter[finger_id] = self.cooldown_frames
        else:
            self.contact_frames[finger_id] = 0

        debug_info['confidence'] = confidence

        if debug and (is_contact or velocity_contact):
            print(f"\n[CONTACT - {finger_id}]")
            print(f"  Depth: {distance_cm:.2f}cm (hysteresis: {depth_ok})")
            print(f"  Velocity: {vy:.1f}px/s - State: {debug_info['tap_state']}")
            print(f"  Peak velocity: {debug_info['peak_velocity']:.1f}px/s")
            print(f"  Confidence: {confidence:.2f}")

        return (is_contact, confidence, debug_info)


class VRKeyboardCVPR2026:
    SPECIAL_KEYS = {
        'backspace': 'BACKSPACE', 'Backspace': 'BACKSPACE', 'back': 'BACKSPACE',
        'delete': 'DELETE', 'Delete': 'DELETE', 'del': 'DELETE',
        'space': 'SPACE', 'Space': 'SPACE', 'SPACE': 'SPACE',
        'enter': 'ENTER', 'Enter': 'ENTER', 'return': 'ENTER',
        'shift': 'SHIFT', 'Shift': 'SHIFT',
        'tab': 'TAB', 'Tab': 'TAB', 'caps': 'CAPS', 'Caps': 'CAPS',
        'ctrl': 'CTRL', 'Ctrl': 'CTRL', 'alt': 'ALT', 'Alt': 'ALT',
        'win': 'WIN', 'Win': 'WIN', 'esc': 'ESC', 'Esc': 'ESC',
        'B.Spa': 'BACKSPACE', 'B.spa': 'BACKSPACE',
    }

    STRICT_KEYS = {'backspace', 'Backspace', 'B.Spa', 'B.spa',
                   'delete', 'Delete', 'del', 'enter', 'Enter', 'return',
                   'esc', 'Esc', 'tab', 'Tab', 'alt', 'Alt',
                   'ctrl', 'Ctrl', 'win', 'Win', 'shift', 'Shift'}

    def __init__(self, annotation_file='keyboard_annotations.json',
                 depth_checkpoint=None, threshold_cm=0.8, camera_id=0,
                 use_real_keyboard=True, track_all_fingers=False,
                 debug_mode=False, target_phrase=None):
        self.target_phrase = target_phrase
        print("=" * 70)
        print("  VR KEYBOARD - CVPR 2026 IMPLEMENTATION (V1)")
        print("  Real-Time Multimodal Fingertip Contact Detection")
        print("=" * 70)
        self._log_data = []

        # Depth estimator (default model — instant tap response)
        if DEPTH_MODEL_AVAILABLE:
            print("\n[LOADING] Depth model ...")
            self.depth_checkpoint_path = depth_checkpoint
            self.depth_estimator = DepthEstimator(
                model_type='depth_anything_v2',
                custom_checkpoint=self.depth_checkpoint_path
            )
        else:
            print("\n[WARNING] Depth model not available")
            self.depth_checkpoint_path = None
            self.depth_estimator = None

        # Contact detector (proven: 0% CER at 5.7 WPM)
        print("[LOADING] Velocity-based contact detector...")
        self.contact_detector = VelocityBasedContactDetector(
            contact_entry_threshold_cm=0.45,
            contact_exit_threshold_cm=0.6,
            required_contact_frames=1,
            cooldown_frames=8,
            confidence_threshold=0.50
        )

        # MediaPipe hands
        print("[LOADING] Hand tracking (MediaPipe)...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.4, min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Keyboard layout
        print("[LOADING] Keyboard layout...")
        self.keys = self._load_keyboard_annotation(annotation_file)

        # Word prediction
        self.word_predictor = None
        self._predictions = []
        if WORD_PREDICTOR_AVAILABLE:
            self.word_predictor = WordPredictor(extra_words=[
                'cvpr', 'kaist', 'hello', 'keyboard', 'typing', 'depth', 'wpm'])
            print("[LOADED] Word Predictor")

        # Camera
        print("[LOADING] Camera...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)

        # Keyboard controller
        self.use_real_keyboard = use_real_keyboard and PYNPUT_AVAILABLE
        if self.use_real_keyboard:
            self.keyboard_controller = Controller()
            print("[ENABLED] Real keyboard simulation")
        else:
            self.keyboard_controller = None

        # State
        self.depth_scale_factor = None
        self.actual_distance_m = None
        self.typing_threshold_m = threshold_cm / 100.0
        self.keyboard_surface_depth = None
        self.is_calibrated = False
        self.typed_text = ""
        self.last_keys_pressed = {}
        self.shift_active = False
        self.caps_lock = False
        self.depth_frame_skip = 2
        self.depth_frame_counter = 0
        self.cached_depth_map = None
        self._last_keypress_pos = None
        self._last_keypress_time = 0.0
        self._last_keypress_finger = ""

        # Index finger only (proven reliable)
        self.track_all_fingers = track_all_fingers
        self.fingertip_landmarks = [self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        if track_all_fingers:
            self.fingertip_landmarks = [
                self.mp_hands.HandLandmark.THUMB_TIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP,
            ]

        self.fps = 0
        self.frame_times = []
        self.debug_mode = debug_mode
        self.last_pressed_key_visual = None
        self.last_pressed_key_frames = 0
        self.typing_metrics = TypingMetrics()

        self._print_config()

    def _print_config(self):
        model = os.path.basename(self.depth_checkpoint_path) if self.depth_checkpoint_path else "Default"
        print("\n" + "=" * 70)
        print("  CONFIGURATION")
        print("=" * 70)
        print(f"  Depth model: {model}")
        print(f"  Hysteresis: {self.contact_detector.contact_entry_threshold_cm*10:.1f}mm / {self.contact_detector.contact_exit_threshold_cm*10:.1f}mm")
        print(f"  Cooldown: {self.contact_detector.cooldown_frames} frames")
        print(f"  Confidence: {self.contact_detector.confidence_threshold}")
        print(f"  Tracking: {'All fingers' if self.track_all_fingers else 'Index only'}")
        print(f"  Depth smoothing: One Euro Filter")
        print(f"  Word Prediction: {'Enabled (1/2/3)' if self.word_predictor else 'Disabled'}")
        print("=" * 70)
        print("[CONTROLS] C=Manual-Cal | D=Debug | M=Metrics | Q=Quit")
        if self.word_predictor:
            print("[PREDICT]  1/2/3 = Accept word prediction")
        print("=" * 70 + "\n")

    def auto_calibrate_keyboard(self, depth_map, known_distance_cm=37.0):
        print("\n" + "=" * 70)
        print("[AUTO-CALIBRATION] Measuring keyboard plane...")
        print("=" * 70)
        h, w = depth_map.shape
        points = [(w//4, h//3), (w//2, h//3), (3*w//4, h//3),
                  (w//4, h//2), (w//2, h//2), (3*w//4, h//2),
                  (w//4, 2*h//3), (w//2, 2*h//3), (3*w//4, 2*h//3)]
        depths = [self._get_depth_at_point(depth_map, x, y) for x, y in points]
        for (x, y), d in zip(points, depths):
            print(f"  Point ({x:3d}, {y:3d}): {d:.3f}m")

        median_raw = np.median(depths)
        std_raw = np.std(depths)
        filtered = [d for d in depths if abs(d - median_raw) < 2 * std_raw]
        if len(filtered) < 5:
            print("[ERROR] Too many outliers!")
            return
        median = np.median(filtered)
        print(f"\n  Median: {median:.3f}m, Std: {np.std(filtered):.3f}m")

        self.actual_distance_m = known_distance_cm / 100.0
        self.depth_scale_factor = self.actual_distance_m / median if median > 0 else 1.0
        self.keyboard_surface_depth = self.actual_distance_m
        self.is_calibrated = True
        print(f"  Scale: {self.depth_scale_factor:.4f}, Surface: {self.keyboard_surface_depth*100:.1f}cm")
        print("=" * 70 + "\n[READY] Start typing!")
        self.typing_metrics = TypingMetrics()

    def _load_keyboard_annotation(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            keys = []
            if isinstance(data, list):
                for obj in data:
                    name = obj.get('key', 'UNKNOWN')
                    pts = obj.get('points', [])
                    if len(pts) == 4:
                        corners = [[p['x'], p['y']] for p in pts]
                        keys.append({'name': name,
                                     'corners': np.array(corners, dtype=np.float32),
                                     'center': np.mean(corners, axis=0).astype(int)})
            print(f"   [SUCCESS] Loaded {len(keys)} keys")
            return keys
        except FileNotFoundError:
            print(f"[ERROR] {filename} not found")
            return []

    def _get_depth_at_point(self, dm, x, y):
        h, w = dm.shape[:2]
        return float(dm[max(0, min(y, h-1)), max(0, min(x, w-1))])

    def _correct_depth(self, raw):
        return raw * self.depth_scale_factor if self.depth_scale_factor else raw

    def _estimate_depth(self, frame):
        if self.depth_estimator is None:
            return np.full(frame.shape[:2], 0.35, dtype=np.float32)
        self.depth_frame_counter += 1
        if self.depth_frame_counter % self.depth_frame_skip == 0 or self.cached_depth_map is None:
            self.cached_depth_map = self.depth_estimator.estimate_depth(frame)
        return self.cached_depth_map

    def _get_fingertip_depth(self, dm, landmarks, tip_id, shape):
        tip = landmarks.landmark[tip_id]
        h, w = shape
        x, y = max(0, min(int(tip.x*w), w-1)), max(0, min(int(tip.y*h), h-1))
        raw = self._get_depth_at_point(dm, x, y)
        return (x, y, self._correct_depth(raw), raw)

    def _normalize_key_name(self, name):
        return self.SPECIAL_KEYS.get(name, name)

    def _simulate_keypress(self, name):
        if not self.use_real_keyboard or not self.keyboard_controller:
            return
        try:
            n = self._normalize_key_name(name)
            if n == 'BACKSPACE':
                self.keyboard_controller.press(Key.backspace)
                self.keyboard_controller.release(Key.backspace)
            elif n == 'SPACE':
                self.keyboard_controller.press(Key.space)
                self.keyboard_controller.release(Key.space)
            elif n == 'ENTER':
                self.keyboard_controller.press(Key.enter)
                self.keyboard_controller.release(Key.enter)
            elif n not in ['SHIFT','CAPS','CTRL','ALT','WIN','ESC','TAB','DELETE']:
                ch = name.lower() if not (self.shift_active or self.caps_lock) else name.upper()
                if len(ch) == 1:
                    self.keyboard_controller.type(ch)
        except Exception as e:
            print(f"[ERROR] Keypress: {e}")

    def _handle_key_press(self, name):
        n = self._normalize_key_name(name)
        if n == 'BACKSPACE':
            if self.typed_text:
                self.typed_text = self.typed_text[:-1]
            self._simulate_keypress(name)
            print("[BACKSPACE]")
        elif n == 'DELETE':
            self.typed_text = ""
            print("[DELETE]")
        elif n == 'SPACE':
            self.typed_text += ' '
            self._simulate_keypress(name)
            print("[SPACE]")
        elif n == 'ENTER':
            self.typed_text += '\n'
            self._simulate_keypress(name)
            print("[ENTER]")
        elif n == 'SHIFT':
            self.shift_active = not self.shift_active
            print(f"[SHIFT {'ON' if self.shift_active else 'OFF'}]")
            return True
        elif n == 'CAPS':
            self.caps_lock = not self.caps_lock
            return True
        elif n in ['CTRL','ALT','WIN','ESC','TAB']:
            self.contact_detector.metrics.false_positives += 1
            return False
        else:
            ch = name
            if len(ch) == 1:
                ch = ch.upper() if (self.shift_active or self.caps_lock) else ch.lower()
                if self.shift_active:
                    self.shift_active = False
            self.typed_text += ch
            self.typing_metrics.total_characters += 1
            self._simulate_keypress(name)
            print(f"[Key: {ch}]")

        self.contact_detector.metrics.total_taps += 1
        self.contact_detector.metrics.true_positives += 1
        return True

    def _check_key_press(self, finger_id, frame, x, y, depth_corrected, timestamp):
        if not self.is_calibrated or self.keyboard_surface_depth is None:
            return (None, 0.0, {})

        distance_cm = (self.keyboard_surface_depth - depth_corrected) * 100
        if abs(distance_cm) > 5.0:
            return (None, 0.0, {'rejected': 'too_far'})

        is_contact, confidence, debug_info = self.contact_detector.check_contact(
            finger_id, frame, x, y, depth_corrected,
            self.keyboard_surface_depth, timestamp, debug=self.debug_mode)

        if is_contact:
            contact_x, contact_y = x, y + 6

            best_key = None
            best_dist = float('inf')
            for key in self.keys:
                cx, cy = key['center']
                d = np.sqrt((contact_x-cx)**2 + (contact_y-cy)**2)
                max_d = 10 if key['name'] in self.STRICT_KEYS else 35
                if d < max_d and d < best_dist:
                    best_dist = d
                    best_key = key

            if best_key:
                if self.debug_mode:
                    print(f"[KEY SELECTED] {best_key['name']} (dist: {best_dist:.1f}px)")
                return (best_key['name'], confidence, debug_info)

        return (None, confidence, debug_info)

    def _draw_keyboard(self, frame):
        overlay = frame.copy()
        for key in self.keys:
            corners = key['corners'].astype(int)
            kn = self._normalize_key_name(key['name'])
            fill = (50, 70, 50) if kn in ['SPACE','ENTER','BACKSPACE'] else (50, 50, 50)
            outline = (100, 150, 100) if kn in ['SPACE','ENTER','BACKSPACE'] else (100, 100, 100)
            if self.last_pressed_key_visual == key['name'] and self.last_pressed_key_frames > 0:
                fill, outline = (0, 200, 0), (0, 255, 0)
            cv2.polylines(frame, [corners], True, outline, 2)
            cv2.fillPoly(overlay, [corners], fill)
            label = 'BKSP' if kn == 'BACKSPACE' else key['name'].upper()
            cv2.putText(frame, label, tuple(key['center'] - [10, -5]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def _draw_ui(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        status = "Calibrated" if self.is_calibrated else "Not Calibrated"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0) if self.is_calibrated else (0, 165, 255), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w-100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        if self.target_phrase:
            cv2.putText(frame, f"Target: {self.target_phrase}", (10, h-45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
        txt = self.typed_text[-50:]
        cv2.putText(frame, txt, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if self.debug_mode:
            cv2.putText(frame, f"WPM: {self.typing_metrics.wpm:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Word predictions
        if self.word_predictor:
            words = self.typed_text.split(' ')
            current = words[-1] if words else ''
            prev = words[-2] if len(words) >= 2 else None
            self._predictions = self.word_predictor.predict(current, prev) if len(current) >= 2 else []
            if self._predictions:
                py = 80 if self.debug_mode else 60
                for i, (word, _) in enumerate(self._predictions[:3]):
                    label = f"[{i+1}] {word}"
                    px = 10 + i * 150
                    sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (px-4, py-16), (px+sz[0]+4, py+4), (60, 60, 120), -1)
                    cv2.putText(frame, label, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)

    def _accept_prediction(self, idx):
        if idx >= len(self._predictions):
            return
        word, _ = self._predictions[idx]
        words = self.typed_text.split(' ')
        partial = words[-1] if words else ''
        if not partial:
            return
        completion = word[len(partial):]
        if not completion:
            return
        self.typed_text += completion
        self.typing_metrics.total_characters += len(completion)
        if self.use_real_keyboard and self.keyboard_controller:
            self.keyboard_controller.type(completion)
        print(f"[PREDICTION] '{partial}' -> '{word}' (saved {len(completion)} chars)")

    @staticmethod
    def _edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    def print_metrics(self):
        print("\n" + "=" * 70)
        print("  PERFORMANCE METRICS")
        print("=" * 70)
        if self.target_phrase:
            typed = self.typed_text.strip()
            ed = self._edit_distance(typed, self.target_phrase)
            cer = ed / max(len(self.target_phrase), 1) * 100
            print(f"  Target: \"{self.target_phrase}\"")
            print(f"  Typed:  \"{typed}\"")
            print(f"    WPM: {self.typing_metrics.wpm:.1f}")
            print(f"    CER: {cer:.1f}% (edit dist: {ed})")
        else:
            print(f"  Typed: \"{self.typed_text.strip()}\"")
            print(f"    WPM: {self.typing_metrics.wpm:.1f}")
        print(f"    Chars: {self.typing_metrics.total_characters}")
        print(f"    Taps: {self.contact_detector.metrics.total_taps}")
        print(f"    Frames: {self.contact_detector.metrics.total_frames}")
        print("=" * 70 + "\n")
        import pandas as pd
        if self._log_data:
            pd.DataFrame(self._log_data).to_csv('depth_velocity_log.csv', index=False)
            print(f"[SAVED] {len(self._log_data)} samples to depth_velocity_log.csv")

    def run(self):
        cv2.namedWindow('VR Keyboard')
        cal_pt = [None]
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cal_pt[0] = (x, y)
        cv2.setMouseCallback('VR Keyboard', mouse_cb)

        print("\n[STARTED] VR Keyboard running...\n")
        print("[INFO] Auto-calibrating...")
        time.sleep(1.0)
        ret, frame = self.cap.read()
        if ret:
            self.auto_calibrate_keyboard(self._estimate_depth(frame), 37.0)

        try:
            while True:
                t0 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    break

                depth_map = self._estimate_depth(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                if not results.multi_hand_landmarks:
                    boosted = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                    results = self.hands.process(cv2.cvtColor(boosted, cv2.COLOR_BGR2RGB))

                if self.last_pressed_key_frames > 0:
                    self.last_pressed_key_frames -= 1

                if results.multi_hand_landmarks:
                    self.contact_detector.metrics.total_frames += 1
                    hands = sorted([(lm.landmark[0].x, lm) for lm in results.multi_hand_landmarks])

                    for hi, (_, lm) in enumerate(hands):
                        self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=4),
                            mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2))

                        for tip_id in self.fingertip_landmarks:
                            x, y, dc, _ = self._get_fingertip_depth(depth_map, lm, tip_id, frame.shape[:2])
                            fid = f"h{hi}_f{tip_id}"

                            key, conf, info = self._check_key_press(fid, frame, x, y, dc, t0)

                            if 'depth_cm' in info and 'velocity_y' in info:
                                self._log_data.append({
                                    'depth_mm': info['depth_cm']*10,
                                    'velocity': abs(info['velocity_y']),
                                    'label': 1 if key else 0
                                })

                            # Draw
                            if key:
                                cv2.circle(frame, (x,y), 14, (0,255,0), -1)
                                cv2.circle(frame, (x,y), 18, (255,255,255), 3)
                            else:
                                cv2.circle(frame, (x,y), 10, (100,100,255), -1)

                            # Handle keypress with cross-hand suppression
                            if self.is_calibrated and key:
                                suppress = False
                                if self._last_keypress_pos and fid != self._last_keypress_finger:
                                    dx = x - self._last_keypress_pos[0]
                                    dy = y - self._last_keypress_pos[1]
                                    if np.sqrt(dx*dx+dy*dy) < 40 and t0 - self._last_keypress_time < 0.3:
                                        suppress = True
                                if not suppress and key != self.last_keys_pressed.get(fid):
                                    self._handle_key_press(key)
                                    self.last_keys_pressed[fid] = key
                                    self.last_pressed_key_visual = key
                                    self.last_pressed_key_frames = 10
                                    self._last_keypress_pos = (x, y)
                                    self._last_keypress_time = t0
                                    self._last_keypress_finger = fid
                            elif not key:
                                self.last_keys_pressed[fid] = None

                # UI
                nh = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                cv2.putText(frame, f"Hands: {nh}/2", (frame.shape[1]-120, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0,255,0) if nh==2 else (0,165,255) if nh==1 else (0,0,255), 2)
                self._draw_keyboard(frame)
                self._draw_ui(frame)
                cv2.imshow('VR Keyboard', frame)

                elapsed = time.time() - t0
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                self.fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    break
                elif k == ord('c'):
                    if cal_pt[0] and depth_map is not None:
                        x, y = cal_pt[0]
                        raw = self._get_depth_at_point(depth_map, x, y)
                        print(f"\n[CAL] Raw depth at ({x},{y}): {raw:.3f}m")
                        try:
                            cm = float(input("Distance (cm): ").strip())
                            self.actual_distance_m = cm / 100.0
                            self.depth_scale_factor = self.actual_distance_m / raw if raw > 0 else 1.0
                            self.keyboard_surface_depth = self.actual_distance_m
                            self.is_calibrated = True
                            print(f"[SUCCESS] Scale={self.depth_scale_factor:.4f}")
                            self.typing_metrics = TypingMetrics()
                        except ValueError:
                            print("[ERROR] Invalid input")
                        cal_pt[0] = None
                elif k == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"[DEBUG {'ON' if self.debug_mode else 'OFF'}]")
                elif k == ord('m'):
                    self.print_metrics()
                elif k == ord('1') and self._predictions:
                    self._accept_prediction(0)
                elif k == ord('2') and self._predictions:
                    self._accept_prediction(1)
                elif k == ord('3') and self._predictions:
                    self._accept_prediction(2)

        finally:
            print("\n[SHUTDOWN]")
            self.print_metrics()
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='VR Keyboard - CVPR 2026')
    parser.add_argument('--annotation', default='keyboard_annotations.json')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--all-fingers', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-keyboard', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()

    try:
        kb = VRKeyboardCVPR2026(
            annotation_file=args.annotation, camera_id=args.camera,
            depth_checkpoint=args.checkpoint, track_all_fingers=args.all_fingers,
            debug_mode=args.debug, use_real_keyboard=not args.no_keyboard,
            target_phrase=args.target)
        kb.run()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
