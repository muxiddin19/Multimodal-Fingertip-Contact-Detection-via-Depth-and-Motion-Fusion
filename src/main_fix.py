import time

import cv2
import json
import tkinter as tk
from tkinter import scrolledtext
import threading
from pynput.keyboard import Controller, Key

from src.camera_manager import CameraManager
from src.hand_tracker import HandTracker
from src.keyboard_manager import KeyboardManager
import src.visualization_utils as viz_utils
from src.one_euro_filter import OneEuroFilter

# --- pynput Key Mapping ---
# Maps string representations from your JSON to pynput's Key objects
KEY_MAP = {
    "BACKSPACE": Key.backspace,
    "ENTER": Key.enter,
    "SPACE": Key.space,
    "SHIFT": Key.shift,
    "CTRL": Key.ctrl,
    "ALT": Key.alt,
    "WIN": Key.cmd,  # 'cmd' is used for the Windows key in pynput
    "ESC": Key.esc,
    "DEL": Key.delete,
    "UP": Key.up,
    "DOWN": Key.down,
    "LEFT": Key.left,
    "RIGHT": Key.right,
    "TAB": Key.tab,
    "CAPS": Key.caps_lock,
}


def ui_thread():
    """Function to run the tkinter UI in a separate thread."""
    try:
        root = tk.Tk()
        root.title("Virtual Keyboard Output")
        root.geometry("600x400")

        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        label = tk.Label(main_frame, text="Click inside this box to start typing with the virtual keyboard.")
        label.pack(pady=(0, 5))

        text_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=20)
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_area.focus()  # Set focus to the text area initially

        root.mainloop()
    except Exception as e:
        print(f"Error in UI thread: {e}")


def run_keyboard_interface():
    """
    Initializes and runs the main loop for the virtual keyboard interface.
    """
    # --- Configuration ---
    ANNOTATION_FILENAME = 'assets/keyboard_annotations.json'
    THRESHOLDS_FILENAME = 'assets/key_thresholds.json'
    POINTS_PER_KEY = 4
    KEY_DEPTH_THRESHOLDS = {}
    KEY_FRAME_COUNT = {}
    FRAME_THRESHOLD = 1

    def load_key_thresholds_from_file(filename: str) -> bool:
        nonlocal KEY_DEPTH_THRESHOLDS
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                KEY_DEPTH_THRESHOLDS = {key: tuple(value) for key, value in data.items()}
            print(f"Successfully loaded key thresholds from '{filename}'.")
            return True
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return False

    def is_finger_pressing_key(key_data: dict, finger_depth: float) -> bool:
        key_name = key_data.get("key")
        if not key_name:
            return False
        threshold = KEY_DEPTH_THRESHOLDS.get(key_name)
        if not threshold:
            return False
        min_depth, max_depth = threshold
        return min_depth <= finger_depth < max_depth

    # --- Initialize ---
    if not load_key_thresholds_from_file(THRESHOLDS_FILENAME):
        return

    KEY_FRAME_COUNT = {key: 0 for key in KEY_DEPTH_THRESHOLDS.keys()}

    # Start the UI in a separate thread
    ui = threading.Thread(target=ui_thread, daemon=True)
    ui.start()

    keyboard = Controller()
    camera_manager = CameraManager()
    hand_tracker = HandTracker()
    keyboard_manager = KeyboardManager(annotation_filename=ANNOTATION_FILENAME, points_per_key=POINTS_PER_KEY)

    # --- Application State ---
    last_detected_keys = set()
    last_pressed_keys = set()

    try:
        if not camera_manager.start_stream():
            print("Failed to start camera stream. Exiting.")
            return

        while True:
            color_image, aligned_depth_frame, depth_frame_dims = camera_manager.get_frames()
            if color_image is None or aligned_depth_frame is None:
                continue

            current_pressed_keys = set()
            current_detected_keys = set()
            # KEY_FRAME_COUNT = dict.fromkeys(KEY_FRAME_COUNT, 0)
            results = hand_tracker.process_frame(color_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_tracker.draw_landmarks(color_image, hand_landmarks)

                    finger_tips = {
                        'thumb': hand_tracker.get_thumb_finger_tip(hand_landmarks, color_image.shape),
                        'index': hand_tracker.get_index_finger_tip(hand_landmarks, color_image.shape),
                        'middle': hand_tracker.get_middle_finger_tip(hand_landmarks, color_image.shape),
                        'ring': hand_tracker.get_ring_finger_tip(hand_landmarks, color_image.shape),
                        'pinky': hand_tracker.get_pinky_finger_tip(hand_landmarks, color_image.shape),
                    }

                    for finger_name, (tip_coords, _) in finger_tips.items():
                        if not tip_coords: continue
                        px, py = tip_coords
                        depth_frame_width, depth_frame_height = depth_frame_dims
                        clamped_px = max(0, min(px, depth_frame_width - 1))
                        clamped_py = max(0, min(py, depth_frame_height - 1))
                        depth_m = aligned_depth_frame.get_distance(clamped_px, clamped_py)
                        viz_utils.draw_finger_tip_info(color_image, px, py, depth_m)

                        finger_point = (px, py)
                        for key_data in keyboard_manager.get_annotated_keys():
                            if keyboard_manager.is_point_in_keycap(finger_point, key_data):
                                if is_finger_pressing_key(key_data, depth_m):
                                    current_pressed_keys.add(key_data['key'])
                                    break  # Assume one finger can only press one key

            # -- Update Number of Frame of each pressed keycap --
            for current_key in current_pressed_keys:
                KEY_FRAME_COUNT[current_key] += 1
                if KEY_FRAME_COUNT[current_key] >= FRAME_THRESHOLD:
                    current_detected_keys.add(current_key)
                print(f"Key {current_key} pressed!")

            # --- Simulate Key Presses using pynput ---
            newly_pressed = current_detected_keys - last_detected_keys
            newly_released = last_detected_keys - current_detected_keys

            for key_str in newly_pressed:
                try:
                    if key_str in KEY_MAP:
                        keyboard.press(KEY_MAP[key_str])
                    elif len(key_str) == 1:  # Handle standard characters
                        keyboard.press(key_str.lower())
                except Exception as e:
                    print(f"Could not press key '{key_str}': {e}")

            for key_str in newly_released:
                try:
                    if key_str in KEY_MAP:
                        keyboard.release(KEY_MAP[key_str])
                    elif len(key_str) == 1:
                        keyboard.release(key_str.lower())
                except Exception as e:
                    print(f"Could not release key '{key_str}': {e}")

            uncontinous_pressed = last_pressed_keys - current_pressed_keys
            for key in uncontinous_pressed:
                KEY_FRAME_COUNT[key] = 0

            for key in newly_released:
                KEY_FRAME_COUNT[key] = 0

            last_detected_keys = current_detected_keys
            last_pressed_keys = current_pressed_keys



            # --- Visualization ---
            viz_utils.draw_keycap_annotations(color_image, keyboard_manager.get_annotated_keys(), current_detected_keys,
                                              POINTS_PER_KEY)
            cv2.imshow('Virtual Keyboard Interface', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --- Clean Up ---
        print("Application stopping...")
        # Release all pressed keys
        for key_str in last_detected_keys:
            try:
                if key_str in KEY_MAP:
                    keyboard.release(KEY_MAP[key_str])
                elif len(key_str) == 1:
                    keyboard.release(key_str.lower())
            except Exception as e:
                print(f"Could not release key '{key_str}' during cleanup: {e}")

        camera_manager.stop_stream()
        hand_tracker.close()
        cv2.destroyAllWindows()
        print("Application stopped.")


if __name__ == "__main__":
    run_keyboard_interface()
