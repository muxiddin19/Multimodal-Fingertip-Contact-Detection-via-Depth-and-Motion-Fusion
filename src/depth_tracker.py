import numpy as np
import cv2
import mediapipe as mp
import time
import json
import os
from camera_manager import CameraManager
from depth_model_manager import DepthEstimator

# Initialize CameraManager
camera_manager = CameraManager()

# Initialize Depth Estimator with CUSTOM MODEL
print("Loading CUSTOM depth estimation model (fine-tuned on typing hands)...")
print("Using Depth Anything V2 - better compatibility!")

# Use Custom Depth Anything V2
depth_estimator = DepthEstimator(
    model_type='custom_depth_anything_v2',
    custom_checkpoint=r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2\latest20.pth"
)

camera_manager.set_depth_estimator(depth_estimator)
print("Custom depth model loaded!")

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# Load keyboard annotations
annotation_filename = 'assets/keyboard_annotations.json'
annotated_keys = []

def load_annotations():
    global annotated_keys
    if os.path.exists(annotation_filename):
        with open(annotation_filename, 'r') as f:
            annotated_keys = json.load(f)
        print(f"Loaded {len(annotated_keys)} keyboard annotations")
    else:
        print(f"Warning: {annotation_filename} not found!")
        print("Please run keyboard_annotation.py first to create the annotation file.")

def is_point_in_key(point, key_data):
    """Check if point is inside key polygon."""
    key_points = key_data['points']
    polygon = np.array([[p['x'], p['y']] for p in key_points], np.int32)
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def find_key_under_finger(finger_pos):
    """Find which key the finger is over."""
    for key_data in annotated_keys:
        if is_point_in_key(finger_pos, key_data):
            return key_data['key']
    return None

def draw_keyboard_overlay(image, current_key=None, key_depths=None):
    """Draw keyboard annotations on image."""
    for key_data in annotated_keys:
        points = np.array([[p['x'], p['y']] for p in key_data['points']], np.int32)
        
        key_name = key_data['key']
        
        # Color based on status
        if key_name == current_key:
            color = (0, 165, 255)  # Orange for active
        elif key_depths and key_name in key_depths and len(key_depths[key_name]) > 0:
            color = (0, 255, 0)  # Green if has data
        else:
            color = (100, 100, 100)  # Gray if not measured
        
        cv2.polylines(image, [points], True, color, 2)
        
        # Draw key label
        center_x = int(np.mean([p['x'] for p in key_data['points']]))
        center_y = int(np.mean([p['y'] for p in key_data['points']]))
        cv2.putText(image, key_name, (center_x - 10, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# Calibration state
calibration_mode = "none"  # "none", "reference", "measuring"
reference_depth_m = None  # Known reference depth
key_depth_data = {}
current_key = None
tracking_active = False
depth_samples = []

# Output file
output_filename = 'assets/key_thresholds.json'

# Main Program
try:
    # Load annotations
    load_annotations()
    
    if not annotated_keys:
        print("ERROR: No keyboard annotations found. Exiting.")
        exit()
    
    # Start streaming
    if not camera_manager.start_stream():
        print("Failed to start camera stream. Exiting.")
        exit()

    print("\n" + "="*70)
    print("DEPTH-BASED KEY CALIBRATION TOOL")
    print("="*70)
    print("\nStep 1: CALIBRATE DEPTH SCALE")
    print("  - Place your hand FLAT on the keyboard surface")
    print("  - Measure actual distance from camera to keyboard (e.g., 30cm)")
    print("  - Press 'C' and enter the distance to calibrate depth model")
    print("\nStep 2: MEASURE KEY DEPTHS")
    print("  - Move finger over a key")
    print("  - Press SPACE to start recording")
    print("  - Press the key multiple times (10-20 presses)")
    print("  - Press SPACE again to stop")
    print("\nStep 3: SAVE")
    print("  - Press 'S' to save thresholds")
    print("\nPress 'Q' to quit")
    print("="*70 + "\n")

    while True:
        # Get frames
        color_image, aligned_depth_frame, depth_frame_dims = camera_manager.get_frames()

        if color_image is None or aligned_depth_frame is None:
            continue

        depth_frame_width, depth_frame_height = depth_frame_dims

        # Get the full depth map for visualization
        depth_map = aligned_depth_frame.depth_map

        # Draw keyboard overlay
        draw_keyboard_overlay(color_image, current_key, key_depth_data)

        # Process hand
        RGB_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)

        current_finger_depth = 0.0
        finger_detected = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                h, w, _ = color_image.shape
                finger_pixel_x = int(index_finger_tip.x * w)
                finger_pixel_y = int(index_finger_tip.y * h)

                # Clamp coordinates
                clamped_x = max(0, min(finger_pixel_x, depth_frame_width - 1))
                clamped_y = max(0, min(finger_pixel_y, depth_frame_height - 1))

                # Get depth AT THE FINGERTIP POSITION (not palm)
                # Sample a small area around fingertip for better accuracy
                sample_radius = 3  # pixels
                depth_samples_local = []
                
                for dx in range(-sample_radius, sample_radius + 1):
                    for dy in range(-sample_radius, sample_radius + 1):
                        sx = max(0, min(clamped_x + dx, depth_frame_width - 1))
                        sy = max(0, min(clamped_y + dy, depth_frame_height - 1))
                        depth_val = aligned_depth_frame.get_distance(sx, sy)
                        if depth_val > 0.001:  # Valid depth
                            depth_samples_local.append(depth_val)
                
                # Use median of local samples for robustness
                if depth_samples_local:
                    depth_at_finger = np.median(depth_samples_local)
                else:
                    depth_at_finger = aligned_depth_frame.get_distance(clamped_x, clamped_y)

                if depth_at_finger > 0:
                    current_finger_depth = depth_at_finger
                    finger_detected = True

                    # Find key under finger
                    key_under_finger = find_key_under_finger((finger_pixel_x, finger_pixel_y))

                    # Draw finger marker with depth - LARGE and VISIBLE
                    circle_color = (0, 255, 0) if depth_at_finger > 0.001 else (0, 0, 255)
                    cv2.circle(color_image, (finger_pixel_x, finger_pixel_y), 12, circle_color, -1)  # Larger circle
                    cv2.circle(color_image, (finger_pixel_x, finger_pixel_y), 15, (255, 255, 255), 2)  # White outline
                    
                    # Draw sampling area (small box showing where depth is measured)
                    sample_size = 6
                    cv2.rectangle(color_image, 
                                (finger_pixel_x - sample_size, finger_pixel_y - sample_size),
                                (finger_pixel_x + sample_size, finger_pixel_y + sample_size),
                                (0, 255, 255), 2)
                    
                    # Show depth in meters AND centimeters - very visible
                    depth_text = f"{current_finger_depth:.3f}m ({current_finger_depth*100:.1f}cm)"
                    # Background box for text visibility
                    text_size = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(color_image,
                                (finger_pixel_x + 15, finger_pixel_y - 25),
                                (finger_pixel_x + 15 + text_size[0] + 10, finger_pixel_y - 5),
                                (0, 0, 0), -1)
                    cv2.putText(color_image, depth_text,
                                (finger_pixel_x + 20, finger_pixel_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if key_under_finger:
                        cv2.putText(color_image, f"Key: {key_under_finger}",
                                   (finger_pixel_x + 10, finger_pixel_y + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        
                        # Auto-select key
                        if current_key != key_under_finger and not tracking_active:
                            current_key = key_under_finger

                    # Record depth if tracking
                    if tracking_active and current_key and current_finger_depth > 0.001:
                        depth_samples.append(current_finger_depth)

                break

        # Display status
        status_y = 30
        
        # Show calibration status
        if reference_depth_m:
            cv2.putText(color_image, f"Calibrated: Reference depth = {reference_depth_m:.3f}m",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            status_y += 30
        else:
            cv2.putText(color_image, "NOT CALIBRATED - Press 'C' to calibrate depth",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            status_y += 30
        
        # Show tracking status
        if tracking_active and current_key:
            cv2.putText(color_image, f"RECORDING: {current_key} - {len(depth_samples)} samples",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            status_y += 30
            
            if len(depth_samples) > 0:
                min_d = min(depth_samples)
                max_d = max(depth_samples)
                cv2.putText(color_image, f"Range: {min_d:.3f}m - {max_d:.3f}m",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                status_y += 30
        elif current_key:
            cv2.putText(color_image, f"Selected: {current_key} - Press SPACE to start",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            status_y += 30
        else:
            cv2.putText(color_image, "Move finger over a key",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            status_y += 30

        # Show depth info
        if finger_detected:
            cv2.putText(color_image, f"Current depth: {current_finger_depth:.3f}m ({current_finger_depth*100:.1f}cm)",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status_y += 30

        # Show calibration progress
        status_y += 10
        calibrated_keys = len([k for k in key_depth_data if len(key_depth_data[k]) > 0])
        total_keys = len(annotated_keys)
        cv2.putText(color_image, f"Progress: {calibrated_keys}/{total_keys} keys calibrated",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show depth map range
        depth_min, depth_max = depth_map.min(), depth_map.max()
        status_y += 35
        cv2.putText(color_image, f"Depth range: {depth_min:.3f}m - {depth_max:.3f}m",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instructions
        cv2.putText(color_image, "C: Calibrate | SPACE: Start/Stop | S: Save | Q: Quit",
                   (10, color_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Depth-Based Key Calibration', color_image)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') or key == ord('C'):  # Calibrate depth
            if finger_detected:
                print("\n" + "="*50)
                print("DEPTH CALIBRATION")
                print("="*50)
                print(f"Current measured depth: {current_finger_depth:.3f}m ({current_finger_depth*100:.1f}cm)")
                print("\nPlace your hand FLAT on the keyboard surface.")
                print("Measure the actual distance from camera to keyboard.")
                
                depth_input = input("\nEnter actual distance in cm (or press Enter to skip): ").strip()
                
                if depth_input:
                    try:
                        actual_depth_cm = float(depth_input)
                        actual_depth_m = actual_depth_cm / 100.0
                        
                        # IMPORTANT: Get RAW depth map BEFORE metric conversion for calibration
                        _, raw_depth_map = depth_estimator.estimate_depth(color_image, return_raw=True)
                        
                        # Calibrate using raw depth map
                        depth_estimator.calibrate(
                            actual_depth_m,
                            raw_depth_map,
                            (clamped_x, clamped_y)
                        )
                        
                        reference_depth_m = actual_depth_m
                        print("✓ Calibration complete! Try moving your hand to verify.\n")
                    except ValueError:
                        print("Invalid input. Calibration cancelled.")
                else:
                    print("Calibration cancelled.")
            else:
                print("No hand detected. Please show your hand and try again.")
        
        elif key == ord(' '):  # Spacebar - Start/Stop tracking
            if not tracking_active:
                # Start tracking
                if current_key and finger_detected:
                    if reference_depth_m is None:
                        print("\n⚠ Warning: Depth not calibrated yet!")
                        print("For best results, press 'C' to calibrate first.")
                        response = input("Continue anyway? (y/n): ").strip().lower()
                        if response != 'y':
                            continue
                    
                    tracking_active = True
                    depth_samples = []
                    print(f"\n--- Started tracking key: {current_key} ---")
                    print("Press the key multiple times with varying pressure")
                    print("Press SPACE when done")
                else:
                    print("Move finger over a key first")
            else:
                # Stop tracking
                tracking_active = False
                if current_key and len(depth_samples) > 0:
                    # Save the depth samples
                    key_depth_data[current_key] = depth_samples.copy()
                    
                    min_depth = min(depth_samples)
                    max_depth = max(depth_samples)
                    
                    print(f"--- Stopped tracking key: {current_key} ---")
                    print(f"Samples: {len(depth_samples)}")
                    print(f"Range: {min_depth:.3f}m ({min_depth*100:.1f}cm) - {max_depth:.3f}m ({max_depth*100:.1f}cm)")
                    print()
                    
                    depth_samples = []
                    current_key = None
                else:
                    print("No samples recorded")
        
        elif key == ord('s') or key == ord('S'):  # Save
            if key_depth_data:
                thresholds = {}
                
                print("\n" + "="*50)
                print("SAVING KEY THRESHOLDS")
                print("="*50)
                
                for key_name, depths in key_depth_data.items():
                    if len(depths) > 0:
                        min_depth = min(depths)
                        max_depth = max(depths)
                        
                        # Add margin for reliability
                        margin = 0.015  # 1.5cm margin
                        thresholds[key_name] = [
                            round(min_depth - margin, 3),
                            round(max_depth + margin, 3)
                        ]
                        
                        print(f"Key '{key_name}': {len(depths)} samples")
                        print(f"  Min: {min_depth:.3f}m ({min_depth*100:.1f}cm)")
                        print(f"  Max: {max_depth:.3f}m ({max_depth*100:.1f}cm)")
                        print(f"  Threshold: [{thresholds[key_name][0]:.3f}, {thresholds[key_name][1]:.3f}]")
                
                if thresholds:
                    # Save to file
                    os.makedirs('assets', exist_ok=True)
                    with open(output_filename, 'w') as f:
                        json.dump(thresholds, f, indent=4)
                    
                    print(f"\n✓ Key thresholds saved to '{output_filename}'")
                    print(f"✓ Calibrated {len(thresholds)} keys")
                else:
                    print("No complete measurements to save")
            else:
                print("No data to save. Measure some keys first.")
        
        elif key == ord('q') or key == ord('Q'):
            break

finally:
    print("\nStopping stream and cleaning up...")
    camera_manager.stop_stream()
    cv2.destroyAllWindows()
    hands.close()
    print("Done!")