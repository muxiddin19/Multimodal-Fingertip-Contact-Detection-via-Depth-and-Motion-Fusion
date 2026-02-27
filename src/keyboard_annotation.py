import numpy as np
import cv2
import json
import os
from camera_manager import CameraManager

# --- Configuration ---
camera_manager = CameraManager()
CAMERA_WIDTH, CAMERA_HEIGHT, _, _, CAMERA_FPS = camera_manager.get_resolution()

# --- Global variables for annotation ---
annotations = []
current_raw_frame = None
window_name = 'Keyboard Annotation Tool'
output_filename = 'assets/keyboard_annotations.json'

# Variables to manage the 4-point annotation process
temp_key_points = []
POINTS_PER_KEY = 4

# --- Zoom and Pan Variables ---
zoom_factor = 1.0
pan_x = 0
pan_y = 0
pan_speed = 20
max_zoom_factor = 5.0
min_zoom_factor = 1.0

# --- Load existing annotations if file exists ---
def load_annotations(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
                validated_annotations = []
                for item in data:
                    if 'key' in item and 'points' in item and len(item['points']) == POINTS_PER_KEY:
                        validated_annotations.append(item)
                    else:
                        print(f"Warning: Skipping malformed annotation entry in {filename}: {item}")
                print(f"Loaded {len(validated_annotations)} existing key(s) from {filename}")
                return validated_annotations
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {filename}. Starting with no annotations.")
                return []
    return []

# Start fresh or load existing
# annotations = load_annotations(output_filename)
annotations = []

# --- Mouse callback function for annotations ---
def mouse_callback(event, x, y, flags, param):
    global annotations, current_raw_frame, temp_key_points, zoom_factor, pan_x, pan_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_raw_frame is not None:
            # Transform clicked coordinates from zoomed view back to original frame coordinates
            original_x = int(pan_x + x / zoom_factor)
            original_y = int(pan_y + y / zoom_factor)

            # Ensure coordinates are within the original frame boundaries
            original_x = max(0, min(original_x, CAMERA_WIDTH - 1))
            original_y = max(0, min(original_y, CAMERA_HEIGHT - 1))

            temp_key_points.append({'x': original_x, 'y': original_y})
            print(f"Point {len(temp_key_points)} of {POINTS_PER_KEY} clicked (original): ({original_x}, {original_y})")

            # Redraw the frame with temporary points for immediate feedback
            draw_current_frame_with_annotations()

            if len(temp_key_points) == POINTS_PER_KEY:
                # All 4 points collected, now prompt for the key value
                key_value = show_input_box("Enter key value for this keycap (e.g., 'A', 'SPACE'):")

                if key_value:  # Only add if a value was entered
                    annotations.append({'key': key_value, 'points': temp_key_points.copy()})
                    print(f"Annotated: Key='{key_value}', Points={temp_key_points}")
                    temp_key_points = []  # Reset for next keycap
                else:
                    print("Annotation cancelled: No key value entered. Points reset.")
                    temp_key_points = []

                # Always redraw after an input box action
                draw_current_frame_with_annotations()


# --- Custom input box function ---
def show_input_box(prompt):
    """Creates a simple input box using OpenCV."""
    input_text = ""
    box_width, box_height = 400, 150
    input_box_img = np.zeros((box_height, box_width, 3), dtype=np.uint8)
    input_box_img.fill(50)

    cv2.putText(input_box_img, "Input Required", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(input_box_img, prompt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    while True:
        temp_img = input_box_img.copy()
        cv2.putText(temp_img, f"Key: {input_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(temp_img, "Press ENTER to confirm, ESC to cancel", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Input Box", temp_img)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter key
            cv2.destroyWindow("Input Box")
            return input_text
        elif key == 27:  # ESC key
            cv2.destroyWindow("Input Box")
            return None
        elif key == 8:  # Backspace
            input_text = input_text[:-1]
        elif 32 <= key <= 126:  # ASCII printable characters
            input_text += chr(key)


# --- Function to apply zoom and pan to a frame ---
def apply_zoom_and_pan(frame):
    global zoom_factor, pan_x, pan_y, CAMERA_WIDTH, CAMERA_HEIGHT

    if frame is None:
        return None

    view_width = int(CAMERA_WIDTH / zoom_factor)
    view_height = int(CAMERA_HEIGHT / zoom_factor)

    pan_x = max(0, min(pan_x, CAMERA_WIDTH - view_width))
    pan_y = max(0, min(pan_y, CAMERA_HEIGHT - view_height))

    cropped_frame = frame[pan_y : pan_y + view_height, pan_x : pan_x + view_width]
    zoomed_frame = cv2.resize(cropped_frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)

    return zoomed_frame


# --- Function to draw existing annotations on the current frame ---
def draw_current_frame_with_annotations():
    global current_raw_frame, temp_key_points, annotations, zoom_factor, pan_x, pan_y

    if current_raw_frame is None:
        return

    display_frame = current_raw_frame.copy()

    # Draw all saved annotations (green)
    for annotation in annotations:
        transformed_points = []
        for point in annotation['points']:
            transformed_x = int((point['x'] - pan_x) * zoom_factor)
            transformed_y = int((point['y'] - pan_y) * zoom_factor)
            transformed_points.append({'x': transformed_x, 'y': transformed_y})

        for point in transformed_points:
            x, y = point['x'], point['y']
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)

        if len(transformed_points) == POINTS_PER_KEY:
            pts = np.array([[p['x'], p['y']] for p in transformed_points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

        if transformed_points:
            first_p = transformed_points[0]
            key_value = annotation['key']
            cv2.putText(display_frame, key_value, (first_p['x'] + 5, first_p['y'] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw temporary points (orange)
    for i, p in enumerate(temp_key_points):
        transformed_x = int((p['x'] - pan_x) * zoom_factor)
        transformed_y = int((p['y'] - pan_y) * zoom_factor)
        cv2.circle(display_frame, (transformed_x, transformed_y), 5, (0, 165, 255), -1)
        cv2.putText(display_frame, str(i + 1), (transformed_x + 5, transformed_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    final_display_frame = apply_zoom_and_pan(display_frame)

    if final_display_frame is not None:
        info_text = f"Zoom: {zoom_factor:.1f}x (Press +/- to zoom, Arrows to pan)"
        capture_text = "Press 'c' to CAPTURE, click 4 points, ENTER key"
        save_text = "Press 's' to SAVE, 'r' to RESET zoom/pan, 'q' to QUIT"

        cv2.putText(final_display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(final_display_frame, capture_text, (10, CAMERA_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(final_display_frame, save_text, (10, CAMERA_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, final_display_frame)


# --- Main program flow ---
try:
    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Start streaming using CameraManager
    if not camera_manager.start_stream():
        print("Failed to start camera stream. Exiting.")
        exit()

    print(f"Webcam started at {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}fps.")
    print(f"Instructions:")
    print(f"  - Press 'c' to CAPTURE a frame for annotation.")
    print(f"  - While a frame is captured, click {POINTS_PER_KEY} points to define a keycap.")
    print(f"  - After 4 clicks, an input box will appear. Type the key value and press ENTER.")
    print(f"  - Use '+' (or '=') to ZOOM IN, '-' to ZOOM OUT.")
    print(f"  - Use ARROW keys to PAN the zoomed view.")
    print(f"  - Press 'r' to RESET zoom and pan.")
    print(f"  - Press 's' to SAVE all annotations to '{output_filename}'.")
    print(f"  - Press 'q' to QUIT the program.")

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    is_live_view = True

    while True:
        # Get frames from CameraManager
        color_image, aligned_depth_frame, (depth_width, depth_height) = camera_manager.get_frames()

        if color_image is None:
            continue

        current_raw_frame = color_image

        if is_live_view:
            draw_current_frame_with_annotations()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            is_live_view = False
            temp_key_points = []
            draw_current_frame_with_annotations()
            print(f"--- Frame CAPTURED --- Click {POINTS_PER_KEY} points for the current keycap.")
        elif key == ord('s'):
            if annotations:
                with open(output_filename, 'w') as f:
                    json.dump(annotations, f, indent=4)
                print(f"✓ Annotations saved to {output_filename}")
                print(f"✓ Total keys annotated: {len(annotations)}")
            else:
                print("No annotations to save.")
        elif key == ord('r'):
            zoom_factor = 1.0
            pan_x = 0
            pan_y = 0
            temp_key_points = []
            print("Zoom and pan RESET.")
            draw_current_frame_with_annotations()
        elif key == ord('+') or key == ord('='):
            old_view_width = CAMERA_WIDTH / zoom_factor
            old_view_height = CAMERA_HEIGHT / zoom_factor
            zoom_factor = min(max_zoom_factor, zoom_factor + 0.2)
            new_view_width = CAMERA_WIDTH / zoom_factor
            new_view_height = CAMERA_HEIGHT / zoom_factor
            pan_x += int((old_view_width - new_view_width) / 2)
            pan_y += int((old_view_height - new_view_height) / 2)
            print(f"Zoom: {zoom_factor:.1f}x")
            draw_current_frame_with_annotations()
        elif key == ord('-'):
            old_view_width = CAMERA_WIDTH / zoom_factor
            old_view_height = CAMERA_HEIGHT / zoom_factor
            zoom_factor = max(min_zoom_factor, zoom_factor - 0.2)
            new_view_width = CAMERA_WIDTH / zoom_factor
            new_view_height = CAMERA_HEIGHT / zoom_factor
            pan_x -= int((new_view_width - old_view_width) / 2)
            pan_y -= int((new_view_height - old_view_height) / 2)
            print(f"Zoom: {zoom_factor:.1f}x")
            draw_current_frame_with_annotations()
        elif key == 82:  # Up arrow
            pan_y = max(0, pan_y - pan_speed)
            draw_current_frame_with_annotations()
        elif key == 84:  # Down arrow
            max_pan_y = CAMERA_HEIGHT - int(CAMERA_HEIGHT / zoom_factor)
            pan_y = min(max_pan_y, pan_y + pan_speed)
            draw_current_frame_with_annotations()
        elif key == 81:  # Left arrow
            pan_x = max(0, pan_x - pan_speed)
            draw_current_frame_with_annotations()
        elif key == 83:  # Right arrow
            max_pan_x = CAMERA_WIDTH - int(CAMERA_WIDTH / zoom_factor)
            pan_x = min(max_pan_x, pan_x + pan_speed)
            draw_current_frame_with_annotations()
        elif key == ord('q'):
            print("Exiting program.")
            break

finally:
    print("Stopping stream and cleaning up...")
    camera_manager.stop_stream()
    cv2.destroyAllWindows()