import cv2
import numpy as np

def draw_finger_tip_info(image, finger_pixel_x, finger_pixel_y, depth_at_finger_m):
    """Draws a circle and depth information for a detected finger tip."""
    cv2.circle(image, (finger_pixel_x, finger_pixel_y), 5, (0, 255, 255), -1)  # Yellow circle
    cv2.putText(image, f"Depth: {depth_at_finger_m:.4f}m",
                (finger_pixel_x + 10, finger_pixel_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

def draw_keycap_annotations(image, annotated_keys, active_key, points_per_key):
    """Draws the keyboard layout and highlights pressed keys."""
    for key_data in annotated_keys:
        key_points_list = key_data['points']
        key_value = key_data['key']

        if len(key_points_list) == points_per_key:
            pts = np.array([[p['x'], p['y']] for p in key_points_list], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Determine color based on whether the key is currently pressed
            if key_value == active_key:
                polygon_color = (0, 255, 0)  # Green for pressed key
                thickness = 2
                # Also draw the key name when pressed
                text_position = (pts[0][0][0] + 5, pts[0][0][1] + 20)
                cv2.putText(image, key_value, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, polygon_color, 2)
            else:
                polygon_color = (0, 0, 255)  # Red for non-pressed keys
                thickness = 1

            # Draw the polygon for the key
            cv2.polylines(image, [pts], True, polygon_color, thickness)

            # Draw circles for the annotation points
            for p in key_points_list:
                cv2.circle(image, (p['x'], p['y']), 1, polygon_color, -1)

# The display_text_overlays function has been removed as the tkinter UI now handles text display.
