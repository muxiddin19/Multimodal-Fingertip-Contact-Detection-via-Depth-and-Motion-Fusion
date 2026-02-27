import json
import os
import numpy as np
import cv2

class KeyboardManager:
    def __init__(self, annotation_filename='src/keyboard_annotations.json', points_per_key=4):
        self.annotation_filename = annotation_filename
        self.points_per_key = points_per_key
        self.annotated_keys = self._load_annotations()

    def _load_annotations(self):
        if os.path.exists(self.annotation_filename):
            with open(self.annotation_filename, 'r') as f:
                data = json.load(f)
                validated_annotations = []
                for item in data:
                    if 'key' in item and 'points' in item and len(item['points']) == self.points_per_key:
                        validated_annotations.append(item)
                    else:
                        print(f"Warning: Skipping malformed annotation entry in {self.annotation_filename}: {item}")
                print(f"Loaded {len(validated_annotations)} annotated key(s) from {self.annotation_filename}")
                return validated_annotations
        else:
            print(f"Warning: Annotation file '{self.annotation_filename}' not found. No keycaps will be displayed.")
            print("Please run the 'Keyboard Annotation Tool' script first to create the annotation file.")
            return []

    def get_annotated_keys(self):
        return self.annotated_keys

    def is_point_in_keycap(self, finger_point, key_data):
        key_points_list = key_data['points']

        if len(key_points_list) == self.points_per_key:
            key_polygon = np.array([[p['x'], p['y']] for p in key_points_list], np.int32)
            # Check if the finger tip is inside the current keycap's polygon
            return cv2.pointPolygonTest(key_polygon, finger_point, False) >= 0
        return False