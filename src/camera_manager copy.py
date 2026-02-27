import cv2
import numpy as np


class MockDepthFrame:
    """Mock depth frame to simulate RealSense depth frame interface."""
    
    def __init__(self, depth_map, width, height):
        self.depth_map = depth_map  # 2D numpy array with depth values in meters
        self.width = width
        self.height = height
    
    def get_distance(self, x, y):
        """Get depth at pixel coordinates (x, y) in meters."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.depth_map[y, x])
        return 0.0
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height


class CameraManager:
    """Webcam-based camera manager with depth estimation."""
    
    def __init__(self, color_width=848, color_height=480, depth_width=848, depth_height=480, fps=30):
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        
        # Webcam capture object
        self.cap = None
        
        # Depth estimator (will be set externally)
        self.depth_estimator = None
        
        # Default depth for fallback
        self.default_depth = 0.458  # meters
        
        # Depth estimation settings
        self.estimate_every_n_frames = 3  # Estimate depth every N frames for performance
        self.frame_count = 0
        self.last_depth_map = None
    
    def get_resolution(self):
        return self.color_width, self.color_height, self.depth_width, self.depth_height, self.fps
    
    def set_depth_estimator(self, depth_estimator):
        """Set the depth estimation model."""
        self.depth_estimator = depth_estimator
        print("Depth estimator connected to camera manager")
    
    def start_stream(self):
        """Initialize webcam stream."""
        print("Starting webcam stream...")
        
        # Try to open webcam (0 is usually the default camera)
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.color_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.color_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Webcam started: {actual_width}x{actual_height} @ {self.fps}fps")
        
        if self.depth_estimator:
            print("Depth estimation: ENABLED")
        else:
            print("Depth estimation: DISABLED (using placeholder values)")
            print("Call set_depth_estimator() to enable depth estimation")
        
        return True
    
    def _generate_placeholder_depth(self, color_image):
        """Generate placeholder depth map with default values."""
        height, width = color_image.shape[:2]
        depth_map = np.full((height, width), self.default_depth, dtype=np.float32)
        return depth_map
    
    def _estimate_depth(self, color_image):
        """Estimate depth from color image using depth model."""
        if self.depth_estimator is None:
            print("Warning: No depth estimator set. Using placeholder values.")
            return self._generate_placeholder_depth(color_image)
        
        try:
            # Estimate depth every N frames for performance
            self.frame_count += 1
            
            if self.frame_count % self.estimate_every_n_frames == 0 or self.last_depth_map is None:
                depth_map = self.depth_estimator.estimate_depth(color_image)
                self.last_depth_map = depth_map
                
                # Debug: print depth range every 30 frames
                if self.frame_count % 30 == 0:
                    stats = self.depth_estimator.get_depth_stats(depth_map)
                    print(f"Depth range: {stats['min']:.3f}m - {stats['max']:.3f}m "
                          f"(avg: {stats['mean']:.3f}m)")
            else:
                # Reuse previous depth map
                depth_map = self.last_depth_map
            
            return depth_map
            
        except Exception as e:
            print(f"Depth estimation error: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_placeholder_depth(color_image)
    
    def get_frames(self):
        """
        Get color frame and depth frame from webcam.
        Returns: (color_image, mock_depth_frame, depth_dimensions)
        """
        if self.cap is None or not self.cap.isOpened():
            return None, None, None
        
        # Read frame from webcam
        ret, color_image = self.cap.read()
        
        if not ret or color_image is None:
            return None, None, None
        
        # Resize if needed
        if color_image.shape[1] != self.color_width or color_image.shape[0] != self.color_height:
            color_image = cv2.resize(color_image, (self.color_width, self.color_height))
        
        # Make image writable for MediaPipe
        color_image.flags.writeable = True
        
        # Generate depth map
        depth_map = self._estimate_depth(color_image)
        
        # Resize depth map to match depth dimensions
        if depth_map.shape[1] != self.depth_width or depth_map.shape[0] != self.depth_height:
            depth_map = cv2.resize(depth_map, (self.depth_width, self.depth_height))
        
        # Create mock depth frame with RealSense-like interface
        mock_depth_frame = MockDepthFrame(
            depth_map, 
            self.depth_width, 
            self.depth_height
        )
        
        depth_dimensions = (self.depth_width, self.depth_height)
        
        return color_image, mock_depth_frame, depth_dimensions
    
    def stop_stream(self):
        """Stop webcam stream."""
        if self.cap is not None:
            print("Stopping webcam stream.")
            self.cap.release()
            self.cap = None