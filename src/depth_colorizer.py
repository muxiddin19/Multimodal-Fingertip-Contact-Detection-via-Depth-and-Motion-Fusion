import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class DepthColorizer:
    """
    A class for dynamic depth colorization of Intel RealSense camera data.
    Provides various colorization schemes and adaptive scaling.
    """

    def __init__(self, min_distance: float = 0.1, max_distance: float = 4.0):
        """
        Initialize the depth colorizer.

        Args:
            min_distance: Minimum distance in meters for colorization
            max_distance: Maximum distance in meters for colorization
        """
        self.min_distance = min_distance
        self.max_distance = max_distance

        # Create colorizer for RealSense
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.visual_preset, 0)  # Dynamic

        # Custom color schemes
        self.color_schemes = {
            'jet': cv2.COLORMAP_JET,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA,
            'inferno': cv2.COLORMAP_INFERNO,
            'magma': cv2.COLORMAP_MAGMA,
            'hot': cv2.COLORMAP_HOT,
            'cool': cv2.COLORMAP_COOL,
            'ocean': cv2.COLORMAP_OCEAN,
            'turbo': cv2.COLORMAP_TURBO
        }

        self.current_scheme = 'jet'
        self.adaptive_scaling = True

    def normalize_depth(self, depth_frame: np.ndarray,
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None) -> np.ndarray:
        """
        Normalize depth values to 0-255 range.

        Args:
            depth_frame: Raw depth frame data
            min_val: Minimum depth value for normalization
            max_val: Maximum depth value for normalization

        Returns:
            Normalized depth frame (0-255)
        """
        if min_val is None or max_val is None:
            if self.adaptive_scaling:
                # Use percentiles for better visualization
                valid_depth = depth_frame[depth_frame > 0]
                if len(valid_depth) > 0:
                    min_val = np.percentile(valid_depth, 5)
                    max_val = np.percentile(valid_depth, 95)
                else:
                    min_val, max_val = self.min_distance * 1000, self.max_distance * 1000
            else:
                min_val = self.min_distance * 1000  # Convert to mm
                max_val = self.max_distance * 1000

        # Normalize to 0-255
        normalized = np.clip((depth_frame - min_val) / (max_val - min_val), 0, 1)
        return (normalized * 255).astype(np.uint8)

    def apply_colormap(self, normalized_depth: np.ndarray,
                       scheme: str = None) -> np.ndarray:
        """
        Apply color scheme to normalized depth data.

        Args:
            normalized_depth: Normalized depth frame (0-255)
            scheme: Color scheme name

        Returns:
            Colorized depth frame
        """
        if scheme is None:
            scheme = self.current_scheme

        if scheme in self.color_schemes:
            colorized = cv2.applyColorMap(normalized_depth, self.color_schemes[scheme])
        else:
            colorized = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        return colorized

    def create_custom_colormap(self, colors: list) -> np.ndarray:
        """
        Create a custom colormap from a list of colors.

        Args:
            colors: List of RGB tuples [(r,g,b), ...]

        Returns:
            Custom colormap lookup table
        """
        n_colors = len(colors)
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)

        for i in range(256):
            # Interpolate between colors
            pos = i / 255.0 * (n_colors - 1)
            idx = int(pos)
            frac = pos - idx

            if idx >= n_colors - 1:
                colormap[i, 0] = colors[-1]
            else:
                color1 = np.array(colors[idx])
                color2 = np.array(colors[idx + 1])
                colormap[i, 0] = (color1 * (1 - frac) + color2 * frac).astype(np.uint8)

        return colormap

    def colorize_depth_frame(self, depth_frame: np.ndarray,
                             scheme: str = None,
                             show_invalid: bool = True) -> np.ndarray:
        """
        Complete depth colorization pipeline.

        Args:
            depth_frame: Raw depth frame data
            scheme: Color scheme to use
            show_invalid: Whether to show invalid (0) depth values

        Returns:
            Colorized depth frame
        """
        # Create mask for invalid depth values
        invalid_mask = depth_frame == 0

        # Normalize depth
        normalized = self.normalize_depth(depth_frame)

        # Apply colormap
        colorized = self.apply_colormap(normalized, scheme)

        # Handle invalid depth values
        if show_invalid:
            colorized[invalid_mask] = [0, 0, 0]  # Black for invalid

        return colorized

    def add_depth_scale(self, image: np.ndarray,
                        min_val: float, max_val: float) -> np.ndarray:
        """
        Add a depth scale bar to the image.

        Args:
            image: Colorized depth image
            min_val: Minimum depth value
            max_val: Maximum depth value

        Returns:
            Image with depth scale
        """
        h, w = image.shape[:2]

        # Create scale bar
        scale_width = 20
        scale_height = h - 60
        scale_x = w - 40
        scale_y = 30

        # Create gradient for scale
        gradient = np.linspace(255, 0, scale_height).astype(np.uint8)
        scale_bar = np.tile(gradient[:, np.newaxis], (1, scale_width))

        # Apply same colormap as image
        scale_colored = self.apply_colormap(scale_bar, self.current_scheme)

        # Add scale to image
        result = image.copy()
        result[scale_y:scale_y + scale_height, scale_x:scale_x + scale_width] = scale_colored

        # Add text labels
        cv2.putText(result, f"{max_val / 1000:.1f}m",
                    (scale_x - 50, scale_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, f"{min_val / 1000:.1f}m",
                    (scale_x - 50, scale_y + scale_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return result

    def get_depth_statistics(self, depth_frame: np.ndarray) -> dict:
        """
        Calculate depth statistics for the current frame.

        Args:
            depth_frame: Raw depth frame data

        Returns:
            Dictionary with depth statistics
        """
        valid_depth = depth_frame[depth_frame > 0]

        if len(valid_depth) == 0:
            return {
                'min': 0, 'max': 0, 'mean': 0, 'median': 0,
                'std': 0, 'valid_pixels': 0, 'total_pixels': depth_frame.size
            }

        return {
            'min': np.min(valid_depth),
            'max': np.max(valid_depth),
            'mean': np.mean(valid_depth),
            'median': np.median(valid_depth),
            'std': np.std(valid_depth),
            'valid_pixels': len(valid_depth),
            'total_pixels': depth_frame.size
        }

    def set_color_scheme(self, scheme: str):
        """Set the current color scheme."""
        if scheme in self.color_schemes:
            self.current_scheme = scheme

    def set_adaptive_scaling(self, adaptive: bool):
        """Enable or disable adaptive scaling."""
        self.adaptive_scaling = adaptive

    def set_distance_range(self, min_dist: float, max_dist: float):
        """Set the distance range for colorization."""
        self.min_distance = min_dist
        self.max_distance = max_dist


class RealSenseDepthVisualizer:
    """
    Complete visualization system for Intel RealSense depth data.
    """

    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize the depth visualizer.

        Args:
            width: Frame width
            height: Frame height
        """
        self.width = width
        self.height = height

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

        # Initialize colorizer
        self.colorizer = DepthColorizer()

        # Alignment object
        self.align = rs.align(rs.stream.color)

        # Current frame data
        self.current_depth = None
        self.current_color = None

    def start(self):
        """Start the camera pipeline."""
        try:
            profile: rs.pipeline_profile = self.pipeline.start(self.config)
            device: rs.device = profile.get_device()
            # print(len(device.query_sensors()))
            # depth_sensor: rs.depth_sensor = device.first_depth_sensor()
            # preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            # for i in range(int(preset_range.max)):
            #     visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
            #     print('%02dd: %s' % (i, visulpreset))
            #     if visulpreset == 'High Accuracy':
            #         depth_sensor.set_option(rs.option.visual_preset, i)
            print("RealSense camera started successfully")
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
        return True

    def stop(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()
        print("RealSense camera stopped")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the current depth and color frames.

        Returns:
            Tuple of (depth_frame, color_frame)
        """
        try:
            frames = self.pipeline.wait_for_frames()

            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            self.current_depth = depth_image
            self.current_color = color_image

            return depth_image, color_image

        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None

    def visualize_depth(self, depth_frame: np.ndarray,
                        scheme: str = 'jet',
                        show_stats: bool = True,
                        show_scale: bool = True) -> np.ndarray:
        """
        Visualize depth frame with colorization.

        Args:
            depth_frame: Raw depth frame
            scheme: Color scheme
            show_stats: Whether to show depth statistics
            show_scale: Whether to show depth scale

        Returns:
            Colorized depth visualization
        """
        # Set color scheme
        self.colorizer.set_color_scheme(scheme)

        # Colorize depth
        colorized = self.colorizer.colorize_depth_frame(depth_frame)

        # Add depth scale
        if show_scale:
            stats = self.colorizer.get_depth_statistics(depth_frame)
            if stats['valid_pixels'] > 0:
                colorized = self.colorizer.add_depth_scale(
                    colorized, stats['min'], stats['max']
                )

        # Add statistics text
        if show_stats:
            stats = self.colorizer.get_depth_statistics(depth_frame)
            if stats['valid_pixels'] > 0:
                text_lines = [
                    f"Min: {stats['min'] / 1000:.2f}m",
                    f"Max: {stats['max'] / 1000:.2f}m",
                    f"Mean: {stats['mean'] / 1000:.2f}m",
                    f"Valid: {stats['valid_pixels']}/{stats['total_pixels']}"
                ]

                for i, line in enumerate(text_lines):
                    cv2.putText(colorized, line, (10, 30 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return colorized

    def run_visualization(self):
        """
        Run the interactive depth visualization.
        """
        if not self.start():
            return

        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'c' to cycle color schemes")
        print("- Press 'a' to toggle adaptive scaling")
        print("- Press 's' to toggle statistics display")
        print("- Press 'r' to toggle scale bar")

        current_scheme_idx = 0
        schemes = list(self.colorizer.color_schemes.keys())
        show_stats = True
        show_scale = True

        try:
            while True:
                # Get frames
                depth_frame, color_frame = self.get_frames()

                if depth_frame is None or color_frame is None:
                    continue

                # Visualize depth
                current_scheme = schemes[current_scheme_idx]
                depth_colorized = self.visualize_depth(
                    depth_frame, current_scheme, show_stats, show_scale
                )

                # Create combined view
                combined = np.hstack([color_frame, depth_colorized])

                # Add title
                cv2.putText(combined, f"Color | Depth ({current_scheme})",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display
                cv2.imshow('RealSense Depth Visualization', combined)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    current_scheme_idx = (current_scheme_idx + 1) % len(schemes)
                    print(f"Switched to {schemes[current_scheme_idx]} color scheme")
                elif key == ord('a'):
                    self.colorizer.set_adaptive_scaling(not self.colorizer.adaptive_scaling)
                    print(f"Adaptive scaling: {self.colorizer.adaptive_scaling}")
                elif key == ord('s'):
                    show_stats = not show_stats
                    print(f"Statistics display: {show_stats}")
                elif key == ord('r'):
                    show_scale = not show_scale
                    print(f"Scale bar: {show_scale}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
            cv2.destroyAllWindows()


def main():
    """Main function to run the depth visualization."""
    visualizer = RealSenseDepthVisualizer()
    visualizer.run_visualization()


if __name__ == "__main__":
    main()