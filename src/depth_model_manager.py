import torch
import cv2
import numpy as np
import os
import torch.nn.functional as F


class DepthEstimator:
    """
    Production-ready depth estimation using Depth Anything V2.
    
    This class provides real-time metric depth estimation with:
    - GPU acceleration support
    - Calibration capabilities for accurate measurements
    - Efficient inference pipeline
    """
    
    def __init__(self, custom_checkpoint=None, device=None):
        """
        Initialize depth estimator with Depth Anything V2.
        
        Args:
            custom_checkpoint: Path to custom Depth Anything V2 checkpoint
                             Default: Uses your fine-tuned model
            device: 'cuda' or 'cpu'. If None, automatically selects GPU if available.
        """
        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("⚠ Using CPU (GPU recommended for better performance)")
        else:
            self.device = device
        
        print(f"Initializing Depth Estimator on {self.device.upper()}")
        
        self.custom_checkpoint = custom_checkpoint
        self.model = None
        
        # Calibration parameters
        self.scale = 1.0
        self.offset = 0.0
        self.is_calibrated = False
        
        self._load_model()
    
    def _load_model(self):
        """Load Depth Anything V2 model."""
        print("\nLoading Depth Anything V2 model...")
        
        checkpoint_path = self.custom_checkpoint or r"D:\Codes\vscode\Pretrained_weights\depth_anything_v2\latest20.pth"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
        
        try:
            # Import Depth Anything V2
            import sys
            sys.path.insert(0, r"D:\Codes\vscode\Depth_Anything_V2_main\metric_depth")
            from depth_anything_v2.dpt import DepthAnythingV2
            
            # Model configuration - must match the encoder used during training
            # Training uses vits (ViT-Small) with max_depth=0.4
            model_config = {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384],
                'max_depth': 0.4
            }
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Strip DDP 'module.' prefix if present (saved by DistributedDataParallel)
            cleaned = {}
            for k, v in state_dict.items():
                cleaned[k.replace('module.', '', 1)] = v
            state_dict = cleaned

            # Initialize model
            self.model = DepthAnythingV2(**model_config)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully!")
            print(f"  Architecture: Vision Transformer Base (ViT-B)")
            print(f"  Ready for inference\n")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def estimate_depth(self, image, return_raw=False):
        """
        Estimate depth map from RGB image.
        
        Args:
            image: RGB image as numpy array (H, W, 3) in BGR format (OpenCV)
                  or RGB format
            return_raw: If True, return both calibrated and raw depth maps
            
        Returns:
            depth_map: Depth map in meters (H, W) as numpy array
            raw_depth_map: (optional) Raw depth before calibration
            
        Example:
            >>> estimator = DepthEstimator()
            >>> image = cv2.imread('photo.jpg')
            >>> depth = estimator.estimate_depth(image)
            >>> print(f"Depth at center: {depth[240, 320]:.2f}m")
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(image)}")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")
        
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        with torch.no_grad():
            h_orig, w_orig = image_rgb.shape[:2]
            
            # Resize to model input size (518x518, divisible by 14)
            target_size = 518
            h_new = (target_size // 14) * 14
            w_new = (target_size // 14) * 14
            
            image_resized = cv2.resize(image_rgb, (w_new, h_new), 
                                      interpolation=cv2.INTER_LINEAR)
            
            # Prepare tensor and normalize
            image_tensor = torch.from_numpy(image_resized).float()
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device) / 255.0
            
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            image_tensor = (image_tensor - mean) / std
            
            # Forward pass
            depth_pred = self.model(image_tensor)
            
            # Resize back to original dimensions
            depth_map = F.interpolate(
                depth_pred[:, None] if depth_pred.dim() == 3 else depth_pred,
                size=(h_orig, w_orig),
                mode='bilinear',
                align_corners=True
            )
            
            if depth_map.dim() == 4:
                depth_map = depth_map[0, 0]
            depth_map = depth_map.cpu().numpy()
        
        # Apply calibration if available
        if self.is_calibrated:
            depth_calibrated = depth_map * self.scale + self.offset
            if return_raw:
                return depth_calibrated, depth_map
            return depth_calibrated
        else:
            if return_raw:
                return depth_map, depth_map.copy()
            return depth_map
    
    def calibrate(self, reference_depth_m, depth_map, point):
        """
        Calibrate depth scale using a known reference point.
        
        This improves absolute depth accuracy by measuring a known distance
        in the scene and adjusting the scale accordingly.
        
        Args:
            reference_depth_m: Known true depth in meters
            depth_map: Current depth map from estimate_depth()
            point: (x, y) pixel coordinates of the reference point
            
        Example:
            >>> depth_map = estimator.estimate_depth(image)
            >>> # If we know point (320, 240) is 1.5 meters away:
            >>> estimator.calibrate(1.5, depth_map, (320, 240))
            >>> # Now future predictions will be calibrated
        """
        x, y = point
        
        # Ensure point is within bounds
        h, w = depth_map.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        measured_depth = depth_map[y, x]
        
        if measured_depth > 0.001:
            self.scale = reference_depth_m / measured_depth
            self.offset = 0.0
            self.is_calibrated = True
            
            print(f"\n{'='*50}")
            print(f"✓ Calibration Complete!")
            print(f"{'='*50}")
            print(f"  Reference point: ({x}, {y})")
            print(f"  Measured depth:  {measured_depth:.3f}m ({measured_depth*100:.1f}cm)")
            print(f"  True depth:      {reference_depth_m:.3f}m ({reference_depth_m*100:.1f}cm)")
            print(f"  Scale factor:    {self.scale:.3f}")
            print(f"  Future predictions will be calibrated automatically")
            print(f"{'='*50}\n")
        else:
            print(f"⚠ Warning: Invalid depth at reference point ({x}, {y})")
            print(f"  Measured depth: {measured_depth:.6f}m (too close to zero)")
    
    def reset_calibration(self):
        """Reset calibration to use raw model outputs."""
        self.scale = 1.0
        self.offset = 0.0
        self.is_calibrated = False
        print("✓ Calibration reset")
    
    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at specific pixel coordinates.
        
        Args:
            depth_map: Depth map from estimate_depth()
            x, y: Pixel coordinates
            
        Returns:
            Depth value in meters (float)
        """
        h, w = depth_map.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return float(depth_map[y, x])
    
    def get_depth_stats(self, depth_map):
        """
        Get statistical summary of depth map.
        
        Args:
            depth_map: Depth map from estimate_depth()
            
        Returns:
            Dictionary with statistics: min, max, mean, median, std
        """
        return {
            'min': float(depth_map.min()),
            'max': float(depth_map.max()),
            'mean': float(depth_map.mean()),
            'median': float(np.median(depth_map)),
            'std': float(depth_map.std())
        }
    
    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_MAGMA):
        """
        Create a colorized visualization of the depth map.
        
        Args:
            depth_map: Depth map from estimate_depth()
            colormap: OpenCV colormap (default: COLORMAP_MAGMA)
                     Options: COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_PLASMA, etc.
            
        Returns:
            Colorized depth map (H, W, 3) as uint8 RGB image
            
        Example:
            >>> depth = estimator.estimate_depth(image)
            >>> depth_viz = estimator.visualize_depth(depth)
            >>> cv2.imshow('Depth', depth_viz)
        """
        # Normalize to 0-255 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min > 1e-6:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_map)
        
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)
        
        # Convert BGR to RGB for consistency
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        return depth_colored
    
    def __repr__(self):
        """String representation of the estimator."""
        calib_status = "Calibrated" if self.is_calibrated else "Uncalibrated"
        return (f"DepthEstimator(model='Depth Anything V2', "
                f"device='{self.device}', status='{calib_status}')")


# Convenience function for quick depth estimation
def estimate_depth_from_image(image_path, checkpoint_path=None, visualize=False):
    """
    Quick one-liner depth estimation from an image file.
    
    Args:
        image_path: Path to image file
        checkpoint_path: Optional custom checkpoint path
        visualize: If True, return both depth map and visualization
        
    Returns:
        depth_map: Depth in meters
        depth_viz: (optional) Colorized visualization
        
    Example:
        >>> depth = estimate_depth_from_image('photo.jpg')
        >>> print(f"Average depth: {depth.mean():.2f}m")
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Create estimator and estimate
    estimator = DepthEstimator(custom_checkpoint=checkpoint_path)
    depth_map = estimator.estimate_depth(image)
    
    if visualize:
        depth_viz = estimator.visualize_depth(depth_map)
        return depth_map, depth_viz
    
    return depth_map