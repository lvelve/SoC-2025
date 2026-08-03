"""
Distortion field generation and data augmentation for ForCenNet.

Provides:
- Random TPS (Thin Plate Spline) distortion field generation
- Forward/backward mapping inversion
- Image warping utilities
- Online data augmentation (crop, flip, rotate)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RBFInterpolator
from typing import Tuple, Optional, List
import cv2


# ============================================================
# TPS Distortion Field Generation
# ============================================================

def generate_tps_control_points(
    grid_size: int = 4,
    perturbation: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate source and target control points for TPS transformation.
    
    Args:
        grid_size: Number of control points per axis (grid_size x grid_size).
        perturbation: Maximum perturbation as fraction of cell size.
        seed: Random seed for reproducibility.
    
    Returns:
        source_pts: (N, 2) source control points in [0, 1].
        target_pts: (N, 2) target control points in [0, 1].
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Create uniform grid of control points
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    source_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (N, 2)
    
    # Add random perturbation to create target points
    cell_size = 1.0 / (grid_size - 1)
    max_offset = perturbation * cell_size
    offsets = rng.uniform(-max_offset, max_offset, size=source_pts.shape)
    
    # Keep boundary points fixed to avoid extreme distortions
    boundary_mask = (
        (source_pts[:, 0] == 0) | (source_pts[:, 0] == 1) |
        (source_pts[:, 1] == 0) | (source_pts[:, 1] == 1)
    )
    offsets[boundary_mask] = 0
    
    target_pts = source_pts + offsets
    # Clip to valid range
    target_pts = np.clip(target_pts, 0, 1)
    
    return source_pts, target_pts


def compute_tps_mapping(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward and backward mapping fields using TPS interpolation.
    
    Args:
        source_pts: (N, 2) source control points.
        target_pts: (N, 2) target control points.
        height: Output image height.
        width: Output image width.
    
    Returns:
        forward_map: (H, W, 2) mapping from distorted to undistorted coordinates (normalized [-1, 1]).
        backward_map: (H, W, 2) mapping from undistorted to distorted coordinates (normalized [-1, 1]).
    """
    # Create output grid (normalized coordinates [0, 1])
    gx = np.linspace(0, 1, width)
    gy = np.linspace(0, 1, height)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (H*W, 2)
    
    # Forward mapping: source -> target
    # Given a point in source space, find where it maps to in target space
    rbf_x = RBFInterpolator(source_pts, target_pts[:, 0], kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(source_pts, target_pts[:, 1], kernel='thin_plate_spline')
    
    forward_x = rbf_x(grid_points).reshape(height, width)
    forward_y = rbf_y(grid_points).reshape(height, width)
    forward_map = np.stack([forward_x, forward_y], axis=-1)  # (H, W, 2) in [0, 1]
    
    # Backward mapping: target -> source
    rbf_x_inv = RBFInterpolator(target_pts, source_pts[:, 0], kernel='thin_plate_spline')
    rbf_y_inv = RBFInterpolator(target_pts, source_pts[:, 1], kernel='thin_plate_spline')
    
    backward_x = rbf_x_inv(grid_points).reshape(height, width)
    backward_y = rbf_y_inv(grid_points).reshape(height, width)
    backward_map = np.stack([backward_x, backward_y], axis=-1)  # (H, W, 2) in [0, 1]
    
    return forward_map, backward_map


def generate_random_distortion_field(
    height: int = 288,
    width: int = 288,
    grid_size: int = 4,
    perturbation: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random distortion field using TPS.
    
    Args:
        height: Image height.
        width: Image width.
        grid_size: TPS control point grid size.
        perturbation: Perturbation strength.
        seed: Random seed.
    
    Returns:
        forward_map: (H, W, 2) normalized to [-1, 1] for grid_sample.
        backward_map: (H, W, 2) normalized to [-1, 1] for grid_sample.
    """
    src_pts, tgt_pts = generate_tps_control_points(grid_size, perturbation, seed)
    forward_map, backward_map = compute_tps_mapping(src_pts, tgt_pts, height, width)
    
    # Normalize from [0, 1] to [-1, 1] for PyTorch grid_sample
    forward_map = forward_map * 2 - 1
    backward_map = backward_map * 2 - 1
    
    return forward_map, backward_map


# ============================================================
# Image Warping Utilities
# ============================================================

def warp_image(
    image: np.ndarray,
    mapping_field: np.ndarray,
    mode: str = 'bilinear'
) -> np.ndarray:
    """
    Warp an image using a mapping field (grid_sample style).
    
    Args:
        image: (H, W, C) image as numpy array.
        mapping_field: (H, W, 2) mapping field in [-1, 1] (grid coordinates for grid_sample).
        mode: Interpolation mode ('bilinear' or 'nearest').
    
    Returns:
        warped: (H, W, C) warped image.
    """
    h, w = image.shape[:2]
    
    # Convert to torch tensors
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    
    img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    grid_tensor = torch.from_numpy(mapping_field).float().unsqueeze(0)  # (1, H, W, 2)
    
    warped_tensor = F.grid_sample(
        img_tensor, grid_tensor,
        mode=mode, padding_mode='zeros', align_corners=True
    )
    
    warped = warped_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
    
    if warped.shape[2] == 1:
        warped = warped.squeeze(2)
    
    return warped


def warp_mask(
    mask: np.ndarray,
    mapping_field: np.ndarray
) -> np.ndarray:
    """
    Warp a binary mask using nearest-neighbor interpolation.
    
    Args:
        mask: (H, W) binary mask.
        mapping_field: (H, W, 2) mapping field in [-1, 1].
    
    Returns:
        warped_mask: (H, W) warped binary mask.
    """
    return warp_image(mask, mapping_field, mode='nearest')


def warp_points(
    points: np.ndarray,
    forward_map: np.ndarray,
    image_height: int,
    image_width: int
) -> np.ndarray:
    """
    Warp point coordinates using a forward mapping field.
    
    Args:
        points: (N, 2) point coordinates [x, y] in pixel space.
        forward_map: (H, W, 2) mapping in [-1, 1] for grid_sample.
        image_height: Image height.
        image_width: Image width.
    
    Returns:
        warped_points: (N, 2) warped point coordinates in pixel space.
    """
    if len(points) == 0:
        return points.copy()
    
    # Normalize points to [-1, 1]
    norm_pts = points.copy().astype(np.float64)
    norm_pts[:, 0] = norm_pts[:, 0] / (image_width - 1) * 2 - 1
    norm_pts[:, 1] = norm_pts[:, 1] / (image_height - 1) * 2 - 1
    
    # Use grid_sample to interpolate the mapping field at point locations
    grid_tensor = torch.from_numpy(forward_map).float().unsqueeze(0)  # (1, H, W, 2)
    pts_tensor = torch.from_numpy(norm_pts).float().unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
    
    sampled = F.grid_sample(
        grid_tensor.permute(0, 3, 1, 2),  # (1, 2, H, W)
        pts_tensor,
        mode='bilinear', padding_mode='border', align_corners=True
    )  # (1, 2, 1, N)
    
    sampled = sampled.squeeze().numpy()  # (2, N)
    warped_points = np.stack([sampled[0], sampled[1]], axis=-1)  # (N, 2)
    
    # Convert back from [-1, 1] to pixel space
    warped_points[:, 0] = (warped_points[:, 0] + 1) / 2 * (image_width - 1)
    warped_points[:, 1] = (warped_points[:, 1] + 1) / 2 * (image_height - 1)
    
    return warped_points


def invert_mapping_field(mapping_field: np.ndarray) -> np.ndarray:
    """
    Invert a mapping field by swapping source/target roles.
    Uses grid_sample to approximate the inverse.
    
    Args:
        mapping_field: (H, W, 2) mapping field in [-1, 1].
    
    Returns:
        inverse_field: (H, W, 2) inverse mapping field in [-1, 1].
    """
    h, w = mapping_field.shape[:2]
    
    # Create identity grid
    gx = np.linspace(-1, 1, w)
    gy = np.linspace(-1, 1, h)
    grid_x, grid_y = np.meshgrid(gx, gy)
    identity = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    
    # The mapping field maps from identity space to distorted space.
    # We want to find, for each point in identity space, where it came from.
    # Simple iterative inverse (fixed-point iteration)
    current = identity.copy()
    for _ in range(5):
        # Sample the mapping at current locations
        grid_tensor = torch.from_numpy(mapping_field).float().unsqueeze(0)
        current_tensor = torch.from_numpy(current).float().unsqueeze(0)
        
        sampled = F.grid_sample(
            grid_tensor.permute(0, 3, 1, 2),
            current_tensor,
            mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0).permute(1, 2, 0).numpy()
        
        # Newton-like step: residual = mapping(current) - identity
        # We want mapping(x) = identity, so adjust
        residual = identity - sampled
        current = current + 0.5 * residual
    
    return current


# ============================================================
# Data Augmentation
# ============================================================

class RandomAugmentation:
    """Online data augmentation for training pairs."""
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
        max_rotate_angle: float = 10.0,
        crop_prob: float = 0.3
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotate_angle = max_rotate_angle
        self.crop_prob = crop_prob
    
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply random augmentations to training pair.
        
        Args:
            image: (H, W, 3) distorted image.
            mask: (H, W) distorted foreground mask.
            bm: (H, W, 2) backward mapping field in [-1, 1].
        
        Returns:
            Augmented (image, mask, bm).
        """
        h, w = image.shape[:2]
        
        # Random horizontal flip
        if np.random.random() < self.flip_prob:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            bm = np.flip(bm, axis=1).copy()
            # Flip x-component of mapping
            bm[:, :, 0] = -bm[:, :, 0]
        
        # Random vertical flip
        if np.random.random() < self.flip_prob:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
            bm = np.flip(bm, axis=0).copy()
            # Flip y-component of mapping
            bm[:, :, 1] = -bm[:, :, 1]
        
        # Random small-angle rotation
        if np.random.random() < self.rotate_prob:
            angle = np.random.uniform(-self.max_rotate_angle, self.max_rotate_angle)
            image, mask, bm = self._rotate(image, mask, bm, angle)
        
        return image.copy(), mask.copy(), bm.copy()
    
    def _rotate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bm: np.ndarray,
        angle: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate all three tensors consistently."""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image_rot = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_rot = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Rotate mapping field components
        bm_rot = np.zeros_like(bm)
        bm_rot[:, :, 0] = cv2.warpAffine(bm[:, :, 0], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bm_rot[:, :, 1] = cv2.warpAffine(bm[:, :, 1], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return image_rot, mask_rot.astype(np.float32), bm_rot


# ============================================================
# Background Composition (Optional)
# ============================================================

def composite_on_background(
    image: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray
) -> np.ndarray:
    """
    Composite distorted document image onto a background.
    
    Args:
        image: (H, W, 3) distorted document image [0, 255].
        mask: (H, W) foreground mask [0, 1].
        background: (H, W, 3) background image [0, 255].
    
    Returns:
        composited: (H, W, 3) composited image.
    """
    if background.shape[:2] != image.shape[:2]:
        background = cv2.resize(background, (image.shape[1], image.shape[0]))
    
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    
    composited = image * mask + background * (1 - mask)
    return composited.astype(np.uint8)


# ============================================================
# Full Training Pair Generation
# ============================================================

def generate_training_pair(
    image: np.ndarray,
    mask: np.ndarray,
    line_points: Optional[List[np.ndarray]] = None,
    grid_size: int = 4,
    perturbation: float = 0.1,
    image_size: int = 288,
    seed: Optional[int] = None
) -> dict:
    """
    Generate a complete training pair from an undistorted document image.
    
    Args:
        image: (H, W, 3) undistorted document image.
        mask: (H, W) foreground mask (binary).
        line_points: List of (N_i, 2) arrays of line control points.
        grid_size: TPS control point grid size.
        perturbation: Distortion strength.
        image_size: Output size (square).
        seed: Random seed.
    
    Returns:
        dict with keys:
            'distorted_image': (image_size, image_size, 3)
            'distorted_mask': (image_size, image_size)
            'backward_map': (image_size, image_size, 2) in [-1, 1]
            'forward_map': (image_size, image_size, 2) in [-1, 1]
            'distorted_line_points': list of (N_i, 2) arrays (optional)
    """
    # Resize input to target size
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    
    # Generate distortion field
    forward_map, backward_map = generate_random_distortion_field(
        image_size, image_size, grid_size, perturbation, seed
    )
    
    # Apply forward map to create distorted image
    distorted_image = warp_image(image, forward_map, mode='bilinear')
    distorted_mask = warp_mask(mask, forward_map)
    
    # Warp line points if provided
    distorted_line_points = None
    if line_points is not None:
        distorted_line_points = []
        for pts in line_points:
            if len(pts) > 0:
                warped_pts = warp_points(pts, forward_map, image_size, image_size)
                distorted_line_points.append(warped_pts)
            else:
                distorted_line_points.append(pts.copy())
    
    # Clip distorted image to valid range
    distorted_image = np.clip(distorted_image, 0, 255).astype(np.uint8)
    distorted_mask = (distorted_mask > 0.5).astype(np.float32)
    
    result = {
        'distorted_image': distorted_image,
        'distorted_mask': distorted_mask,
        'backward_map': backward_map.astype(np.float32),
        'forward_map': forward_map.astype(np.float32),
    }
    
    if distorted_line_points is not None:
        result['distorted_line_points'] = distorted_line_points
    
    return result