"""
Visualization utilities for ForCenNet.

Provides functions for:
- Visualizing backward mapping fields
- Comparing distorted and rectified images
- Visualizing foreground masks
- Saving training progress visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
import os


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a numpy image.
    
    Args:
        tensor: (C, H, W) or (B, C, H, W) tensor in [0, 1]
    
    Returns:
        image: (H, W, C) numpy array in [0, 255] uint8
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        else:
            tensor = tensor.permute(1, 2, 0)
    
    image = tensor.detach().cpu().numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def visualize_mapping_field(
    bm: torch.Tensor,
    grid_size: int = 20
) -> np.ndarray:
    """
    Visualize a backward mapping field as a deformed grid.
    
    Args:
        bm: (2, H, W) or (B, 2, H, W) mapping field in [-1, 1]
        grid_size: spacing between grid lines
    
    Returns:
        vis: (H, W, 3) visualization image
    """
    if bm.dim() == 4:
        bm = bm[0]
    
    C, H, W = bm.shape
    device = bm.device
    dtype = bm.dtype
    
    # Create identity grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing='ij'
    )
    identity = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
    
    # Apply mapping
    deformed = identity + bm
    deformed = deformed.clamp(-1, 1)
    
    # Convert to pixel coordinates
    def_x = ((deformed[0] + 1) / 2 * (W - 1)).cpu().numpy()
    def_y = ((deformed[1] + 1) / 2 * (H - 1)).cpu().numpy()
    
    # Create visualization
    vis = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    # Draw horizontal lines (blue)
    for y in range(0, H, grid_size):
        for x in range(0, W - 1, 1):
            x1, y1 = int(def_x[y, x]), int(def_y[y, x])
            x2, y2 = int(def_x[y, x + 1]), int(def_y[y, x + 1])
            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                cv2.line(vis, (x1, y1), (x2, y2), (200, 100, 50), 1)
    
    # Draw vertical lines (red)
    for x in range(0, W, grid_size):
        for y in range(0, H - 1, 1):
            x1, y1 = int(def_x[y, x]), int(def_y[y, x])
            x2, y2 = int(def_x[y + 1, x]), int(def_y[y + 1, x])
            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                cv2.line(vis, (x1, y1), (x2, y2), (50, 100, 200), 1)
    
    return vis


def visualize_mask(
    mask: torch.Tensor,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Visualize a foreground mask with colormap.
    
    Args:
        mask: (H, W) or (1, H, W) or (B, H, W) mask tensor
        colormap: OpenCV colormap name
    
    Returns:
        vis: (H, W, 3) colored mask visualization
    """
    if mask.dim() == 3:
        mask = mask[0]
    if mask.dim() == 4:
        mask = mask[0, 0]
    
    mask_np = mask.detach().cpu().numpy()
    mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)
    
    # Apply colormap
    cmap = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
    vis = cv2.applyColorMap(mask_np, cmap)
    
    return vis


def create_comparison_figure(
    distorted: torch.Tensor,
    rectified: torch.Tensor,
    bm_pred: torch.Tensor,
    mask_logits: Optional[torch.Tensor] = None,
    bm_gt: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a side-by-side comparison figure.
    
    Args:
        distorted: (3, H, W) distorted image tensor
        rectified: (3, H, W) rectified image tensor
        bm_pred: (2, H, W) predicted backward mapping
        mask_logits: (2, H, W) optional mask logits
        bm_gt: (2, H, W) optional ground truth mapping
        save_path: optional path to save the figure
    
    Returns:
        figure: (H, W_total, 3) concatenated visualization
    """
    # Convert tensors to images
    dist_img = tensor_to_image(distorted)
    rect_img = tensor_to_image(rectified)
    bm_vis = visualize_mapping_field(bm_pred)
    
    H, W = dist_img.shape[:2]
    
    # Resize all to same size
    dist_img = cv2.resize(dist_img, (W, H))
    rect_img = cv2.resize(rect_img, (W, H))
    bm_vis = cv2.resize(bm_vis, (W, H))
    
    images = [dist_img, rect_img, bm_vis]
    
    # Add mask visualization if provided
    if mask_logits is not None:
        mask_prob = F.softmax(mask_logits, dim=0)
        mask_fg = mask_prob[1]
        mask_vis = visualize_mask(mask_fg)
        mask_vis = cv2.resize(mask_vis, (W, H))
        images.append(mask_vis)
    
    # Add GT mapping if provided
    if bm_gt is not None:
        bm_gt_vis = visualize_mapping_field(bm_gt)
        bm_gt_vis = cv2.resize(bm_gt_vis, (W, H))
        images.append(bm_gt_vis)
    
    # Add labels
    labels = ['Distorted', 'Rectified', 'BM Predicted']
    if mask_logits is not None:
        labels.append('FG Mask')
    if bm_gt is not None:
        labels.append('BM Ground Truth')
    
    labeled_images = []
    for img, label in zip(images, labels):
        labeled = img.copy()
        cv2.putText(labeled, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(labeled, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        labeled_images.append(labeled)
    
    # Concatenate horizontally
    figure = np.concatenate(labeled_images, axis=1)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(figure, cv2.COLOR_RGB2BGR) if figure.shape[2] == 3 else figure)
    
    return figure


def save_training_visualization(
    epoch: int,
    iteration: int,
    distorted: torch.Tensor,
    rectified: torch.Tensor,
    bm_pred: torch.Tensor,
    mask_logits: Optional[torch.Tensor] = None,
    bm_gt: Optional[torch.Tensor] = None,
    save_dir: str = './vis'
) -> None:
    """
    Save training visualization to disk.
    
    Args:
        epoch: current epoch
        iteration: current iteration
        distorted: distorted image tensor
        rectified: rectified image tensor
        bm_pred: predicted mapping field
        mask_logits: optional mask logits
        bm_gt: optional GT mapping field
        save_dir: directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'epoch{epoch:03d}_iter{iteration:06d}.png')
    
    create_comparison_figure(
        distorted, rectified, bm_pred,
        mask_logits=mask_logits,
        bm_gt=bm_gt,
        save_path=save_path
    )