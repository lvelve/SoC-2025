"""
Grid sampling utilities for ForCenNet.

Provides functions for:
- Creating identity grids
- Applying backward mapping fields to rectify images
- Grid sampling with various interpolation modes
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def create_identity_grid(
    height: int,
    width: int,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create an identity grid for grid_sample.
    
    Args:
        height: grid height
        width: grid width
        device: torch device
        dtype: torch dtype
    
    Returns:
        grid: (1, H, W, 2) identity grid in [-1, 1]
    """
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device, dtype=dtype),
        torch.linspace(-1, 1, width, device=device, dtype=dtype),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    return grid.unsqueeze(0)  # (1, H, W, 2)


def apply_backward_mapping(
    image: torch.Tensor,
    bm: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'border',
    align_corners: bool = True
) -> torch.Tensor:
    """
    Apply backward mapping field to rectify an image.
    
    Args:
        image: (B, C, H, W) input image
        bm: (B, 2, H, W) backward mapping field in [-1, 1]
        mode: interpolation mode ('bilinear', 'nearest', 'bicubic')
        padding_mode: padding mode ('zeros', 'border', 'reflection')
        align_corners: align corners flag for grid_sample
    
    Returns:
        rectified: (B, C, H, W) rectified image
    """
    B, C, H, W = image.shape
    
    # Create identity grid
    identity = create_identity_grid(H, W, device=image.device, dtype=image.dtype)
    identity = identity.expand(B, -1, -1, -1)  # (B, H, W, 2)
    
    # Resize BM to match image size if needed
    if bm.shape[2] != H or bm.shape[3] != W:
        bm = F.interpolate(bm, size=(H, W), mode='bilinear', align_corners=True)
    
    # BM is (B, 2, H, W), convert to (B, H, W, 2)
    bm_perm = bm.permute(0, 2, 3, 1)
    
    # Apply: grid = identity + BM
    grid = identity + bm_perm
    grid = grid.clamp(-1, 1)
    
    # Sample
    rectified = F.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    
    return rectified


def apply_forward_mapping(
    image: torch.Tensor,
    fm: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'border',
    align_corners: bool = True
) -> torch.Tensor:
    """
    Apply forward mapping field to distort an image.
    
    Args:
        image: (B, C, H, W) input image
        fm: (B, 2, H, W) forward mapping field in [-1, 1]
        mode: interpolation mode
        padding_mode: padding mode
        align_corners: align corners flag
    
    Returns:
        distorted: (B, C, H, W) distorted image
    """
    B, C, H, W = image.shape
    
    # Create identity grid
    identity = create_identity_grid(H, W, device=image.device, dtype=image.dtype)
    identity = identity.expand(B, -1, -1, -1)
    
    # Resize FM to match image size if needed
    if fm.shape[2] != H or fm.shape[3] != W:
        fm = F.interpolate(fm, size=(H, W), mode='bilinear', align_corners=True)
    
    # FM is (B, 2, H, W), convert to (B, H, W, 2)
    fm_perm = fm.permute(0, 2, 3, 1)
    
    # Apply: grid = identity - FM (forward mapping gives where to go, we need inverse)
    # Actually, forward mapping: output[p] = input[p + FM(p)]
    # So grid for grid_sample should be: grid[p] = p + FM(p)
    grid = identity + fm_perm
    grid = grid.clamp(-1, 1)
    
    distorted = F.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    
    return distorted


def compute_mapping_error(
    bm_pred: torch.Tensor,
    bm_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Compute mapping field error metrics.
    
    Args:
        bm_pred: (B, 2, H, W) predicted backward mapping
        bm_gt: (B, 2, H, W) ground truth backward mapping
        mask: (B, 1, H, W) optional foreground mask for weighted evaluation
    
    Returns:
        dict with metrics:
            'mae': mean absolute error
            'mse': mean squared error
            'max_error': maximum absolute error
            'foreground_mae': MAE in foreground regions (if mask provided)
    """
    # Resize prediction to match GT if needed
    if bm_pred.shape[2:] != bm_gt.shape[2:]:
        bm_pred = F.interpolate(bm_pred, size=bm_gt.shape[2:], mode='bilinear', align_corners=True)
    
    # Compute errors
    abs_error = torch.abs(bm_pred - bm_gt)  # (B, 2, H, W)
    sq_error = (bm_pred - bm_gt) ** 2
    
    # Per-pixel error (L2 norm of the 2D error vector)
    pixel_error = torch.sqrt(sq_error.sum(dim=1, keepdim=True))  # (B, 1, H, W)
    
    metrics = {
        'mae': abs_error.mean().item(),
        'mse': sq_error.mean().item(),
        'max_error': pixel_error.max().item(),
    }
    
    # Foreground-weighted metrics
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Resize mask if needed
        if mask.shape[2:] != bm_gt.shape[2:]:
            mask = F.interpolate(mask.float(), size=bm_gt.shape[2:], mode='nearest')
        
        # Weighted MAE
        weighted_error = abs_error * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            metrics['foreground_mae'] = (weighted_error.sum() / (mask_sum * 2)).item()
        else:
            metrics['foreground_mae'] = 0.0
    
    return metrics


def compute_rectification_metrics(
    rectified: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Compute image-level rectification quality metrics.
    
    Args:
        rectified: (B, C, H, W) rectified image
        target: (B, C, H, W) target (undistorted) image
        mask: (B, 1, H, W) optional foreground mask
    
    Returns:
        dict with metrics:
            'psnr': Peak Signal-to-Noise Ratio
            'ssim_approx': approximate SSIM (simplified)
    """
    # MSE
    mse = ((rectified - target) ** 2).mean()
    
    # PSNR
    if mse > 0:
        psnr = 10 * torch.log10(1.0 / mse)
    else:
        psnr = torch.tensor(float('inf'))
    
    # Simple SSIM approximation
    mu_r = rectified.mean()
    mu_t = target.mean()
    sigma_r = ((rectified - mu_r) ** 2).mean()
    sigma_t = ((target - mu_t) ** 2).mean()
    sigma_rt = ((rectified - mu_r) * (target - mu_t)).mean()
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_num = (2 * mu_r * mu_t + C1) * (2 * sigma_rt + C2)
    ssim_den = (mu_r ** 2 + mu_t ** 2 + C1) * (sigma_r + sigma_t + C2)
    ssim = ssim_num / ssim_den
    
    metrics = {
        'psnr': psnr.item(),
        'ssim_approx': ssim.item(),
    }
    
    return metrics