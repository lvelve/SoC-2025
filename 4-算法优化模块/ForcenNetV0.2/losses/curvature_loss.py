"""
Curvature consistency loss for ForCenNet.

Ensures that the curvature of line elements in the predicted mapping field
matches the curvature in the ground truth mapping field.

Steps:
1. Sample control points from line point sets at regular intervals
2. Project points into predicted and ground truth mapping fields (bilinear interpolation)
3. Compute discrete curvature for both sets of deformed points
4. Compute L1 difference between predicted and ground truth curvatures

kappa_i = |x'_i * y''_i - y'_i * x''_i| / (x'_i^2 + y'_i^2)^{3/2} + epsilon

L_k = (1/(N-1)) * sum_{i=1}^{N-1} (kappa_pred_i - kappa_gt_i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def bilinear_sample_map(
    mapping_field: torch.Tensor,
    points: torch.Tensor
) -> torch.Tensor:
    """
    Sample mapping field values at given point locations using bilinear interpolation.
    
    Args:
        mapping_field: (B, 2, H, W) mapping field in [-1, 1]
        points: (N, 2) point coordinates [x, y] in pixel space
    
    Returns:
        sampled: (N, 2) sampled mapping values
    """
    B, C, H, W = mapping_field.shape
    N = points.shape[0]
    
    if N == 0:
        return torch.zeros(0, 2, device=mapping_field.device, dtype=mapping_field.dtype)
    
    # Normalize points to [-1, 1] for grid_sample
    # Move to same device as mapping_field to avoid device mismatch
    norm_points = points.clone().float().to(mapping_field.device)
    norm_points[:, 0] = norm_points[:, 0] / (W - 1) * 2 - 1  # x
    norm_points[:, 1] = norm_points[:, 1] / (H - 1) * 2 - 1  # y
    
    # Reshape for grid_sample: (1, 1, N, 2) -> (B, 1, N, 2)
    grid = norm_points.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
    grid = grid.expand(B, -1, -1, -1)  # (B, 1, N, 2)
    
    # grid_sample expects grid in (x, y) format with last dim=2
    sampled = F.grid_sample(
        mapping_field,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # (B, 2, 1, N)
    
    sampled = sampled.squeeze(2).transpose(1, 2)  # (B, N, 2)
    
    return sampled


def compute_curvature(
    points: torch.Tensor,
    epsilon: float = 1e-4
) -> torch.Tensor:
    """
    Compute discrete curvature for a sequence of 2D points.
    
    kappa_i = |x'_i * y''_i - y'_i * x''_i| / (x'_i^2 + y'_i^2)^{3/2} + epsilon
    
    Uses central differences for interior points, forward/backward for boundaries.
    
    Args:
        points: (N, 2) ordered point sequence [x, y]
        epsilon: small constant to avoid division by zero
    
    Returns:
        curvature: (N,) curvature values
    """
    N = points.shape[0]
    
    if N < 3:
        return torch.zeros(N, device=points.device, dtype=points.dtype)
    
    x = points[:, 0]  # (N,)
    y = points[:, 1]  # (N,)
    
    # First derivatives using central differences
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(y)
    
    # Interior: central difference
    dx[1:-1] = (x[2:] - x[:-2]) / 2.0
    dy[1:-1] = (y[2:] - y[:-2]) / 2.0
    
    # Boundary: forward/backward difference
    dx[0] = x[1] - x[0]
    dy[0] = y[1] - y[0]
    dx[-1] = x[-1] - x[-2]
    dy[-1] = y[-1] - y[-2]
    
    # Second derivatives using central differences
    ddx = torch.zeros_like(x)
    ddy = torch.zeros_like(y)
    
    # Interior: central difference of first derivatives
    ddx[1:-1] = (dx[2:] - dx[:-2]) / 2.0
    ddy[1:-1] = (dy[2:] - dy[:-2]) / 2.0
    
    # Boundary: forward/backward difference
    ddx[0] = dx[1] - dx[0]
    ddy[0] = dy[1] - dy[0]
    ddx[-1] = dx[-1] - dx[-2]
    ddy[-1] = dy[-1] - dy[-2]
    
    # Curvature formula
    numerator = torch.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** 1.5 + epsilon
    
    curvature = numerator / denominator
    
    return curvature


def project_points_to_map(
    mapping_field: torch.Tensor,
    points: torch.Tensor
) -> torch.Tensor:
    """
    Project points through a mapping field.
    
    For each point p_i, the projected coordinate is:
    p_i + BM(p_i)
    where BM(p_i) is sampled from the mapping field at p_i using bilinear interpolation.
    
    Args:
        mapping_field: (B, 2, H, W) mapping field (normalized [-1, 1])
        points: (N, 2) point coordinates [x, y] in pixel space
    
    Returns:
        projected: (B, N, 2) projected point coordinates
    """
    B = mapping_field.shape[0]
    N = points.shape[0]
    
    if N == 0:
        return torch.zeros(B, 0, 2, device=mapping_field.device, dtype=mapping_field.dtype)
    
    # Sample mapping values at point locations
    map_values = bilinear_sample_map(mapping_field, points)  # (B, N, 2)
    
    # Normalize points to [-1, 1]
    H, W = mapping_field.shape[2], mapping_field.shape[3]
    norm_points = points.clone().float().to(mapping_field.device)
    norm_points[:, 0] = norm_points[:, 0] / (W - 1) * 2 - 1
    norm_points[:, 1] = norm_points[:, 1] / (H - 1) * 2 - 1
    
    # Project: add mapping offset
    projected = norm_points.unsqueeze(0).expand(B, -1, -1) + map_values  # (B, N, 2)
    
    # Clamp to valid range
    projected = projected.clamp(-1, 1)
    
    # Convert back to pixel space
    projected_pixel = projected.clone()
    projected_pixel[:, :, 0] = (projected[:, :, 0] + 1) / 2 * (W - 1)
    projected_pixel[:, :, 1] = (projected[:, :, 1] + 1) / 2 * (H - 1)
    
    return projected_pixel


class CurvatureLoss(nn.Module):
    """
    Curvature consistency loss.
    
    L_k = (1/(N-1)) * sum_{i=1}^{N-1} (kappa_pred_i - kappa_gt_i)
    
    For each line of control points:
    1. Project points into predicted and GT mapping fields
    2. Compute curvature of projected point sequences
    3. Compute L1 difference of curvatures
    """
    
    def __init__(
        self,
        epsilon: float = 1e-4,
        sample_interval: int = 4
    ):
        super().__init__()
        self.epsilon = epsilon
        self.sample_interval = sample_interval
    
    def forward(
        self,
        bm_pred: torch.Tensor,
        bm_gt: torch.Tensor,
        line_points_list: Optional[List[torch.Tensor]] = None,
        image_size: int = 288
    ) -> torch.Tensor:
        """
        Compute curvature consistency loss.
        
        Args:
            bm_pred: (B, 2, H, W) predicted backward mapping field
            bm_gt: (B, 2, H, W) ground truth backward mapping field
            line_points_list: list of (N_i, 2) line point tensors (per sample in batch)
                          or list of list of (N_ij, 2) tensors
            image_size: image size for normalization
        
        Returns:
            loss: scalar tensor
        """
        B = bm_pred.shape[0]
        device = bm_pred.device
        dtype = bm_pred.dtype
        
        # Resize prediction to match ground truth
        if bm_pred.shape[2:] != bm_gt.shape[2:]:
            bm_pred = F.interpolate(
                bm_pred, size=bm_gt.shape[2:], mode='bilinear', align_corners=True
            )
        
        # If no line points provided, generate a synthetic grid of lines
        if line_points_list is None:
            line_points_list = self._generate_default_lines(image_size, device, dtype, B)
        
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        valid_count = 0
        
        for b in range(B):
            # Get line points for this sample
            if b < len(line_points_list):
                sample_lines = line_points_list[b]
            else:
                sample_lines = line_points_list[0]
            
            if isinstance(sample_lines, torch.Tensor) and sample_lines.dim() == 1:
                # Skip invalid data
                continue
            
            # Handle different formats of line points
            if isinstance(sample_lines, (list, tuple)):
                lines = sample_lines
            elif isinstance(sample_lines, torch.Tensor):
                if sample_lines.dim() == 2:
                    lines = [sample_lines]
                else:
                    lines = [sample_lines]
            else:
                continue
            
            for line_pts in lines:
                if not isinstance(line_pts, torch.Tensor):
                    line_pts = torch.tensor(line_pts, device=device, dtype=dtype)
                
                # Filter out zero-padded points
                valid_mask = (line_pts[:, 0] > 0) | (line_pts[:, 1] > 0)
                line_pts = line_pts[valid_mask]
                
                if len(line_pts) < 3:
                    continue
                
                # Subsample if too many points
                if len(line_pts) > 500:
                    indices = torch.linspace(0, len(line_pts) - 1, 500, device=device).long()
                    line_pts = line_pts[indices]
                
                # Project points through predicted and GT mapping fields
                pred_projected = project_points_to_map(
                    bm_pred[b:b+1], line_pts
                ).squeeze(0)  # (N, 2)
                
                gt_projected = project_points_to_map(
                    bm_gt[b:b+1], line_pts
                ).squeeze(0)  # (N, 2)
                
                # Compute curvatures
                pred_curvature = compute_curvature(pred_projected, self.epsilon)
                gt_curvature = compute_curvature(gt_projected, self.epsilon)
                
                # L1 difference of curvatures (excluding first and last points)
                if len(pred_curvature) > 2:
                    curv_diff = torch.abs(
                        pred_curvature[1:-1] - gt_curvature[1:-1]
                    ).mean()
                    total_loss = total_loss + curv_diff
                    valid_count += 1
        
        if valid_count > 0:
            total_loss = total_loss / valid_count
        
        return total_loss
    
    def _generate_default_lines(
        self,
        image_size: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int
    ) -> List[torch.Tensor]:
        """
        Generate default horizontal and vertical lines for curvature computation
        when no explicit line points are provided.
        """
        lines = []
        num_lines = 10
        
        # Horizontal lines
        for i in range(num_lines):
            y = (i + 1) * image_size / (num_lines + 1)
            xs = torch.linspace(0, image_size - 1, image_size // self.sample_interval, device=device, dtype=dtype)
            ys = torch.full_like(xs, y)
            line = torch.stack([xs, ys], dim=-1)
            lines.append(line)
        
        # Vertical lines
        for i in range(num_lines):
            x = (i + 1) * image_size / (num_lines + 1)
            ys = torch.linspace(0, image_size - 1, image_size // self.sample_interval, device=device, dtype=dtype)
            xs = torch.full_like(ys, x)
            line = torch.stack([xs, ys], dim=-1)
            lines.append(line)
        
        # Return same lines for each batch element
        return [lines for _ in range(batch_size)]