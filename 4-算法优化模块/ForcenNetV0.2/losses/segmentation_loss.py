"""
Segmentation loss for ForCenNet.

L_seg = ||M - M_d||_1

where:
- M: predicted foreground mask (after softmax)
- M_d: ground truth distorted foreground mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """
    L1 segmentation loss for foreground mask prediction.
    
    L_seg = ||softmax(M_logits) - M_gt||_1
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        mask_logits: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mask_logits: (B, 2, H, W) predicted foreground logits
            mask_gt: (B, H, W) ground truth binary mask (0 or 1)
        
        Returns:
            loss: scalar tensor
        """
        # Convert logits to probabilities
        mask_pred = F.softmax(mask_logits, dim=1)  # (B, 2, H, W)
        
        # Extract foreground probability (class 1)
        mask_fg = mask_pred[:, 1, :, :]  # (B, H, W)
        
        # L1 loss
        loss = F.l1_loss(mask_fg, mask_gt, reduction='mean')
        
        return loss