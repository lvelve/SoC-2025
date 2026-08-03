"""
Mapping regression loss for ForCenNet.

L_map = ||BM_pred - BM_gt||_1

where:
- BM_pred: predicted backward mapping field
- BM_gt: ground truth backward mapping field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingLoss(nn.Module):
    """
    L1 mapping regression loss for backward mapping field prediction.
    
    L_map = ||BM_pred - BM_gt||_1
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        bm_pred: torch.Tensor,
        bm_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            bm_pred: (B, 2, H, W) predicted backward mapping field
            bm_gt: (B, 2, H, W) ground truth backward mapping field
        
        Returns:
            loss: scalar tensor
        """
        # Resize prediction to match ground truth if needed
        if bm_pred.shape[2:] != bm_gt.shape[2:]:
            bm_pred = F.interpolate(
                bm_pred, size=bm_gt.shape[2:], mode='bilinear', align_corners=True
            )
        
        loss = F.l1_loss(bm_pred, bm_gt, reduction='mean')
        
        return loss