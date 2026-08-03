"""
Foreground segmentation branch for ForCenNet.

Takes encoder features and predicts foreground/background segmentation mask.
Lightweight CNN head that produces 2-class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundSegmentation(nn.Module):
    """
    Foreground segmentation module.
    
    Input: Encoder feature (last scale)
    Output: Foreground mask (288, 288, 2)
    
    Architecture:
        1x1 conv (unify channels) -> upsample -> multiple 1x1 convs -> 2-class logits
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_classes: int = 2,
        output_size: int = 288
    ):
        super().__init__()
        self.output_size = output_size
        
        # Channel reduction (use GroupNorm instead of BatchNorm for small spatial dims)
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Refinement layers
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.classifier = nn.Conv2d(hidden_channels // 2, num_classes, 1)
    
    def forward(self, encoder_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_feature: (B, C, H, W) encoder feature map (e.g., last scale)
        
        Returns:
            mask_logits: (B, num_classes, output_size, output_size) foreground logits
        """
        x = self.channel_reduce(encoder_feature)
        
        # Upsample to target resolution
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
        
        # Refine
        x = self.refine(x)
        
        # Classify
        mask_logits = self.classifier(x)
        
        return mask_logits