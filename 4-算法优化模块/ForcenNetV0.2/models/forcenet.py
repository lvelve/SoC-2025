"""
ForCenNet: Foreground-Centric Network for Document Image Rectification.

Main model that combines:
1. Feature extraction + Transformer encoder
2. Foreground segmentation branch
3. Mask-guided Transformer decoder
4. Backward mapping field prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .encoder import ForCenNetEncoder
from .segmentation import ForegroundSegmentation
from .decoder import MaskGuidedDecoder


class ForCenNet(nn.Module):
    """
    ForCenNet: Foreground-Centric Network for Document Image Rectification.
    
    Architecture:
        Input: distorted image I_d (B, 3, 288, 288)
            |
        Feature Extraction (large kernel conv + residual)
            | features F (B, 256, 96, 96)
            |
            +---> Transformer Encoder (3 layers, SPW attention)
            |       | multi-scale features {E1, E2, E3}
            |
            +---> Foreground Segmentation (lightweight CNN)
                    | foreground mask M (B, 2, 288, 288)
                    |
        Mask-guided Transformer Decoder (3 layers)
            | (uses mask M to guide attention)
            |
        Progressive Upsampler
            |
        Output: backward mapping field BM_pred (B, 2, 288, 288)
    """
    
    def __init__(
        self,
        # Feature extraction params
        in_channels: int = 3,
        feature_channels: int = 256,
        kernel_size: int = 7,
        num_res_blocks: int = 4,
        # Encoder params
        embed_dim: int = 256,
        num_heads: int = 8,
        encoder_layers: int = 3,
        mlp_ratio: float = 4.0,
        patch_kernel: int = 3,
        patch_stride: int = 2,
        pool_sizes: list = [1, 2, 4],
        # Segmentation params
        seg_hidden_channels: int = 128,
        num_classes: int = 2,
        # Decoder params
        decoder_layers: int = 3,
        num_queries: int = 64,
        mask_gamma: float = 0.8,
        mask_sigma: float = 0.005,
        # Output params
        output_size: int = 288,
        # Dropout
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.output_size = output_size
        
        # Encoder (feature extraction + transformer)
        self.encoder = ForCenNetEncoder(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=kernel_size,
            num_res_blocks=num_res_blocks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=encoder_layers,
            mlp_ratio=mlp_ratio,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            pool_sizes=pool_sizes,
            drop=drop,
            attn_drop=attn_drop
        )
        
        # Foreground segmentation branch
        self.segmentation = ForegroundSegmentation(
            in_channels=feature_channels,
            hidden_channels=seg_hidden_channels,
            num_classes=num_classes,
            output_size=output_size
        )
        
        # Mask-guided decoder (DETR-style with learned queries)
        self.decoder = MaskGuidedDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=decoder_layers,
            mlp_ratio=mlp_ratio,
            mask_gamma=mask_gamma,
            mask_sigma=mask_sigma,
            output_size=output_size,
            num_queries=num_queries,
            drop=drop,
            attn_drop=attn_drop
        )
    
    def forward(
        self,
        image: torch.Tensor,
        return_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            image: (B, 3, H, W) distorted document image
            return_mask: whether to return the predicted mask logits
        
        Returns:
            dict with:
                'bm_pred': (B, 2, output_size, output_size) predicted backward mapping field
                'mask_logits': (B, 2, output_size, output_size) foreground mask logits (if return_mask=True)
        """
        B = image.shape[0]
        
        # 1. Encode: extract features and get multi-scale encoder outputs
        multi_scale_features, feature_map = self.encoder(image)
        # multi_scale_features: list of (B, N_i, D) at different scales
        # feature_map: (B, 256, 96, 96)
        
        # 2. Segmentation: predict foreground mask from the last encoder feature
        # Reshape last encoder feature to spatial
        last_enc = multi_scale_features[-1]  # (B, N_last, D)
        N_last = last_enc.shape[1]
        H_last = int(N_last ** 0.5)
        W_last = N_last // H_last
        if H_last * W_last != N_last:
            # Use feature_map directly
            seg_input = feature_map
        else:
            seg_input = last_enc.transpose(1, 2).reshape(B, -1, H_last, W_last)
        
        mask_logits = self.segmentation(seg_input)  # (B, 2, 288, 288)
        
        # 3. Decode: mask-guided backward mapping prediction
        bm_pred = self.decoder(multi_scale_features, feature_map, mask_logits)
        # bm_pred: (B, 2, output_size, output_size)
        
        result = {'bm_pred': bm_pred}
        if return_mask:
            result['mask_logits'] = mask_logits
        
        return result
    
    def rectify(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method: predict and apply backward mapping to rectify image.
        
        Args:
            image: (B, 3, H, W) distorted document image, normalized to [0, 1]
        
        Returns:
            dict with:
                'rectified': (B, 3, H, W) rectified image
                'bm_pred': (B, 2, H, W) predicted backward mapping field
                'mask_logits': (B, 2, H, W) foreground mask logits
        """
        H, W = image.shape[2], image.shape[3]
        
        # Forward pass
        output = self.forward(image, return_mask=True)
        bm_pred = output['bm_pred']  # (B, 2, H_out, W_out)
        
        # Resize BM to input image size if needed
        if bm_pred.shape[2] != H or bm_pred.shape[3] != W:
            bm_pred = F.interpolate(bm_pred, size=(H, W), mode='bilinear', align_corners=True)
        
        # Create identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device, dtype=image.dtype),
            torch.linspace(-1, 1, W, device=image.device, dtype=image.dtype),
            indexing='ij'
        )
        identity_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        identity_grid = identity_grid.unsqueeze(0).expand(image.shape[0], -1, -1, -1)  # (B, H, W, 2)
        
        # Apply backward mapping
        # BM_pred contains the sampling grid coordinates
        # Use grid_sample to apply the backward mapping
        bm_grid = bm_pred.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        # Apply grid_sample to get rectified image
        rectified = F.grid_sample(
            image, bm_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return {
            'rectified': rectified,
            'bm_pred': bm_pred,
            'mask_logits': output['mask_logits']
        }


def build_forcenet(config: dict = None) -> ForCenNet:
    """
    Build ForCenNet model from configuration dict.
    
    Args:
        config: Configuration dict with model parameters under 'model' key
    
    Returns:
        ForCenNet model instance
    """
    if config is None:
        config = {}
    
    model_config = config.get('model', {})
    
    # Feature extraction params
    fe_config = model_config.get('feature_extraction', {})
    # Encoder params
    enc_config = model_config.get('encoder', {})
    # Segmentation params
    seg_config = model_config.get('segmentation', {})
    # Decoder params
    dec_config = model_config.get('decoder', {})
    
    model = ForCenNet(
        # Feature extraction
        in_channels=fe_config.get('in_channels', 3),
        feature_channels=fe_config.get('out_channels', 256),
        kernel_size=fe_config.get('kernel_size', 7),
        num_res_blocks=fe_config.get('num_res_blocks', 4),
        # Encoder
        embed_dim=enc_config.get('embed_dim', 256),
        num_heads=enc_config.get('num_heads', 8),
        encoder_layers=enc_config.get('num_layers', 3),
        mlp_ratio=enc_config.get('mlp_ratio', 4.0),
        patch_kernel=enc_config.get('patch_kernel', 3),
        patch_stride=enc_config.get('patch_stride', 2),
        pool_sizes=enc_config.get('pool_sizes', [1, 2, 4]),
        # Segmentation
        seg_hidden_channels=seg_config.get('hidden_channels', 128),
        num_classes=seg_config.get('num_classes', 2),
        # Decoder
        decoder_layers=dec_config.get('num_layers', 3),
        num_queries=dec_config.get('num_queries', 64),
        mask_gamma=model_config.get('mask_gamma', 0.8),
        mask_sigma=model_config.get('mask_sigma', 0.005),
        # Output
        output_size=model_config.get('output_size', 288),
        # Dropout
        drop=model_config.get('drop', 0.0),
        attn_drop=model_config.get('attn_drop', 0.0),
    )
    
    return model


if __name__ == '__main__':
    # Quick test
    model = ForCenNet()
    x = torch.randn(2, 3, 288, 288)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"BM pred shape: {output['bm_pred'].shape}")
    print(f"Mask logits shape: {output['mask_logits'].shape}")
    
    # Test rectification
    rect_output = model.rectify(x)
    print(f"Rectified shape: {rect_output['rectified'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")