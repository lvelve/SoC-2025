"""
Transformer Encoder with Spatial Pooling Window (SPW) attention for ForCenNet.

Components:
- Large-kernel convolution feature extraction with residual blocks
- Overlapping Patch Embedding
- SPW (Spatial Pooling Window) Transformer layers
- Multi-scale feature output {E1, E2, E3}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from typing import List, Tuple


# ============================================================
# Feature Extraction Module (Large Kernel Conv + Residual)
# ============================================================

class LargeKernelConv(nn.Module):
    """Large kernel convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """ResNet-style residual block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class DownsampleBlock(nn.Module):
    """Downsample by factor of 2 using strided convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FeatureExtraction(nn.Module):
    """
    Feature extraction module: large kernel conv + residual blocks + downsample.
    
    Input: (B, 3, H, W)
    Output: (B, out_channels, H//3, W//3)  -- 288 -> 96, 8 -> ~3
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        kernel_size: int = 7,
        num_res_blocks: int = 4
    ):
        super().__init__()
        
        # Large kernel conv
        self.large_conv = LargeKernelConv(in_channels, 64, kernel_size)
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Downsample: H -> H/2
        self.down1 = DownsampleBlock(64, 128)
        
        # Downsample: H/2 -> H/4
        self.down2 = DownsampleBlock(128, out_channels)
        
        # Note: adaptive_pool removed — output size is now proportional to input
        # For 288 input: 288/4 = 72 (was 96 with hardcoded pool)
        # For 8 input: 8/4 = 2 (was 96 with hardcoded pool — caused OOM!)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            features: (B, out_channels, H//4, W//4)
        """
        x = self.large_conv(x)       # (B, 64, H, W)
        x = self.res_blocks(x)       # (B, 64, H, W)
        x = self.down1(x)            # (B, 128, H/2, W/2)
        x = self.down2(x)            # (B, 256, H/4, W/4)
        return x


# ============================================================
# Overlapping Patch Embedding
# ============================================================

class OverlappingPatchEmbedding(nn.Module):
    """
    Overlapping patch embedding with kernel=3, stride=2.
    Preserves boundary information better than non-overlapping patches.
    """
    
    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            tokens: (B, N, D) where N = H'*W'
            H': height after patch embedding
            W': width after patch embedding
        """
        x = self.proj(x)  # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D) where N = H*W
        x = self.norm(x)
        return x, H, W


# ============================================================
# Spatial Pooling Window (SPW) Attention
# ============================================================

class SpatialPoolingAttention(nn.Module):
    """
    Multi-head attention with Spatial Pooling Window (SPW).
    
    Applies spatial pooling to Key and Value tensors to reduce
    attention complexity while preserving spatial structure.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        pool_size: int = 1,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool_size = pool_size
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Spatial pooling for K and V
        if pool_size > 1:
            self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0)
        else:
            self.pool = None
    
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        mask_bias: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input tokens
            H: spatial height
            W: spatial width
            mask_bias: (B, N, N) optional mask attention bias
        
        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)
        
        # Apply spatial pooling to K and V if pool_size > 1 and feature map is large enough
        if self.pool is not None and H >= self.pool_size and W >= self.pool_size:
            # Reshape K and V to spatial form
            k_spatial = k.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
            v_spatial = v.transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H, W)
            
            k_pooled = self.pool(k_spatial)  # (B*heads, head_dim, H', W')
            v_pooled = self.pool(v_spatial)
            
            H_p, W_p = k_pooled.shape[2], k_pooled.shape[3]
            k = k_pooled.reshape(B, self.num_heads, self.head_dim, -1).transpose(2, 3)  # (B, heads, N', head_dim)
            v = v_pooled.reshape(B, self.num_heads, self.head_dim, -1).transpose(2, 3)
            
            N_kv = k.shape[2]
        else:
            N_kv = N
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        # This avoids materializing the full N×N attention matrix, saving massive GPU memory
        attn_mask = None
        if mask_bias is not None:
            # mask_bias: (B, N, N) -> (B, 1, N, N_kv)
            if self.pool is not None and mask_bias.shape[-1] != N_kv:
                mask_bias_spatial = mask_bias.reshape(B, N, H, W)
                mask_bias_pooled = F.adaptive_avg_pool2d(mask_bias_spatial, (H_p, W_p))
                mask_bias = mask_bias_pooled.reshape(B, N, N_kv)
            attn_mask = mask_bias.unsqueeze(1)  # (B, 1, N, N_kv)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )  # (B, heads, N, head_dim)
        
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with SPW attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        pool_size: int = 1,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SpatialPoolingAttention(
            embed_dim, num_heads, pool_size, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int, mask_bias: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W, mask_bias)
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Transformer Encoder with Multi-scale Output
# ============================================================

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder producing multi-scale features {E1, E2, E3}.
    
    Uses overlapping patch embedding at each scale and SPW attention.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        patch_kernel: int = 3,
        patch_stride: int = 2,
        pool_sizes: List[int] = [1, 2, 4],
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # Patch embeddings at each scale
        self.patch_embeds = nn.ModuleList()
        # Layer 0: from feature map (96x96) to scale 0
        self.patch_embeds.append(
            OverlappingPatchEmbedding(in_channels, embed_dim, patch_kernel, stride=1)  # Keep same size
        )
        # Layer 1, 2: progressively downsample
        for i in range(1, num_layers):
            self.patch_embeds.append(
                OverlappingPatchEmbedding(embed_dim, embed_dim, patch_kernel, stride=patch_stride)
            )
        
        # Transformer layers at each scale
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_pool_size = pool_sizes[i] if i < len(pool_sizes) else 1
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dim, num_heads, mlp_ratio,
                    pool_size=layer_pool_size,
                    drop=drop, attn_drop=attn_drop
                )
            )
        
        # Norms for each scale output
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
    
    def _encoder_layer_forward(self, i, tokens, H, W, mask_bias):
        """Forward through a single encoder layer, suitable for gradient checkpointing."""
        tokens = self.layers[i](tokens, H, W, mask_bias)
        tokens = self.norms[i](tokens)
        return tokens
    
    def forward(
        self,
        features: torch.Tensor,
        mask_bias: torch.Tensor = None
    ) -> List[torch.Tensor]:
        """
        Args:
            features: (B, C, H, W) input feature map (96x96)
            mask_bias: optional attention bias from foreground mask
        
        Returns:
            List of multi-scale feature tensors [E1, E2, E3]
            Each element: (B, N_i, D) token sequence
        """
        outputs = []
        x = features
        
        for i in range(self.num_layers):
            # Patch embedding
            tokens, H, W = self.patch_embeds[i](x)
            
            # Transformer encoding with gradient checkpointing to save memory
            if self.training:
                tokens = gradient_checkpoint(
                    self._encoder_layer_forward, i, tokens, H, W, mask_bias,
                    use_reentrant=False
                )
            else:
                tokens = self._encoder_layer_forward(i, tokens, H, W, mask_bias)
            
            outputs.append(tokens)
            
            # Reshape back to spatial for next patch embedding
            x = tokens.transpose(1, 2).reshape(-1, tokens.shape[2], H, W)
        
        return outputs


# ============================================================
# Complete Encoder (Feature Extraction + Transformer)
# ============================================================

class ForCenNetEncoder(nn.Module):
    """
    Complete encoder for ForCenNet:
    1. Feature extraction (large kernel conv + residual)
    2. Transformer encoding with multi-scale SPW attention
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        kernel_size: int = 7,
        num_res_blocks: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        patch_kernel: int = 3,
        patch_stride: int = 2,
        pool_sizes: List[int] = [1, 2, 4],
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        # Feature extraction
        self.feature_extraction = FeatureExtraction(
            in_channels, out_channels, kernel_size, num_res_blocks
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            out_channels, embed_dim, num_heads, num_layers,
            mlp_ratio, patch_kernel, patch_stride, pool_sizes,
            drop, attn_drop
        )
        
        self.embed_dim = embed_dim
    
    def forward(
        self,
        x: torch.Tensor,
        mask_bias: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image
            mask_bias: optional attention bias
        
        Returns:
            multi_scale_features: List of [E1, E2, E3] token sequences
            feature_map: (B, C, 96, 96) spatial feature map
        """
        # Feature extraction
        feature_map = self.feature_extraction(x)  # (B, 256, 96, 96)
        
        # Transformer encoding
        multi_scale_features = self.transformer(feature_map, mask_bias)
        
        return multi_scale_features, feature_map