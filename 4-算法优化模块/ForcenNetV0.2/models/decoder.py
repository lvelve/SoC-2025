"""
Mask-guided Transformer Decoder for ForCenNet.

Core innovation: Uses foreground mask to guide attention mechanism,
focusing on readable regions (text, lines, graphics).

Components:
- Mask-guided self-attention (MSA)
- Encoder-decoder cross-attention
- Progressive upsampling for backward mapping field prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class MaskGuidedSelfAttention(nn.Module):
    """
    Mask-guided self-attention (MSA).
    
    MSA(Q,K,V) = Softmax((QK^T + sigma * Seq(M_tilde) @ Seq(M_tilde)^T) / sqrt(d)) V
    
    where:
    - M_tilde = sum_{i=0}^{1} i * softmax(gamma * M)_i
    - gamma = 0.8 (smoothing coefficient)
    - sigma = 0.005 (scaling factor)
    - Seq(.) = sequence unfolding along feature dimension
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mask_gamma: float = 0.8,
        mask_sigma: float = 0.005,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mask_gamma = mask_gamma
        self.mask_sigma = mask_sigma
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)
    
    def compute_mask_bias(
        self,
        mask_logits: torch.Tensor,
        H: int,
        W: int,
        query_length: int
    ) -> torch.Tensor:
        """
        Compute mask-guided attention bias.
        
        Args:
            mask_logits: (B, num_classes, H_full, W_full) foreground mask logits
            H: spatial height of current scale
            W: spatial width of current scale
            query_length: number of query tokens (N_q)
        
        Returns:
            bias: (B, query_length, query_length) attention bias from mask
        """
        B = mask_logits.shape[0]
        
        # Apply softmax with gamma scaling to get probability mask
        # mask_logits: (B, 2, H_full, W_full) -> foreground probability
        mask_prob = F.softmax(self.mask_gamma * mask_logits, dim=1)  # (B, 2, H_full, W_full)
        
        # M_tilde = sum_{i=0}^{1} i * softmax(gamma * M)_i
        # This gives the foreground probability (weighted by class index 0, 1)
        weights = torch.tensor([0.0, 1.0], device=mask_logits.device, dtype=mask_logits.dtype)
        M_tilde = (mask_prob * weights.view(1, 2, 1, 1)).sum(dim=1, keepdim=True)  # (B, 1, H_full, W_full)
        
        # Resize to current scale
        M_tilde = F.interpolate(M_tilde, size=(H, W), mode='bilinear', align_corners=True)  # (B, 1, H, W)
        
        # Flatten to sequence
        M_seq = M_tilde.flatten(2).squeeze(1)  # (B, N) where N = H*W
        
        # Trim or pad to match query_length
        N = M_seq.shape[1]
        if N < query_length:
            pad_size = query_length - N
            M_seq = F.pad(M_seq, (0, pad_size), value=0)
        elif N > query_length:
            M_seq = M_seq[:, :query_length]
        
        # Compute mask bias: sigma * Seq(M_tilde) @ Seq(M_tilde)^T
        # M_seq: (B, N_q) -> (B, N_q, 1) @ (B, 1, N_q) -> (B, N_q, N_q)
        mask_bias = self.mask_sigma * torch.einsum('bi,bj->bij', M_seq, M_seq)  # (B, N_q, N_q)
        
        return mask_bias
    
    def forward(
        self,
        x: torch.Tensor,
        mask_logits: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) query tokens
            mask_logits: (B, 2, H_full, W_full) foreground mask logits
            H: spatial height at current scale
            W: spatial width at current scale
        
        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)
        
        # Compute mask-guided bias
        mask_bias = self.compute_mask_bias(mask_logits, H, W, N)  # (B, N, N)
        attn_mask = mask_bias.unsqueeze(1)  # (B, 1, N, N)
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        # This avoids materializing the full N×N attention matrix, saving massive GPU memory
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )  # (B, heads, N, head_dim)
        
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class CrossAttention(nn.Module):
    """Encoder-decoder cross-attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D) decoder tokens
            key_value: (B, N_kv, D) encoder tokens
        
        Returns:
            output: (B, N_q, D)
        """
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]
        
        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )  # (B, heads, N_q, head_dim)
        
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class MaskGuidedDecoderLayer(nn.Module):
    """Single mask-guided Transformer decoder layer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_gamma: float = 0.8,
        mask_sigma: float = 0.005,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        # Mask-guided self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MaskGuidedSelfAttention(
            embed_dim, num_heads, mask_gamma, mask_sigma, drop, attn_drop
        )
        
        # Cross-attention with encoder features
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads, drop, attn_drop)
        
        # FFN
        self.norm3 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        encoder_feature: torch.Tensor,
        mask_logits: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D) decoder query tokens
            encoder_feature: (B, N_kv, D) encoder feature tokens
            mask_logits: (B, 2, H_full, W_full) foreground mask logits
            H: spatial height at current scale
            W: spatial width at current scale
        
        Returns:
            output: (B, N_q, D)
        """
        # Mask-guided self-attention
        query = query + self.self_attn(self.norm1(query), mask_logits, H, W)
        
        # Cross-attention
        query = query + self.cross_attn(self.norm2(query), encoder_feature)
        
        # FFN
        query = query + self.mlp(self.norm3(query))
        
        return query


class ProgressiveUpsampler(nn.Module):
    """
    Progressive upsampling module (inspired by DocTr/DocGeoNet).
    
    Takes decoder output at a coarse scale and progressively upsamples
    to the target resolution (288x288).
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        output_channels: int = 2,
        output_size: int = 288
    ):
        super().__init__()
        self.output_size = output_size
        
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, embed_dim // 2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, embed_dim // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, embed_dim // 8),
                nn.ReLU(inplace=True)
            ),
        ])
        
        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Conv2d(embed_dim // 8, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 1)
        )
        
        # Adaptive pool for final size adjustment
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_size, output_size))
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) decoder output tokens
            H: spatial height of current scale
            W: spatial width of current scale
        
        Returns:
            output: (B, output_channels, output_size, output_size) mapping field
        """
        B, N, D = x.shape
        
        # Reshape to spatial
        x = x.transpose(1, 2).reshape(B, D, H, W)  # (B, D, H, W)
        
        # Progressive upsampling
        for up_block in self.upsample_blocks:
            x = up_block(x)
        
        # Predict
        x = self.pred_head(x)
        
        # Adjust to target size
        x = self.adaptive_pool(x)
        
        return x


class MaskGuidedDecoder(nn.Module):
    """
    Mask-guided Transformer Decoder for ForCenNet.
    
    Takes multi-scale encoder features and foreground mask,
    uses mask-guided attention to predict the backward mapping field.
    
    Uses learned query embeddings (DETR-style) instead of all encoder tokens
    to keep the attention sequence length manageable.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        mask_gamma: float = 0.8,
        mask_sigma: float = 0.005,
        output_size: int = 288,
        num_queries: int = 64,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.output_size = output_size
        
        # Learned query embeddings (DETR-style)
        # num_queries tokens that attend to encoder features via cross-attention
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        # Spatial dimensions for query grid (e.g., 64 -> 8x8)
        self.query_h = int(num_queries ** 0.5)
        self.query_w = num_queries // self.query_h
        assert self.query_h * self.query_w == num_queries, \
            f"num_queries ({num_queries}) must be a perfect square or at least factorizable"
        
        # Decoder layers
        self.layers = nn.ModuleList([
            MaskGuidedDecoderLayer(
                embed_dim, num_heads, mlp_ratio,
                mask_gamma, mask_sigma, drop, attn_drop
            )
            for _ in range(num_layers)
        ])
        
        # Norms
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Progressive upsampler
        self.upsampler = ProgressiveUpsampler(embed_dim, 2, output_size)
    
    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        feature_map: torch.Tensor,
        mask_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            multi_scale_features: List of [E1, E2, E3] encoder features at different scales
            feature_map: (B, C, H_feat, W_feat) spatial feature map from feature extraction
            mask_logits: (B, 2, H_full, W_full) predicted foreground mask logits
        
        Returns:
            bm_pred: (B, 2, output_size, output_size) predicted backward mapping field
        """
        B = multi_scale_features[0].shape[0]
        
        # Use learned query embeddings (DETR-style)
        # query: (B, num_queries, D) — much smaller than using all encoder tokens
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, D)
        
        H_q = self.query_h
        W_q = self.query_w
        
        # Process through decoder layers
        # Each layer cross-attends to a different encoder scale (cycling if needed)
        for i in range(self.num_layers):
            # Select encoder feature for cross-attention
            enc_idx = i % len(multi_scale_features)
            enc_feature = multi_scale_features[enc_idx]
            
            # Apply decoder layer
            query = self.layers[i](query, enc_feature, mask_logits, H_q, W_q)
            query = self.norms[i](query)
        
        # Progressive upsampling to target resolution
        bm_pred = self.upsampler(query, H_q, W_q)
        
        return bm_pred
