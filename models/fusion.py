import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """Attention-based fusion of spatial and frequency features"""
    
    def __init__(
            self, 
            spatial_dim: int, freq_dim: int, hidden_dim: int = 256, 
            drop_out: float = 0.3):
        """
        Args:
            spatial_dim: Dimension of spatial features
            freq_dim: Dimension of frequency features
            hidden_dim: Hidden dimension for attention
            drop_out: Drop_out layer prob
        """
        super(AttentionFusion, self).__init__()

        # Normalization
        self.spatial_norm = nn.LayerNorm(spatial_dim)
        self.freq_norm = nn.LayerNorm(freq_dim)
        
        # Attention weights
        self.attention_context = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        spatial_features: torch.Tensor, 
        freq_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: (B, spatial_dim)
            freq_features: (B, freq_dim)
        Returns:
            Fused features (B, spatial_dim + freq_dim)
        """
        # Normalize feature
        spatial_norm_feat = self.spatial_norm(spatial_features)
        freq_norm_feat = self.freq_norm(freq_features)

        # Joint context -> observe 2 branch and decide
        joint = torch.cat([spatial_norm_feat, freq_norm_feat], dim=1)

        # Calculate weight
        attn_gate = self.attention_context(joint)

        spatial_weight, freq_weight = attn_gate[:, 0:1], attn_gate[:, 1:2]

        # Apply attention
        weighted_spatial = spatial_features * spatial_weight
        weighted_freq = freq_features * freq_weight

        # Concatenate
        fused = torch.cat([
            weighted_spatial, weighted_freq
        ], dim=1)
        
        return fused
    
    @torch.no_grad()
    def get_attention_weights(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor
    ):
        """
        Return attention weights used in forward pass.
        Values are in [0, 1] independently (sigmoid gate).
        """

        spatial_norm = self.spatial_norm(spatial_feat)
        freq_norm = self.freq_norm(freq_feat)

        joint = torch.cat([spatial_norm, freq_norm], dim=1)

        # attention_context already outputs sigmoid-ed values
        attn_gate = self.attention_context(joint)

        spatial_weight = attn_gate[:, 0:1]
        freq_weight = attn_gate[:, 1:2]

        return spatial_weight, freq_weight

