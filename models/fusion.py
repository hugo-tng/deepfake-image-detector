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

        self.align_spatial = nn.Identity()
        self.align_freq = nn.Identity()

        if spatial_dim != freq_dim:
            common_dim = min(spatial_dim, freq_dim)
            self.align_spatial = nn.Linear(spatial_dim, common_dim)
            self.align_freq = nn.Linear(freq_dim, common_dim)
            self.out_dim = common_dim * 2
        else:
            self.out_dim = spatial_dim + freq_dim
        
        # Attention weights
        self.gate = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, 2)
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
        gate_logits = self.gate(joint)
        gates = torch.softmax(gate_logits, dim=1)

        spatial_weight, freq_weight = gates[:, 0:1], gates[:, 1:2]

        # Apply attention
        weighted_spatial = spatial_norm_feat * spatial_weight
        weighted_freq = freq_norm_feat * freq_weight

        spatial_out = self.align_spatial(weighted_spatial)
        freq_out = self.align_freq(weighted_freq)

        # Concatenate
        fused = torch.cat([
            spatial_out, freq_out
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
        attn_gate = self.gate(joint)

        spatial_weight = attn_gate[:, 0:1]
        freq_weight = attn_gate[:, 1:2]

        return spatial_weight, freq_weight


# Dùng sigmoid thay vì softmax
class AttentionFusionSigmoid(nn.Module):
    """
    Sigmoid-gated attention fusion
    Each branch is gated independently (no competition).
    """

    def __init__(
        self,
        spatial_dim: int,
        freq_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        # Normalize each branch
        self.spatial_norm = nn.LayerNorm(spatial_dim)
        self.freq_norm = nn.LayerNorm(freq_dim)

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spatial_feat: (B, spatial_dim)
            freq_feat: (B, freq_dim)
        Returns:
            fused: (B, spatial_dim + freq_dim)
        """

        # Normalize features
        s = self.spatial_norm(spatial_feat)
        f = self.freq_norm(freq_feat)

        # Joint context
        joint = torch.cat([s, f], dim=1)

        # Independent gates
        gates = self.gate(joint)
        g_spatial = gates[:, 0:1]
        g_freq = gates[:, 1:2]

        # Apply gates
        s_weighted = s * g_spatial
        f_weighted = f * g_freq

        # Concatenate
        fused = torch.cat([s_weighted, f_weighted], dim=1)

        return fused

    @torch.no_grad()
    def get_attention_weights(self, spatial_feat, freq_feat):
        s = self.spatial_norm(spatial_feat)
        f = self.freq_norm(freq_feat)
        joint = torch.cat([s, f], dim=1)
        gates = self.gate(joint)
        return gates[:, 0:1], gates[:, 1:2]