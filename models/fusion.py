import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """Attention-based fusion of spatial and frequency features"""
    
    def __init__(self, spatial_dim: int, freq_dim: int, hidden_dim: int = 224):
        """
        Args:
            spatial_dim: Dimension of spatial features
            freq_dim: Dimension of frequency features
            hidden_dim: Hidden dimension for attention
        """
        super(AttentionFusion, self).__init__()
        
        # Attention weights
        self.spatial_attention = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.freq_attention = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
        # Calculate attention weights
        spatial_weight = self.spatial_attention(spatial_features)
        freq_weight = self.freq_attention(freq_features)
        
        # Normalize weights
        total_weight = spatial_weight + freq_weight
        spatial_weight = spatial_weight / (total_weight + 1e-8)
        freq_weight = freq_weight / (total_weight + 1e-8)
        
        # Apply attention
        weighted_spatial = spatial_features * spatial_weight
        weighted_freq = freq_features * freq_weight
        
        # Concatenate
        fused = torch.cat([weighted_spatial, weighted_freq], dim=1)
        
        return fused