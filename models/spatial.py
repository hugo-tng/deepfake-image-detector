import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,  
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights
)

class SpatialBranch(nn.Module):
    """Spatial feature extraction using EfficientNet-B4"""
    
    def __init__(
        self, 
        model_name: str = 'efficientnet_b0',
        output_dim: int = 512
    ):
        """
        Args:
            model_name: EfficientNet variant
            pretrained: Use ImageNet pretrained weights
            output_dim: Output feature dimension
        """
        super(SpatialBranch, self).__init__()
        
        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b4':
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b4(weights=weights)
            backbone_output_dim = 1792  
        elif model_name == 'efficientnet_b3':
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b3(weights=weights)
            backbone_output_dim = 1536
        elif model_name == 'efficientnet_b2':
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b2(weights=weights)
            backbone_output_dim = 1408
        elif model_name == 'efficientnet_b1':
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b1(weights=weights)
            backbone_output_dim = 1280
        elif model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
            backbone_output_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove classification head
        self.backbone.classifier = nn.Identity()
        
        self.output_dim = output_dim
        self.backbone_output_dim = backbone_output_dim
        
        # Add projection layer
        if output_dim != backbone_output_dim:
            self.projection = nn.Linear(backbone_output_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            Spatial features (B, output_dim)
        """
        # Extract features
        features = self.backbone(x)
        
        # Project
        features = self.projection(features)
        
        return features