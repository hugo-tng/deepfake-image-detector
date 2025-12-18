import torch
import torch.nn as nn
from .spatial import SpatialBranch
from .frequency import FrequencyBranch
from .fusion import AttentionFusion
from typing import Tuple

class DeepfakeDetector(nn.Module):
    """EfficientNet-based model with frequency domain features for DeepFake detection
    """
    
    def __init__(
        self,
        mode='hybrid',
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of output classes (2 for binary)
            dropout_rate: Dropout rate before classifier
        """
        super(DeepfakeDetector, self).__init__()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.spatial_frozen = False
        self.img_size = kwargs.get("img_size", 240)
        self.attn_hidden_dim = kwargs.get("attention_hidden_dim", 256)

        # Spatial branch parameters
        if self.mode in ['spatial', 'hybrid']:
            efficientnet_model = kwargs.get('efficientnet_model', 'efficientnet_b0')
            spatial_dim = kwargs.get('spatial_dim', 512)

            self.spatial_branch = SpatialBranch(
                model_name=efficientnet_model,
                output_dim=spatial_dim
            )
            self._spatial_dim = spatial_dim
        else:
            self.spatial_branch = None
            self._spatial_dim = 0

        # Frequency branch parameters
        if self.mode in ['frequency', 'hybrid']:
            freq_dim = kwargs.get('freq_dim', 512)

            self.frequency_branch = FrequencyBranch(
                input_channels=3,
                output_dim=freq_dim,
                img_size=self.img_size
            )
            self._freq_dim = freq_dim
        else:
            self.frequency_branch = None
            self._freq_dim = 0

        # Fusion module
        fusion_dim = self._spatial_dim + self._freq_dim

        if self.mode == 'hybrid':
            use_attention_fusion = kwargs.get('use_attention_fusion', True)
            if use_attention_fusion:
                self.fusion = AttentionFusion(
                    spatial_dim=self._spatial_dim,
                    freq_dim=self._freq_dim,
                    hidden_dim=self.attn_hidden_dim
                )
            else:
                self.fusion = None
        else:
            self.fusion = None            

        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            Logits (B, num_classes)
        """
       # Extract Spatial
        if self.mode == 'spatial':
            features = self.spatial_branch(x)
            
        # Extract Frequency
        elif self.mode == 'frequency':
            features = self.frequency_branch(x)
            
        # Hybrid mode
        else:
            spatial_feat = self.spatial_branch(x)
            freq_feat = self.frequency_branch(x)
            
            if self.fusion:
                features = self.fusion(spatial_feat, freq_feat)
            else:
                features = torch.cat([spatial_feat, freq_feat], dim=1)
        
        # classifier
        logits = self.classifier(features)
        return logits
    

    def freeze_spatial_backbone(self, freeze: bool = True):
        """
        Đóng băng hoặc mở băng nhánh Spatial (EfficientNet).
        Args:
            freeze: True để đóng băng, False để cho phép train.
        """
        if self.spatial_branch is None:
            return
        
        self.spatial_frozen = freeze
        # Duyệt qua tất cả tham số của spatial_branch
        for param in self.spatial_branch.parameters():
            param.requires_grad = not freeze

        for m in self.spatial_branch.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                if freeze:
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
                else:
                    m.train()
                    for p in m.parameters():
                        p.requires_grad = True

        for param in self.spatial_branch.projection.parameters():
            param.requires_grad = True
            
        print(f"== Spatial Branch is now {'Frozen' if freeze else 'Unfrozen'}.")

    def unfreeze_spatial_backbone(self):
        """Hàm tiện ích để mở băng nhanh"""
        self.freeze_spatial_backbone(freeze=False)

    def is_spatial_frozen(self):
        return self.spatial_frozen
    
    @torch.no_grad()
    def get_feature_importance(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for interpretability
        Returns: (spatial_weight, freq_weight)
        """
        if self.fusion is None:
            return None, None
        
        spatial_features = self.spatial_branch(x)
        freq_features = self.frequency_branch(x)
        
        spatial_weight, freq_weight = self.fusion.get_attention_weights(
            spatial_features,
            freq_features
        )
            
        return spatial_weight, freq_weight