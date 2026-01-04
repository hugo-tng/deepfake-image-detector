import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyExtractor(nn.Module):
    """
    Improved Frequency Extractor with proper normalization and masking
    """

    def __init__(self, high_freq_ratio=0.7, img_size=240):
        super().__init__()
        self.ratio = high_freq_ratio
        
        # Create initial mask
        mask = self._create_high_pass_mask(img_size, img_size, high_freq_ratio)
        self.register_buffer("high_pass_mask", mask, persistent=False)
        
        # Learnable temperature for mask
        self.mask_temperature = nn.Parameter(torch.ones(1))

    def _create_high_pass_mask(self, H, W, ratio):
        """Create high-pass filter mask"""
        center_h, center_w = H // 2, W // 2
        radius = int(min(H, W) * (1 - ratio) / 2)

        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij"
        )
        dist = torch.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Smooth mask instead of hard threshold
        # Using sigmoid for smooth transition
        mask = torch.sigmoid((dist - radius) * 0.1)
        
        return mask[None, None]  # (1, 1, H, W)

    def _normalize_fft(self, magnitude):
        """
        Proper per-sample, per-channel normalization
        Args:
            magnitude: (B, C, H, W)
        Returns:
            normalized: (B, C, H, W)
        """
        B, C, H, W = magnitude.shape
        
        # Compute statistics per sample, per channel
        # Reshape to (B, C, H*W)
        mag_flat = magnitude.view(B, C, -1)
        
        # Compute mean and std along spatial dimension
        mean = mag_flat.mean(dim=2, keepdim=True)  # (B, C, 1)
        std = mag_flat.std(dim=2, keepdim=True) + 1e-6  # (B, C, 1)
        
        # Normalize
        mag_normalized = (mag_flat - mean) / std
        
        # Reshape back
        mag_normalized = mag_normalized.view(B, C, H, W)
        
        return mag_normalized

    def extract_fft_features(self, x):
        """
        Extract FFT features with proper normalization
        Args:
            x: (B, C, H, W)
        Returns:
            magnitude: (B, C, H, W)
        """
        # FFT
        fft = torch.fft.fft2(x, norm='ortho')  # Orthogonal normalization
        
        # Shift zero frequency to center (proper way)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        
        # Compute magnitude with numerical stability
        magnitude = torch.log1p(torch.abs(fft_shifted))
        
        # Proper normalization (per-sample, per-channel)
        magnitude = self._normalize_fft(magnitude)
        
        return magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            filtered_freq: (B, C, H, W)
        """
        # Extract FFT features
        freq = self.extract_fft_features(x)
        
        # Dynamic mask resizing if needed
        if freq.shape[-2:] != self.high_pass_mask.shape[-2:]:
            current_mask = F.interpolate(
                self.high_pass_mask, 
                size=freq.shape[-2:], 
                mode='bilinear',
                align_corners=False
            )
        else:
            current_mask = self.high_pass_mask
        
        # Apply mask with learnable temperature
        filtered_freq = freq * torch.clamp(current_mask * self.mask_temperature, 0, 1)
        
        return filtered_freq


class FrequencyBranch(nn.Module):
    """Fast frequency processing branch"""

    def __init__(
        self,
        input_channels=3,
        output_dim=256,
        img_size=240,
        dropout=0.3
    ):
        super().__init__()

        self.freq_extractor = FrequencyExtractor(
            high_freq_ratio=0.7,
            img_size=img_size
        )

        self.block1 = self._make_block(input_channels, 64, dropout)
        self.block2 = self._make_block(64, 128, dropout)
        self.block3 = self._make_block(128, 256, dropout)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

    def _make_block(self, in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns: (B, output_dim)
        """
        freq_features = self.freq_extractor(x)

        # CNN blocks
        x1 = self.block1(freq_features)  # (B, 64, H/2, W/2)
        x2 = self.block2(x1)             # (B, 128, H/4, W/4)
        x3 = self.block3(x2)             # (B, 256, H/8, W/8)

        # Global pooling
        features = self.global_pool(x3)  # (B, 256, 1, 1)

        features = features.flatten(1)   # (B, 256)

        features = self.projection(features) # (B, output_dim)

        return features