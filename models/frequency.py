import torch
import torch.nn as nn

class FrequencyExtractor(nn.Module):
    """Fast, vectorized frequency-domain feature extractor (FFT / DCT / both)"""

    def __init__(self, method='fft', high_freq_ratio=0.5, img_size=224):
        super().__init__()
        self.method = method

        # Precomputed high-pass mask (moves with model.device)
        mask = self._create_high_pass_mask(img_size, img_size, high_freq_ratio)
        self.register_buffer("high_pass_mask", mask, persistent=False)

    def _create_high_pass_mask(self, H, W, ratio):
        center_h, center_w = H // 2, W // 2
        radius = int(min(H, W) * (1 - ratio) / 2)

        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij"
        )
        dist = torch.sqrt((x - center_w)**2 + (y - center_h)**2)
        mask = (dist > radius).float()
        return mask[None, None]  # shape (1, 1, H, W)

    def extract_fft_features(self, x):
        """
        Vectorized FFT for entire batch.
        Args: x (B, C, H, W)
        Returns: (B, C, H, W)
        """
        fft = torch.fft.fft2(x)  # Complex output

        # Faster fftshift using roll
        h2 = x.size(-2) // 2
        w2 = x.size(-1) // 2
        fft_shifted = torch.roll(fft, shifts=(h2, w2), dims=(-2, -1))

        magnitude = torch.log(torch.abs(fft_shifted) + 1e-8)
        return magnitude

    def extract_dct_features(self, x):
        """
        Vectorized DCT if available, fast FFT-based fallback otherwise.
        """
        try:
            import torch_dct as dct_lib
            return dct_lib.dct_2d(x)   # Vectorized
        except ImportError:
            # Fallback: fast approximation (vectorized)
            x_padded = torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)
            fft = torch.fft.fft(x_padded)
            return torch.real(fft[..., :x.shape[-1]]) * 0.5

    def forward(self, x):
        if self.method == 'fft':
            freq = self.extract_fft_features(x)
        elif self.method == 'dct':
            freq = self.extract_dct_features(x)
        elif self.method == 'both':
            freq = torch.cat([
                self.extract_fft_features(x),
                self.extract_dct_features(x)
            ], dim=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply high-pass mask (broadcasted)
        return freq * self.high_pass_mask


class FrequencyBranch(nn.Module):
    """Fast frequency processing branch"""

    def __init__(
        self,
        input_channels=3,
        method='fft',
        output_dim=512,
        img_size=224
    ):
        super().__init__()

        self.freq_extractor = FrequencyExtractor(
            method=method,
            img_size=img_size
        )

        # channels = 3 (RGB) or 6 (FFT+DCT)
        freq_channels = input_channels * (2 if method == 'both' else 1)

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(freq_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.projection = nn.Linear(256, output_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, output_dim)
        """
        freq_features = self.freq_extractor(x)
        features = self.conv_layers(freq_features)  # (B, 256, 1, 1)

        features = features.view(features.size(0), -1)  # (B, 256)
        features = self.projection(features)

        return features