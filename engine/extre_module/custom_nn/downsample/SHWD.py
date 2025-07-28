import torch
import torch.nn as nn
from calflops import calculate_flops


class LAWD(nn.Module):
    """
    Lightweight Asymmetric Wavelet Downsampling (LAWD) module.

    This module improves upon the Haar Wavelet Downsampling (HWD) by introducing
    an asymmetric architecture to process low-frequency and high-frequency components
    separately and more efficiently, making it suitable for micro-object detection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_attn (bool): Whether to use the channel attention mechanism in the high-frequency path.
    """

    def __init__(self, in_channels, out_channels, use_attn=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attn = use_attn

        # The low-frequency path processes the 'A' component (C channels)
        # It's the main path for contextual information.
        self.low_freq_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU(inplace=True)
        )

        # The high-frequency path processes the 'H', 'V', 'D' components (3*C channels)
        # It's designed to be lightweight and focus on detail enhancement.
        # We use a grouped convolution to process H, V, D components somewhat independently.
        self.high_freq_path = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(in_channels * 3),
            nn.SiLU(inplace=True)
        )

        # Optional micro-attention block for the high-frequency path
        if self.use_attn:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Squeeze
                nn.Conv2d(in_channels * 3, in_channels // 4, kernel_size=1, bias=False),  # Excite (reduce)
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, in_channels * 3, kernel_size=1, bias=False),  # Excite (expand)
                nn.Sigmoid()
            )

        # Final fusion layer to combine the two paths and project to the output dimension
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3 + out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        # Input tensor shape:
        B, C, H, W = x.shape

        # Ensure height and width are even
        assert H % 2 == 0 and W % 2 == 0, "Input height and width must be even for Haar wavelet transform."

        # Step 1: Haar Wavelet Decomposition using tensor slicing
        # This is more efficient than using a fixed convolution kernel.
        # Decompose along height
        x_top = x[:, :, 0:H:2, :]
        x_bottom = x[:, :, 1:H:2, :]
        L_h = (x_top + x_bottom) / 2
        H_h = (x_top - x_bottom) / 2

        # Decompose along width
        # Low-frequency component (A)
        LL = (L_h[:, :, :, 0:W:2] + L_h[:, :, :, 1:W:2]) / 2
        # High-frequency components (H, V, D)
        LH = (L_h[:, :, :, 0:W:2] - L_h[:, :, :, 1:W:2]) / 2  # Horizontal details
        HL = (H_h[:, :, :, 0:W:2] + H_h[:, :, :, 1:W:2]) / 2  # Vertical details
        HH = (H_h[:, :, :, 0:W:2] - H_h[:, :, :, 1:W:2]) / 2  # Diagonal details

        # Step 2: Asymmetric Processing
        # Path 1: Low-frequency main path
        low_freq_out = self.low_freq_path(LL)  # Shape:

        # Path 2: High-frequency bypass
        high_freq_in = torch.cat([LH, HL, HH], dim=1)  # Shape:
        high_freq_feat = self.high_freq_path(high_freq_in)

        if self.use_attn:
            attention_weights = self.attn(high_freq_feat)
            high_freq_out = high_freq_feat * attention_weights
        else:
            high_freq_out = high_freq_feat

        # Step 3: Feature Fusion
        # Concatenate the outputs from both paths
        fusion_in = torch.cat([low_freq_out, high_freq_out], dim=1)

        # Final fusion and projection
        output = self.fusion(fusion_in)  # Shape:

        return output

