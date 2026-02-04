"""
Noise2Void U-Net Model - Optimized Configuration

Based on optimization study findings:
- Narrow network (4 base channels) prevents hallucinations
- Depth 3 is optimal for single-image denoising
- Kaiming (He) initialization for ReLU activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias
    )


def conv1x1(in_channels, out_channels):
    """1x1 convolution for channel reduction"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1
    )


class DownBlock(nn.Module):
    """
    Encoder block: 2 convolutions + optional max pooling
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.pooling = pooling
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        skip = x
        if self.pooling:
            x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """
    Decoder block: upsample + concatenate skip + 2 convolutions
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat, channels = out_channels (from up) + out_channels (from skip)
        self.conv1 = conv3x3(out_channels * 2, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class N2VUNet(nn.Module):
    """
    Noise2Void U-Net with Optimized Configuration
    
    Key design choices from optimization study:
    - start_channels=4: Narrow network prevents overfitting/hallucinations
    - depth=3: Optimal for single-image denoising
    - Kaiming initialization: Proper for ReLU activations
    - Skip connections via concatenation
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale)
    out_channels : int
        Number of output channels (1 for grayscale prediction)
    depth : int
        Number of pooling operations (default: 3)
    start_channels : int
        Number of filters in first layer, doubles each depth (default: 4)
    """
    
    def __init__(self, in_channels=1, out_channels=1, depth=3, start_channels=4):
        super(N2VUNet, self).__init__()
        
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_channels = start_channels
        
        # Store normalization parameters (set during training)
        self.mean = 0.0
        self.std = 1.0
        
        # Build encoder
        self.encoders = nn.ModuleList()
        channels = start_channels
        in_ch = in_channels
        
        for i in range(depth):
            pooling = (i < depth - 1)  # No pooling at bottom
            self.encoders.append(DownBlock(in_ch, channels, pooling=pooling))
            in_ch = channels
            if i < depth - 1:
                channels *= 2
        
        # Build decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = channels
            out_ch = channels // 2
            self.decoders.append(UpBlock(in_ch, out_ch))
            channels = out_ch
        
        # Final 1x1 convolution
        self.final_conv = conv1x1(channels, out_channels)
        
        # Initialize weights with Kaiming (He) initialization
        self._init_weights()
    
    def _init_weights(self):
        """Apply Kaiming (He) initialization for ReLU activations"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path - save skip connections
        skips = []
        for i, encoder in enumerate(self.encoders):
            x, skip = encoder(x)
            if i < self.depth - 1:  # Don't save skip from bottom
                skips.append(skip)
        
        # Decoder path - use skip connections in reverse order
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_n2v_model(depth=3, start_channels=4):
    """
    Factory function to create N2V model with optimal configuration
    
    Parameters
    ----------
    depth : int
        U-Net depth (default: 3)
    start_channels : int
        Base channel count (default: 4, as per optimization study)
    
    Returns
    -------
    N2VUNet
        Initialized model
    """
    model = N2VUNet(
        in_channels=1,
        out_channels=1,
        depth=depth,
        start_channels=start_channels
    )
    print(f"Created N2V U-Net:")
    print(f"  Depth: {depth}")
    print(f"  Base channels: {start_channels}")
    print(f"  Parameters: {model.count_parameters():,}")
    return model
