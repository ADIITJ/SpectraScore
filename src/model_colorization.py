#!/usr/bin/env python3
"""
Colorization model architectures. 

Implements:
- PaperNet: VGG-styled network from "Colorful Image Colorization" paper
- MobileLiteVariant: Memory-efficient variant for low-VRAM training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class PaperNet(nn.Module):
    """
    Colorization network from Zhang et al. ECCV 2016.
    
    Architecture follows Table 4 from paper with dilated convolutions.
    Input: L channel (1, H, W)
    Output: Distribution over Q=313 ab bins (313, H, W)
    """
    
    def __init__(self, num_classes: int = 313, input_channels: int = 1, use_checkpointing: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.use_checkpointing = use_checkpointing
        
        # Encoder with dilated convolutions
        # conv1: 64 filters, stride 1
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # conv2: 128 filters, stride 1
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # conv3: 256 filters, stride 1
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # conv4: 512 filters, stride 1
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # conv5: 512 filters, dilation 2
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn5 = nn.BatchNorm2d(512)
        
        # conv6: 512 filters, dilation 2
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn6 = nn.BatchNorm2d(512)
        
        # conv7: 512 filters, stride 1
        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        
        # conv8: 256 filters, upsample
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        
        # Output layer
        self.conv_out = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, 1, H, W) L channel input
            
        Returns:
            out: (B, Q, H, W) logits over ab bins
        """
        # Encoder
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.bn1(self.conv1_2(x)))  # 1/2 resolution
        
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.bn2(self.conv2_2(x)))  # 1/4 resolution
        
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.bn3(self.conv3_3(x)))  # 1/8 resolution
        
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.bn4(self.conv4_3(x)))  # 1/8 resolution
        
        # Dilated convolutions
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.bn5(self.conv5_3(x)))
        
        x = self.relu(self.conv6_1(x))
        x = self.relu(self.conv6_2(x))
        x = self.relu(self.bn6(self.conv6_3(x)))
        
        x = self.relu(self.conv7_1(x))
        x = self.relu(self.conv7_2(x))
        x = self.relu(self.bn7(self.conv7_3(x)))
        
        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.relu(self.conv8_1(x))
        x = self.relu(self.conv8_2(x))
        x = self.relu(self.bn8(self.conv8_3(x)))  # 1/4 resolution
        
        # Upsample to original resolution
        x = self.conv_out(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        return x

