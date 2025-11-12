#!/usr/bin/env python3
"""Colorization model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional
import numpy as np


class ResNetEncoder(nn.Module):
    """ResNet-18 encoder (pretrained, truncated before avgpool)."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Remove avgpool and fc
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Output channels: 512 for resnet18
        self.out_channels = 512
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class ConvEncoder(nn.Module):
    """Lightweight convolutional encoder."""
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            # 176 -> 88
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 88 -> 44
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 44 -> 22
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 22 -> 11
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.out_channels = 512
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder with upsampling blocks."""
    
    def __init__(self, in_channels: int, out_channels: int, num_classes: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        
        # Upsample blocks
        self.up1 = self._make_upsample_block(in_channels, 256)
        self.up2 = self._make_upsample_block(256, 128)
        self.up3 = self._make_upsample_block(128, 64)
        self.up4 = self._make_upsample_block(64, 64)
        
        # Final output head
        if num_classes is not None:
            # Classification: output logits over bins
            self.head = nn.Conv2d(64, num_classes, 1)
        else:
            # Regression: output 2-channel ab
            self.head = nn.Sequential(
                nn.Conv2d(64, out_channels, 1),
                nn.Tanh()  # ab in [-1, 1] (scaled to [-128, 127])
            )
    
    def _make_upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.head(x)
        return x


class ColorizationModel(nn.Module):
    """Full colorization model with encoder-decoder architecture."""
    
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, 
                 num_classes: Optional[int] = None, input_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder
        if backbone == "resnet18":
            # ResNet expects 3-channel input, replicate L channel
            self.encoder = ResNetEncoder(pretrained=pretrained)
            self.replicate_channels = True
        elif backbone == "custom":
            self.encoder = ConvEncoder(in_channels=input_channels)
            self.replicate_channels = False
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Decoder
        out_channels = 2 if num_classes is None else num_classes
        self.decoder = Decoder(self.encoder.out_channels, out_channels, num_classes)
    
    def forward(self, L):
        """
        Args:
            L: [B, 1, H, W] grayscale L channel
        
        Returns:
            output: [B, 2, H, W] for regression or [B, num_classes, H, W] for classification
        """
        # ResNet expects 3 channels
        if self.replicate_channels:
            x = L.repeat(1, 3, 1, 1)
        else:
            x = L
        
        features = self.encoder(x)
        output = self.decoder(features)
        
        return output
    
    def predict_ab_from_logits(self, logits: torch.Tensor, centers: np.ndarray, 
                                T: float = 0.38) -> torch.Tensor:
        """
        Convert classification logits to ab values via annealed softmax.
        
        Args:
            logits: [B, num_classes, H, W]
            centers: [num_classes, 2] bin centers
            T: temperature
        
        Returns:
            ab: [B, 2, H, W]
        """
        B, K, H, W = logits.shape
        
        # Temperature-scaled softmax
        probs = F.softmax(logits / T, dim=1)
        
        centers_torch = torch.from_numpy(centers).float().to(logits.device)
        
        # Weighted sum over bins
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, K)
        ab_flat = probs_flat @ centers_torch
        ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab
    
    def predict(self, L: torch.Tensor, centers: Optional[np.ndarray] = None, 
                T: float = 0.38) -> torch.Tensor:
        """
        Inference method.
        
        Args:
            L: [B, 1, H, W]
            centers: required if num_classes is not None
            T: temperature for classification
        
        Returns:
            ab: [B, 2, H, W]
        """
        output = self.forward(L)
        
        if self.num_classes is not None:
            if centers is None:
                raise ValueError("centers required for classification model")
            ab = self.predict_ab_from_logits(output, centers, T)
            # Scale ab from bin centers (which are in [-128, 127] range)
            return ab
        else:
            # Regression: scale from [-1, 1] to [-128, 127]
            return output * 128.0


def build_model(backbone: str = "resnet18", pretrained: bool = True, 
                loss_type: str = "classification", num_bins: int = 313) -> ColorizationModel:
    """
    Factory function to build colorization model.
    
    Args:
        backbone: "resnet18" or "custom"
        pretrained: use pretrained weights
        loss_type: "classification" or "regression"
        num_bins: number of ab bins for classification
    
    Returns:
        model: ColorizationModel instance
    """
    num_classes = num_bins if loss_type == "classification" else None
    return ColorizationModel(backbone=backbone, pretrained=pretrained, num_classes=num_classes)
