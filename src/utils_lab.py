#!/usr/bin/env python3
"""Lab color space utilities for colorization."""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional
from sklearn.cluster import KMeans
from skimage import color
import cv2


def rgb_to_lab_tensor(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor [B,3,H,W] in [0,1] to Lab tensor [B,3,H,W]."""
    # Lab ranges: L[0,100], a[-128,127], b[-128,127]
    rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy()
    lab_np = np.stack([color.rgb2lab(img) for img in rgb_np])
    lab = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float().to(rgb.device)
    return lab


def lab_to_rgb_tensor(lab: torch.Tensor) -> torch.Tensor:
    """Convert Lab tensor [B,3,H,W] to RGB tensor [B,3,H,W] in [0,1]."""
    lab_np = lab.permute(0, 2, 3, 1).cpu().numpy()
    rgb_np = np.stack([color.lab2rgb(img) for img in lab_np])
    rgb = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float().to(lab.device)
    return rgb.clamp(0, 1)


def compute_ab_bins(dataset_ab_samples: np.ndarray, k: int = 313, cache_path: Optional[Path] = None) -> np.ndarray:
    """
    Compute k ab-bin centers via KMeans clustering.
    
    Args:
        dataset_ab_samples: [N, 2] array of ab values from dataset
        k: number of bins (default 313 as in Zhang)
        cache_path: optional path to save/load centers
    
    Returns:
        centers: [k, 2] array of ab cluster centers
    """
    if cache_path and cache_path.exists():
        centers = np.load(cache_path)
        if centers.shape == (k, 2):
            return centers
    
    # Subsample for efficiency
    if len(dataset_ab_samples) > 100000:
        indices = np.random.choice(len(dataset_ab_samples), 100000, replace=False)
        samples = dataset_ab_samples[indices]
    else:
        samples = dataset_ab_samples
    
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=300, n_init=10)
    kmeans.fit(samples)
    centers = kmeans.cluster_centers_
    
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, centers)
    
    return centers


def ab_to_bin_indices(ab: torch.Tensor, centers: np.ndarray, sigma: float = 5.0) -> torch.Tensor:
    """
    Map ab values to nearest bin indices.
    
    Args:
        ab: [B, 2, H, W] ab tensor
        centers: [K, 2] bin centers
        sigma: temperature for soft assignment (unused in hard assignment)
    
    Returns:
        indices: [B, H, W] long tensor of bin indices
    """
    B, _, H, W = ab.shape
    ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2).cpu().numpy()  # [B*H*W, 2]
    centers_torch = torch.from_numpy(centers).float()
    ab_torch = torch.from_numpy(ab_flat).float()
    
    # Compute distances: [B*H*W, K]
    dists = torch.cdist(ab_torch, centers_torch)
    indices = dists.argmin(dim=1).reshape(B, H, W)
    
    return indices.long()


def bin_indices_to_ab(indices: torch.Tensor, centers: np.ndarray) -> torch.Tensor:
    """
    Convert bin indices to ab values.
    
    Args:
        indices: [B, H, W] long tensor
        centers: [K, 2] centers
    
    Returns:
        ab: [B, 2, H, W] tensor
    """
    centers_torch = torch.from_numpy(centers).float().to(indices.device)
    ab_flat = centers_torch[indices.flatten()]  # [B*H*W, 2]
    B, H, W = indices.shape
    ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
    return ab


def compute_class_rebalancing_weights(bin_indices_list: list, num_bins: int, 
                                       lambda_smooth: float = 0.5) -> np.ndarray:
    """
    Compute inverse-frequency class weights with Gaussian smoothing (Zhang et al.).
    
    Args:
        bin_indices_list: list of bin index arrays from dataset
        num_bins: total number of bins
        lambda_smooth: smoothing parameter
    
    Returns:
        weights: [num_bins] array of rebalancing weights
    """
    # Compute empirical probability
    counts = np.zeros(num_bins)
    for indices in bin_indices_list:
        unique, cnt = np.unique(indices, return_counts=True)
        counts[unique] += cnt
    
    total = counts.sum()
    p = counts / (total + 1e-8)
    
    # Smooth with Gaussian kernel (simplified: use uniform mixing)
    p_smooth = (1 - lambda_smooth) * p + lambda_smooth / num_bins
    
    # Inverse frequency
    weights = 1.0 / (p_smooth + 1e-8)
    
    # Normalize to [0, 1] range then scale
    weights = weights / weights.sum() * num_bins
    
    return weights.astype(np.float32)


def visualize_colorization(L: torch.Tensor, ab: torch.Tensor, out_path: Path, 
                           denorm_L: bool = True) -> None:
    """
    Save colorized image from L and ab channels.
    
    Args:
        L: [1, 1, H, W] L channel tensor
        ab: [1, 2, H, W] ab channel tensor
        out_path: output file path
        denorm_L: whether to denormalize L from [-1,1] to [0,100]
    """
    if denorm_L:
        L = (L + 1.0) * 50.0  # [-1,1] -> [0,100]
    
    lab = torch.cat([L, ab], dim=1)  # [1, 3, H, W]
    rgb = lab_to_rgb_tensor(lab)
    
    # Convert to numpy and save
    img = rgb[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr)


def soft_encode_ab(ab: torch.Tensor, centers: np.ndarray, sigma: float = 5.0, T: float = 0.38) -> torch.Tensor:
    """
    Soft encode ab to distribution over bins (for training with soft targets).
    
    Args:
        ab: [B, 2, H, W]
        centers: [K, 2]
        sigma: kernel bandwidth
        T: temperature for normalization
    
    Returns:
        dist: [B, K, H, W] soft distribution
    """
    B, _, H, W = ab.shape
    K = len(centers)
    
    ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)  # [B*H*W, 2]
    centers_torch = torch.from_numpy(centers).float().to(ab.device)
    
    # Gaussian kernel: exp(-||ab - c||^2 / (2*sigma^2))
    dists_sq = torch.cdist(ab_flat, centers_torch).pow(2)  # [B*H*W, K]
    weights = torch.exp(-dists_sq / (2 * sigma ** 2))
    
    # Normalize
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Reshape
    dist = weights.reshape(B, H, W, K).permute(0, 3, 1, 2)
    
    return dist


def annealed_mean_ab(logits: torch.Tensor, centers: np.ndarray, T: float = 0.38) -> torch.Tensor:
    """
    Compute expected ab from logits using annealed softmax (Zhang et al.).
    
    Args:
        logits: [B, K, H, W]
        centers: [K, 2]
        T: temperature
    
    Returns:
        ab: [B, 2, H, W]
    """
    B, K, H, W = logits.shape
    
    # Temperature-scaled softmax
    probs = F.softmax(logits / T, dim=1)  # [B, K, H, W]
    
    centers_torch = torch.from_numpy(centers).float().to(logits.device)  # [K, 2]
    
    # Weighted sum: sum_k p_k * center_k
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, K)  # [B*H*W, K]
    ab_flat = probs_flat @ centers_torch  # [B*H*W, 2]
    
    ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
    
    return ab
