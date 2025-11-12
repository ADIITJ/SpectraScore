#!/usr/bin/env python3
"""
Train all three colorization models and evaluate with SPCR.

Trains:
1. PaperNet (full model)
2. MobileLiteVariant (lightweight)
3. L2RegressionNet (regression baseline)
"""

import argparse
import csv
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model_colorization import PaperNet, MobileLiteVariant, L2RegressionNet
from utils_lab import (
    rgb_to_lab_tensor, lab_to_rgb_tensor, compute_ab_bins, ab_to_bin_indices,
    bin_indices_to_ab, compute_class_rebalancing_weights, annealed_mean_ab
)


def rgb_to_lab_single(rgb_np):
    """Convert single RGB numpy image to L and ab tensors."""
    # Normalize to [0, 1]
    rgb_np = rgb_np.astype(np.float32) / 255.0
    # Convert to tensor and add batch dim
    rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    lab = rgb_to_lab_tensor(rgb_tensor).squeeze(0)  # [3, H, W]
    L = lab[0:1]  # [1, H, W]
    ab = lab[1:3]  # [2, H, W]
    return L, ab


class ColorizationDataset(Dataset):
    """Dataset for colorization training."""
    
    def __init__(self, image_dir: Path, img_size: int = 176, centers: Optional[np.ndarray] = None, 
                 is_regression: bool = False):
        self.image_dir = image_dir
        self.img_size = img_size
        self.centers = centers
        self.is_regression = is_regression
        
        # Collect image paths
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and resize
        img = cv2.imread(str(img_path))
        if img is None:
            # Return a black image if loading fails
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to Lab
        L, ab_gt = rgb_to_lab_single(img)
        
        if self.is_regression:
            # Regression: return ab directly
            return L, ab_gt
        else:
            # Classification: convert ab to bin indices
            if self.centers is None:
                raise ValueError("centers required for classification")
            bin_indices = ab_to_bin_indices(ab_gt.unsqueeze(0), self.centers).squeeze(0)
            return L, bin_indices


def train_epoch(model, dataloader, optimizer, device, is_regression=False, 
                class_weights=None, epoch=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for L, target in pbar:
        L = L.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(L)
        
        if is_regression:
            # L2 loss for regression
            loss = F.mse_loss(output, target)
        else:
            # Cross-entropy for classification
            if class_weights is not None:
                loss = F.cross_entropy(output, target, weight=class_weights)
            else:
                loss = F.cross_entropy(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, out_dir, model_name):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    path = out_dir / f"{model_name}_epoch_{epoch+1:03d}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def evaluate_with_spcr(model, test_images_dir, device, centers=None, is_regression=False,
                       output_dir=None, model_name="model"):
    """Colorize test images and evaluate with SPCR."""
    model.eval()
    
    if output_dir is None:
        output_dir = Path("results") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colorized_dir = output_dir / "colorized"
    colorized_dir.mkdir(exist_ok=True)
    
    # Colorize test images
    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    print(f"\nColorizing {len(test_images)} test images...")
    with torch.no_grad():
        for img_path in tqdm(test_images):
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H_orig, W_orig = img.shape[:2]
            
            # Resize to model input size
            img_resized = cv2.resize(img, (176, 176))
            
            # Convert to Lab
            L, _ = rgb_to_lab_single(img_resized)
            L = L.unsqueeze(0).to(device)
            
            # Predict
            output = model(L)
            
            if is_regression:
                ab_pred = output[0]  # (2, H, W)
            else:
                # Classification: convert logits to ab
                logits = output[0]  # (Q, H, W)
                ab_pred = annealed_mean_ab(logits, centers, T=0.38)
            
            # Get L channel
            L_np = L[0, 0].cpu().numpy()
            ab_np = ab_pred.cpu().numpy()
            
            # Resize ab back to original size
            ab_np = np.stack([
                cv2.resize(ab_np[0], (W_orig, H_orig)),
                cv2.resize(ab_np[1], (W_orig, H_orig))
            ], axis=0)
            
            # Resize L back to original size
            L_np = cv2.resize(L_np, (W_orig, H_orig))
            
            # Combine L and ab, convert to RGB
            Lab_full = np.stack([L_np, ab_np[0], ab_np[1]], axis=0)  # [3, H, W]
            Lab_full_tensor = torch.from_numpy(Lab_full).unsqueeze(0)  # [1, 3, H, W]
            rgb_tensor = lab_to_rgb_tensor(Lab_full_tensor).squeeze(0)  # [3, H, W]
            rgb = (rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Save
            out_path = colorized_dir / img_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    print(f"Colorized images saved to: {colorized_dir}")
    
    # Run SPCR evaluation
    print("\nRunning SPCR evaluation...")
    project_root = Path(__file__).parent.parent
    spcr_script = project_root / "src" / "spcr_full.py"
    results_csv = output_dir / "spcr_results.csv"
    
    cmd = [
        sys.executable,
        str(spcr_script),
        "--original", str(test_images_dir),
        "--colorized", str(colorized_dir),
        "--output", str(results_csv),
        "--device", str(device)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"SPCR results saved to: {results_csv}")
    except subprocess.CalledProcessError as e:
        print(f"SPCR evaluation failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Directory with training images")
    parser.add_argument("--out_dir", type=str, default="outputs/all_models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=176, help="Image size")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--test_images", type=str, default="assets/original", help="Test images for SPCR")
    args = parser.parse_args()
    
    # Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_root)
    test_images_dir = Path(args.test_images)
    
    # Device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Compute ab bins and class weights for classification models
    print("\nCollecting ab samples from training data...")
    ab_samples = []
    bin_indices_list = []
    
    for img_path in list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.img_size, args.img_size))
        
        # Convert to Lab
        L, ab = rgb_to_lab_single(img)
        ab_np = ab.cpu().numpy().reshape(2, -1).T  # [H*W, 2]
        ab_samples.append(ab_np)
    
    ab_samples = np.vstack(ab_samples)
    print(f"Collected {len(ab_samples)} ab samples")
    
    print("Computing ab bins via KMeans...")
    centers = compute_ab_bins(ab_samples, k=313, cache_path=out_dir / "ab_centers.npy")
    print(f"Centers shape: {centers.shape}")
    
    print("Computing class rebalancing weights...")
    # Compute bin indices for all training images
    for img_path in list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.img_size, args.img_size))
        L, ab = rgb_to_lab_single(img)
        indices = ab_to_bin_indices(ab.unsqueeze(0), centers).squeeze(0)
        bin_indices_list.append(indices.cpu().numpy().flatten())
    
    class_weights = compute_class_rebalancing_weights(bin_indices_list, num_bins=313, lambda_smooth=0.5)
    class_weights_tensor = torch.from_numpy(class_weights).float().to(device)
    print(f"Class weights shape: {class_weights.shape}")
    
    # Generate test images if needed
    if not test_images_dir.exists() or len(list(test_images_dir.glob("*.jpg"))) == 0:
        print("\nGenerating test images...")
        test_script = Path(__file__).parent / "test_spcr.py"
        subprocess.run([sys.executable, str(test_script)], check=True)
    
    # Models to train
    models_config = [
        {
            "name": "PaperNet",
            "model_class": PaperNet,
            "kwargs": {"num_classes": 313},
            "is_regression": False,
        },
        {
            "name": "MobileLite",
            "model_class": MobileLiteVariant,
            "kwargs": {"num_classes": 313, "base_channels": 32},
            "is_regression": False,
        },
        {
            "name": "L2Regression",
            "model_class": L2RegressionNet,
            "kwargs": {"base_channels": 32},
            "is_regression": True,
        }
    ]
    
    # Train each model
    for config in models_config:
        print(f"\n{'='*60}")
        print(f"Training {config['name']}")
        print(f"{'='*60}")
        
        model_out_dir = out_dir / config['name']
        model_out_dir.mkdir(exist_ok=True)
        
        # Create model
        model = config['model_class'](**config['kwargs']).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Dataset and dataloader
        dataset = ColorizationDataset(
            data_dir, 
            img_size=args.img_size,
            centers=centers if not config['is_regression'] else None,
            is_regression=config['is_regression']
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training log
        log_file = model_out_dir / "training_log.csv"
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'lr'])
        
        # Train
        for epoch in range(args.epochs):
            avg_loss = train_epoch(
                model, dataloader, optimizer, device,
                is_regression=config['is_regression'],
                class_weights=class_weights_tensor if not config['is_regression'] else None,
                epoch=epoch
            )
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            
            # Log
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_loss, current_lr])
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, model_out_dir, config['name'])
            
            scheduler.step()
        
        # Evaluate with SPCR
        print(f"\nEvaluating {config['name']} with SPCR...")
        evaluate_with_spcr(
            model, test_images_dir, device,
            centers=centers if not config['is_regression'] else None,
            is_regression=config['is_regression'],
            output_dir=model_out_dir,
            model_name=config['name']
        )
        
        print(f"\n{config['name']} training complete!")
        print(f"Checkpoints saved to: {model_out_dir}")
    
    print(f"\n{'='*60}")
    print("All models trained and evaluated!")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
