#!/usr/bin/env python3
"""Training pipeline for image colorization."""

import argparse
import csv
import os
import random
import subprocess
import sys
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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_colorization import build_model
from utils_lab import (
    rgb_to_lab_tensor, lab_to_rgb_tensor, compute_ab_bins, ab_to_bin_indices,
    bin_indices_to_ab, compute_class_rebalancing_weights, visualize_colorization,
    annealed_mean_ab
)


class ColorizationDataset(Dataset):
    """Dataset for colorization training."""
    
    def __init__(self, image_dir: Path, img_size: int = 176, mode: str = "train",
                 centers: Optional[np.ndarray] = None, loss_type: str = "classification"):
        self.image_dir = image_dir
        self.img_size = img_size
        self.mode = mode
        self.centers = centers
        self.loss_type = loss_type
        
        # Collect image paths
        # Collect image paths (supports nested directories like ImageNet)
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG'}  # Added .JPEG for ImageNet
        self.image_paths = []
        
        # Check if images are directly in image_dir (COCO style)
        direct_images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        
        if len(direct_images) > 0:
            # Flat structure (COCO): images directly in directory
            self.image_paths = direct_images
            print(f"Found {len(self.image_paths)} images in flat directory structure")
        else:
            # Nested structure (ImageNet): images in subdirectories
            for class_dir in sorted(image_dir.iterdir()):
                if class_dir.is_dir():
                    class_images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
                    self.image_paths.extend(class_images)
            print(f"Found {len(self.image_paths)} images across {len([d for d in image_dir.iterdir() if d.is_dir()])} subdirectories")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir} or its subdirectories")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load RGB
        img = cv2.imread(str(img_path))
        if img is None:
            # Return black image on error
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize or crop
        if self.mode == "train":
            # Random crop
            h, w = img.shape[:2]
            if h < self.img_size or w < self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size))
            else:
                y = random.randint(0, h - self.img_size)
                x = random.randint(0, w - self.img_size)
                img = img[y:y+self.img_size, x:x+self.img_size]
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # RGB to Lab
        lab = rgb_to_lab_tensor(img.unsqueeze(0)).squeeze(0)
        L, ab = lab[0:1], lab[1:3]
        
        # Normalize L to [-1, 1]
        L = (L / 50.0) - 1.0
        
        if self.loss_type == "classification":
            # Convert ab to bin indices
            if self.centers is None:
                raise ValueError("centers required for classification")
            
            # ab is [2, H, W], need [1, 2, H, W] for function
            bin_indices = ab_to_bin_indices(ab.unsqueeze(0), self.centers).squeeze(0)
            return L, bin_indices, ab, img_path.name
        else:
            # Regression: normalize ab to [-1, 1]
            ab = ab / 128.0
            return L, ab, ab, img_path.name
    
    def get_all_ab_values(self, max_samples: int = 10000):
        """Collect ab values for computing bin centers."""
        ab_list = []
        indices = np.random.choice(len(self), min(max_samples, len(self)), replace=False)
        
        for idx in tqdm(indices, desc="Collecting ab values"):
            img_path = self.image_paths[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            lab = rgb_to_lab_tensor(img.unsqueeze(0)).squeeze(0)
            ab = lab[1:3].permute(1, 2, 0).reshape(-1, 2).numpy()
            ab_list.append(ab)
        
        return np.concatenate(ab_list, axis=0)


def train_epoch(model, dataloader, optimizer, criterion, device, class_weights=None, 
                loss_type="classification"):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if loss_type == "classification":
            L, target_bins, _, _ = batch
            L = L.to(device)
            target_bins = target_bins.to(device)
        else:
            L, target_ab, _, _ = batch
            L = L.to(device)
            target_ab = target_ab.to(device)
        
        optimizer.zero_grad()
        
        output = model(L)
        
        if loss_type == "classification":
            # output: [B, num_classes, H, W], target: [B, H, W]
            loss = criterion(output, target_bins)
        else:
            # output: [B, 2, H, W], target: [B, 2, H, W]
            loss = criterion(output, target_ab)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, epoch, out_dir, centers=None,
                   loss_type="classification", save_images=True):
    """Validation epoch with optional image saving."""
    model.eval()
    total_loss = 0.0
    
    if save_images:
        colorized_dir = out_dir / f"eval/epoch_{epoch:03d}/colorized"
        original_dir = out_dir / f"eval/epoch_{epoch:03d}/original"
        colorized_dir.mkdir(parents=True, exist_ok=True)
        original_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if loss_type == "classification":
                L, target_bins, gt_ab, filenames = batch
                L = L.to(device)
                target_bins = target_bins.to(device)
            else:
                L, target_ab, gt_ab, filenames = batch
                L = L.to(device)
                target_ab = target_ab.to(device)
            
            output = model(L)
            
            if loss_type == "classification":
                loss = criterion(output, target_bins)
            else:
                loss = criterion(output, target_ab)
            
            total_loss += loss.item()
            
            # Save images
            if save_images:
                for i in range(L.size(0)):
                    L_img = L[i:i+1]
                    
                    if loss_type == "classification":
                        pred_ab = annealed_mean_ab(output[i:i+1], centers, T=0.38)
                    else:
                        pred_ab = output[i:i+1] * 128.0
                    
                    # Save colorized
                    visualize_colorization(L_img, pred_ab, 
                                          colorized_dir / filenames[i], denorm_L=True)
                    
                    # Save original
                    gt_ab_scaled = gt_ab[i:i+1].to(device)
                    visualize_colorization(L_img, gt_ab_scaled,
                                          original_dir / filenames[i], denorm_L=True)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, colorized_dir if save_images else None, original_dir if save_images else None


def run_spcr_evaluation(original_dir: Path, colorized_dir: Path, epoch: int, 
                        out_dir: Path) -> Tuple[float, float]:
    """Run SPCR evaluation scripts and parse results."""
    spcr_full_csv = out_dir / f"eval/epoch_{epoch:03d}/spcr_full.csv"
    spcr_light_csv = out_dir / f"eval/epoch_{epoch:03d}/spcr_light.csv"
    
    # Get path to SPCR scripts in src/
    src_dir = Path(__file__).parent
    
    # Run spcr_full.py
    try:
        subprocess.run([
            sys.executable, str(src_dir / "spcr_full.py"),
            "--original", str(original_dir),
            "--colorized", str(colorized_dir),
            "--output", str(spcr_full_csv),
            # "--device", "mps" if torch.backends.mps.is_available() else "cpu"
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ], check=True, capture_output=True, timeout=600)
    except Exception as e:
        print(f"Warning: spcr_full.py failed: {e}")
        spcr_full_csv = None
    
    # Run spcr_light.py
    try:
        subprocess.run([
            sys.executable, str(src_dir / "spcr_light.py"),
            "--original", str(original_dir),
            "--colorized", str(colorized_dir),
            "--output", str(spcr_light_csv),
            # "--device", "mps" if torch.backends.mps.is_available() else "cpu"
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ], check=True, capture_output=True, timeout=600)
    except Exception as e:
        print(f"Warning: spcr_light.py failed: {e}")
        spcr_light_csv = None
    
    # Parse results
    spcr_full_score = 0.0
    spcr_light_score = 0.0
    
    if spcr_full_csv and spcr_full_csv.exists():
        with open(spcr_full_csv, 'r') as f:
            reader = csv.DictReader(f)
            scores = [float(row['spcr']) for row in reader]
            spcr_full_score = np.mean(scores) if scores else 0.0
    
    if spcr_light_csv and spcr_light_csv.exists():
        with open(spcr_light_csv, 'r') as f:
            reader = csv.DictReader(f)
            scores = [float(row['spcr_light']) for row in reader]
            spcr_light_score = np.mean(scores) if scores else 0.0
    
    return spcr_full_score, spcr_light_score


def main():
    parser = argparse.ArgumentParser(description="Train colorization model")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--images_csv", type=str, default=None, help="Optional CSV with image list")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss_type", type=str, default="classification", 
                       choices=["classification", "regression"], help="Loss type")
    parser.add_argument("--use_mps", action="store_true", help="Use MPS device on Mac")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--eval_after_epoch", action="store_true", help="Run SPCR eval after each epoch")
    parser.add_argument("--ab_centers_path", type=str, default=None, help="Path to precomputed ab centers")
    parser.add_argument("--num_bins", type=int, default=313, help="Number of ab bins")
    parser.add_argument("--img_size", type=int, default=176, help="Image size")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "custom"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # data_root = Path(args.data_root)
    # if not data_root.exists():
    #     raise ValueError(f"data_root does not exist: {data_root}")
    
    # # Find images directory
    # if (data_root / "images").exists():
    #     image_dir = data_root / "images"
    # else:
    #     image_dir = data_root
    
    # # Load or compute ab bin centers
    # if args.loss_type == "classification":
    #     if args.ab_centers_path and Path(args.ab_centers_path).exists():
    #         centers = np.load(args.ab_centers_path)
    #     else:
    #         print("Computing ab bin centers...")
    #         temp_dataset = ColorizationDataset(image_dir, args.img_size, mode="train", 
    #                                           loss_type="regression")
    #         ab_samples = temp_dataset.get_all_ab_values(max_samples=10000)
    #         centers_path = out_dir / "ab_centers.npy"
    #         centers = compute_ab_bins(ab_samples, k=args.num_bins, cache_path=centers_path)
    #         print(f"Saved ab centers to {centers_path}")
    # else:
    #     centers = None
    
    # # Create datasets
    # all_dataset = ColorizationDataset(image_dir, args.img_size, mode="train", 
    #                                  centers=centers, loss_type=args.loss_type)
    
    # # Train/val split
    # n_val = int(len(all_dataset) * args.val_split)
    # n_train = len(all_dataset) - n_val
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     all_dataset, [n_train, n_val], 
    #     generator=torch.Generator().manual_seed(args.seed)
    # )
    
    # # Update val_dataset mode
    # val_dataset.dataset.mode = "val"
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                          num_workers=args.num_workers, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
    #                        num_workers=args.num_workers, pin_memory=True)
    
    # print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise ValueError(f"data_root does not exist: {data_root}")
    
    # Check if separate train/val directories exist (ImageNet style)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    
    if train_dir.exists() and val_dir.exists():
        # ImageNet style: separate train/val directories
        print(f"Detected ImageNet-style directory structure")
        print(f"Train directory: {train_dir}")
        print(f"Val directory: {val_dir}")
        train_image_dir = train_dir
        val_image_dir = val_dir
        use_split = False
    else:
        # COCO style: single directory with split
        print(f"Detected COCO-style directory structure")
        if (data_root / "images").exists():
            train_image_dir = data_root / "images"
        else:
            train_image_dir = data_root
        val_image_dir = train_image_dir
        use_split = True
    
    # Load or compute ab bin centers
    if args.loss_type == "classification":
        if args.ab_centers_path and Path(args.ab_centers_path).exists():
            print(f"Loading precomputed ab centers from {args.ab_centers_path}")
            centers = np.load(args.ab_centers_path)
        else:
            print("Computing ab bin centers from training data...")
            print("This may take a few minutes...")
            temp_dataset = ColorizationDataset(train_image_dir, args.img_size, mode="train", 
                                              loss_type="regression")
            ab_samples = temp_dataset.get_all_ab_values(max_samples=10000)
            centers_path = out_dir / "ab_centers.npy"
            centers = compute_ab_bins(ab_samples, k=args.num_bins, cache_path=centers_path)
            print(f"Saved ab centers to {centers_path}")
    else:
        centers = None
    
    # Create datasets
    if use_split:
        # COCO style: create one dataset and split
        print("Creating single dataset with train/val split...")
        all_dataset = ColorizationDataset(train_image_dir, args.img_size, mode="train", 
                                         centers=centers, loss_type=args.loss_type)
        
        n_val = int(len(all_dataset) * args.val_split)
        n_train = len(all_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            all_dataset, [n_train, n_val], 
            generator=torch.Generator().manual_seed(args.seed)
        )
        val_dataset.dataset.mode = "val"
    else:
        # ImageNet style: separate train/val datasets
        print("Creating separate train and val datasets...")
        train_dataset = ColorizationDataset(train_image_dir, args.img_size, mode="train",
                                           centers=centers, loss_type=args.loss_type)
        val_dataset = ColorizationDataset(val_image_dir, args.img_size, mode="val",
                                         centers=centers, loss_type=args.loss_type)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Compute class weights for classification
    class_weights = None
    if args.loss_type == "classification":
        print("Computing class rebalancing weights...")
        bin_indices_list = []
        for i in tqdm(range(min(1000, len(train_dataset))), desc="Sampling bins"):
            _, bins, _, _ = train_dataset[i]
            bin_indices_list.append(bins.numpy())
        
        weights = compute_class_rebalancing_weights(bin_indices_list, args.num_bins)
        class_weights = torch.from_numpy(weights).float().to(device)
    
    # Build model
    model = build_model(backbone=args.backbone, pretrained=True, 
                       loss_type=args.loss_type, num_bins=args.num_bins)
    model = model.to(device)
    
    # Loss
    if args.loss_type == "classification":
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        criterion = nn.L1Loss()
    
    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume checkpoint
    start_epoch = 0
    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training log
    log_path = out_dir / "training_log.csv"
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'spcr_full', 'spcr_light', 'lr'])
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                class_weights, args.loss_type)
        
        val_loss, colorized_dir, original_dir = validate_epoch(
            model, val_loader, criterion, device, epoch, out_dir, centers,
            args.loss_type, save_images=args.eval_after_epoch
        )
        
        scheduler.step()
        
        # SPCR evaluation
        spcr_full, spcr_light = 0.0, 0.0
        if args.eval_after_epoch and colorized_dir:
            print("Running SPCR evaluation...")
            spcr_full, spcr_light = run_spcr_evaluation(original_dir, colorized_dir, 
                                                        epoch, out_dir)
            print(f"SPCR Full: {spcr_full:.4f}, SPCR Light: {spcr_light:.4f}")
        
        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, spcr_full, spcr_light, 
                           optimizer.param_groups[0]['lr']])
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        
        # Save best model
        if epoch == start_epoch or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
    
    print(f"\nTraining complete. Outputs in {out_dir}")


if __name__ == "__main__":
    # Usage: python train_colorization.py --data_root data/coco --loss_type classification --use_mps --eval_after_epoch
    main()
