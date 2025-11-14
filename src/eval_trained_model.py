#!/usr/bin/env python3
"""Evaluate trained model on ImageNet val set."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import cv2
from tqdm import tqdm
from model_colorization import PaperNet
from utils_lab import rgb_to_lab_tensor, lab_to_rgb_tensor, annealed_mean_ab

def rgb_to_lab_single(rgb_np):
    """Convert RGB numpy to Lab tensor."""
    rgb_np = rgb_np.astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)
    lab = rgb_to_lab_tensor(rgb_tensor).squeeze(0)
    return lab[0:1], lab[1:3]

def main():
    # Paths
    model_path = "outputs/papernet_200k_7epochs/PaperNet/PaperNet_epoch_007.pt"
    ab_centers_path = "outputs/papernet_200k_7epochs/ab_centers.npy"
    val_dir = Path("/home/ayaan/Image-Hue/data/val")
    output_dir = Path("/home/ayaan/Image-Hue/data/val_colorized_papernet")
    colorized_dir = output_dir / "colorized"
    colorized_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device('cuda')
    print(f"Loading model from: {model_path}")
    model = PaperNet(num_classes=313).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    # Load ab centers
    centers = np.load(ab_centers_path)
    print(f"✓ Loaded {len(centers)} color bins")
    
    # Find all images (recursively, include .JPEG)
    print("\nFinding images...")
    test_images = list(val_dir.rglob('*.JPEG')) + list(val_dir.rglob('*.jpg')) + list(val_dir.rglob('*.png'))
    print(f"✓ Found {len(test_images)} images")
    
    if len(test_images) == 0:
        print("ERROR: No images found!")
        return
    
    # Colorize
    print(f"\n{'='*60}")
    print("COLORIZING IMAGENET VALIDATION SET")
    print(f"{'='*60}\n")
    
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(test_images, desc="Colorizing"):
            # Get class and filename
            class_id = img_path.parent.name
            base_name = img_path.stem
            
            # Output filename: flat structure with class prefix
            output_name = f"{class_id}_{base_name}.png"
            output_path = colorized_dir / output_name
            
            # Skip if exists
            if output_path.exists():
                continue
            
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H_orig, W_orig = img_rgb.shape[:2]
                
                # Resize to model input
                img_resized = cv2.resize(img_rgb, (224, 224))
                
                # Convert to Lab
                L, _ = rgb_to_lab_single(img_resized)
                L_norm = (L / 50.0) - 1.0
                L_norm = L_norm.unsqueeze(0).to(device)
                
                # Predict colors
                output = model(L_norm)
                ab_pred = annealed_mean_ab(output, centers, T=0.38)[0]
                
                # Combine L + ab
                lab_full = torch.cat([L, ab_pred.cpu()], dim=0)
                rgb_pred = lab_to_rgb_tensor(lab_full.unsqueeze(0)).squeeze(0)
                rgb_np = (rgb_pred.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                # Resize back to original size
                rgb_np = cv2.resize(rgb_np, (W_orig, H_orig))
                
                # Save
                cv2.imwrite(str(output_path), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
    
    print(f"\n✓ Colorization complete!")
    print(f"✓ Colorized images saved to: {colorized_dir}")
    print(f"✓ Total colorized: {len(list(colorized_dir.glob('*.png')))}")
    
    # Now run SPCR evaluation
    print(f"\n{'='*60}")
    print("RUNNING SPCR EVALUATION")
    print(f"{'='*60}\n")
    
    import subprocess
    spcr_script = Path("src/spcr_full_imagenet.py")
    results_csv = output_dir / "spcr_results.csv"
    
    cmd = [
        sys.executable,
        str(spcr_script),
        "--original", str(val_dir),
        "--colorized", str(colorized_dir),
        "--output", str(results_csv),
        "--device", "cuda"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ SPCR results saved to: {results_csv}")
    except subprocess.CalledProcessError as e:
        print(f"SPCR evaluation failed: {e}")

if __name__ == "__main__":
    main()