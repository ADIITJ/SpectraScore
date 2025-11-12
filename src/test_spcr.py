#!/usr/bin/env python3
"""
Example script to test SPCR metric implementations.

This script creates sample test images and demonstrates how to use both
spcr_full.py and spcr_light.py.

Usage:
    python test_spcr.py
"""

import os
import numpy as np
import cv2
from pathlib import Path

def create_test_images():
    """
    Create sample test images for demonstration.
    
    Creates:
    - assets/original/ folder with sample "ground truth" images
    - assets/colorized/ folder with simulated colorized versions
    """
    print("Creating test image directories...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create directories
    original_dir = project_root / "assets" / "original"
    colorized_dir = project_root / "assets" / "colorized"
    original_dir.mkdir(parents=True, exist_ok=True)
    colorized_dir.mkdir(parents=True, exist_ok=True)
    
    # Image size
    height, width = 256, 256
    
    # Create 5 test image pairs
    num_images = 5
    
    for i in range(num_images):
        print(f"Generating test image pair {i+1}/{num_images}...")
        
        # Create original image with varied content
        original = np.zeros((height, width, 3), dtype=np.uint8)
        
        if i == 0:
            # Blue sky gradient
            for y in range(height):
                original[y, :] = [200 - y//3, 150 - y//4, 50 + y//5]  # BGR
        elif i == 1:
            # Green landscape
            original[:height//2, :] = [100, 180, 80]  # Upper half - green
            original[height//2:, :] = [80, 120, 60]   # Lower half - darker green
        elif i == 2:
            # Colorful scene
            original[:, :width//3] = [50, 50, 200]    # Red region
            original[:, width//3:2*width//3] = [200, 150, 50]  # Blue region
            original[:, 2*width//3:] = [50, 200, 50]  # Green region
        elif i == 3:
            # Natural scene simulation
            original[:height//3, :] = [200, 180, 120]  # Sky
            original[height//3:2*height//3, :] = [60, 150, 80]  # Trees
            original[2*height//3:, :] = [80, 140, 100]  # Grass
        else:
            # Sunset gradient
            for y in range(height):
                original[y, :] = [
                    100 + int(100 * y/height),    # B
                    100 + int(50 * y/height),     # G
                    150 + int(100 * (1 - y/height))  # R
                ]
        
        # Add some noise for realism
        noise = np.random.randint(-10, 10, original.shape, dtype=np.int16)
        original = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Create colorized version (simulate good colorization with slight variations)
        colorized = original.copy()
        
        # Add slight color shift to simulate colorization artifacts
        shift = np.random.randint(-15, 15, (3,), dtype=np.int16)
        colorized = np.clip(colorized.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        
        # Add slight blur to simulate model output
        colorized = cv2.GaussianBlur(colorized, (3, 3), 0.5)
        
        # Save images
        filename = f"test_image_{i+1:02d}.png"
        cv2.imwrite(str(original_dir / filename), original)
        cv2.imwrite(str(colorized_dir / filename), colorized)
    
    print(f"\n✓ Created {num_images} test image pairs")
    print(f"  Original images: {original_dir.relative_to(project_root)}/")
    print(f"  Colorized images: {colorized_dir.relative_to(project_root)}/")
    print("\nYou can now run from project root:")
    print(f"  python src/spcr_full.py --original assets/original/ --colorized assets/colorized/ --output results/results_full.csv")
    print(f"  python src/spcr_light.py --original assets/original/ --colorized assets/colorized/ --output results/results_light.csv")


def cleanup_test_images():
    """Remove test image directories."""
    import shutil
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    original_dir = project_root / "assets" / "original"
    colorized_dir = project_root / "assets" / "colorized"
    
    if original_dir.exists() and any(original_dir.iterdir()):
        for f in original_dir.iterdir():
            f.unlink()
        print(f"Cleaned {original_dir.relative_to(project_root)}/")
    
    if colorized_dir.exists() and any(colorized_dir.iterdir()):
        for f in colorized_dir.iterdir():
            f.unlink()
        print(f"Cleaned {colorized_dir.relative_to(project_root)}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SPCR metric implementations")
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove test images instead of creating them'
    )
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_test_images()
    else:
        create_test_images()
