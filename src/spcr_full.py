#!/usr/bin/env python3
"""
Semantic-Perceptual Colour Realism (SPCR) Metric - Full Version
================================================================

This script evaluates image colorization quality using three components:
1. Semantic Plausibility: Measures if colors match semantic categories (uses DeepLabV3)
2. Perceptual Realism: Compares deep features between original and colorized images (uses VGG16)
3. Colour Diversity: Measures color variance in Lab space

Final SPCR Score = 0.4 * Semantic + 0.4 * Perceptual + 0.2 * Diversity

Usage:
    python spcr_full.py --original original/ --colorized colorized/ --output results_full.csv

Author: Computer Vision Final Project
Date: November 2025
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage import color
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Semantic class hue ranges (in HSV, H in degrees [0-360])
# DeepLabV3 uses PASCAL VOC/COCO classes
SEMANTIC_HUE_RANGES = {
    'sky': (140, 220),           # Blue sky
    'vegetation': (60, 140),      # Green vegetation
    'grass': (60, 140),           # Green grass
    'tree': (60, 140),            # Green trees
    'person': [(0, 40), (300, 360)],  # Skin tones (red-yellow, magenta)
    'water': (160, 220),          # Blue water
    'sea': (160, 220),            # Blue sea
}

# DeepLabV3 class names (PASCAL VOC 2012 + COCO)
DEEPLABV3_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


# ============================================================================
# MODEL LOADING AND INITIALIZATION
# ============================================================================

class SPCRMetric:
    """
    Semantic-Perceptual Colour Realism (SPCR) Metric Calculator.
    
    This class handles model initialization and metric computation for
    evaluating image colorization quality.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize SPCR metric with required models.
        
        Args:
            device: Device to run models on ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        # Detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load segmentation model (DeepLabV3)
        logger.info("Loading DeepLabV3 segmentation model...")
        self.segmentation_model = models.segmentation.deeplabv3_resnet101(
            pretrained=True
        ).to(self.device)
        self.segmentation_model.eval()
        
        # Load perceptual model (VGG16)
        logger.info("Loading VGG16 perceptual model...")
        vgg16 = models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        
        # Extract features from relu4_2 layer (after conv4_2)
        # VGG16 features: conv layers are in features module
        self.perceptual_model = nn.Sequential(*list(vgg16.features[:23])).to(self.device)
        self.perceptual_model.eval()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("Models loaded successfully!")
    
    def compute_semantic_score(self, colorized_img: np.ndarray) -> float:
        """
        Compute semantic plausibility score.
        
        Measures if colors in the colorized image are plausible given the
        semantic segmentation of the scene.
        
        Args:
            colorized_img: Colorized image in BGR format (H, W, 3)
        
        Returns:
            Semantic plausibility score in [0, 1]
        """
        try:
            # Prepare image for segmentation
            img_rgb = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            
            # Get segmentation mask
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)['out'][0]
                segmentation_mask = output.argmax(0).cpu().numpy()
            
            # Convert colorized image to HSV
            img_hsv = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2HSV)
            hue = img_hsv[:, :, 0].astype(np.float32) * 2  # OpenCV H is [0-180], convert to [0-360]
            
            # Compute plausibility for each semantic class
            total_pixels = 0
            plausible_pixels = 0
            
            for class_idx, class_name in enumerate(DEEPLABV3_CLASSES):
                if class_name not in SEMANTIC_HUE_RANGES:
                    continue
                
                # Get pixels belonging to this class
                class_mask = (segmentation_mask == class_idx)
                class_pixel_count = np.sum(class_mask)
                
                if class_pixel_count == 0:
                    continue
                
                total_pixels += class_pixel_count
                
                # Get hue range for this class
                hue_range = SEMANTIC_HUE_RANGES[class_name]
                
                # Handle multiple ranges (e.g., for person/skin tones)
                if isinstance(hue_range[0], tuple):
                    plausible_mask = np.zeros_like(class_mask, dtype=bool)
                    for h_min, h_max in hue_range:
                        plausible_mask |= (hue >= h_min) & (hue <= h_max)
                else:
                    h_min, h_max = hue_range
                    plausible_mask = (hue >= h_min) & (hue <= h_max)
                
                # Count plausible pixels for this class
                plausible_pixels += np.sum(class_mask & plausible_mask)
            
            # Compute semantic score
            if total_pixels > 0:
                semantic_score = plausible_pixels / total_pixels
            else:
                # If no relevant semantic classes found, return neutral score
                semantic_score = 0.5
            
            return float(semantic_score)
        
        except Exception as e:
            logger.warning(f"Error computing semantic score: {e}")
            return 0.5  # Return neutral score on error
    
    def compute_perceptual_score(
        self, 
        original_img: np.ndarray, 
        colorized_img: np.ndarray
    ) -> float:
        """
        Compute perceptual realism score using deep features.
        
        Extracts VGG16 features and computes cosine similarity between
        original and colorized images.
        
        Args:
            original_img: Original image in BGR format (H, W, 3)
            colorized_img: Colorized image in BGR format (H, W, 3)
        
        Returns:
            Perceptual realism score in [0, 1]
        """
        try:
            # Prepare images
            orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            col_rgb = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)
            
            # Resize to same size if needed
            if orig_rgb.shape != col_rgb.shape:
                col_rgb = cv2.resize(col_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
            
            # Transform and batch
            orig_tensor = self.transform(orig_rgb).unsqueeze(0).to(self.device)
            col_tensor = self.transform(col_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                orig_features = self.perceptual_model(orig_tensor)
                col_features = self.perceptual_model(col_tensor)
            
            # Flatten features
            orig_flat = orig_features.view(orig_features.size(0), -1)
            col_flat = col_features.view(col_features.size(0), -1)
            
            # Compute cosine similarity
            cos_sim = nn.functional.cosine_similarity(orig_flat, col_flat)
            
            # Normalize to [0, 1] (cosine similarity is in [-1, 1])
            perceptual_score = (cos_sim.item() + 1) / 2
            
            return float(perceptual_score)
        
        except Exception as e:
            logger.warning(f"Error computing perceptual score: {e}")
            return 0.0
    
    def compute_diversity_score(self, colorized_img: np.ndarray) -> float:
        """
        Compute colour diversity score.
        
        Measures the variance of chroma values in Lab color space.
        Higher variance indicates more diverse colors.
        
        Args:
            colorized_img: Colorized image in BGR format (H, W, 3)
        
        Returns:
            Colour diversity score in [0, 1]
        """
        try:
            # Convert to Lab color space
            img_rgb = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)
            img_lab = color.rgb2lab(img_rgb)
            
            # Extract a and b channels (chroma)
            a_channel = img_lab[:, :, 1]
            b_channel = img_lab[:, :, 2]
            
            # Compute chroma magnitude: C = sqrt(a^2 + b^2)
            chroma = np.sqrt(a_channel**2 + b_channel**2)
            
            # Compute variance of chroma
            chroma_var = np.var(chroma)
            
            # Normalize variance to [0, 1]
            # Typical chroma variance ranges from 0 to ~2000
            # Using a sigmoid-like normalization
            max_expected_var = 2000.0
            diversity_score = min(chroma_var / max_expected_var, 1.0)
            
            return float(diversity_score)
        
        except Exception as e:
            logger.warning(f"Error computing diversity score: {e}")
            return 0.0
    
    def compute_spcr(
        self,
        original_img: np.ndarray,
        colorized_img: np.ndarray,
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ) -> Dict[str, float]:
        """
        Compute full SPCR score with all components.
        
        Args:
            original_img: Original image in BGR format (H, W, 3)
            colorized_img: Colorized image in BGR format (H, W, 3)
            weights: Tuple of (semantic_weight, perceptual_weight, diversity_weight)
        
        Returns:
            Dictionary with 'semantic', 'perceptual', 'diversity', and 'spcr' scores
        """
        semantic_score = self.compute_semantic_score(colorized_img)
        perceptual_score = self.compute_perceptual_score(original_img, colorized_img)
        diversity_score = self.compute_diversity_score(colorized_img)
        
        # Compute weighted SPCR score
        w_sem, w_per, w_div = weights
        spcr_score = (
            w_sem * semantic_score + 
            w_per * perceptual_score + 
            w_div * diversity_score
        )
        
        return {
            'semantic': semantic_score,
            'perceptual': perceptual_score,
            'diversity': diversity_score,
            'spcr': spcr_score
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array in BGR format, or None if loading fails
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        
        # Check if image is grayscale (convert to BGR if needed)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    
    except Exception as e:
        logger.warning(f"Error loading image {image_path}: {e}")
        return None


def get_image_pairs(
    original_folder: str,
    colorized_folder: str
) -> List[Tuple[str, str, str]]:
    """
    Get matching pairs of original and colorized images.
    
    Args:
        original_folder: Path to folder with original images
        colorized_folder: Path to folder with colorized images
    
    Returns:
        List of tuples (filename, original_path, colorized_path)
    """
    original_path = Path(original_folder)
    colorized_path = Path(colorized_folder)
    
    if not original_path.exists():
        raise ValueError(f"Original folder does not exist: {original_folder}")
    if not colorized_path.exists():
        raise ValueError(f"Colorized folder does not exist: {colorized_folder}")
    
    # Get all image files from original folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    original_files = [
        f for f in original_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    # Match with colorized files
    pairs = []
    for orig_file in original_files:
        col_file = colorized_path / orig_file.name
        if col_file.exists():
            pairs.append((orig_file.name, str(orig_file), str(col_file)))
        else:
            logger.warning(f"No matching colorized image for: {orig_file.name}")
    
    logger.info(f"Found {len(pairs)} matching image pairs")
    return pairs


def save_results_to_csv(
    results: List[Dict[str, any]],
    output_path: str
) -> None:
    """
    Save evaluation results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        logger.warning("No results to save")
        return
    
    fieldnames = ['filename', 'semantic', 'perceptual', 'diversity', 'spcr']
    
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Results saved to: {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def main():
    """
    Main function to evaluate colorization quality using SPCR metric.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate image colorization quality using SPCR metric (Full Version)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Path to folder containing original images'
    )
    parser.add_argument(
        '--colorized',
        type=str,
        required=True,
        help='Path to folder containing colorized images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results_full.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to run models on (auto-detect if not specified)'
    )
    parser.add_argument(
        '--weights',
        type=float,
        nargs=3,
        default=[0.4, 0.4, 0.2],
        metavar=('SEMANTIC', 'PERCEPTUAL', 'DIVERSITY'),
        help='Weights for semantic, perceptual, and diversity components'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Validate weights
    weights = tuple(args.weights)
    if not np.isclose(sum(weights), 1.0):
        logger.warning(f"Weights sum to {sum(weights)}, not 1.0. Results may be unexpected.")
    
    # Initialize SPCR metric
    logger.info("Initializing SPCR metric...")
    spcr_metric = SPCRMetric(device=args.device)
    
    # Get image pairs
    logger.info("Finding image pairs...")
    image_pairs = get_image_pairs(args.original, args.colorized)
    
    if not image_pairs:
        logger.error("No matching image pairs found!")
        sys.exit(1)
    
    # Evaluate each pair
    results = []
    scores = {
        'semantic': [],
        'perceptual': [],
        'diversity': [],
        'spcr': []
    }
    
    logger.info(f"Evaluating {len(image_pairs)} image pairs...")
    
    for filename, orig_path, col_path in tqdm(image_pairs, desc="Processing images"):
        # Load images
        original_img = load_image(orig_path)
        colorized_img = load_image(col_path)
        
        if original_img is None or colorized_img is None:
            logger.warning(f"Skipping {filename} due to loading error")
            continue
        
        # Compute SPCR scores
        try:
            result = spcr_metric.compute_spcr(
                original_img,
                colorized_img,
                weights=weights
            )
            
            # Store result
            result['filename'] = filename
            results.append(result)
            
            # Accumulate scores for statistics
            for key in scores:
                scores[key].append(result[key])
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue
    
    # Save results to CSV
    if results:
        save_results_to_csv(results, args.output)
        
        # Compute and print statistics
        print("\n" + "=" * 60)
        print("SPCR EVALUATION RESULTS (Full Version)")
        print("=" * 60)
        print(f"Total images evaluated: {len(results)}")
        print(f"Weights: Semantic={weights[0]:.2f}, Perceptual={weights[1]:.2f}, Diversity={weights[2]:.2f}")
        print("-" * 60)
        
        for metric_name, metric_scores in scores.items():
            if metric_scores:
                mean_score = np.mean(metric_scores)
                std_score = np.std(metric_scores)
                print(f"{metric_name.capitalize():12s}: {mean_score:.4f} ± {std_score:.4f}")
        
        print("=" * 60)
        print(f"\nResults saved to: {args.output}")
    
    else:
        logger.error("No images were successfully processed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
