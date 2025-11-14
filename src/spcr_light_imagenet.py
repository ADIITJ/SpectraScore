#!/usr/bin/env python3
"""
Semantic-Perceptual Colour Realism (SPCR) Metric - Lightweight Version

This script evaluates image colorization quality using two components:
1. Perceptual Realism: Compares deep features (VGG16) or histogram similarity (fallback)
2. Colour Diversity: Measures color variance in Lab space

Final SPCR-Light Score = 0.7 * Perceptual + 0.3 * Diversity

This version does NOT use semantic segmentation, making it faster and more
lightweight than the full version.

Usage:
    python spcr_light.py --original original/ --colorized colorized/ --output results_light.csv
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
from skimage import color
from tqdm import tqdm

# Try to import PyTorch for deep features (
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Will use histogram-based fallback for perceptual score.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class SPCRLightMetric:
    """
    Lightweight Semantic-Perceptual Colour Realism (SPCR) Metric Calculator.
    
    This class computes colorization quality without semantic segmentation,
    focusing on perceptual realism and color diversity.
    """
    
    def __init__(self, device: Optional[str] = None, use_deep_features: bool = True):
        """
        Initialize SPCR-Light metric.
        
        Args:
            device: Device to run models on ('cuda', 'mps', 'cpu', or None for auto-detect)
            use_deep_features: Whether to use deep features (requires PyTorch)
        """
        self.use_deep_features = use_deep_features and TORCH_AVAILABLE
        
        if self.use_deep_features:
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
            
            # Load perceptual model (VGG16)
            logger.info("Loading VGG16 perceptual model...")
            try:
                vgg16 = models.vgg16(pretrained=True).to(self.device)
                vgg16.eval()
                
                # Extract features from relu4_2 layer (after conv4_2)
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
                
                logger.info("VGG16 model loaded successfully!")
            except Exception as e:
                logger.warning(f"Failed to load VGG16: {e}. Falling back to histogram method.")
                self.use_deep_features = False
        else:
            logger.info("Using histogram-based perceptual score (no deep features)")
    
    def compute_perceptual_score_deep(
        self,
        original_img: np.ndarray,
        colorized_img: np.ndarray
    ) -> float:
        """
        Compute perceptual realism score using deep features (VGG16).
        
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
            logger.warning(f"Error computing deep perceptual score: {e}")
            return 0.0
    
    def compute_perceptual_score_histogram(
        self,
        original_img: np.ndarray,
        colorized_img: np.ndarray
    ) -> float:
        """
        Compute perceptual realism score using histogram comparison (fallback method).
        
        Compares 2D histograms of (a,b) chroma channels in Lab space using
        Bhattacharyya distance, then converts to similarity score.
        
        Args:
            original_img: Original image in BGR format (H, W, 3)
            colorized_img: Colorized image in BGR format (H, W, 3)
        
        Returns:
            Perceptual realism score in [0, 1]
        """
        try:
            # Convert to Lab color space
            orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            col_rgb = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)
            
            # Resize to same size if needed
            if orig_rgb.shape != col_rgb.shape:
                col_rgb = cv2.resize(col_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
            
            orig_lab = color.rgb2lab(orig_rgb)
            col_lab = color.rgb2lab(col_rgb)
            
            # Extract a and b channels
            orig_a = orig_lab[:, :, 1]
            orig_b = orig_lab[:, :, 2]
            col_a = col_lab[:, :, 1]
            col_b = col_lab[:, :, 2]
            
            # Create 2D histograms for (a,b) chroma
            # Lab a,b typically range from -128 to 127
            bins = 64  # Reduced bins for efficiency
            
            hist_orig, _, _ = np.histogram2d(
                orig_a.ravel(),
                orig_b.ravel(),
                bins=bins,
                range=[[-128, 127], [-128, 127]]
            )
            
            hist_col, _, _ = np.histogram2d(
                col_a.ravel(),
                col_b.ravel(),
                bins=bins,
                range=[[-128, 127], [-128, 127]]
            )
            
            # Normalize histograms
            hist_orig = hist_orig / (hist_orig.sum() + 1e-10)
            hist_col = hist_col / (hist_col.sum() + 1e-10)
            
            # Compute Bhattacharyya coefficient
            bc = np.sum(np.sqrt(hist_orig * hist_col))
            
            # Bhattacharyya coefficient is already in [0, 1]
            # 1 means identical distributions, 0 means completely different
            perceptual_score = bc
            
            return float(perceptual_score)
        
        except Exception as e:
            logger.warning(f"Error computing histogram perceptual score: {e}")
            return 0.0
    
    def compute_perceptual_score(
        self,
        original_img: np.ndarray,
        colorized_img: np.ndarray
    ) -> float:
        """
        Compute perceptual realism score.
        
        Uses deep features if available, otherwise falls back to histogram method.
        
        Args:
            original_img: Original image in BGR format (H, W, 3)
            colorized_img: Colorized image in BGR format (H, W, 3)
        
        Returns:
            Perceptual realism score in [0, 1]
        """
        if self.use_deep_features:
            return self.compute_perceptual_score_deep(original_img, colorized_img)
        else:
            return self.compute_perceptual_score_histogram(original_img, colorized_img)
    
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
    
    def compute_spcr_light(
        self,
        original_img: np.ndarray,
        colorized_img: np.ndarray,
        weights: Tuple[float, float] = (0.7, 0.3)
    ) -> Dict[str, float]:
        """
        Compute SPCR-Light score.
        
        Args:
            original_img: Original image in BGR format (H, W, 3)
            colorized_img: Colorized image in BGR format (H, W, 3)
            weights: Tuple of (perceptual_weight, diversity_weight)
        
        Returns:
            Dictionary with 'perceptual', 'diversity', and 'spcr_light' scores
        """
        perceptual_score = self.compute_perceptual_score(original_img, colorized_img)
        diversity_score = self.compute_diversity_score(colorized_img)
        
        # Compute weighted SPCR-Light score
        w_per, w_div = weights
        spcr_light_score = w_per * perceptual_score + w_div * diversity_score
        
        return {
            'perceptual': perceptual_score,
            'diversity': diversity_score,
            'spcr_light': spcr_light_score
        }


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
    Get matching pairs of original and colorized images for ImageNet structure.
    
    Original folder has nested structure: val/{class_id}/{filename}.JPEG
    Colorized folder is flat: val_colorized/{class_id}_{filename}_siggraph17.png
    
    Args:
        original_folder: Path to folder with original images (nested by class)
        colorized_folder: Path to folder with colorized images (flat)
    
    Returns:
        List of tuples (display_name, original_path, colorized_path)
    """
    original_path = Path(original_folder)
    colorized_path = Path(colorized_folder)
    
    if not original_path.exists():
        raise ValueError(f"Original folder does not exist: {original_folder}")
    if not colorized_path.exists():
        raise ValueError(f"Colorized folder does not exist: {colorized_folder}")
    
    # Get all class directories
    class_dirs = [d for d in original_path.iterdir() if d.is_dir()]
    
    pairs = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPEG', '.JPG'}
    
    for class_dir in class_dirs:
        class_id = class_dir.name  # e.g., n01440764
        
        # Get all images in this class directory
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix in image_extensions]
        
        for orig_file in image_files:
            # Original: val/n01440764/ILSVRC2012_val_00000293.JPEG
            # Colorized: val_colorized/n01440764_ILSVRC2012_val_00000293_siggraph17.png
            
            # Extract base filename without extension
            base_name = orig_file.stem  # e.g., ILSVRC2012_val_00000293
            
            # Construct colorized filename
            colorized_filename = f"{class_id}_{base_name}_siggraph17.png"
            col_file = colorized_path / colorized_filename
            
            if col_file.exists():
                # Use a unique display name that includes class
                display_name = f"{class_id}_{base_name}"
                pairs.append((display_name, str(orig_file), str(col_file)))
            else:
                logger.warning(f"No matching colorized image for: {class_id}/{orig_file.name}")
    
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
    
    fieldnames = ['filename', 'perceptual', 'diversity', 'spcr_light']
    
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Results saved to: {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")


def main():
    """
    Main function to evaluate colorization quality using SPCR-Light metric.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate image colorization quality using SPCR-Light metric',
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
        default='results_light.csv',
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
        nargs=2,
        default=[0.7, 0.3],
        metavar=('PERCEPTUAL', 'DIVERSITY'),
        help='Weights for perceptual and diversity components'
    )
    parser.add_argument(
        '--no-deep-features',
        action='store_true',
        help='Disable deep features and use histogram method only'
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
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Validate weights
    weights = tuple(args.weights)
    if not np.isclose(sum(weights), 1.0):
        logger.warning(f"Weights sum to {sum(weights)}, not 1.0. Results may be unexpected.")
    
    # Initialize SPCR-Light metric
    logger.info("Initializing SPCR-Light metric...")
    use_deep = not args.no_deep_features
    spcr_metric = SPCRLightMetric(device=args.device, use_deep_features=use_deep)
    
    # Get image pairs
    logger.info("Finding image pairs...")
    image_pairs = get_image_pairs(args.original, args.colorized)
    
    if not image_pairs:
        logger.error("No matching image pairs found!")
        sys.exit(1)
    
    # Evaluate each pair
    results = []
    scores = {
        'perceptual': [],
        'diversity': [],
        'spcr_light': []
    }
    
    logger.info(f"Evaluating {len(image_pairs)} image pairs...")
    
    for filename, orig_path, col_path in tqdm(image_pairs, desc="Processing images"):
        # Load images
        original_img = load_image(orig_path)
        colorized_img = load_image(col_path)
        
        if original_img is None or colorized_img is None:
            logger.warning(f"Skipping {filename} due to loading error")
            continue
        
        # Compute SPCR-Light scores
        try:
            result = spcr_metric.compute_spcr_light(
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
        print("SPCR EVALUATION RESULTS (Lightweight Version)")
        print("=" * 60)
        print(f"Total images evaluated: {len(results)}")
        print(f"Deep features: {'Enabled (VGG16)' if spcr_metric.use_deep_features else 'Disabled (Histogram)'}")
        print(f"Weights: Perceptual={weights[0]:.2f}, Diversity={weights[1]:.2f}")
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
