# Image Colorization with SPCR Evaluation

Modern implementation of Zhang et al. (2016) "Colorful Image Colorization" with a novel **Semantic-Perceptual Colour Realism (SPCR)** metric for evaluation.

## Features

### Colorization Training
- **Two loss modes**: Classification (313 bins, Zhang et al.) and Regression
- **Two backbones**: Pretrained ResNet-18 or Custom lightweight encoder
- **Class rebalancing**: Inverse-frequency weights for rare colors
- **Annealed softmax**: Temperature-based inference (T=0.38)
- **MPS/CUDA/CPU**: Auto device detection with Apple Silicon support

### SPCR Evaluation Metric (Novel Contribution)
Comprehensive quality assessment combining three components:
- **Semantic Plausibility** (40%): DeepLabV3 segmentation validates color-object consistency
- **Perceptual Realism** (40%): VGG16 deep features measure perceptual similarity
- **Colour Diversity** (20%): Lab chroma variance penalizes desaturation

Two implementations provided:
- `spcr_full.py`: Full metric with semantic segmentation
- `spcr_light.py`: Lightweight version (histogram fallback, no segmentation)

## Quick Start

### Installation
```bash
# Clone repository
git clone <repo_url>
cd CV_final_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Test SPCR Metrics
```bash
# Generate test images
python src/test_spcr.py

# Evaluate with SPCR Full
python src/spcr_full.py \
    --original assets/original/ \
    --colorized assets/colorized/ \
    --output results/results_full.csv \
    --device mps  # or cuda/cpu

# Evaluate with SPCR Light
python src/spcr_light.py \
    --original assets/original/ \
    --colorized assets/colorized/ \
    --output results/results_light.csv \
    --device mps

# Cleanup
python src/test_spcr.py --cleanup
```

### Train Colorization Model
```bash
# Download COCO dataset (100 images for quick test)
python src/data_download.py \
    --out_dir data/coco \
    --num_images 100 \
    --workers 4

# Train with classification loss (5 epochs, quick)
python src/train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/quick \
    --loss_type classification \
    --epochs 5 \
    --batch_size 16 \
    --use_mps

# Full training with SPCR evaluation (30 epochs)
python src/train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/full \
    --loss_type classification \
    --epochs 30 \
    --batch_size 16 \
    --use_mps \
    --eval_after_epoch

# Train with regression loss (faster alternative)
python src/train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/regression \
    --loss_type regression \
    --epochs 20 \
    --use_mps
```

### Interactive Demo
```bash
python src/demo.py
```

## Project Structure

```
CV_final_project/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── src/                        # Source code
│   ├── train_colorization.py  # Training pipeline
│   ├── data_download.py       # COCO dataset downloader
│   ├── model_colorization.py  # Network architecture
│   ├── utils_lab.py           # Lab color utilities
│   ├── spcr_full.py           # Full SPCR metric
│   ├── spcr_light.py          # Lightweight SPCR metric
│   ├── test_spcr.py           # Test image generator
│   └── demo.py                # Interactive demo
│
├── data/                       # Datasets
│   └── coco/                  # COCO images (downloaded)
│
├── assets/                     # Test images
│   ├── original/              # Ground truth (generated)
│   └── colorized/             # Colorized (generated)
│
├── results/                    # Evaluation outputs
│   ├── results_full.csv       # SPCR full results
│   └── results_light.csv      # SPCR light results
│
├── outputs/                    # Training outputs
│   └── <experiment>/
│       ├── training_log.csv   # Training metrics
│       ├── best_model.pt      # Best checkpoint
│       └── eval/              # Per-epoch evaluations
│
└── docs/                       # Additional documentation
```

## SPCR Metric Details

### Full SPCR (with Segmentation)
```
SPCR = 0.4 × Semantic + 0.4 × Perceptual + 0.2 × Diversity
```

**Semantic Plausibility**: Uses DeepLabV3-ResNet101 to segment the image and validates that colors match expected hue ranges for each class (e.g., sky→blue, grass→green).

**Perceptual Realism**: Extracts VGG16 conv4_2 features from both original and colorized images, computes cosine similarity.

**Colour Diversity**: Measures variance of chroma magnitude in Lab space, normalized to [0,1].

### Light SPCR (without Segmentation)
```
SPCR-Light = 0.7 × Perceptual + 0.3 × Diversity
```

Faster alternative using 2D histogram comparison (Bhattacharyya coefficient) instead of deep features when PyTorch unavailable.

## Training Details

### Classification Mode (Zhang et al. 2016)
- Quantize ab space into 313 bins via KMeans clustering
- Predict distribution over bins per pixel
- Apply class rebalancing weights (inverse frequency)
- Use annealed softmax for inference

### Regression Mode
- Direct ab prediction (2 channels)
- L1 or MSE loss
- Faster training, simpler pipeline
- May produce less saturated colors

### Model Architecture
```
Input: L channel [B,1,H,W]
↓
Encoder: ResNet-18 (pretrained) or Custom Conv
↓
Features: [B,512,H/16,W/16]
↓
Decoder: 4× upsampling blocks
↓
Output: [B,313,H,W] (classification) or [B,2,H,W] (regression)
```

## Command Reference

### SPCR Evaluation
```bash
# Full SPCR
python src/spcr_full.py --original <orig_dir> --colorized <col_dir> --output <csv> --device mps

# Light SPCR
python src/spcr_light.py --original <orig_dir> --colorized <col_dir> --output <csv> --device mps

# Custom weights (semantic, perceptual, diversity)
python src/spcr_full.py ... --weights 0.5 0.3 0.2
```

### Data Download
```bash
# Download 100 images
python src/data_download.py --out_dir data/coco --num_images 100 --workers 4

# Download 1000 images (full training)
python src/data_download.py --out_dir data/coco --num_images 1000 --workers 8
```

### Training
```bash
# Classification (recommended)
python src/train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/exp1 \
    --loss_type classification \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_mps \
    --eval_after_epoch

# Regression (faster)
python src/train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/exp2 \
    --loss_type regression \
    --epochs 20 \
    --batch_size 16 \
    --use_mps

# Resume training
python src/train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/exp1 \
    --resume_checkpoint outputs/exp1/checkpoint_epoch_010.pt \
    --epochs 50 \
    --use_mps

# Custom backbone
python src/train_colorization.py \
    --backbone custom \
    --loss_type classification \
    --epochs 20 \
    --use_mps
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with MPS/CUDA support)
- 4-6 GB GPU memory (classification), 3-4 GB (regression)
- ~10 GB disk space (1000 COCO images)

See `requirements.txt` for complete list.

## Performance

### SPCR Evaluation (MPS, Apple Silicon)
- Full SPCR: ~2-3 sec/image
- Light SPCR: ~1-1.5 sec/image

### Training (1000 images, batch_size=16, MPS)
- Classification: ~3-5 min/epoch
- Regression: ~2-3 min/epoch

## Citation

This implementation is based on:

```bibtex
@inproceedings{zhang2016colorful,
  title={Colorful image colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}
```

SPCR metric is our novel contribution for colorization quality assessment.

## License

Academic and research use. See original paper for details.

## Troubleshooting

### PyTorch Installation
```bash
# Apple Silicon (M1/M2/M3)
pip install torch torchvision

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Out of Memory
- Reduce `--batch_size` (try 8 or 4)
- Reduce `--img_size` (try 128)
- Use CPU: remove `--use_mps` flag

### SPCR Evaluation Fails
- Ensure scripts run from project root
- Check image paths are correct
- Verify MPS/CUDA availability

### Module Import Errors
```bash
# Ensure you're in project root when running scripts
cd /path/to/CV_final_project
python src/train_colorization.py ...
```

## Contact

For questions or issues, please open an issue on GitHub.
