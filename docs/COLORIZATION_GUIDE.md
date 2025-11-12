# Colorization Training Pipeline - Quick Start

## Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Install additional dependencies
pip install scikit-learn pycocotools
```

## Quick Start (5 Steps)

### 1. Download Dataset (~1000 COCO images)
```bash
python data_download.py --out_dir data/coco --num_images 1000 --workers 8
```

### 2. Train with Classification Loss (Zhang et al. method)
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/classification \
    --loss_type classification \
    --epochs 50 \
    --batch_size 16 \
    --use_mps \
    --eval_after_epoch
```

### 3. Train with Regression Loss (faster alternative)
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/regression \
    --loss_type regression \
    --epochs 50 \
    --batch_size 16 \
    --use_mps
```

### 4. Resume Training
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/classification \
    --resume_checkpoint outputs/classification/checkpoint_epoch_010.pt \
    --loss_type classification \
    --use_mps
```

### 5. Monitor Results
```bash
# Check training log
cat outputs/classification/training_log.csv

# View colorized images
open outputs/classification/eval/epoch_049/colorized/
```

## Architecture Options

### ResNet-18 Backbone (Default, Recommended)
```bash
python train_colorization.py \
    --data_root data/coco \
    --backbone resnet18 \
    --loss_type classification \
    --use_mps
```

### Custom Lightweight Encoder
```bash
python train_colorization.py \
    --data_root data/coco \
    --backbone custom \
    --loss_type classification \
    --use_mps
```

## Loss Types

### Classification (313 bins, Zhang et al.)
- More accurate colors
- Class rebalancing for rare colors
- Slower training
- Use `--loss_type classification`

### Regression (Direct ab prediction)
- Faster training
- Simpler implementation
- May produce desaturated colors
- Use `--loss_type regression`

## SPCR Evaluation Integration

When `--eval_after_epoch` is enabled:
- Automatically runs `spcr_full.py` and `spcr_light.py` after each validation
- Saves results to CSV in `outputs/eval/epoch_XXX/`
- Logs SPCR scores in `training_log.csv`

## File Outputs

```
outputs/
├── ab_centers.npy              # Cached ab bin centers (classification only)
├── training_log.csv            # Epoch, losses, SPCR scores
├── best_model.pt               # Best model weights
├── checkpoint_epoch_XXX.pt     # Per-epoch checkpoints
└── eval/
    └── epoch_XXX/
        ├── colorized/          # Model predictions
        ├── original/           # Ground truth
        ├── spcr_full.csv       # Full SPCR results
        └── spcr_light.csv      # Light SPCR results
```

## Advanced Usage

### Custom Image Size
```bash
python train_colorization.py \
    --data_root data/coco \
    --img_size 224 \
    --batch_size 8 \
    --use_mps
```

### Precomputed ab Centers
```bash
# First run computes and saves centers
python train_colorization.py --data_root data/coco --loss_type classification --epochs 1

# Subsequent runs reuse them
python train_colorization.py \
    --data_root data/coco \
    --ab_centers_path outputs/ab_centers.npy \
    --loss_type classification
```

### CPU Training
```bash
python train_colorization.py \
    --data_root data/coco \
    --loss_type classification
# Automatically uses CPU if no GPU/MPS available
```

### Different Dataset
```bash
# Use your own images (put in a folder)
python train_colorization.py \
    --data_root /path/to/your/images \
    --loss_type classification \
    --use_mps
```

## Performance Notes

### Training Speed (per epoch, 1000 images, batch_size=16)
- **MPS (M1/M2 Mac)**: ~3-5 minutes (classification), ~2-3 minutes (regression)
- **CUDA (GPU)**: ~2-4 minutes (classification), ~1-2 minutes (regression)
- **CPU**: ~15-30 minutes (not recommended)

### Memory Requirements
- **Classification**: ~4-6 GB GPU memory
- **Regression**: ~3-4 GB GPU memory
- Reduce `--batch_size` if OOM

## Troubleshooting

### "No images found"
- Check `--data_root` points to correct directory
- Images should be in `data_root/images/` or directly in `data_root/`

### "SPCR evaluation failed"
- Ensure `spcr_full.py` and `spcr_light.py` are in same directory
- Check evaluation images were saved correctly

### MPS errors on Mac
- Update to latest PyTorch: `pip install --upgrade torch torchvision`
- Fall back to CPU if needed (remove `--use_mps`)

### Out of memory
- Reduce `--batch_size` (try 8 or 4)
- Reduce `--img_size` (try 128)
- Use CPU: remove `--use_mps`

## Dataset Download Issues

### COCO download fails
- Check internet connection
- Try reducing `--workers` (e.g., `--workers 4`)
- Manually download annotations from http://images.cocodataset.org/annotations/annotations_trainval2017.zip

### Partial downloads
- Script resumes automatically (skips existing images)
- Re-run `data_download.py` to complete

## Example Complete Workflow

```bash
# 1. Install dependencies
source venv/bin/activate
pip install scikit-learn pycocotools

# 2. Download 1000 COCO images
python data_download.py --out_dir data/coco --num_images 1000

# 3. Train for 30 epochs with SPCR evaluation
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/exp1 \
    --loss_type classification \
    --epochs 30 \
    --batch_size 16 \
    --use_mps \
    --eval_after_epoch

# 4. Check results
cat outputs/exp1/training_log.csv
open outputs/exp1/eval/epoch_029/colorized/

# 5. Resume for more epochs
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/exp1 \
    --resume_checkpoint outputs/exp1/checkpoint_epoch_029.pt \
    --loss_type classification \
    --epochs 50 \
    --use_mps \
    --eval_after_epoch
```

## Citation

Based on:
```
Zhang, R., Isola, P., & Efros, A. A. (2016).
Colorful image colorization.
In ECCV 2016.
```
