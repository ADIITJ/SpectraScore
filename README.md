# Image Colorization with SPCR Evaluation

Modern implementation of Zhang et al. (2016) "Colorful Image Colorization" with a novel **Semantic-Perceptual Colour Realism (SPCR)** metric for evaluation.

## Features

### Colorization Training
- **Two loss modes**: Classification (313 bins, Zhang et al.) and Regression
- **Multiple backbones**: PaperNet (ResNet-18), MobileLiteVariant, L2RegressionNet
- **Class rebalancing**: Inverse-frequency weights for rare colors
- **Annealed softmax**: Temperature-based inference (T=0.38)
- **GPU Optimization**: Mixed precision training (AMP) for NVIDIA GPUs
- **ImageNet Support**: Native support for ImageNet dataset structure

### SPCR Evaluation Metric (Novel Contribution)
Comprehensive quality assessment combining three components:
- **Semantic Plausibility** (40%): DeepLabV3 segmentation validates color-object consistency
- **Perceptual Realism** (40%): VGG16 deep features measure perceptual similarity
- **Colour Diversity** (20%): Lab chroma variance penalizes desaturation

Two implementations provided:
- `spcr_full_imagenet.py`: Full metric with semantic segmentation (ImageNet-compatible)
- `spcr_light_imagenet.py`: Lightweight version (histogram fallback, no segmentation)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ADIITJ/SpectraScore.git
cd SpectraScore

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (for NVIDIA GPUs)
pip install torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/cu124

# Install other dependencies
pip install -r requirements.txt

# Verify CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}, PyTorch: {torch.__version__}')"
```

**Expected Output:**
```
CUDA: True, Device: NVIDIA GeForce RTX 5070 Ti, PyTorch: 2.6.0+cu124
```

---

## ImageNet Training (GPU-Optimized)

### ImageNet Dataset Structure

The training scripts expect ImageNet dataset with the following structure:

```
/path/to/imagenet/
├── train/
│   ├── n01484850/          # Class folder (e.g., "great white shark")
│   │   ├── n01484850_1.JPEG
│   │   ├── n01484850_2.JPEG
│   │   └── ...
│   ├── n01491361/          # Another class folder
│   │   └── ...
│   └── ...                 # 1000 class folders total
│
└── val/
    ├── n01484850/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    ├── n01491361/
    │   └── ...
    └── ...                 # 1000 class folders total
```

**Key Points:**
- Each split (`train`/`val`) contains 1000 subfolders (one per ImageNet class)
- Folder names follow ImageNet synset IDs (e.g., `n01484850`)
- Image files are named with `.JPEG` extension (case-sensitive)
- Training images: typically `<synset>_<id>.JPEG`
- Validation images: typically `ILSVRC2012_val_<id>.JPEG`

---

## Training Pipeline: Complete Workflow

### Step 1: Prepare AB Bin Centers (One-Time Setup)

The classification-based colorization model quantizes the `ab` color space into 313 bins using k-means clustering. This step computes the bin centers from a sample of 10,000 images.

```bash
python src/train_all_models_imagenet.py \
    --data_root /home/ayaan/Image-Hue/data/train \
    --out_dir outputs/papernet_imagenet \
    --epochs 0 \
    --batch_size 32 \
    --img_size 224 \
    --num_workers 8 \
    --device cuda
```

**What This Does:**
1. Scans all images in `/home/ayaan/Image-Hue/data/train` (including nested class folders)
2. Randomly samples 10,000 images for AB bin computation
3. Extracts all `ab` pixel values from these images (no subsampling)
4. Runs k-means clustering (k=313) to compute bin centers
5. Saves bin centers to `outputs/papernet_imagenet/ab_centers.npy`
6. Computes class rebalancing weights (inverse frequency)
7. Saves weights to `outputs/papernet_imagenet/class_weights.pt`
8. Exits without training (epochs=0)

**Time:** ~15-20 minutes (one-time)

**Output Files:**
- `outputs/papernet_imagenet/ab_centers.npy` (313 bin centers)
- `outputs/papernet_imagenet/class_weights.pt` (class rebalancing weights)

---

### Step 2: Train PaperNet Model

Now train the PaperNet model on 200,000 images (stratified sampling across 1000 classes).

```bash
python src/train_all_models_imagenet.py \
    --data_root /home/ayaan/Image-Hue/data/train \
    --out_dir outputs/papernet_imagenet \
    --load_ab_centers outputs/papernet_imagenet/ab_centers.npy \
    --epochs 7 \
    --batch_size 32 \
    --img_size 224 \
    --num_workers 8 \
    --device cuda
```

**What This Does:**
1. Loads precomputed AB bin centers from `ab_centers.npy` (skips recomputation)
2. Performs **stratified sampling**: selects 200,000 images with equal representation from each of the 1000 classes (200 images per class)
3. Saves the list of training images to `outputs/papernet_imagenet/training_images_list.txt`
4. Trains PaperNet model for 7 epochs
5. **Mixed Precision Training**: Uses `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` for 2x speed boost
6. Saves checkpoints:
   - Every 2500 iterations: `checkpoint_iter_<N>.pt`
   - Every epoch: `checkpoint_epoch_<N>.pt`
   - Best model: `best_model.pt`
7. Logs training metrics to `outputs/papernet_imagenet/training_log.csv`

**Training Time (RTX 5070 Ti 16GB):**
- Speed: ~3-5 iterations/second
- Epoch duration: ~2.5 hours (for 200k images, batch_size=32)
- Total time: ~17.5 hours (7 epochs)

**GPU Utilization:**
- VRAM usage: ~8-10 GB / 16 GB
- GPU utilization: 85-95%

**Output Files:**
- `outputs/papernet_imagenet/PaperNet/checkpoint_epoch_<N>.pt` (per epoch)
- `outputs/papernet_imagenet/PaperNet/checkpoint_iter_<N>.pt` (every 2500 iterations)
- `outputs/papernet_imagenet/PaperNet/best_model.pt` (best validation loss)
- `outputs/papernet_imagenet/training_log.csv` (loss, metrics)
- `outputs/papernet_imagenet/training_images_list.txt` (list of 200k training images)

---

### Step 3: Colorize Validation Images

Use the trained model to colorize ImageNet validation images.

```bash
python src/eval_trained_model.py \
    --model_path outputs/papernet_imagenet/PaperNet/best_model.pt \
    --input_dir /home/ayaan/Image-Hue/data/val \
    --output_dir outputs/papernet_eval_imagenet_val \
    --ab_centers outputs/papernet_imagenet/ab_centers.npy \
    --batch_size 32 \
    --device cuda
```

**What This Does:**
1. Loads the trained PaperNet model from `best_model.pt`
2. Recursively finds all `.JPEG` images in `/home/ayaan/Image-Hue/data/val` (including class subfolders)
3. Colorizes each image:
   - Converts to Lab color space
   - Passes `L` channel through the model
   - Predicts `ab` channels
   - Combines `L` (original) + `ab` (predicted) → full color image
4. Saves colorized images to `outputs/papernet_eval_imagenet_val/<class_id>/<filename>_siggraph17.png`
   - Example: `outputs/papernet_eval_imagenet_val/n02837789/n02837789_ILSVRC2012_val_00007559_siggraph17.png`
5. The `_siggraph17.png` suffix is required for SPCR evaluation (next step)

**Time:** ~30-45 minutes (for 50,000 validation images)

**Output:**
- Colorized images in `outputs/papernet_eval_imagenet_val/`
- Preserves ImageNet class folder structure
- Filename format: `<class_id>_<original_name>_siggraph17.png`

---

### Step 4: Evaluate with SPCR Metric

Compute the SPCR score for the colorized images.

```bash
# Create output directory first
mkdir -p outputs/papernet_eval_imagenet_val

# Run SPCR evaluation
python src/spcr_full_imagenet.py \
    --original /home/ayaan/Image-Hue/data/val \
    --colorized outputs/papernet_eval_imagenet_val \
    --output outputs/papernet_eval_imagenet_val/spcr_full_results.csv \
    --device cuda \
    --batch_size 16
```

**What This Does:**
1. Matches original images (`.JPEG`) with colorized images (`_siggraph17.png`) by filename
2. For each image pair, computes:
   - **Semantic Plausibility** (40%): DeepLabV3 segmentation + color-class consistency
   - **Perceptual Realism** (40%): VGG16 feature similarity
   - **Colour Diversity** (20%): Lab chroma variance
3. Computes overall SPCR score: `0.4×Semantic + 0.4×Perceptual + 0.2×Diversity`
4. Saves per-image scores to CSV: `spcr_full_results.csv`
5. Prints summary statistics (mean, median, std dev)

**Time:** ~2-3 seconds per image (for full SPCR with segmentation)

**Output Files:**
- `outputs/papernet_eval_imagenet_val/spcr_full_results.csv`

**Example Output:**
```
SPCR Full Results
Total Pairs Evaluated: 50000
Mean SPCR: 0.7234
Median SPCR: 0.7456
Std Dev: 0.1123
Min: 0.3421
Max: 0.9567
```

---

## Training Configuration Details

### GPU Optimizations Applied

The `train_all_models_imagenet.py` script includes extensive optimizations for NVIDIA GPUs:

#### 1. **Mixed Precision Training (AMP)**
```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    outputs = model(images)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
- Uses FP16 for computation, FP32 for accumulation
- 2x faster training, ~30% less VRAM usage

#### 2. **Optimized DataLoader**
```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,           # Parallel data loading
    pin_memory=True,         # Faster CPU→GPU transfer
    persistent_workers=True, # Reuse worker processes
    prefetch_factor=3,       # Preload 3 batches per worker
    drop_last=True,          # Consistent batch sizes
    shuffle=True
)
```

#### 3. **No Pixel Subsampling**
- Uses **all pixels** for AB bin computation (no `ab_np[::4]` subsampling)
- Higher quality color distribution

#### 4. **Stratified Sampling**
- 200,000 training images sampled equally from 1000 classes
- 200 images per class → balanced training

#### 5. **Regular Checkpointing**
- Saves every 2500 iterations (prevents data loss)
- Per-epoch checkpoints
- Best model tracking

### Command-Line Arguments

```bash
python src/train_all_models_imagenet.py --help
```

**Key Arguments:**
- `--data_root`: Path to ImageNet train directory (required)
- `--out_dir`: Output directory for checkpoints and logs (default: `outputs/papernet_imagenet`)
- `--load_ab_centers`: Path to precomputed AB bins (skip recomputation)
- `--epochs`: Number of training epochs (default: 7)
- `--batch_size`: Batch size (default: 32, adjust based on VRAM)
- `--img_size`: Input image size (default: 224)
- `--num_workers`: DataLoader workers (default: 8, adjust based on CPU cores)
- `--device`: Device (default: `cuda`, options: `cuda`, `cpu`)
- `--lr`: Learning rate (default: 1e-4)

---

## Model Architecture: PaperNet

**PaperNet** is based on Zhang et al. (2016) "Colorful Image Colorization":

```
Input: L channel [B, 1, 224, 224]
         ↓
Encoder: ResNet-18 (pretrained on ImageNet)
         ↓ (feature extraction)
Features: [B, 512, H/16, W/16]
         ↓
Decoder: 4 upsampling blocks (ConvTranspose2d + BatchNorm + ReLU)
         ↓
Output: [B, 313, 224, 224]  (probability distribution over 313 AB bins)
```

**Key Components:**
- **Encoder**: ResNet-18 backbone (layers 1-4, pretrained)
- **Decoder**: 4 upsampling layers (512→256→128→64→313 channels)
- **Loss**: Cross-entropy with class rebalancing weights
- **Inference**: Annealed softmax (T=0.38) → weighted sum of top-K bins → predicted `ab`

**Why Classification (313 bins) vs. Regression?**
- **Classification**: More vibrant colors, better diversity, industry standard
- **Regression**: Faster, simpler, but tends to produce desaturated colors
- PaperNet uses classification for highest quality

---

## Project Structure

```
SpectraScore/
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
│
├── src/                                 # Source code
│   ├── train_all_models_imagenet.py    # ImageNet training pipeline (GPU-optimized)
│   ├── eval_trained_model.py           # Colorize images with trained model
│   ├── model_colorization.py           # Network architectures (PaperNet, etc.)
│   ├── utils_lab.py                    # Lab color space utilities
│   ├── spcr_full_imagenet.py           # Full SPCR metric (ImageNet-compatible)
│   ├── spcr_light_imagenet.py          # Lightweight SPCR metric
│   ├── test_spcr.py                    # Test image generator
│   └── demo.py                         # Interactive demo
│
├── outputs/                             # Training outputs
│   └── papernet_imagenet/
│       ├── ab_centers.npy              # 313 AB bin centers (precomputed)
│       ├── class_weights.pt            # Class rebalancing weights
│       ├── training_images_list.txt    # List of 200k training images
│       ├── training_log.csv            # Training metrics
│       └── PaperNet/
│           ├── checkpoint_epoch_<N>.pt # Per-epoch checkpoints
│           ├── checkpoint_iter_<N>.pt  # Iteration checkpoints (every 2500)
│           └── best_model.pt           # Best model (lowest validation loss)
│
├── data/                                # Datasets (not in repo)
│   └── imagenet/
│       ├── train/                      # 1.28M images, 1000 classes
│       └── val/                        # 50K images, 1000 classes
```

---

## Performance Benchmarks

### Training Performance (NVIDIA RTX 5070 Ti 16GB VRAM)

| Configuration | Speed (it/s) | Epoch Time | VRAM Usage |
|--------------|--------------|------------|------------|
| **PaperNet** (batch=32, img=224, AMP) | 3-5 | ~2.5 hours | 8-10 GB |
| PaperNet (batch=64, img=224, AMP) | 5-7 | ~1.5 hours | 14-15 GB |
| PaperNet (batch=16, img=224, AMP) | 2-3 | ~4 hours | 5-6 GB |

**Recommendations:**
- **RTX 5070 Ti (16GB)**: Use `batch_size=32` or `batch_size=64` for optimal speed
- **RTX 4060 Ti (8GB)**: Use `batch_size=16` with `img_size=224`
- **RTX 3060 (12GB)**: Use `batch_size=32` with `img_size=224`

### SPCR Evaluation Performance

| Metric | Time per Image | Hardware |
|--------|----------------|----------|
| **SPCR Full** (with segmentation) | ~2-3 sec | RTX 5070 Ti, CUDA |
| **SPCR Light** (no segmentation) | ~1-1.5 sec | RTX 5070 Ti, CUDA |

---

## Complete Command Workflow Summary

### First-Time Training (All Steps)

```bash
# Compute AB bin centers (one-time, ~15-20 min)
python src/train_all_models_imagenet.py \
    --data_root /home/ayaan/Image-Hue/data/train \
    --out_dir outputs/papernet_imagenet \
    --epochs 0 \
    --batch_size 32 \
    --device cuda

# Train PaperNet model (7 epochs, ~17.5 hours)
python src/train_all_models_imagenet.py \
    --data_root /home/ayaan/Image-Hue/data/train \
    --out_dir outputs/papernet_imagenet \
    --load_ab_centers outputs/papernet_imagenet/ab_centers.npy \
    --epochs 7 \
    --batch_size 32 \
    --device cuda

# Colorize validation images (~30-45 min)
python src/eval_trained_model.py \
    --model_path outputs/papernet_imagenet/PaperNet/best_model.pt \
    --input_dir /home/ayaan/Image-Hue/data/val \
    --output_dir outputs/papernet_eval_imagenet_val \
    --ab_centers outputs/papernet_imagenet/ab_centers.npy \
    --batch_size 32 \
    --device cuda

# Evaluate with SPCR metric (~2-3 hours)
mkdir -p outputs/papernet_eval_imagenet_val
python src/spcr_full_imagenet.py \
    --original /home/ayaan/Image-Hue/data/val \
    --colorized outputs/papernet_eval_imagenet_val \
    --output outputs/papernet_eval_imagenet_val/spcr_full_results.csv \
    --device cuda \
    --batch_size 16
```

### Resume Training (If Interrupted)

```bash
python src/train_all_models_imagenet.py \
    --data_root /home/ayaan/Image-Hue/data/train \
    --out_dir outputs/papernet_imagenet \
    --load_ab_centers outputs/papernet_imagenet/ab_centers.npy \
    --resume_checkpoint outputs/papernet_imagenet/PaperNet/checkpoint_epoch_3.pt \
    --epochs 7 \
    --device cuda
```

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 4060 Ti, RTX 5070 Ti, etc.)
- **CUDA**: 12.x or 11.8+
- **RAM**: 16GB+ (32GB+ recommended for large batch sizes)
- **Disk**: ~200GB free (ImageNet dataset + outputs)

### Software
- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA support (nightly recommended for AMP)
- **CUDA Toolkit**: 12.x or 11.8+

See `requirements.txt` for complete Python package list.

---

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

Faster alternative using 2D histogram comparison (Bhattacharyya coefficient) instead of deep features when semantic segmentation is unavailable.

---

## Troubleshooting

### PyTorch CUDA Installation

```bash
# PyTorch Nightly (recommended for latest AMP features)
pip install torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/cu124

# PyTorch Stable (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# PyTorch Stable (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. **Reduce batch size**:
   ```bash
   --batch_size 16  # or 8
   ```

2. **Reduce image size**:
   ```bash
   --img_size 128  # or 160
   ```

3. **Reduce DataLoader workers**:
   ```bash
   --num_workers 4
   ```

4. **Clear GPU cache** (before running):
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

5. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

### ImageNet Directory Issues

**Error: "Found 0 total images"**

**Cause:** Incorrect ImageNet structure or file extensions.

**Solution:**
1. Verify directory structure:
   ```bash
   ls /path/to/imagenet/train | head -10  # Should show class folders (n01484850, etc.)
   ls /path/to/imagenet/train/n01484850 | head -5  # Should show .JPEG files
   ```

2. Check file extensions (case-sensitive):
   ```bash
   find /path/to/imagenet/train -name "*.JPEG" | wc -l  # Should be >1M
   ```

3. Update file extension in script if needed (line 365 in `train_all_models_imagenet.py`):
   ```python
   all_images = list(data_root.rglob('*.JPEG'))  # Change to *.jpg if needed
   ```

### SPCR Evaluation: "Found 0 matching image pairs"

**Cause:** Colorized images missing `_siggraph17.png` suffix.

**Solution:**
1. Ensure `eval_trained_model.py` saves with correct suffix (line 55):
   ```python
   out_filename = f"{class_id}_{base_name}_siggraph17.png"
   ```

2. Or manually rename files:
   ```bash
   cd outputs/papernet_eval_imagenet_val
   for f in */*.png; do mv "$f" "${f%.png}_siggraph17.png"; done
   ```

### Module Import Errors

```bash
# Ensure you're in project root when running scripts
cd /path/to/SpectraScore
python src/train_all_models_imagenet.py ...
```

### UserWarning: "negative Z values" during colorization

**This is normal and harmless.** Some predicted colors fall slightly outside the RGB gamut and are automatically clipped by `scikit-image`.

To suppress warnings:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*negative Z values.*')
```

---

## Citation

This implementation is based on:

```bibtex
@inproceedings{zhang2016colorful,
  title={Colorful image colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={649--666},
  year={2016},
  organization={Springer}
}
```

**SPCR metric** is our novel contribution for comprehensive colorization quality assessment.

---

## License

Academic and research use. See original paper for details.

---

## Contact

For questions or issues, please open an issue on GitHub: [https://github.com/ADIITJ/SpectraScore](https://github.com/ADIITJ/SpectraScore)

---

## Acknowledgments

- Original colorization paper: Zhang et al. (2016)
- Pretrained models: torchvision (ResNet-18, DeepLabV3, VGG16)
- ImageNet dataset: ILSVRC
