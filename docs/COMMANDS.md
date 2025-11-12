# 📝 Complete Command Reference

## Environment Setup

```bash
# Navigate to project
cd "/Users/ashishdate/Documents/IITJ/4th year/CV_final_project"

# Activate virtual environment
source venv/bin/activate

# Install additional dependencies (if needed)
pip install scikit-learn pycocotools
```

## SPCR Evaluation Metrics

### Generate Test Images
```bash
python test_spcr.py
```

### Run Full SPCR (with semantic segmentation)
```bash
python spcr_full.py \
    --original original/ \
    --colorized colorized/ \
    --output results_full.csv \
    --device mps
```

### Run Light SPCR (without segmentation)
```bash
python spcr_light.py \
    --original original/ \
    --colorized colorized/ \
    --output results_light.csv \
    --device mps
```

### Cleanup Test Images
```bash
python test_spcr.py --cleanup
```

### View SPCR Results
```bash
cat results_full.csv
cat results_light.csv
```

## Data Download

### Download 100 Images (Quick Test)
```bash
python data_download.py \
    --out_dir data/coco \
    --num_images 100 \
    --workers 4
```

### Download 1000 Images (Full Training)
```bash
python data_download.py \
    --out_dir data/coco \
    --num_images 1000 \
    --workers 8
```

### Download Custom Amount
```bash
python data_download.py \
    --out_dir data/coco \
    --num_images 500 \
    --workers 6 \
    --seed 42
```

## Colorization Training

### Quick Training (5 epochs, Classification)
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/quick_class \
    --loss_type classification \
    --epochs 5 \
    --batch_size 16 \
    --use_mps
```

### Full Training (30 epochs, Classification + SPCR)
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/full_class \
    --loss_type classification \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_mps \
    --eval_after_epoch
```

### Regression Training (Faster)
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/regression \
    --loss_type regression \
    --epochs 20 \
    --batch_size 16 \
    --use_mps
```

### Custom Backbone Training
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/custom \
    --backbone custom \
    --loss_type classification \
    --epochs 20 \
    --use_mps
```

### Resume Training
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/full_class \
    --resume_checkpoint outputs/full_class/checkpoint_epoch_010.pt \
    --loss_type classification \
    --epochs 30 \
    --use_mps \
    --eval_after_epoch
```

### Training with Custom Parameters
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/custom_params \
    --loss_type classification \
    --epochs 40 \
    --batch_size 8 \
    --lr 5e-5 \
    --img_size 224 \
    --num_bins 313 \
    --val_split 0.15 \
    --use_mps \
    --eval_after_epoch
```

### CPU Training (No GPU/MPS)
```bash
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/cpu_train \
    --loss_type regression \
    --epochs 10 \
    --batch_size 4
```

## Results & Analysis

### View Training Log
```bash
cat outputs/full_class/training_log.csv
```

### View SPCR Evaluation Results
```bash
cat outputs/full_class/eval/epoch_029/spcr_full.csv
cat outputs/full_class/eval/epoch_029/spcr_light.csv
```

### Open Colorized Images
```bash
open outputs/full_class/eval/epoch_029/colorized/
```

### Open Original Images
```bash
open outputs/full_class/eval/epoch_029/original/
```

### List All Checkpoints
```bash
ls -lh outputs/*/checkpoint_*.pt
```

### Find Best Model
```bash
ls -lh outputs/*/best_model.pt
```

## Interactive Demo

### Run Menu-Driven Demo
```bash
python demo.py
```

## Advanced Usage

### Precompute ab Centers
```bash
# First run (computes centers)
python train_colorization.py \
    --data_root data/coco \
    --loss_type classification \
    --epochs 1

# Subsequent runs (reuses centers)
python train_colorization.py \
    --data_root data/coco \
    --ab_centers_path outputs/ab_centers.npy \
    --loss_type classification \
    --epochs 30
```

### Train on Custom Dataset
```bash
# Organize your images in a folder
mkdir -p data/my_dataset/images
# Copy your images there

# Train
python train_colorization.py \
    --data_root data/my_dataset \
    --loss_type classification \
    --use_mps
```

### Multiple Experiments
```bash
# Experiment 1: Classification
python train_colorization.py --data_root data/coco --out_dir outputs/exp1_class --loss_type classification --epochs 20 --use_mps

# Experiment 2: Regression
python train_colorization.py --data_root data/coco --out_dir outputs/exp2_reg --loss_type regression --epochs 20 --use_mps

# Experiment 3: Custom backbone
python train_colorization.py --data_root data/coco --out_dir outputs/exp3_custom --backbone custom --loss_type classification --epochs 20 --use_mps
```

### Batch SPCR Evaluation
```bash
# Evaluate multiple epoch folders
for epoch in {000..029}; do
    python spcr_full.py \
        --original outputs/full_class/eval/epoch_$epoch/original/ \
        --colorized outputs/full_class/eval/epoch_$epoch/colorized/ \
        --output outputs/full_class/eval/epoch_$epoch/spcr_full.csv \
        --device mps
done
```

## Utility Commands

### Check Environment Status
```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Count Images
```bash
ls -1 data/coco/images/ | wc -l
```

### Check Disk Space
```bash
du -sh data/
du -sh outputs/
```

### Clean Up Old Checkpoints
```bash
# Keep only last 5 checkpoints
cd outputs/full_class
ls -t checkpoint_*.pt | tail -n +6 | xargs rm
cd ../..
```

### Monitor GPU/MPS Memory
```bash
# During training (another terminal)
watch -n 1 'python -c "import torch; print(torch.mps.current_allocated_memory()/1024**3, \"GB\")"'
```

### View Documentation
```bash
cat README.md
cat QUICKSTART.md
cat COLORIZATION_GUIDE.md
cat PROJECT_SUMMARY.md
cat CHECKLIST.md
```

## Troubleshooting Commands

### Fix Import Errors
```bash
pip install --upgrade torch torchvision scikit-image scikit-learn pycocotools
```

### Clear Cache
```bash
rm -rf __pycache__/
rm -rf .pytest_cache/
```

### Verify Installation
```bash
python -c "import utils_lab; import model_colorization; import train_colorization; print('✓ All imports OK')"
```

### Test SPCR Scripts
```bash
python spcr_full.py --help
python spcr_light.py --help
```

### Test Training Script
```bash
python train_colorization.py --help
```

## Complete Workflow Example

```bash
# 1. Setup
cd "/Users/ashishdate/Documents/IITJ/4th year/CV_final_project"
source venv/bin/activate

# 2. Test SPCR
python test_spcr.py
python spcr_full.py --original original/ --colorized colorized/ --device mps
python spcr_light.py --original original/ --colorized colorized/ --device mps

# 3. Download data
python data_download.py --out_dir data/coco --num_images 100

# 4. Train model
python train_colorization.py \
    --data_root data/coco \
    --out_dir outputs/demo \
    --loss_type classification \
    --epochs 10 \
    --batch_size 16 \
    --use_mps \
    --eval_after_epoch

# 5. View results
cat outputs/demo/training_log.csv
open outputs/demo/eval/epoch_009/colorized/

# 6. Cleanup
python test_spcr.py --cleanup
```

## Quick Reference Table

| Task | Command |
|------|---------|
| Activate env | `source venv/bin/activate` |
| Demo | `python demo.py` |
| Test SPCR | `python test_spcr.py && python spcr_full.py --original original/ --colorized colorized/ --device mps` |
| Download data | `python data_download.py --out_dir data/coco --num_images 100` |
| Train quick | `python train_colorization.py --data_root data/coco --loss_type classification --epochs 5 --use_mps` |
| Train full | `python train_colorization.py --data_root data/coco --loss_type classification --epochs 30 --use_mps --eval_after_epoch` |
| View log | `cat outputs/*/training_log.csv` |
| Open images | `open outputs/*/eval/epoch_*/colorized/` |

---

**Tip**: For quick testing, always start with small datasets (100 images) and few epochs (5-10).
