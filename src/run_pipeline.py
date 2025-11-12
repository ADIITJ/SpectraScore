#!/usr/bin/env python3
"""
Complete pipeline: Download → Train → Evaluate

Chains together:
1. Download COCO dataset
2. Train all three models
3. Evaluate with SPCR
4. Generate summary report
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Complete colorization pipeline")
    parser.add_argument("--num_images", type=int, default=10000, help="Number of images to download")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--workers", type=int, default=8, help="Download workers")
    parser.add_argument("--skip_download", action="store_true", help="Skip download if data exists")
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "coco_pipeline"
    output_dir = project_root / "outputs" / f"pipeline_{args.num_images}imgs_{args.epochs}epochs"
    test_images_dir = project_root / "assets" / "original"
    
    # Use venv python explicitly
    venv_python = project_root / "venv" / "bin" / "python"
    if not venv_python.exists():
        print(f"❌ Virtual environment not found at {venv_python}")
        print("Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt")
        return 1
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║          IMAGE COLORIZATION - COMPLETE PIPELINE                   ║
╚═══════════════════════════════════════════════════════════════════╝

Configuration:
  • Images:      {args.num_images:,}
  • Epochs:      {args.epochs}
  • Batch size:  {args.batch_size}
  • Device:      {args.device}
  • Data dir:    {data_dir}
  • Output dir:  {output_dir}

Models to train:
  1. PaperNet (31M params)
  2. MobileLiteVariant (2M params)
  3. L2RegressionNet (1M params)

Estimated total time: {args.num_images/1000 * 5 + args.epochs * 3:.0f}-{args.num_images/1000 * 8 + args.epochs * 4:.0f} minutes
""")
    
    print("Starting pipeline in 3 seconds...")
    time.sleep(3)
    
    pipeline_start = time.time()
    
    # Step 1: Download data
    if not args.skip_download or not (data_dir / "images").exists():
        download_cmd = [
            str(venv_python),
            str(project_root / "src" / "data_download.py"),
            "--out_dir", str(data_dir),
            "--num_images", str(args.num_images),
            "--workers", str(args.workers)
        ]
        
        if not run_command(download_cmd, f"Downloading {args.num_images:,} COCO images"):
            print("\n❌ Pipeline failed at download step")
            return 1
    else:
        print(f"\n✅ Skipping download - using existing data at {data_dir}")
    
    # Verify download
    image_dir = data_dir / "images"
    if not image_dir.exists():
        print(f"\n❌ Image directory not found: {image_dir}")
        return 1
    
    num_images = len(list(image_dir.glob("*.jpg")))
    print(f"\n✅ Verified: {num_images:,} images available")
    
    # Step 2: Generate test images (if needed)
    if not test_images_dir.exists() or len(list(test_images_dir.glob("*.jpg"))) == 0:
        test_cmd = [
            str(venv_python),
            str(project_root / "src" / "test_spcr.py")
        ]
        if not run_command(test_cmd, "Generating test images"):
            print("\n⚠️  Warning: Test image generation failed, continuing...")
    
    # Step 3: Train all models
    train_cmd = [
        str(venv_python),
        str(project_root / "src" / "train_all_models.py"),
        "--data_root", str(image_dir),
        "--out_dir", str(output_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--test_images", str(test_images_dir)
    ]
    
    if not run_command(train_cmd, f"Training all models ({args.epochs} epochs each)"):
        print("\n❌ Pipeline failed at training step")
        return 1
    
    # Step 4: Generate summary report
    print(f"\n{'='*70}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*70}\n")
    
    summary_file = output_dir / "PIPELINE_SUMMARY.md"
    
    with open(summary_file, 'w') as f:
        f.write(f"""# Pipeline Results Summary

## Configuration
- **Images**: {num_images:,}
- **Epochs**: {args.epochs}
- **Batch size**: {args.batch_size}
- **Device**: {args.device}
- **Total time**: {(time.time() - pipeline_start) / 60:.1f} minutes

## Models Trained
1. PaperNet
2. MobileLiteVariant  
3. L2RegressionNet

## Results

### PaperNet
""")
        
        # Read training logs
        for model_name in ["PaperNet", "MobileLiteVariant", "L2RegressionNet"]:
            model_dir = output_dir / model_name
            
            # Training log
            training_log = model_dir / "training_log.csv"
            if training_log.exists():
                f.write(f"\n### {model_name}\n\n")
                f.write("**Training Progress:**\n```\n")
                with open(training_log) as log:
                    f.write(log.read())
                f.write("```\n\n")
            
            # SPCR results
            spcr_results = model_dir / "spcr_results.csv"
            if spcr_results.exists():
                f.write("**SPCR Evaluation:**\n```\n")
                with open(spcr_results) as res:
                    f.write(res.read())
                f.write("```\n\n")
    
    print(f"✅ Summary report saved to: {summary_file}")
    
    # Final summary
    total_time = time.time() - pipeline_start
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    PIPELINE COMPLETED ✅                          ║
╚═══════════════════════════════════════════════════════════════════╝

Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)

Results saved to:
  {output_dir}

Check the following:
  • Training logs:  {output_dir}/<model>/training_log.csv
  • Checkpoints:    {output_dir}/<model>/*.pt
  • Colorized:      {output_dir}/<model>/colorized/
  • SPCR scores:    {output_dir}/<model>/spcr_results.csv
  • Summary:        {summary_file}

To view results:
  cat {summary_file}
  open {output_dir}
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
