#!/usr/bin/env python3
"""Interactive demo for Image Colorization + SPCR Evaluation."""

import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, desc):
    """Execute command and display output."""
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    print(f"Command: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nFailed with exit code {result.returncode}")
    else:
        print(f"\n✓ {desc} completed")
    input("\nPress Enter to continue...")


def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    while True:
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("="*60)
        print("IMAGE COLORIZATION + SPCR EVALUATION - DEMO")
        print("="*60)
        print("\nSPCR EVALUATION")
        print("  1. Generate test images")
        print("  2. Run SPCR Full evaluation")
        print("  3. Run SPCR Light evaluation")
        print("  4. Cleanup test images")
        
        print("\nCOLORIZATION")
        print("  5. Download 100 COCO images")
        print("  6. Train (5 epochs, quick)")
        print("  7. Train (30 epochs, full)")
        
        print("\nRESULTS")
        print("  8. View SPCR results")
        print("  9. View training log")
        print(" 10. Check environment")
        
        print("\n  0. Exit")
        print("="*60)
        
        choice = input("Choice: ").strip()
        
        if choice == '0':
            sys.exit(0)
        elif choice == '1':
            run_cmd("python src/test_spcr.py", "Generate test images")
        elif choice == '2':
            run_cmd("python src/spcr_full.py --original assets/original/ --colorized assets/colorized/ --output results/results_full.csv --device mps", "SPCR Full")
        elif choice == '3':
            run_cmd("python src/spcr_light.py --original assets/original/ --colorized assets/colorized/ --output results/results_light.csv --device mps", "SPCR Light")
        elif choice == '4':
            run_cmd("python src/test_spcr.py --cleanup", "Cleanup test images")
        elif choice == '5':
            run_cmd("python src/data_download.py --out_dir data/coco --num_images 100 --workers 4", "Download COCO")
        elif choice == '6':
            run_cmd("python src/train_colorization.py --data_root data/coco --out_dir outputs/quick --loss_type classification --epochs 5 --batch_size 16 --use_mps", "Quick training")
        elif choice == '7':
            run_cmd("python src/train_colorization.py --data_root data/coco --out_dir outputs/full --loss_type classification --epochs 30 --batch_size 16 --use_mps --eval_after_epoch", "Full training")
        elif choice == '8':
            print("\n" + "="*60 + "\nSPCR RESULTS\n" + "="*60)
            for f in ['results/results_full.csv', 'results/results_light.csv']:
                if Path(f).exists():
                    print(f"\n📄 {f}\n" + "-"*60)
                    print(Path(f).read_text())
            input("\nPress Enter...")
        elif choice == '9':
            print("\n" + "="*60 + "\nTRAINING LOGS\n" + "="*60)
            for log in Path("outputs").rglob("training_log.csv"):
                print(f"\n📄 {log}\n" + "-"*60)
                print(log.read_text())
            input("\nPress Enter...")
        elif choice == '10':
            run_cmd("python --version && python -c \"import torch; print(f'PyTorch {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')\"", "Environment check")
        else:
            print("\nInvalid choice")
            input("Press Enter...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
