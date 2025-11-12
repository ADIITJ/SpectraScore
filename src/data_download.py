#!/usr/bin/env python3
"""COCO 2017 dataset downloader for colorization training."""

import argparse
import csv
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


COCO_ANNO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_BASE = "http://images.cocodataset.org/train2017"


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download file with retry logic."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            if attempt == retries - 1:
                return False
    return False


def download_coco_annotations(anno_dir: Path) -> Path:
    """Download and extract COCO annotations."""
    anno_dir.mkdir(parents=True, exist_ok=True)
    anno_file = anno_dir / "instances_train2017.json"
    
    if anno_file.exists():
        return anno_file
    
    print("Downloading COCO 2017 annotations...")
    zip_path = anno_dir / "annotations_trainval2017.zip"
    
    if not download_file(COCO_ANNO_URL, zip_path):
        raise RuntimeError(f"Failed to download annotations from {COCO_ANNO_URL}")
    
    print("Extracting annotations...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(anno_dir)
    
    zip_path.unlink()
    
    extracted = anno_dir / "annotations" / "instances_train2017.json"
    if extracted.exists():
        import shutil
        extracted.rename(anno_file)
        shutil.rmtree(anno_dir / "annotations", ignore_errors=True)
    
    return anno_file


def download_image(img_id: int, filename: str, out_dir: Path) -> Tuple[int, str, str, bool]:
    """Download single COCO image."""
    url = f"{COCO_IMG_BASE}/{filename}"
    dest = out_dir / filename
    
    if dest.exists():
        return img_id, filename, url, True
    
    success = download_file(url, dest)
    return img_id, filename, url, success


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 subset for colorization")
    parser.add_argument("--out_dir", type=str, default="data/coco", help="Output directory")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to download")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent downloads")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Download annotations
    anno_file = download_coco_annotations(out_dir)
    
    # Load COCO
    print("Loading COCO annotations...")
    coco = COCO(str(anno_file))
    
    # Sample images
    all_img_ids = list(coco.imgs.keys())
    random.shuffle(all_img_ids)
    selected_ids = all_img_ids[:args.num_images]
    
    print(f"Downloading {len(selected_ids)} images...")
    
    # Prepare download tasks
    tasks = []
    for img_id in selected_ids:
        img_info = coco.imgs[img_id]
        filename = img_info['file_name']
        tasks.append((img_id, filename))
    
    # Concurrent download
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_image, img_id, fn, img_dir): (img_id, fn) 
                   for img_id, fn in tasks}
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                img_id, filename, url, success = future.result()
                if success:
                    results.append((img_id, filename, url))
                else:
                    print(f"\nWarning: Failed to download {filename}")
                pbar.update(1)
    
    # Save manifest
    csv_path = out_dir / "images.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'filename', 'coco_url'])
        writer.writerows(results)
    
    print(f"\nDownloaded {len(results)}/{args.num_images} images")
    print(f"Images: {img_dir}")
    print(f"Manifest: {csv_path}")


if __name__ == "__main__":
    main()
