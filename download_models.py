#!/usr/bin/env python3
"""
Downloads the SAM (Segment Anything Model) checkpoint required by
automatic_mask_borders.py into the segmentation/ directory next to this script.

Run once after cloning or setting up the environment:
    python download_models.py

The downloaded file (~1.2 GB) is gitignored and will not be committed.
"""

import urllib.request
import sys
from pathlib import Path

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
MODEL_NAME = "sam_vit_l_0b3195.pth"

def download_with_progress(url: str, dest: Path):
    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(count * block_size / total_size * 100, 100)
        downloaded = min(count * block_size, total_size) / (1024 ** 2)
        total = total_size / (1024 ** 2)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  {downloaded:.0f} / {total:.0f} MB")
        sys.stdout.flush()

    print(f"Downloading {MODEL_NAME} (~1.2 GB) from Meta...")
    urllib.request.urlretrieve(url, dest, reporthook)
    print()

def main():
    script_dir = Path(__file__).resolve().parent
    seg_dir = script_dir / "segmentation"
    seg_dir.mkdir(exist_ok=True)

    dest = seg_dir / MODEL_NAME

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 ** 2)
        print(f"Model already present: {dest} ({size_mb:.0f} MB)")
        print("Delete the file and re-run if you want to re-download.")
        return

    download_with_progress(MODEL_URL, dest)

    size_mb = dest.stat().st_size / (1024 ** 2)
    print(f"Saved to: {dest} ({size_mb:.0f} MB)")
    print("automatic_mask_borders.py will auto-load it from segmentation/ on next launch.")

if __name__ == "__main__":
    main()
