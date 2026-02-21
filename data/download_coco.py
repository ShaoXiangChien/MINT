#!/usr/bin/env python3
"""Download COCO 2017 validation images.

Downloads and extracts the COCO 2017 validation set, which is used
as the image source for several MINT experiments (object patching,
multimodal fusion, global image fusion, text patching).

Usage::

    python -m data.download_coco --output_dir data/val2017
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"


def download_file(url, dest, chunk_size=8192):
    """Download a file with progress reporting."""
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)",
                          end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="Download COCO val2017")
    parser.add_argument("--output_dir", type=str, default="data/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "val2017.zip"
    if not zip_path.exists():
        download_file(COCO_VAL_URL, zip_path)
    else:
        print(f"Zip already exists: {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    print(f"Done. Images in {output_dir / 'val2017'}/")


if __name__ == "__main__":
    main()
