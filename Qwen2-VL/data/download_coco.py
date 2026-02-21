#!/usr/bin/env python3
"""
Download and extract COCO 2017 validation set using pure Python
No unzip or apt install needed!
"""

import urllib.request
import zipfile
import os
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest_path):
    """Download file with progress bar"""
    print(f"📥 Downloading from {url}")
    print(f"   Saving to: {dest_path}")
    
    # Get file size
    with urllib.request.urlopen(url) as response:
        file_size = int(response.headers.get('Content-Length', 0))
    
    # Download with progress bar
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        def reporthook(block_num, block_size, total_size):
            if block_num > 0:
                pbar.update(block_size)
        
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    
    print(f"✅ Download complete!")
    return dest_path

def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar"""
    print(f"\n📦 Extracting {zip_path}")
    print(f"   Destination: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files
        file_list = zip_ref.namelist()
        
        # Extract with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_to)
    
    print(f"✅ Extraction complete!")
    print(f"   Extracted {len(file_list)} files")

def main():
    # Configuration
    COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
    BASE_DIR = Path("/home/sc305/VLM/Qwen2-VL/data")
    ZIP_PATH = BASE_DIR / "val2017.zip"
    
    # Create directory if it doesn't exist
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🖼️  COCO 2017 Validation Set Downloader")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  URL: {COCO_URL}")
    print(f"  Download to: {ZIP_PATH}")
    print(f"  Extract to: {BASE_DIR}")
    print(f"  Expected size: ~1 GB (5000 images)")
    
    # Check if already downloaded
    if ZIP_PATH.exists():
        print(f"\n⚠️  Zip file already exists: {ZIP_PATH}")
        response = input("   Delete and re-download? (y/n): ")
        if response.lower() != 'y':
            print("   Skipping download...")
        else:
            os.remove(ZIP_PATH)
            download_file(COCO_URL, ZIP_PATH)
    else:
        # Download the zip file
        download_file(COCO_URL, ZIP_PATH)
    
    # Check if already extracted
    val2017_dir = BASE_DIR / "val2017"
    if val2017_dir.exists():
        num_images = len(list(val2017_dir.glob("*.jpg")))
        print(f"\n⚠️  Directory already exists: {val2017_dir}")
        print(f"   Found {num_images} images")
        response = input("   Re-extract? (y/n): ")
        if response.lower() != 'y':
            print("   Skipping extraction...")
            return
    
    # Extract the zip file
    extract_zip(ZIP_PATH, BASE_DIR)
    
    # Verify extraction
    val2017_dir = BASE_DIR / "val2017"
    if val2017_dir.exists():
        num_images = len(list(val2017_dir.glob("*.jpg")))
        print(f"\n✅ SUCCESS!")
        print(f"   Location: {val2017_dir}")
        print(f"   Images found: {num_images}")
        print(f"   Expected: 5000 images")
        
        if num_images == 5000:
            print(f"   ✅ All images present!")
        else:
            print(f"   ⚠️  Image count mismatch!")
        
        # Show some example files
        print(f"\n📸 Example images:")
        for img_path in list(val2017_dir.glob("*.jpg"))[:5]:
            print(f"   - {img_path.name}")
    else:
        print(f"\n❌ ERROR: Extraction failed!")
    
    # Optional: Delete zip file to save space
    print(f"\n💾 Zip file size: {ZIP_PATH.stat().st_size / (1024**3):.2f} GB")
    response = input("   Delete zip file to save space? (y/n): ")
    if response.lower() == 'y':
        os.remove(ZIP_PATH)
        print(f"   ✓ Deleted {ZIP_PATH}")
    
    print("\n" + "=" * 70)
    print("✅ COCO 2017 Download Complete!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()