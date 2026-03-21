"""Simple Python-based extraction utility.

Extracts ZIP and TAR.GZ archives using only Python's built-in modules.
No external tools (unzip, tar, 7z, etc.) are required.

Usage::

    # Extract a zip file to the same directory
    python data/prepare/unzip.py val2014.zip

    # Extract a tar.gz file to a specific directory
    python data/prepare/unzip.py controlled_images.tar.gz --dest /path/to/whatsup/

    # Extract multiple files at once
    python data/prepare/unzip.py val2014.zip sceneGraphs.zip --dest /data/
"""

import argparse
import tarfile
import zipfile
from pathlib import Path


def unzip(zip_path: str, dest_dir: str = None, verbose: bool = True) -> None:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"ERROR: File not found: {zip_path}")
        return

    # Default destination: same directory as the archive file
    dest = Path(dest_dir) if dest_dir else zip_path.parent
    dest.mkdir(parents=True, exist_ok=True)

    name = zip_path.name

    # Handle .tar.gz / .tgz / .tar.bz2 / .tar
    if tarfile.is_tarfile(zip_path):
        with tarfile.open(zip_path, "r:*") as tf:
            members = tf.getmembers()
            total = len(members)
            print(f"Extracting {name} → {dest}  ({total} entries)")
            for i, member in enumerate(members, 1):
                tf.extract(member, dest)
                if verbose and (i % 500 == 0 or i == total):
                    print(f"  {i}/{total} entries extracted...", end="\r")
        print(f"\nDone: {name} extracted to {dest}")
        return

    # Handle .zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        print(f"Extracting {name} → {dest}  ({total} files)")

        for i, member in enumerate(members, 1):
            zf.extract(member, dest)
            if verbose and (i % 1000 == 0 or i == total):
                print(f"  {i}/{total} files extracted...", end="\r")

    print(f"\nDone: {name} extracted to {dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Unzip files using Python's built-in zipfile module")
    parser.add_argument("zip_files", nargs="+",
                        help="One or more ZIP files to extract")
    parser.add_argument("--dest", type=str, default=None,
                        help="Destination directory (default: same as zip file)")
    args = parser.parse_args()

    for zip_file in args.zip_files:
        unzip(zip_file, args.dest)


if __name__ == "__main__":
    main()
