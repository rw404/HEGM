#!/usr/bin/env python3
"""
Download processed datasets and pre-trained checkpoints for HEGM reproduction.

Usage:
    python scripts/download_artifacts.py --splits
    python scripts/download_artifacts.py --checkpoints
    python scripts/download_artifacts.py --all
"""

import os
import sys
import argparse
import tempfile
import zipfile
import shutil

# Google Drive file IDs
SPLITS_FILE_ID = "1z1jjwJrJFOpwD-jf4Dmy-LSMNkno2XxZ"
CHECKPOINTS_FILE_ID = "1fyrOO0mhnT6cjKUofwRIE1mkQW5nSwB0"


def check_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("Error: gdown is not installed.")
        print("Install it with: pip install gdown")
        sys.exit(1)


def clean_macos_junk(base_dir: str):
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for f in files:
            if f == ".DS_Store" or f.startswith("._"):
                os.remove(os.path.join(root, f))
        for d in dirs:
            if d == "__MACOSX":
                shutil.rmtree(os.path.join(root, d))


def download_and_extract(file_id: str, target_dir: str, desc: str):
    gdown = check_gdown()

    if not file_id:
        print(f"Error: {desc} file ID is not set.")
        print(f"Please set the FILE_ID in scripts/download_artifacts.py")
        return False, []

    print(f"Downloading {desc}...")
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    extracted_files = []
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, tmp_path, quiet=False)

        with zipfile.ZipFile(tmp_path, 'r') as zf:
            for name in zf.namelist():
                if "__MACOSX" in name or name.endswith(".DS_Store") or os.path.basename(name).startswith("._"):
                    continue
                if not name.endswith("/"):
                    extracted_files.append(name)
            zf.extractall(target_dir)

        clean_macos_junk(target_dir)

        print(f"Extracted to {target_dir}/")
        return True, sorted(extracted_files)
    except Exception as e:
        print(f"Error: Failed to download or extract {desc}: {e}")
        return False, []
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_splits():
    print("\nDownloading processed dataset splits...")

    if not SPLITS_FILE_ID:
        print("Error: SPLITS_FILE_ID is not set.")
        print("Please edit scripts/download_artifacts.py and set SPLITS_FILE_ID.")
        return False

    success, files = download_and_extract(SPLITS_FILE_ID, "data/processed", "splits.zip")

    if success:
        print(f"Files: {len(files)}")
        for f in files[:10]:
            print(f"  {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")

    return success


def download_checkpoints():
    print("\nDownloading pre-trained checkpoints...")

    if not CHECKPOINTS_FILE_ID:
        print("Error: CHECKPOINTS_FILE_ID is not set.")
        print("Please edit scripts/download_artifacts.py and set CHECKPOINTS_FILE_ID.")
        return False

    success, files = download_and_extract(CHECKPOINTS_FILE_ID, "checkpoints", "checkpoints.zip")

    if success:
        print(f"Checkpoints: {len(files)}")
        for f in files:
            print(f"  {f}")

        print("\nExample:")
        print("  python evaluate.py --config configs/vklsvd_k3.yaml --checkpoint checkpoints/vklsvd_k3/last.ckpt")

    return success


def main():
    parser = argparse.ArgumentParser(description="Download HEGM artifacts from Google Drive")
    parser.add_argument("--splits", action="store_true", help="Download processed dataset splits")
    parser.add_argument("--checkpoints", action="store_true", help="Download pre-trained checkpoints")
    parser.add_argument("--all", action="store_true", help="Download everything")
    args = parser.parse_args()

    if not args.splits and not args.checkpoints and not args.all:
        parser.print_help()
        print("\nNote: Set SPLITS_FILE_ID and CHECKPOINTS_FILE_ID in the script before use.")
        return

    if args.splits or args.all:
        download_splits()

    if args.checkpoints or args.all:
        download_checkpoints()

    print("\nDone.")


if __name__ == "__main__":
    main()
