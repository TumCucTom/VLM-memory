#!/usr/bin/env python
"""
Download QA pairs from Hugging Face instead of generating them locally.
This script downloads the VLM-3R-DATA dataset which includes pre-generated QA pairs.
"""

import argparse
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

def download_full_dataset(repo_id, output_dir):
    """Download the entire dataset using snapshot_download"""
    print(f"Downloading full dataset from {repo_id}...")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            repo_type="dataset",
            local_dir_use_symlinks=False
        )
        print(f"\n✓ Successfully downloaded dataset to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def download_using_datasets_lib(repo_id, output_dir, split=None):
    """Download using datasets library (allows filtering by split)"""
    print(f"Loading dataset from {repo_id}...")
    
    try:
        if split:
            dataset = load_dataset(repo_id, split=split)
        else:
            dataset = load_dataset(repo_id)
        
        # Save to disk
        if isinstance(dataset, dict):
            # Multiple splits
            for split_name, split_data in dataset.items():
                split_output = os.path.join(output_dir, split_name)
                os.makedirs(split_output, exist_ok=True)
                split_data.save_to_disk(split_output)
                print(f"✓ Saved {split_name} split to: {split_output}")
        else:
            # Single split
            dataset.save_to_disk(output_dir)
            print(f"✓ Saved dataset to: {os.path.abspath(output_dir)}")
        
        return output_dir
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Download QA pairs from Hugging Face VLM-3R-DATA dataset"
    )
    parser.add_argument(
        "--repo_id",
        default="Journey9ni/VLM-3R-DATA",
        help="Hugging Face repository ID (default: Journey9ni/VLM-3R-DATA)"
    )
    parser.add_argument(
        "--output_dir",
        default="data/scannet_vsibench",
        help="Output directory for downloaded data (default: data/scannet_vsibench)"
    )
    parser.add_argument(
        "--method",
        choices=["snapshot", "datasets"],
        default="snapshot",
        help="Download method: 'snapshot' for full download, 'datasets' for using datasets library"
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Specific split to download (only used with --method datasets)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Downloading QA pairs from Hugging Face")
    print("=" * 60)
    print(f"Repository: {args.repo_id}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print("=" * 60)
    print()
    
    try:
        if args.method == "snapshot":
            download_full_dataset(args.repo_id, args.output_dir)
        else:
            download_using_datasets_lib(args.repo_id, args.output_dir, args.split)
        
        print()
        print("=" * 60)
        print("Download completed successfully!")
        print("=" * 60)
        print(f"\nData location: {os.path.abspath(args.output_dir)}")
        print("\nNote: The dataset includes QA pairs for multiple datasets:")
        print("  - ScanNet")
        print("  - ScanNet++")
        print("  - ARKitScenes")
        print("\nYou may need to filter for ScanNet-specific files based on your needs.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have huggingface_hub and datasets installed:")
        print("   pip install huggingface_hub datasets")
        print("2. Check your internet connection")
        print("3. Verify the repository ID is correct")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
