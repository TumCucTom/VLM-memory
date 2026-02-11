#!/usr/bin/env python3
"""
Export per-frame camera .npz (intrinsic + extrinsic c2w) from COLMAP for step 1.2.
Requires ScanNet++ toolbox: set SCANNETPP_ROOT or --scannetpp_root to repo path.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_txt_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Export COLMAP to per-frame camera .npz for step 1.2.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="ScanNet++ data root containing <scene_id>/dslr/colmap and train_test_lists.json")
    parser.add_argument("--scene_list_file", type=str, required=True,
                        help="Text file with one scene ID per line (e.g. nvs_sem_train.txt)")
    parser.add_argument("--scannetpp_root", type=str, default=None,
                        help="Path to ScanNet++ toolbox repo (for common.utils.colmap). Default: SCANNETPP_ROOT env")
    args = parser.parse_args()

    scannetpp_root = args.scannetpp_root or __import__("os").environ.get("SCANNETPP_ROOT")
    if not scannetpp_root or not Path(scannetpp_root).exists():
        print("Error: ScanNet++ toolbox required. Set SCANNETPP_ROOT or --scannetpp_root to repo path.", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, str(Path(scannetpp_root).resolve()))
    from common.utils.colmap import read_model
    from common.scene_release import ScannetppScene_Release

    data_root = Path(args.data_root)
    scene_ids = read_txt_list(args.scene_list_file)

    for scene_id in tqdm(scene_ids, desc="Exporting camera npz"):
        scene = ScannetppScene_Release(scene_id, data_root=data_root)
        colmap_dir = scene.dslr_colmap_dir
        if not colmap_dir or not colmap_dir.exists():
            continue
        train_list_path = scene.scene_root_dir / "dslr" / "train_test_lists.json"
        if not train_list_path.exists():
            continue
        with open(train_list_path) as f:
            train_names = set(json.load(f).get("train", []))
        if not train_names:
            continue

        try:
            cameras, images, _ = read_model(colmap_dir, ".txt")
        except Exception:
            continue
        if not cameras or not images:
            continue
        # name -> image (match case-insensitively for basename)
        by_name = {img.name: img for img in images.values()}
        by_basename_lower = {Path(img.name).stem.lower(): img for img in images.values()}
        cam = next(iter(cameras.values()))
        K = cam.K if hasattr(cam, "K") else np.array([
            [cam.params[0], 0, cam.params[2]],
            [0, cam.params[1], cam.params[3]],
            [0, 0, 1]
        ])

        out_camera_dir = scene.scene_root_dir / "dslr" / "camera"
        out_camera_dir.mkdir(parents=True, exist_ok=True)

        for name in train_names:
            img = by_name.get(name) or (by_basename_lower.get(Path(name).stem.lower()) if by_basename_lower else None)
            if img is None:
                continue
            w2c = img.world_to_camera
            c2w = np.linalg.inv(w2c)
            basename = Path(name).stem
            np.savez(
                out_camera_dir / f"{basename}.npz",
                intrinsic=K.astype(np.float64),
                extrinsic=c2w.astype(np.float64),
            )


if __name__ == "__main__":
    main()
