#!/usr/bin/env python3
"""
Create filtered vsibench QA JSONs that contain only samples whose video path exists.
Writes *_valid.json alongside originals and prints a report (counts, filtered-out).
Run from repo root: python scripts/VLM_3R/filter_vsibench_valid_paths.py
"""
import json
import os
import sys

VIDEO_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vlm_3r_data")
VIDEO_FOLDER = os.path.normpath(os.path.abspath(VIDEO_FOLDER))

# Same datasets as vsibench_data.yaml
DATASETS = [
    ("data/vlm_3r_data/vsibench/merged_qa_scannet_train.json", "scannet"),
    ("data/vlm_3r_data/vsibench/merged_qa_scannetpp_train.json", "scannetpp"),
    ("data/vlm_3r_data/vsibench/merged_qa_route_plan_train.json", "route_plan"),
]


def main():
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(repo_root)
    video_folder = VIDEO_FOLDER
    if not os.path.isdir(video_folder):
        print("Error: VIDEO_FOLDER not found:", video_folder, file=sys.stderr)
        sys.exit(1)

    total_original = 0
    total_valid = 0
    total_filtered_out = 0
    filtered_videos = set()
    report = []

    for json_path, name in DATASETS:
        if not os.path.isfile(json_path):
            report.append((name, json_path, 0, 0, 0, "FILE_NOT_FOUND"))
            continue
        with open(json_path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            report.append((name, json_path, len(data), 0, 0, "NOT_LIST"))
            continue

        valid = []
        removed_videos = set()
        for s in data:
            v = s.get("video")
            if not v:
                valid.append(s)
                continue
            full = os.path.join(video_folder, v)
            if os.path.exists(full):
                valid.append(s)
            else:
                removed_videos.add(v)

        out_path = json_path.replace(".json", "_valid.json")
        with open(out_path, "w") as f:
            json.dump(valid, f, indent=2)

        orig_n = len(data)
        valid_n = len(valid)
        out_n = orig_n - valid_n
        total_original += orig_n
        total_valid += valid_n
        total_filtered_out += out_n
        filtered_videos |= removed_videos
        report.append((name, out_path, orig_n, valid_n, out_n, None))

    # Print report
    print("=" * 60)
    print("Vsibench path filter report (VIDEO_FOLDER=%s)" % video_folder)
    print("=" * 60)
    for name, out_path, orig, valid_n, out_n, err in report:
        if err:
            print("%s: %s" % (name, err))
            continue
        print("%s: %d -> %d valid (filtered out %d samples)" % (name, orig, valid_n, out_n))
        print("  output: %s" % out_path)
    print("-" * 60)
    print("Total samples: %d -> %d valid (filtered out %d)" % (total_original, total_valid, total_filtered_out))
    print("Unique video paths removed: %d" % len(filtered_videos))
    if filtered_videos:
        for p in sorted(filtered_videos)[:20]:
            print("  - %s" % p)
        if len(filtered_videos) > 20:
            print("  ... and %d more" % (len(filtered_videos) - 20))
    print("=" * 60)
    return 0 if total_filtered_out == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
