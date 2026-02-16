#!/usr/bin/env python3
"""Generate train.txt and val.txt from ARKitScenes metadata.csv (video_id per line by fold)."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Generate train/val split files from metadata CSV.")
    p.add_argument("metadata_csv", type=str, help="Path to metadata.csv (video_id, visit_id, fold).")
    p.add_argument("output_dir", type=str, help="Directory to write train.txt and val.txt.")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ids = []
    val_ids = []
    with open(args.metadata_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            vid = str(row.get("video_id", "")).strip()
            if not vid:
                continue
            fold = str(row.get("fold", "")).strip()
            if fold == "Training":
                train_ids.append(vid)
            elif fold == "Validation":
                val_ids.append(vid)

    (out / "train.txt").write_text("\n".join(train_ids) + "\n")
    (out / "val.txt").write_text("\n".join(val_ids) + "\n")
    print(f"Wrote {len(train_ids)} ids to {out / 'train.txt'}, {len(val_ids)} ids to {out / 'val.txt'}")


if __name__ == "__main__":
    main()
