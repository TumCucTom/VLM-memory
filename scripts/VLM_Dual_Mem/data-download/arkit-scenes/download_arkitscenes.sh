#!/bin/bash
# Download ARKitScenes raw data and metadata for VLM-3R.
# Run from project root (VLM-memory). Uses project dirs: data/raw_arkitscenes/raw, metadata.csv.

set -e

# Project root (set by SLURM or default to script's repo root)
PROJECT_DIR="${PROJECT_DIR:-$(cd -P "$(dirname "$0")/../../../../.." && pwd)}"
cd "$PROJECT_DIR"

RAW_DIR="${RAW_DIR:-data/raw_arkitscenes/raw}"
REPO_DIR="${ARKITSCENES_REPO_DIR:-data/raw_arkitscenes/ARKitScenes_repo}"
METADATA_CSV_URL="${METADATA_CSV_URL:-https://raw.githubusercontent.com/apple/ARKitScenes/main/raw/raw_train_val_splits.csv}"

# Assets needed for VLM-3R (mov, annotation, mesh; optional: lowres_wide, depth, etc.)
RAW_DATASET_ASSETS=(
    mov
    annotation
    mesh
    confidence
    lowres_depth
    "lowres_wide.traj"
    lowres_wide
    lowres_wide_intrinsics
)

mkdir -p "$RAW_DIR"
METADATA_CSV="${RAW_DIR}/metadata.csv"

echo "=========================================="
echo "ARKitScenes download"
echo "=========================================="
echo "Project dir: $PROJECT_DIR"
echo "Raw dir:     $RAW_DIR"
echo "Metadata CSV: $METADATA_CSV"
echo ""

# 1. Fetch metadata CSV (same columns as processor expects: video_id, visit_id, fold)
if [ ! -f "$METADATA_CSV" ]; then
    echo "Downloading metadata CSV to $METADATA_CSV ..."
    if command -v curl &>/dev/null; then
        curl -sL -o "$METADATA_CSV" "$METADATA_CSV_URL"
    elif command -v wget &>/dev/null; then
        wget -q -O "$METADATA_CSV" "$METADATA_CSV_URL"
    else
        echo "ERROR: Need curl or wget to download metadata CSV."
        exit 1
    fi
    echo "Downloaded metadata CSV."
else
    echo "Metadata CSV already exists: $METADATA_CSV"
fi

# 2. Clone ARKitScenes repo if needed (for download_data.py)
if [ ! -f "${REPO_DIR}/download_data.py" ]; then
    echo "Cloning ARKitScenes repository to ${REPO_DIR} ..."
    mkdir -p "$(dirname "$REPO_DIR")"
    if [ -d "$REPO_DIR" ]; then
        echo "Directory exists but download_data.py missing; trying git pull."
        (cd "$REPO_DIR" && git pull --depth 1) || true
    else
        git clone --depth 1 https://github.com/apple/ARKitScenes.git "$REPO_DIR"
    fi
fi
if [ ! -f "${REPO_DIR}/download_data.py" ]; then
    echo "ERROR: download_data.py not found in ${REPO_DIR}"
    exit 1
fi

# 2b. Patch Apple's script to avoid pandas/numpy read_csv bug (TypeError: Cannot convert numpy.ndarray to numpy.ndarray)
# Replace all pd.read_csv with csv module or DataFrame.from_records
if grep -q "metadata = pd.read_csv(dst_file)" "${REPO_DIR}/download_data.py" \
   || grep -q "pd.DataFrame.from_records(rows)" "${REPO_DIR}/download_data.py" \
   || grep -q "pd.DataFrame({k: \[r\[k\]" "${REPO_DIR}/download_data.py" \
   || grep -q "engine='pyarrow'" "${REPO_DIR}/download_data.py" \
   || ! grep -q "csv.DictReader" "${REPO_DIR}/download_data.py"; then
    echo "Patching download_data.py for pandas/numpy CSV read compatibility..."
    python3 - "$REPO_DIR" << 'PATCHEOF'
import sys
path = sys.argv[1]
with open(f"{path}/download_data.py") as f:
    content = f.read()

# Patch 1: video_id_csv block
if "csv.DictReader" not in content:
    new_block = '''    elif args.video_id_csv is not None:
        import csv
        video_ids_ = []
        splits_ = []
        with open(args.video_id_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.split is None or row.get('fold') == args.split:
                    video_ids_.append(str(row['video_id']))
                    splits_.append(row['fold'])'''
    variants = [
        "df = pd.read_csv(args.video_id_csv, engine='python')",
        "df = pd.read_csv(args.video_id_csv, dtype=str)",
        "df = pd.read_csv(args.video_id_csv)",
    ]
    patched = False
    for read_line in variants:
        old_block = '''    elif args.video_id_csv is not None:
        ''' + read_line + '''
        if args.split is not None:
            df = df[df["fold"] == args.split]
        video_ids_ = df["video_id"].to_list()
        video_ids_ = list(map(str, video_ids_))  # Expecting video id to be a string
        splits_ = df["fold"].to_list()'''
        if old_block in content:
            content = content.replace(old_block, new_block)
            patched = True
            break
    if not patched:
        sys.stderr.write("Could not find video_id_csv block to patch\n")
        sys.exit(1)

# Patch 2: get_metadata - avoid pandas (DataFrame/Index triggers numpy bug). Use csv + wrapper for .loc/.iat.
new_get_meta_standalone = (
    "    import csv\n"
    "    with open(dst_file, newline='') as f:\n"
    "        rows = list(csv.DictReader(f))\n"
    "    if not rows:\n"
    "        return\n"
    "    def _float_eq(a, b):\n"
    "        try: return float(a) == float(b)\n"
    "        except (TypeError, ValueError): return a == b\n"
    "    class _Column:\n"
    "        def __init__(self, rows, col): self._rows, self._col = rows, col\n"
    "        def __eq__(self, x):\n"
    "            try: x = float(x)\n"
    "            except (TypeError, ValueError): pass\n"
    "            return [_float_eq(r.get(self._col), x) for r in self._rows]\n"
    "    class _IatVal:\n"
    "        def __init__(self, data): self._d = data if isinstance(data, list) else [data]\n"
    "        @property\n"
    "        def iat(self):\n"
    "            class _Idx:\n"
    "                def __init__(self, d): self._d = d\n"
    "                def __getitem__(self, k):\n"
    "                    if isinstance(k, tuple): return self._d[k[1]] if k[0]==0 and k[1]<len(self._d) else None\n"
    "                    return self._d[k] if k < len(self._d) else None\n"
    "            return _Idx(self._d)\n"
    "    class _Row(dict):\n"
    "        def __getitem__(self, k):\n"
    "            v = dict.get(self, k)\n"
    "            return _IatVal([v] if v is not None else [])\n"
    "    class _Loc:\n"
    "        def __init__(self, rows): self._rows = rows\n"
    "        def __getitem__(self, key):\n"
    "            if isinstance(key, tuple) and len(key) == 2:\n"
    "                mask, cols = key\n"
    "                idx = next((i for i, m in enumerate(mask) if m), None)\n"
    "                return _IatVal([self._rows[idx].get(c) for c in cols] if idx is not None else [])\n"
    "            idx = next((i for i, m in enumerate(key) if m), None)\n"
    "            return _Row(self._rows[idx]) if idx is not None else _Row()\n"
    "    class _Meta:\n"
    "        def __getitem__(self, col): return _Column(rows, col)\n"
    "        @property\n"
    "        def loc(self): return _Loc(rows)\n"
    "    metadata = _Meta()\n"
    "    return metadata\n"
)
old_variants = [
    "    metadata = pd.read_csv(dst_file)\n    return metadata",
    "    metadata = pd.read_csv(dst_file, engine='pyarrow')\n    return metadata",
    "    import csv\n    with open(dst_file, newline='') as f:\n        reader = csv.DictReader(f)\n        rows = list(reader)\n    if not rows:\n        return\n    metadata = pd.DataFrame.from_records(rows)\n    return metadata",
    "    import csv\n    with open(dst_file, newline='') as f:\n        reader = csv.DictReader(f)\n        rows = list(reader)\n    if not rows:\n        return\n    metadata = pd.DataFrame({k: [r[k] for r in rows] for k in rows[0].keys()})\n    return metadata",
]
get_meta_patched = False
for old in old_variants:
    if old in content:
        content = content.replace(old, new_get_meta_standalone)
        get_meta_patched = True
        break
if not get_meta_patched and "class _Meta:" not in content:
    sys.stderr.write("Could not find get_metadata block to patch\n")
    sys.exit(1)

with open(f"{path}/download_data.py", 'w') as f:
    f.write(content)
print("Patched download_data.py")
PATCHEOF
fi

# 3. Run Apple's download script. Apple expects download_dir = parent of dataset dir (it creates raw/Training, raw/Validation under that).
DOWNLOAD_PARENT="$(realpath "$(dirname "$RAW_DIR")")"
ASSETS_STRING="${RAW_DATASET_ASSETS[*]}"
echo "Starting raw data download (this can take hours; dataset is large)..."
python3 "${REPO_DIR}/download_data.py" raw \
    --video_id_csv "$METADATA_CSV" \
    --download_dir "$DOWNLOAD_PARENT" \
    --raw_dataset_assets $ASSETS_STRING

echo ""
echo "=========================================="
echo "Download complete"
echo "=========================================="
echo "Raw data: $RAW_DIR (Training/ and Validation/ per video_id)"
echo "Metadata: $METADATA_CSV"
echo "Next: run 02_preprocess_arkitscenes.slurm to generate metadata and videos in data/vlm_3r_data/arkitscenes/videos/"
echo ""
a/arkitscenes/videos/"
echo ""