#!/bin/bash
#SBATCH --job-name=scannetpp_prep_1_2
#SBATCH --output=logs/scannetpp_prep_1_2_%j.out
#SBATCH --error=logs/scannetpp_prep_1_2_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=workq

# Prepares data needed for step 1.2 (export sampled frames):
# 1) Clone ScanNet++ toolbox to ~/scratch/scannetpp if missing
# 2) Install renderpy and run depth rendering (DSLR render_depth)
# 3) Export COLMAP to per-frame camera/*.npz
# Requires: data/raw_data/scannetpp already downloaded (data/ and splits/)

set -e
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

# Paths (override via env if needed)
PROJECT_ROOT="${PROJECT_ROOT:-/home/u5fj/trvbale.u5fj/scratch/VLM-memory}"
SCRATCH_PARENT="${SCRATCH_PARENT:-$HOME/scratch}"
SCANNETPP_DIR="${SCRATCH_PARENT}/scannetpp"
RENDERPY_DIR="${SCRATCH_PARENT}/renderpy"
RAW_SCANNETPP="${PROJECT_ROOT}/data/raw_data/scannetpp"
DATA_ROOT="${RAW_SCANNETPP}"
SPLITS_DIR="${RAW_SCANNETPP}/splits"

mkdir -p "$SCRATCH_PARENT"
mkdir -p "${PROJECT_ROOT}/logs"

# Activate env
source ~/miniforge3/bin/activate
conda activate scannetpp

# ---------- 1) Clone ScanNet++ toolbox ----------
if [ ! -d "${SCANNETPP_DIR}/.git" ]; then
  echo "Cloning ScanNet++ toolbox into ${SCANNETPP_DIR}..."
  git clone --depth 1 https://github.com/scannetpp/scannetpp.git "${SCANNETPP_DIR}"
else
  echo "ScanNet++ toolbox already at ${SCANNETPP_DIR}"
fi
cd "${SCANNETPP_DIR}"
pip install -q -r requirements.txt

# ---------- 2) Install renderpy and run render ----------
if [ ! -d "${RENDERPY_DIR}/.git" ]; then
  echo "Cloning renderpy into ${RENDERPY_DIR}..."
  git clone --recursive --depth 1 https://github.com/liu115/renderpy.git "${RENDERPY_DIR}"
fi
if ! python -c "import renderpy" 2>/dev/null; then
  echo "Installing renderpy from source (requires pip package cmake)..."
  pip install -q cmake
  cd "${RENDERPY_DIR}"
  if ! pip install . 2>&1; then
    echo "renderpy install failed (may need cmake/OpenGL). Skipping render step."
    RENDER_SKIP=1
  fi
  cd "${SCANNETPP_DIR}"
else
  echo "renderpy already importable"
fi

if [ "${RENDER_SKIP:-0}" != "1" ]; then
  # Render config: data_root = raw_data/scannetpp (has data/ and splits/)
  RENDER_CFG="${PROJECT_ROOT}/scripts/VLM_Dual_Mem/data-download/scannetpp/preprocessing/render_scannetpp_generated.yml"
  sed "s|DATA_ROOT_PLACEHOLDER|${DATA_ROOT}|g" \
    "${PROJECT_ROOT}/scripts/VLM_Dual_Mem/data-download/scannetpp/preprocessing/render_scannetpp.yml.template" \
    > "${RENDER_CFG}"
  echo "Running depth render (DSLR)... output under ${DATA_ROOT}/data/<scene_id>/dslr/render_depth"
  cd "${SCANNETPP_DIR}"
  PYTHONPATH="${SCANNETPP_DIR}" python -m common.render "${RENDER_CFG}" || true
fi

# ---------- 3) Export COLMAP to camera/*.npz ----------
# ScannetppScene_Release expects data_root = dir containing scene dirs = raw_data/scannetpp/data
CAMERA_DATA_ROOT="${RAW_SCANNETPP}/data"
export SCANNETPP_ROOT="${SCANNETPP_DIR}"

for split in nvs_sem_train nvs_sem_val; do
  SPLIT_FILE="${SPLITS_DIR}/${split}.txt"
  if [ ! -f "${SPLIT_FILE}" ]; then
    echo "Split file not found: ${SPLIT_FILE}, skip."
    continue
  fi
  echo "Exporting camera .npz for split ${split}..."
  python "${PROJECT_ROOT}/scripts/VLM_Dual_Mem/data-download/scannetpp/preprocessing/export_colmap_to_camera_npz.py" \
    --data_root "${CAMERA_DATA_ROOT}" \
    --scene_list_file "${SPLIT_FILE}" \
    --scannetpp_root "${SCANNETPP_DIR}"
done

echo "End Time: $(date)"
echo "Done. Next: run step1_2_export_sampled_frames.slurm to export sampled frames."
