#!/bin/bash
#SBATCH --job-name=scannetpp_camera_export
#SBATCH --output=logs/scannetpp_camera_export_%j.out
#SBATCH --error=logs/scannetpp_camera_export_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=workq

# Export COLMAP to per-frame camera/*.npz only (no renderpy, no render).
# Requires: ScanNet++ toolbox at ~/scratch/scannetpp and raw_data/scannetpp downloaded.

set -e
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

PROJECT_ROOT="${PROJECT_ROOT:-/home/u5fj/trvbale.u5fj/scratch/VLM-memory}"
SCRATCH_PARENT="${SCRATCH_PARENT:-$HOME/scratch}"
SCANNETPP_DIR="${SCRATCH_PARENT}/scannetpp"
RAW_SCANNETPP="${PROJECT_ROOT}/data/raw_data/scannetpp"
SPLITS_DIR="${RAW_SCANNETPP}/splits"
CAMERA_DATA_ROOT="${RAW_SCANNETPP}/data"

mkdir -p "${PROJECT_ROOT}/logs"

if [ ! -d "${SCANNETPP_DIR}" ]; then
  echo "ERROR: ScanNet++ toolbox not found at ${SCANNETPP_DIR}. Run step1_2_prep_render_and_camera.slurm once or clone manually."
  exit 1
fi

source ~/miniforge3/bin/activate 2>/dev/null || true
conda activate vlm3r 2>/dev/null || true

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
echo "Done. Next: run step1_2_export_sampled_frames.slurm after render_depth is available (or use step 1.3 for iPhone)."
