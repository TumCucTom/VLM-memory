#!/usr/bin/env bash
# Create a dedicated conda env for ScanNet++ prep (renderpy + toolbox).
# Run once, then the prep SLURM job will use it automatically if it exists.
#
# Usage: bash setup_scannetpp_env.sh

set -e
source ~/miniforge3/bin/activate 2>/dev/null || source ~/miniconda3/bin/activate 2>/dev/null || true

if conda env list | grep -q '^scannetpp '; then
  echo "Env 'scannetpp' already exists. Activate with: conda activate scannetpp"
  exit 0
fi

echo "Creating conda env 'scannetpp' (Python 3.10, opencv, cmake, etc.)..."
conda create -n scannetpp python=3.10 -y
conda activate scannetpp

# OpenCV with CMake config (so renderpy's CMake can find it)
conda install -y -c conda-forge opencv cmake ninja
# ScanNet++ toolbox lightweight deps (no torch pin)
pip install munch imageio lz4 open3d POT omegaconf hydra-core pyyaml tqdm numpy Pillow

echo ""
echo "Done. Use this env for the prep job:"
echo "  conda activate scannetpp"
echo "  sbatch step1_2_prep_render_and_camera.slurm"
echo "Or just sbatch — the job will activate 'scannetpp' if it exists."
