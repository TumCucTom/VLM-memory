#!/bin/bash
#SBATCH --job-name=upgrade_bnb
#SBATCH --output=logs/upgrade_bnb_%j.out
#SBATCH --error=logs/upgrade_bnb_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

source ~/miniforge3/bin/activate
conda activate vlm3r-2

echo "Current bitsandbytes version:"
pip show bitsandbytes | grep Version

echo "Upgrading bitsandbytes..."
pip install bitsandbytes --upgrade

echo "Verifying new version:"
pip show bitsandbytes | grep Version

echo "Testing import..."
python -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__} imported OK')"

echo "Running bitsandbytes CUDA check..."
python -m bitsandbytes 2>&1 | tail -20

echo "Done."
echo "End Time: $(date)"