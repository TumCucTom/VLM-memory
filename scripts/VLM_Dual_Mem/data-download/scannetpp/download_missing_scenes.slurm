#!/bin/bash
#SBATCH --job-name=download_scannetpp_missing
#SBATCH --output=logs/download_scannetpp_missing_%j.out
#SBATCH --error=logs/download_scannetpp_missing_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

source ~/miniforge3/bin/activate
conda activate vlm3r

cd /home/b6ar/trvbale.b6ar/scratch/VLM-memory

MISSING_FILE="scripts/VLM_Dual_Mem/data-download/scannetpp/missing_scenes.txt"
CONFIG_IN="scripts/VLM_Dual_Mem/data-download/scannetpp/data_download/download_scannetpp.yml"
CONFIG_OUT="scripts/VLM_Dual_Mem/data-download/scannetpp/data_download/download_scannetpp_missing.yml"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannetpp/download_scannetpp_wrapper.py"

# Build download_scenes list from missing_scenes.txt
SCENES_LIST=$(tr '\n' ',' < "$MISSING_FILE" | sed 's/,$//')

# Create new config that downloads only missing scenes
python3 << PYEOF
import yaml

with open("$CONFIG_IN", 'r') as f:
    config = yaml.safe_load(f)

# Remove download_splits, add download_scenes
config.pop('download_splits', None)
config['download_scenes'] = [$SCENES_LIST]

with open("$CONFIG_OUT", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created config with {len(config['download_scenes'])} scenes")
PYEOF

echo "Starting download of missing ScanNet++ scenes..."
python "$WRAPPER_SCRIPT" "$CONFIG_OUT"

echo "End Time: $(date)"