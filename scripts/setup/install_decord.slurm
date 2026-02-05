#!/bin/bash
#SBATCH --job-name=install_decord
#SBATCH --output=logs/install_decord_%j.out
#SBATCH --error=logs/install_decord_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=workq

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate vlm3r

# Install compatible FFmpeg version (4.x) for decord
echo "Installing compatible FFmpeg 4.x..."
conda install -c conda-forge "ffmpeg<5" -y

# Install build dependencies
echo "Installing build dependencies..."
pip install cmake ninja

# Set up FFmpeg paths for cmake
FFMPEG_PREFIX=$(conda info --base)/envs/vlm3r
export PKG_CONFIG_PATH="${FFMPEG_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PATH="${FFMPEG_PREFIX}/bin:${PATH}"

echo "FFmpeg prefix: $FFMPEG_PREFIX"
echo "Checking for FFmpeg libraries..."
ls -la ${FFMPEG_PREFIX}/lib/libav* 2>/dev/null | head -5 || echo "FFmpeg libs not found in expected location"

# Build decord from source
BUILD_DIR=/tmp/decord-build-$$
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Cloning decord repository..."
git clone --recursive https://github.com/dmlc/decord.git .

echo "Building decord C++ library..."
# Build the C++ library from the root directory
mkdir -p build
cd build
# Point cmake to FFmpeg installation
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_FFMPEG=ON \
    -DFFMPEG_DIR="${FFMPEG_PREFIX}" \
    -DFFMPEG_ROOT="${FFMPEG_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${FFMPEG_PREFIX}" \
    -DCMAKE_LIBRARY_PATH="${FFMPEG_PREFIX}/lib" \
    -DCMAKE_INCLUDE_PATH="${FFMPEG_PREFIX}/include"
make -j$SLURM_CPUS_PER_TASK

# Verify the library was built
cd ..
if [ ! -f build/libdecord.so ]; then
    echo "Searching for libdecord.so..."
    find build -name "libdecord.so" -type f
    LIB_SRC=$(find build -name "libdecord.so" -type f | head -1)
    if [ -z "$LIB_SRC" ] || [ ! -f "$LIB_SRC" ]; then
        echo "ERROR: libdecord.so not found after build"
        exit 1
    fi
    echo "Found libdecord.so at: $LIB_SRC"
else
    echo "Found libdecord.so at: build/libdecord.so"
fi

# Remove any existing library in python/decord/ to avoid setup.py copy conflicts
cd python
if [ -f decord/libdecord.so ]; then
    echo "Removing existing libdecord.so to avoid setup.py copy conflict..."
    rm -f decord/libdecord.so
fi

# Install Python package - setup.py will copy the library from build directory
echo "Installing Python package..."
pip install . --no-build-isolation

# Clean up
cd /
rm -rf $BUILD_DIR

echo "End Time: $(date)"
echo "Decord installation completed successfully!"
