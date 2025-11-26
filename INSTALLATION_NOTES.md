# Installation Notes

## Quick Install (Recommended)

```bash
# 1. Create and activate environment
conda create -n glass_env python=3.10 -y
conda activate glass_env

# 2. Install core dependencies (should complete without errors)
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; import numpy; import pandas; print('✓ Core dependencies installed')"
```

This installs all dependencies needed for:
- Eye gaze feature extraction (with OpenFace)
- Emotion prediction inference
- Visualization

## Optional Dependencies

### Batch Processing with Threadward

```bash
pip install threadward
```

Only needed if you want to process multiple videos in parallel using `scripts/process_videos_batch.py`.

### PyFeat (Not Recommended)

```bash
pip install py-feat
```

**Note:** PyFeat is an alternative to OpenFace but has complex dependencies including:
- h5py (requires HDF5 C library)
- Multiple deep learning models
- Additional compilation requirements

**We recommend using OpenFace instead** (see main README.md).

## OpenFace Installation

OpenFace must be installed separately as it's a C++ application. See the main README.md for detailed instructions.

Quick summary for Ubuntu/Debian:
```bash
# Install dependencies
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-all-dev

# Or with conda (no sudo required):
conda install -c conda-forge cmake openblas lapack boost

# Clone and build OpenFace
cd ~
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
bash download_models.sh
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j$(nproc)

# Test
./bin/FeatureExtraction -help
```

## Troubleshooting

### "Cannot import torch"
Make sure you activated the conda environment:
```bash
conda activate glass_env
```

### "OpenFace binary not found"
Specify the path explicitly:
```bash
python scripts/process_videos.py \
    --openface-bin ~/OpenFace/build/bin/FeatureExtraction \
    --videos video.mp4 --output results/
```

### "No module named feat"
This is expected if you haven't installed py-feat. The default workflow uses OpenFace, not PyFeat.

### Build errors during pip install
If you see compilation errors during `pip install -r requirements.txt`:
1. Make sure you're using Python 3.10: `python --version`
2. Try upgrading pip: `pip install --upgrade pip setuptools wheel`
3. Install numpy first: `pip install numpy>=1.23.0`
4. Then install the rest: `pip install -r requirements.txt`

If issues persist, the requirements.txt has been designed to avoid problematic packages like py-feat.

## Testing Your Installation

```bash
# Test with the provided example video (if available)
python scripts/inference_pipeline.py \
    --csv path/to/openface/result.csv \
    --output test_predictions.csv

# You should see output like:
# Loading model from: emotion_prediction/best_model/model.ckpt
# Using device: cuda
# ...
# Results saved to: test_predictions.csv
```

## System Requirements

- Python 3.10+
- 8GB+ RAM
- GPU with 4GB+ VRAM (recommended for inference)
- ~500MB disk space for dependencies
- ~48MB for GLASS model checkpoint
- Linux/macOS (Windows may require additional setup for OpenFace)

