# GLASS: Gaze-based Learning for Affective State Sensing

This repository contains the official implementation for emotion prediction from eye gaze patterns using self-supervised learning. The model predicts continuous Valence, Arousal, and Dominance (VAD) dimensions from eye movement data extracted via OpenFace.

## Quick Start: Inference

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenFace 2.0+ installed (see Installation section)

### Step 1: Install Dependencies

```bash
# Create conda environment
conda create -n glass_env python=3.10
conda activate glass_env

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Extract Eye Gaze Features with OpenFace

Process your video to extract eye gaze features:

```bash
python scripts/process_videos.py \
    --videos /path/to/your/video.mp4 \
    --output openface_results/ \
    --openface-bin /path/to/OpenFace/build/bin/FeatureExtraction
```

This creates a CSV file at `openface_results/video_name/result.csv` containing frame-by-frame gaze coordinates.

### Step 3: Predict Emotions

Run emotion prediction using the pre-trained model:

```bash
python scripts/inference_pipeline.py \
    --csv openface_results/video_name/result.csv \
    --output predictions.csv
```

The output CSV contains columns: `timestamp`, `valence`, `arousal`, `dominance`.

**Output format:** The model produces predictions at 2.5-second intervals using 5-second windows of eye gaze data. Each row represents:
- `timestamp`: End time of the analysis window (seconds)
- `valence`: Emotional valence [0, 1] (negative → positive)
- `arousal`: Emotional arousal [0, 1] (calm → excited)
- `dominance`: Emotional dominance [0, 1] (submissive → dominant)

### Step 4: Visualize Predictions (Optional)

Create a video with emotion predictions overlaid:

```bash
python scripts/visualize_eyegaze_emotion.py \
    --video /path/to/your/video.mp4 \
    --predictions predictions.csv \
    --output visualization.mp4
```

### Batch Processing

For processing multiple videos in parallel:

```bash
pip install threadward

python scripts/process_videos_batch.py \
    --videos video1.mp4 video2.mp4 video3.mp4 \
    --output openface_results/ \
    --num-workers 4 \
    --num-gpus-per-worker 1
```

## Installation

### 1. Install Conda Environment

```bash
# Create environment
conda create -n glass_env python=3.10
conda activate glass_env

# Install core dependencies
pip install -r requirements.txt

# Optional: Install threadward for batch processing
# pip install threadward
```

**Note:** All core dependencies should install without issues. `py-feat` is NOT required - we use OpenFace for gaze extraction.

### 2. Install OpenFace

OpenFace 2.0 is required for eye gaze extraction. Install from source:

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev \
    liblapack-dev libx11-dev libgtk-3-dev libboost-all-dev

# Clone OpenFace
cd ~
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace

# Download pre-trained models
bash download_models.sh

# Build OpenFace
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j$(nproc)

# Test installation
./build/bin/FeatureExtraction -help
```

**Note:** If you don't have sudo access:
1. Install dependencies in conda: `conda install -c conda-forge cmake openblas lapack boost`
2. Build OpenFace with conda libraries by adding `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX` to the cmake command
3. Set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` before running

For other operating systems, see [OpenFace documentation](https://github.com/TadasBaltrusaitis/OpenFace/wiki).

### 3. Verify Installation

```bash
# Activate environment
conda activate glass_env

# Set OpenFace path (adjust as needed)
export OPENFACE_BIN=$HOME/OpenFace/build/bin/FeatureExtraction

# Test on a sample video
python scripts/process_videos.py \
    --videos test_video.mp4 \
    --output test_output/ \
    --openface-bin $OPENFACE_BIN
```

## Repository Structure

```
GLASS/
├── gaze_detection/          # OpenFace wrapper and feature extraction
├── self_supervised_gaze/    # Self-supervised gaze encoder training
│   ├── models/              # Model architectures (PatchSeq2Seq)
│   ├── configs/             # Training configurations
│   ├── train.py             # Training script
│   ├── gaze_forecast_dataset.py  # Dataset loader
│   └── convert_csv_to_npy.py     # Convert OpenFace CSV to numpy
├── emotion_prediction/      # Emotion prediction from gaze embeddings
│   ├── best_model/          # Pre-trained model checkpoint
│   ├── glass.py             # Model definition and training
│   ├── baselines.py         # Baseline models and metrics
│   └── build_eyegaze_emotion_dataset.py  # Dataset preparation
├── scripts/                 # Inference and processing scripts
│   ├── process_videos.py            # Sequential OpenFace processing
│   ├── process_videos_batch.py     # Parallel batch processing
│   ├── inference_pipeline.py       # Emotion prediction inference
│   └── visualize_eyegaze_emotion.py  # Visualization tool
├── requirements.txt
└── README.md
```

## Model Details

### Pre-trained Model

The included `emotion_prediction/best_model/model.ckpt` is a GLASS model trained with:
- **Architecture:** PatchSeq2Seq encoder (256-dim, 6 layers) + ChunkedRichPoolMLP head
- **Training data:** 5-second windows of eye gaze at 30 FPS (150 frames)
- **Input:** 6-dimensional eye gaze vectors (2 eyes × 3D gaze directions)
- **Output:** Continuous VAD predictions normalized to [0, 1]

### Performance

On held-out test data:
- Valence: Pearson r = 0.30
- Arousal: Pearson r = 0.06
- Dominance: Pearson r = 0.31
- Average Pearson r = 0.22

See `emotion_prediction/best_model/glass.json` for complete metrics.

## Training

For reproducibility and custom model training, see below.

### Step 1: Prepare Raw Gaze Data

Extract eye gaze from your video dataset using OpenFace:

```bash
# Process videos (sequential or batch)
python scripts/process_videos.py \
    --videos /path/to/videos/*.mp4 \
    --output openface_results/
```

### Step 2: Convert to Training Format

Convert OpenFace CSV files to numpy arrays:

```bash
cd self_supervised_gaze

python convert_csv_to_npy.py \
    --in_root /path/to/openface_results/ \
    --raw_out gaze_raw/ \
    --norm_out gaze_norm/
```

This creates:
- `gaze_raw/`: Raw gaze coordinates
- `gaze_norm/`: Head-pose normalized gaze
- `index.csv`: Dataset statistics

### Step 3: Train Self-Supervised Gaze Encoder

Train the gaze forecasting model:

```bash
cd self_supervised_gaze

python train.py --config configs/preset_small.yaml
```

**Configuration options:**
- `preset_small.yaml`: 256-dim model, 6 encoder + 4 decoder layers
- `preset_base.yaml`: 512-dim model
- `preset_large.yaml`: 768-dim model

**Key hyperparameters** (edit YAML file):
- `data.past_len`: Input sequence length (frames)
- `data.future_len`: Prediction sequence length (frames)
- `model.patch_size`: Temporal patch size
- `optim.lr`: Learning rate
- `optim.batch_size`: Batch size

Checkpoints are saved to `checkpoints/<run_name>/`.

### Step 4: Prepare Emotion Dataset

Build a dataset pairing gaze windows with VAD labels:

```bash
cd emotion_prediction

python build_eyegaze_emotion_dataset.py \
    --eye-dir /path/to/openface_results/ \
    --emotion-json /path/to/vad_labels.json \
    --win-sec 5.0 \
    --stride-sec 2.5 \
    --seed 123
```

**Input format for VAD labels** (`arousal_valence_dominance.json`):
```json
{
  "video_id": [
    [start_time, end_time, valence, arousal, dominance],
    [start_time, end_time, valence, arousal, dominance],
    ...
  ],
  ...
}
```

This creates `data/5.0_seconds/seed_123/train.csv` and `test.csv`.

### Step 5: Train Emotion Prediction Model

Fine-tune the encoder with emotion prediction head:

```bash
cd emotion_prediction

python glass.py \
    --data_folder data/5.0_seconds/seed_123/ \
    --ckpt-loc ../self_supervised_gaze/checkpoints/small_5_sec_5_sec/best.ckpt \
    --epochs 15 \
    --batch-size 128 \
    --output-dir results/ \
    --save_model
```

**Key arguments:**
- `--ckpt-loc`: Path to pre-trained gaze encoder
- `--epochs`: Training epochs
- `--head-arch`: Prediction head architecture (`mlp`, `tcn`, `gru`, `transformer`)
- `--chunk-split`: Temporal chunking (seconds)
- `--calibrate`: Enable affine calibration on test set

Results are saved to `results/` with metrics in JSON format.

### Hyperparameter Sweeps

For systematic hyperparameter search, use the threadward framework:

```bash
cd self_supervised_gaze

python threadward_seconds_sweep.py \
    --results-folder sweep_results/
```

This trains models across different temporal window combinations.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{glass2024,
  title={GLASS: Gaze-based Learning for Affective State Sensing},
  author={Your Authors},
  journal={Your Venue},
  year={2024}
}
```

## License

See LICENSE file for details.

## Troubleshooting

### OpenFace not found
Ensure OpenFace is built and the binary path is correct:
```bash
which FeatureExtraction  # Should show OpenFace binary location
export OPENFACE_BIN=/path/to/OpenFace/build/bin/FeatureExtraction
```

### CUDA out of memory
Reduce batch size in config files or inference scripts:
```bash
python scripts/inference_pipeline.py --csv data.csv --output out.csv --batch-size 64
```

### Missing columns in CSV
OpenFace output requires gaze columns: `gaze_0_x`, `gaze_0_y`, `gaze_0_z`, `gaze_1_x`, `gaze_1_y`, `gaze_1_z`. Ensure OpenFace was run with `-gaze` flag.

### Conda environment issues
If OpenFace requires system libraries:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Contact

For questions or issues, please open a GitHub issue.

