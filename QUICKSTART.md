# GLASS Quick Start Guide

This guide provides the fastest path to running emotion prediction on your videos.

## Setup (One-time)

```bash
# 1. Create environment and install dependencies
conda create -n glass_env python=3.10
conda activate glass_env
pip install -r requirements.txt

# 2. Optional: Install threadward for batch processing
# pip install threadward

# 3. Install OpenFace (see README.md for detailed instructions)
# Ensure OpenFace binary is at: ~/OpenFace/build/bin/FeatureExtraction
```

**Note:** If you get build errors, make sure you're not trying to install optional dependencies. The core requirements should install cleanly.

## Running Inference (3 Steps)

### Step 1: Extract Eye Gaze Features

```bash
conda activate glass_env

python scripts/process_videos.py \
    --videos /path/to/your/video.mp4 \
    --output results/ \
    --openface-bin ~/OpenFace/build/bin/FeatureExtraction
```

**Output:** `results/video_name/result.csv`

### Step 2: Predict Emotions

```bash
python scripts/inference_pipeline.py \
    --csv results/video_name/result.csv \
    --output predictions.csv
```

**Output:** `predictions.csv` with columns: `timestamp`, `valence`, `arousal`, `dominance`

### Step 3: Visualize (Optional)

```bash
python scripts/visualize_eyegaze_emotion.py \
    --video /path/to/your/video.mp4 \
    --predictions predictions.csv \
    --output output_video.mp4
```

**Output:** `output_video.mp4` with emotion graphs overlaid

## Batch Processing

For multiple videos:

```bash
pip install threadward

python scripts/process_videos_batch.py \
    --videos video1.mp4 video2.mp4 video3.mp4 \
    --output results/ \
    --num-workers 4
```

## Understanding the Output

**Prediction CSV format:**
- `timestamp`: End time of 5-second analysis window (seconds)
- `valence`: Emotional valence [0, 1] (0=negative, 1=positive)
- `arousal`: Emotional arousal [0, 1] (0=calm, 1=excited)
- `dominance`: Emotional dominance [0, 1] (0=submissive, 1=dominant)

**Prediction frequency:** New prediction every 2.5 seconds using a 5-second sliding window.

## Troubleshooting

**OpenFace not found:**
```bash
export OPENFACE_BIN=/path/to/OpenFace/build/bin/FeatureExtraction
python scripts/process_videos.py --openface-bin $OPENFACE_BIN ...
```

**CUDA out of memory:**
```bash
# Use smaller batch size
python scripts/inference_pipeline.py --csv data.csv --output out.csv
# Model internally uses batch_size=256, which should fit most GPUs
```

**Missing gaze columns:**
Ensure OpenFace was run with the `-gaze` flag (this script does it automatically).

## Next Steps

- See `README.md` for complete documentation
- See `README.md` for training your own models
- See `emotion_prediction/best_model/glass.json` for model performance metrics

