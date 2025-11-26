#!/usr/bin/env python3
"""
Emotion Prediction Inference Pipeline.

This script performs emotion prediction (VAD: Valence, Arousal, Dominance) 
from eye gaze data extracted by OpenFace.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_path):
    """
    Load the pre-trained emotion prediction model.
    
    Args:
        model_path: Path to model checkpoint (.ckpt file)
    
    Returns:
        model: Loaded PyTorch model
        input_standardizer: Tuple of (mu, sd) for input normalization
        device: PyTorch device
    """
    # Set seeds for deterministic behavior (ensures lazy initialization is consistent)
    import random
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load checkpoint - handle numpy compatibility issues
    import sys
    import pickle
    
    # Add numpy compatibility for older checkpoints
    try:
        import numpy._core as _np_core
    except ImportError:
        import numpy.core as _np_core
        sys.modules['numpy._core'] = _np_core
        sys.modules['numpy._core.multiarray'] = _np_core.multiarray
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Parse checkpoint structure
    if not isinstance(ckpt, dict):
        raise ValueError("Unexpected checkpoint format")
    
    # Extract components
    from emotion_prediction.glass import FinetuneRegressor
    
    backbone_cfg = ckpt['backbone_config']
    head_kwargs = ckpt['head_kwargs'].copy()
    state_dict = ckpt['state_dict']
    input_std = ckpt.get('input_standardizer', {})
    
    print(f"Model type: {ckpt.get('type', 'unknown')}")
    
    # Fix parameter name mismatch (chunk_splits_seconds -> chunk_splits)
    if 'chunk_splits_seconds' in head_kwargs:
        head_kwargs['chunk_splits'] = head_kwargs.pop('chunk_splits_seconds')
    
    # Reconstruct model
    model = FinetuneRegressor(
        cfg=backbone_cfg,
        **head_kwargs
    )
    
    # Get input standardizer
    if 'mu' in input_std and 'sd' in input_std:
        mu = input_std['mu']
        sd = input_std['sd']
    else:
        print("Warning: No input standardizer found, using identity")
        mu = np.zeros((1, 1, 6), dtype=np.float32)
        sd = np.ones((1, 1, 6), dtype=np.float32)
    
    # Move model to device FIRST
    model.to(device)
    model.eval()
    
    # Force initialization of lazy components with a dummy forward pass
    # This ensures all layers exist before loading weights
    print("Initializing model layers...")
    with torch.no_grad():
        Tp = int(backbone_cfg["data"]["past_len"])  # 150 frames for 5 sec at 30 fps
        D_in = int(backbone_cfg["model"]["D_in"])    # 6 gaze dimensions
        dummy_input = torch.zeros(1, Tp, D_in, dtype=torch.float32).to(device)
        _ = model(dummy_input)
    
    # Now load the trained weights (will overwrite the random lazy initialization)
    print("Loading trained weights...")
    missing = model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys:
        print(f"Missing keys: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"Unexpected keys: {missing.unexpected_keys}")
    
    return model, (mu, sd), device


def load_openface_csv(csv_path, window_sec=5.0, stride_sec=2.5, fps=30):
    """
    Load OpenFace CSV and extract gaze data windows.
    
    Args:
        csv_path: Path to OpenFace result CSV
        window_sec: Window size in seconds
        stride_sec: Stride between windows in seconds
        fps: Frames per second (default: 30)
    
    Returns:
        windows: List of numpy arrays, each (T, 6)
        timestamps: List of end timestamps for each window
        df: Original dataframe
    """
    print(f"Loading OpenFace data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Gaze columns
    gaze_cols = [
        "gaze_0_x", "gaze_0_y", "gaze_0_z",
        "gaze_1_x", "gaze_1_y", "gaze_1_z",
    ]
    
    # Check columns exist
    missing = [c for c in gaze_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing gaze columns: {missing}")
    
    # Extract gaze data
    gaze_data = df[gaze_cols].to_numpy(dtype=np.float32)
    
    # Get timestamps (if available)
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy()
    else:
        # Assume 30 fps
        timestamps = np.arange(len(gaze_data)) / fps
    
    # Create windows
    window_frames = int(window_sec * fps)
    stride_frames = int(stride_sec * fps)
    
    windows = []
    window_timestamps = []
    
    for end_idx in range(window_frames, len(gaze_data) + 1, stride_frames):
        start_idx = end_idx - window_frames
        window = gaze_data[start_idx:end_idx]
        
        # Check if window has valid data (not all zeros)
        if not np.all(np.abs(window) < 1e-8):
            windows.append(window)
            window_timestamps.append(timestamps[end_idx - 1])
    
    print(f"Extracted {len(windows)} windows (window_sec={window_sec}, stride_sec={stride_sec})")
    
    return windows, window_timestamps, df


def predict_emotions(model, gaze_windows, input_standardizer, device, batch_size=256):
    """
    Predict emotions from gaze windows.
    
    Args:
        model: Pre-trained model
        gaze_windows: List of gaze data arrays, each (T, 6)
        input_standardizer: Tuple of (mu, sd) for input normalization
        device: PyTorch device
        batch_size: Batch size for inference
    
    Returns:
        predictions: numpy array of shape (N, 3) with VAD predictions
    """
    print(f"Running emotion prediction on {len(gaze_windows)} windows...")
    
    mu, sd = input_standardizer
    
    # Standardize and batch process
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(gaze_windows), batch_size):
            # Get batch of windows
            batch_windows = gaze_windows[i:i+batch_size]
            
            # Ensure all windows have the same length (pad/trim if needed)
            target_len = 150  # 5 seconds * 30 fps
            processed = []
            for w in batch_windows:
                if len(w) < target_len:
                    # Pad by repeating last frame
                    pad_len = target_len - len(w)
                    w = np.vstack([w, np.tile(w[-1:], (pad_len, 1))])
                elif len(w) > target_len:
                    # Trim to target length (take most recent)
                    w = w[-target_len:]
                processed.append(w)
            
            # Stack into batch
            batch_np = np.stack(processed, axis=0).astype(np.float32)  # (B, T, 6)
            
            # Standardize
            batch_np = (batch_np - mu) / sd
            
            # Convert to tensor and ensure contiguity
            batch_tensor = torch.from_numpy(batch_np).to(device).contiguous()
            
            # Forward pass
            pred = model(batch_tensor)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    print(f"Predictions shape: {predictions.shape}")
    
    return predictions


def save_results(predictions, timestamps, output_path):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: numpy array of shape (N, 3) with VAD predictions
        timestamps: List of timestamps for each prediction
        output_path: Path to save results
    """
    # Create dataframe
    results_df = pd.DataFrame(predictions, columns=["valence", "arousal", "dominance"])
    results_df["timestamp"] = timestamps
    
    # Reorder columns
    results_df = results_df[["timestamp", "valence", "arousal", "dominance"]]
    
    # Save
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY:")
    print("="*60)
    print(f"Number of predictions: {len(predictions)}")
    print(f"Time range: {timestamps[0]:.2f}s - {timestamps[-1]:.2f}s")
    print("\nMean VAD:")
    print(f"  Valence:   {predictions[:, 0].mean():.3f} (±{predictions[:, 0].std():.3f})")
    print(f"  Arousal:   {predictions[:, 1].mean():.3f} (±{predictions[:, 1].std():.3f})")
    print(f"  Dominance: {predictions[:, 2].mean():.3f} (±{predictions[:, 2].std():.3f})")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Predict emotions (VAD) from OpenFace gaze data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on a single video's OpenFace output
  python inference_pipeline.py --csv results/video1/result.csv --output predictions.csv
  
  # Use custom model and window parameters
  python inference_pipeline.py --csv results/video1/result.csv \\
      --model ../emotion_prediction/best_model/model.ckpt \\
      --window-sec 5.0 --stride-sec 2.5 \\
      --output predictions.csv
        """
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to OpenFace result CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save prediction results"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (default: use best_model)"
    )
    
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Window size in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=2.5,
        help="Stride between windows in seconds (default: 2.5)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frames per second (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect model if not specified
    if args.model is None:
        script_dir = Path(__file__).parent
        default_model = script_dir.parent / "emotion_prediction/best_model/model.ckpt"
        
        if default_model.exists():
            args.model = str(default_model)
            print(f"Using default model: {args.model}")
        else:
            print("ERROR: Could not find default model. Please specify with --model")
            sys.exit(1)
    
    # Validate inputs
    if not Path(args.csv).exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"ERROR: Model checkpoint not found: {args.model}")
        sys.exit(1)
    
    # Load model
    model, input_standardizer, device = load_model(args.model)
    
    # Load OpenFace data
    gaze_windows, timestamps, df = load_openface_csv(
        args.csv,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        fps=args.fps
    )
    
    if len(gaze_windows) == 0:
        print("ERROR: No valid windows extracted from CSV")
        sys.exit(1)
    
    # Run prediction
    predictions = predict_emotions(model, gaze_windows, input_standardizer, device)
    
    # Save results
    save_results(predictions, timestamps, args.output)


if __name__ == "__main__":
    main()

