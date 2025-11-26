#!/usr/bin/env python3
"""
Emotion Visualization Script.

Creates a video with emotion predictions overlaid as:
- Text showing 5-second averaged VAD values
- Three scrolling line graphs showing Valence, Arousal, Dominance over time
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import cv2
from collections import deque

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Install it with: pip install matplotlib")
    sys.exit(1)


def load_predictions(csv_path):
    """Load emotion predictions from CSV."""
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required = ["timestamp", "valence", "arousal", "dominance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {missing}")
    
    return df


def create_graph_image(values, labels, window_sec=5.0, width=800, height=200):
    """
    Create a line graph image showing recent VAD values.
    
    Args:
        values: Dict with keys "valence", "arousal", "dominance", each a list of recent values
        labels: Dict with current averaged values for each dimension
        window_sec: Time window to display
        width: Image width
        height: Image height per graph
    
    Returns:
        Image as numpy array (RGB)
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(width/100, height*3/100), dpi=100)
    fig.patch.set_facecolor('white')
    
    colors = {
        "valence": "#2E7D32",      # Green
        "arousal": "#C62828",      # Red
        "dominance": "#1565C0"     # Blue
    }
    
    dimensions = ["valence", "arousal", "dominance"]
    titles = ["Valence", "Arousal", "Dominance"]
    
    for i, (dim, title) in enumerate(zip(dimensions, titles)):
        ax = axes[i]
        
        # Get values for this dimension
        vals = values.get(dim, [])
        if not vals:
            vals = [0.5]  # Default to neutral
        
        # Plot
        x = np.arange(len(vals))
        ax.plot(x, vals, color=colors[dim], linewidth=2, label=title)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Current value text
        current_val = labels.get(dim, 0.5)
        ax.text(0.02, 0.95, f"{title}: {current_val:.3f}", 
                transform=ax.transAxes, fontsize=12, weight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        # Styling
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(1, len(vals)-1))
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i == 2:  # Last subplot
            ax.set_xlabel("Time (recent)", fontsize=10)
        else:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    
    # Convert to image
    fig.canvas.draw()
    
    # Try different matplotlib API versions
    try:
        # Newer matplotlib
        img_data = fig.canvas.buffer_rgba()
        img = np.frombuffer(img_data, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Drop alpha channel
    except AttributeError:
        try:
            # Older matplotlib
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Alternative approach
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    
    plt.close(fig)
    
    return img


def interpolate_predictions(df, fps=30):
    """
    Interpolate predictions to match video frame rate.
    
    Args:
        df: DataFrame with predictions
        fps: Video FPS
    
    Returns:
        Dictionary mapping frame_idx -> {valence, arousal, dominance}
    """
    timestamps = df["timestamp"].values
    valence = df["valence"].values
    arousal = df["arousal"].values
    dominance = df["dominance"].values
    
    # Determine frame range
    max_time = timestamps[-1]
    num_frames = int(max_time * fps) + 1
    
    # Interpolate for each frame
    frame_times = np.arange(num_frames) / fps
    
    valence_interp = np.interp(frame_times, timestamps, valence)
    arousal_interp = np.interp(frame_times, timestamps, arousal)
    dominance_interp = np.interp(frame_times, timestamps, dominance)
    
    # Create frame-wise predictions
    frame_predictions = {}
    for i in range(num_frames):
        frame_predictions[i] = {
            "valence": float(valence_interp[i]),
            "arousal": float(arousal_interp[i]),
            "dominance": float(dominance_interp[i]),
        }
    
    return frame_predictions, num_frames


def create_visualized_video(video_path, predictions_csv, output_path, 
                           window_sec=5.0, graph_height=600):
    """
    Create a video with emotion predictions visualized.
    
    Args:
        video_path: Path to input video
        predictions_csv: Path to predictions CSV
        output_path: Path to save output video
        window_sec: Time window for graphs (seconds)
        graph_height: Height of graph section (pixels)
    """
    print(f"Loading video: {video_path}")
    print(f"Loading predictions: {predictions_csv}")
    
    # Load predictions
    df = load_predictions(predictions_csv)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Interpolate predictions to frame level
    frame_predictions, pred_frames = interpolate_predictions(df, fps)
    
    # Setup output video
    output_height = height + graph_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, output_height))
    
    # History for scrolling graphs
    history_frames = int(window_sec * fps)
    value_history = {
        "valence": deque(maxlen=history_frames),
        "arousal": deque(maxlen=history_frames),
        "dominance": deque(maxlen=history_frames),
    }
    
    print(f"Creating visualization video...")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get predictions for this frame
        if frame_idx in frame_predictions:
            pred = frame_predictions[frame_idx]
        else:
            pred = {"valence": 0.5, "arousal": 0.5, "dominance": 0.5}
        
        # Update history
        value_history["valence"].append(pred["valence"])
        value_history["arousal"].append(pred["arousal"])
        value_history["dominance"].append(pred["dominance"])
        
        # Compute 5-second averages
        avg_window = min(history_frames, len(value_history["valence"]))
        avg_labels = {
            "valence": np.mean(list(value_history["valence"])[-avg_window:]),
            "arousal": np.mean(list(value_history["arousal"])[-avg_window:]),
            "dominance": np.mean(list(value_history["dominance"])[-avg_window:]),
        }
        
        # Create graph section
        graph_values = {
            "valence": list(value_history["valence"]),
            "arousal": list(value_history["arousal"]),
            "dominance": list(value_history["dominance"]),
        }
        
        graph_img = create_graph_image(
            graph_values, avg_labels, 
            window_sec=window_sec, 
            width=width, 
            height=graph_height//3
        )
        
        # Resize graph to match video width
        graph_img = cv2.resize(graph_img, (width, graph_height))
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
        
        # Combine frame and graph
        combined = np.vstack([frame, graph_img])
        
        # Write frame
        out.write(combined)
        
        frame_idx += 1
        
        # Progress
        if frame_idx % 100 == 0:
            progress = frame_idx / total_frames * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Output saved to: {output_path}")
    print(f"Total frames processed: {frame_idx}")
    print('='*60)


def main():
    parser = argparse.ArgumentParser(
        description="Create a video with emotion predictions visualized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Create visualization video
  python visualize_eyegaze_emotion.py \\
      --video input_video.mp4 \\
      --predictions predictions.csv \\
      --output visualization.mp4
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to emotion predictions CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output video"
    )
    
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Time window for graphs in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--graph-height",
        type=int,
        default=600,
        help="Height of graph section in pixels (default: 600)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.video).exists():
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)
    
    if not Path(args.predictions).exists():
        print(f"ERROR: Predictions CSV not found: {args.predictions}")
        sys.exit(1)
    
    # Create visualization
    create_visualized_video(
        video_path=args.video,
        predictions_csv=args.predictions,
        output_path=args.output,
        window_sec=args.window_sec,
        graph_height=args.graph_height
    )


if __name__ == "__main__":
    main()

