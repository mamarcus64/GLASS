#!/usr/bin/env python3
"""
Sequential OpenFace processing script.
Process multiple videos sequentially and extract facial features using OpenFace.
"""

import os
import sys
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaze_detection import extract_facial_features


def process_videos(video_paths, output_dir, openface_bin, openface_model_dir=None):
    """
    Process a list of videos sequentially with OpenFace.
    
    Args:
        video_paths: List of paths to video files
        output_dir: Directory to save results
        openface_bin: Path to OpenFace FeatureExtraction binary
        openface_model_dir: Optional path to OpenFace model directory
    """
    # Setup models dictionary
    models = {
        "openface_bin": openface_bin,
        "pyfeat_detector": None,
        "openface_model_dir": openface_model_dir
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(video_paths)} videos sequentially...")
    
    for video_path in video_paths:
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"[WARNING] Video not found: {video_path}, skipping...")
            continue
        
        # Create output folder for this video
        video_name = video_path.stem
        save_folder = output_dir / video_name
        save_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {video_path.name}")
        print(f"Output folder: {save_folder}")
        
        try:
            result_path = extract_facial_features(
                models, 
                save_folder=save_folder, 
                video_path=video_path, 
                output_format="csv"
            )
            print(f"✓ Success: {result_path}")
        except Exception as e:
            print(f"✗ Error processing {video_path.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process videos with OpenFace sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  python process_videos.py --videos /path/to/video.mp4 --output results/

  # Process multiple videos
  python process_videos.py --videos video1.mp4 video2.mp4 video3.mp4 --output results/
  
  # Specify custom OpenFace paths
  python process_videos.py --videos *.mp4 --output results/ \\
      --openface-bin /path/to/OpenFace/build/bin/FeatureExtraction
        """
    )
    
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Path(s) to video file(s) to process"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--openface-bin",
        type=str,
        default=None,
        help="Path to OpenFace FeatureExtraction binary (default: auto-detect)"
    )
    
    parser.add_argument(
        "--openface-model-dir",
        type=str,
        default=None,
        help="Path to OpenFace model directory (optional)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect OpenFace binary if not specified
    if args.openface_bin is None:
        # Try common locations
        possible_paths = [
            Path.home() / "OpenFace/build/bin/FeatureExtraction",
            Path("/usr/local/bin/FeatureExtraction"),
            Path("../OpenFace/build/bin/FeatureExtraction"),
        ]
        
        for p in possible_paths:
            if p.exists():
                args.openface_bin = str(p)
                print(f"Auto-detected OpenFace binary: {args.openface_bin}")
                break
        
        if args.openface_bin is None:
            print("ERROR: Could not find OpenFace binary. Please specify with --openface-bin")
            sys.exit(1)
    
    # Validate OpenFace binary exists
    if not Path(args.openface_bin).exists():
        print(f"ERROR: OpenFace binary not found: {args.openface_bin}")
        sys.exit(1)
    
    # Process videos
    process_videos(
        video_paths=args.videos,
        output_dir=args.output,
        openface_bin=args.openface_bin,
        openface_model_dir=args.openface_model_dir
    )


if __name__ == "__main__":
    main()

