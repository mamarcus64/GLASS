#!/usr/bin/env python3
"""
Batch OpenFace processing script using threadward.
Process multiple videos in parallel across multiple GPUs.
"""

import os
import sys
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import threadward
except ImportError:
    print("ERROR: threadward not installed. Install it with: pip install threadward")
    sys.exit(1)

from gaze_detection import extract_facial_features


class VideoProcessingRunner(threadward.Threadward):
    """Threadward runner for parallel video processing with OpenFace."""
    
    def __init__(self, openface_bin, openface_model_dir=None, num_workers=4, 
                 num_gpus_per_worker=1, debug=False, results_folder="openface_results"):
        super().__init__(debug=debug, results_folder=results_folder)
        
        self.openface_bin = openface_bin
        self.openface_model_dir = openface_model_dir
        
        self.set_constraints(
            SUCCESS_CONDITION="NO_ERROR_AND_VERIFY",
            OUTPUT_MODE="LOG_FILE_ONLY",
            NUM_WORKERS=num_workers,
            NUM_GPUS_PER_WORKER=num_gpus_per_worker,
            AVOID_GPUS=None,
            INCLUDE_GPUS=None,  # Use all available GPUs
            FAILURE_HANDLING="PRINT_FAILURE_AND_CONTINUE",
            TASK_FOLDER_LOCATION="VARIABLE_SUBFOLDER",
            EXISTING_FOLDER_HANDLING="VERIFY",
            TASK_TIMEOUT=-1  # No timeout
        )
    
    def task_method(self, variables, task_folder, log_file):
        """Process a single video."""
        video_path = variables["video_path"]
        
        # Setup models dictionary
        models = {
            "openface_bin": self.openface_bin,
            "pyfeat_detector": None,
            "openface_model_dir": self.openface_model_dir
        }
        
        # Extract features
        result_path = extract_facial_features(
            models,
            save_folder=task_folder,
            video_path=video_path,
            output_format="csv"
        )
        
        print(f"Processed {Path(video_path).name} -> {result_path}")
    
    def verify_task_success(self, variables, task_folder, log_file):
        """Verify that the result CSV was created."""
        result_csv = Path(task_folder) / "result.csv"
        return result_csv.exists()
    
    def setup_variable_set(self, variable_set):
        """Setup variables from video list."""
        # This will be populated dynamically
        pass


def process_videos_batch(video_paths, output_dir, openface_bin, openface_model_dir=None,
                        num_workers=4, num_gpus_per_worker=1, debug=False):
    """
    Process videos in parallel using threadward.
    
    Args:
        video_paths: List of paths to video files
        output_dir: Directory to save results
        openface_bin: Path to OpenFace FeatureExtraction binary
        openface_model_dir: Optional path to OpenFace model directory
        num_workers: Number of parallel workers
        num_gpus_per_worker: Number of GPUs per worker
        debug: Enable debug mode
    """
    print(f"Setting up batch processing for {len(video_paths)} videos...")
    print(f"Workers: {num_workers}, GPUs per worker: {num_gpus_per_worker}")
    
    # Create runner
    runner = VideoProcessingRunner(
        openface_bin=openface_bin,
        openface_model_dir=openface_model_dir,
        num_workers=num_workers,
        num_gpus_per_worker=num_gpus_per_worker,
        debug=debug,
        results_folder=output_dir
    )
    
    # Add videos as variables
    video_nicknames = []
    for video_path in video_paths:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"[WARNING] Video not found: {video_path}, skipping...")
            continue
        
        nickname = video_path.stem
        video_nicknames.append(nickname)
        runner.variable_set.add_variable(
            "video_path",
            values=[str(video_path.absolute())],
            nicknames=[nickname]
        )
    
    if not video_nicknames:
        print("ERROR: No valid videos to process")
        sys.exit(1)
    
    print(f"Processing videos: {', '.join(video_nicknames)}")
    
    # Run processing
    runner.run()
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process videos with OpenFace in parallel using threadward",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process multiple videos in parallel (4 workers, 1 GPU each)
  python process_videos_batch.py --videos video1.mp4 video2.mp4 video3.mp4 \\
      --output results/ --num-workers 4
  
  # Process with 2 workers, no GPUs
  python process_videos_batch.py --videos *.mp4 --output results/ \\
      --num-workers 2 --num-gpus-per-worker 0
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
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--num-gpus-per-worker",
        type=int,
        default=1,
        help="Number of GPUs per worker (default: 1, use 0 for CPU only)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Auto-detect OpenFace binary if not specified
    if args.openface_bin is None:
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
    process_videos_batch(
        video_paths=args.videos,
        output_dir=args.output,
        openface_bin=args.openface_bin,
        openface_model_dir=args.openface_model_dir,
        num_workers=args.num_workers,
        num_gpus_per_worker=args.num_gpus_per_worker,
        debug=args.debug
    )


if __name__ == "__main__":
    main()

