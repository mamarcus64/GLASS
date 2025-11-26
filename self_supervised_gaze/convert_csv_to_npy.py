import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

REQUIRED_RAW = ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_1_x","gaze_1_y","gaze_1_z"]
REQUIRED_NORM = [c + "_z" for c in REQUIRED_RAW]  # normalized columns end with _z

def infer_video_name(csv_path: Path) -> str:
    # threadward_results/{video_name}/result.csv  → video_name
    return csv_path.parent.name

def convert_one(csv_path: Path, raw_out: Path, norm_out: Path, frames_out: Path | None = None):
    df = pd.read_csv(csv_path)

    missing_raw = [c for c in REQUIRED_RAW if c not in df.columns]
    missing_norm = [c for c in REQUIRED_NORM if c not in df.columns]

    if missing_raw:
        raise ValueError(f"{csv_path} missing raw columns: {missing_raw}")
    if missing_norm:
        raise ValueError(f"{csv_path} missing normalized columns: {missing_norm}")

    raw = df[REQUIRED_RAW].to_numpy(dtype=np.float32)     # (T, 6)
    norm = df[REQUIRED_NORM].to_numpy(dtype=np.float32)   # (T, 6)

    vid = infer_video_name(csv_path)
    raw_path = raw_out / f"{vid}.npy"
    norm_path = norm_out / f"{vid}.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(raw_path, raw)
    np.save(norm_path, norm)

    # Optional: save frame numbers if present
    if frames_out is not None and "frame" in df.columns:
        frames_out.mkdir(parents=True, exist_ok=True)
        np.save(frames_out / f"{vid}.npy", df["frame"].to_numpy(dtype=np.int64))

    # Basic stats
    t = raw.shape[0]
    fin_raw = np.isfinite(raw).all(axis=1).sum()
    fin_norm = np.isfinite(norm).all(axis=1).sum()
    return vid, t, fin_raw, fin_norm

def main():
    ap = argparse.ArgumentParser(description="Convert threadward gaze CSVs to raw/normalized .npy")
    ap.add_argument("--in_root", type=str, default="/data2/mjma/voices/threadward_results_normalized",
                    help="Root containing {video_name}/result.csv")
    ap.add_argument("--raw_out", type=str, default="/data2/mjma/voices/self_supervised_gaze/gaze_raw",
                    help="Output dir for raw arrays")
    ap.add_argument("--norm_out", type=str, default="/data2/mjma/voices/self_supervised_gaze/gaze_norm",
                    help="Output dir for normalized arrays")
    ap.add_argument("--save_frames", action="store_true",
                    help="If set, also save frame numbers (when available)")
    ap.add_argument("--frames_out", type=str, default="/data2/mjma/voices/self_supervised_gaze/frames",
                    help="Output dir for frame arrays (used only if --save_frames)")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    raw_out = Path(args.raw_out)
    norm_out = Path(args.norm_out)
    frames_out = Path(args.frames_out) if args.save_frames else None

    csvs = sorted(in_root.glob("*/result.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs found under {in_root}/**/result.csv")
    
    print(f"Found {len(csvs)} CSV(s). Converting...")
    rows = []
    for p in tqdm(csvs):
        vid, t, fin_raw, fin_norm = convert_one(p, raw_out, norm_out, frames_out)
        rows.append((vid, t, fin_raw, fin_norm))

    # Write an index for convenience
    idx_path = Path("/data2/mjma/voices/self_supervised_gaze") / "index.csv"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["video","T","finite_raw","finite_norm"]).to_csv(idx_path, index=False)
    print(f"Done. Wrote index to {idx_path}")

if __name__ == "__main__":
    main()
