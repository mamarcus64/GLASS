# build_eyegaze_emotion_dataset.py
from __future__ import annotations
# ===================================

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass(frozen=True)
class BuildConfig:
    win_sec: float = 5.0                # history length (seconds)
    stride_sec: float = 2.5            # window end-time stride (seconds)
    seed: int = 123                     # for subject split
    gaze_cols: Tuple[str, ...] = (
        "gaze_0_x", "gaze_0_y", "gaze_0_z",
        "gaze_1_x", "gaze_1_y", "gaze_1_z",
    )
    face_cols: Tuple[str, ...] = (
        "AU01_r", "AU02_r", "AU04_r",        # brows
        "AU23_r",                            # lips
        "AU45_c",                            # blinks
    )
    timestamp_col: str = "timestamp"    # seconds
    csv_name: str = "result.csv"        # per-video file name (changed)
    # If True, drop windows with zero frames (e.g., before first timestamp)
    drop_empty_windows: bool = True
    # Output CSV settings
    output_dir: str = "."  # directory to save output files
    # Undersampling settings - fixed values
    std_threshold: float = 2.0          # standard deviation threshold for undersampling (fixed at 2.0)
    within_std_ratio: float = 1.0       # ratio of within-std to outside-std samples during training (fixed at 1.0)
    debug_mode_on: bool = False  # enable debug mode for faster processing


def _subject_of(video_id: str) -> str:
    # video_id format: "{subject_num}.{video_num}"
    return video_id.split(".")[0]


def _load_emotions(json_path: Path) -> Dict[str, List[List[float]]]:
    with json_path.open("r") as f:
        data = json.load(f)
    return data


def _emotion_lookup_fast(emotion_lists: List[List[float]]):
    """
    Build a fast lookup f(t) -> Optional[np.ndarray(3,)] for V-A-D if t is inside any segment.
    Assumes segments are mostly non-overlapping; if overlaps exist, prefers the segment with
    the latest start <= t.
    """
    starts, stops, vad = [], [], []
    for rec in emotion_lists:
        if not rec or len(rec) < 5:
            continue
        s, e = float(rec[0]), float(rec[1])
        if e <= s:
            continue
        v, a, d = float(rec[-3]), float(rec[-2]), float(rec[-1])
        starts.append(s); stops.append(e); vad.append((v, a, d))
    if not starts:
        # No labels: always None
        def lookup_none(t: float): return None
        return lookup_none

    starts = np.asarray(starts, dtype=np.float64)
    stops  = np.asarray(stops,  dtype=np.float64)
    vad    = np.asarray(vad,    dtype=np.float32)
    order = np.argsort(starts, kind="mergesort")
    starts, stops, vad = starts[order], stops[order], vad[order]

    def lookup(t: float) -> Optional[np.ndarray]:
        # index of last segment with start <= t
        idx = np.searchsorted(starts, t, side="right") - 1
        if idx >= 0 and t < stops[idx]:
            return vad[idx]
        return None

    return lookup


def _load_gaze_csv(csv_path: Path, cfg: BuildConfig) -> Optional[pd.DataFrame]:
    try:
        usecols = [cfg.timestamp_col] + list(cfg.gaze_cols) + list(cfg.face_cols)
        dtypes = {cfg.timestamp_col: "float64"}
        dtypes.update({c: "float32" for c in cfg.gaze_cols})
        dtypes.update({c: "float32" for c in cfg.face_cols})
        df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, engine="c", low_memory=False)
        df.dropna(subset=[cfg.timestamp_col], inplace=True)
        df.sort_values(cfg.timestamp_col, inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        print(f"[error] Failed to load {csv_path}: {e}")
        return None


def _is_repetitive_gaze(gaze_window: np.ndarray, num_sampled: int = 5, repetitive_fraction: float = 0.8) -> bool:
    """
    Check if eye gaze data is repetitive by sampling frames.
    
    Args:
        gaze_window: np.ndarray of shape [n_frames, 6] containing eye gaze data
        num_sampled: int, number of frames to sample (default 20)
        repetitive_fraction: float, fraction of sampled frames that must be identical to consider repetitive (default 0.8)
    
    Returns:
        bool: True if the data is repetitive, False otherwise
    """
    if gaze_window.size == 0:
        return False
    
    n_frames = gaze_window.shape[0]
    if n_frames == 1:
        return False
    
    # Sample frames (or all frames if less than num_sampled)
    sample_size = min(num_sampled, n_frames)
    if sample_size < num_sampled:
        # If we have fewer than num_sampled frames, check if they're all the same
        return np.allclose(gaze_window[0], gaze_window[1:])
    
    # Sample random frame indices
    sample_indices = np.random.choice(n_frames, size=num_sampled, replace=False)
    
    # Count how many sampled frames are identical to the first frame
    identical_count = 0
    for idx in sample_indices:
        if np.allclose(gaze_window[0], gaze_window[idx]):
            identical_count += 1
    
    # Check if the fraction of identical frames meets the threshold
    threshold_count = int(repetitive_fraction * num_sampled)
    return identical_count >= threshold_count


def _apply_undersampling(
    X_list: List[np.ndarray],
    Y_list: List[Optional[np.ndarray]],
    Z_list: List[Optional[np.ndarray]],
    meta_list: List[Tuple[str, float]],
    std_threshold: float = 2.0,
    tail_fraction: float = 1/3,
    bin_num: int = 100,
    seed: Optional[int] = 42,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[Optional[np.ndarray]], List[Tuple[str, float]]]:
    """
    Apply undersampling based on V-A-D distribution using z-scores and binning.

    Args:
        X_list: List of gaze windows
        Y_list: List of emotion labels (V-A-D)
        Z_list: List of face features (brows, lips, blinking)
        meta_list: List of metadata (video_id, timestamp)
        std_threshold: Z-score threshold for separating bulk vs. tails (default 2.0)
        tail_fraction: Desired fraction of tail samples in final dataset (default 1/3)
        bin_num: Number of bins for bulk sampling (default 100)
        seed: Random seed for reproducibility

    Returns:
        Undersampled X_list, Y_list, Z_list, and meta_list
    """
    if std_threshold < 0:
        # No undersampling
        return X_list, Y_list, Z_list, meta_list

    # Filter out samples without labels
    labeled_indices = [i for i, y in enumerate(Y_list) if y is not None]
    if not labeled_indices:
        print("[warn] No labeled samples found for undersampling")
        return X_list, Y_list, Z_list, meta_list

    # Collect VAD values
    vad_values = np.array([Y_list[i] for i in labeled_indices if Y_list[i] is not None])
    if vad_values.size == 0:
        print("[warn] No valid VAD values found for undersampling")
        return X_list, Y_list, Z_list, meta_list

    # Compute distances from center
    center = vad_values.mean(axis=0)
    dist = np.linalg.norm(vad_values - center, axis=1)

    # Z-scores of distances
    z_scores = (dist - dist.mean()) / (dist.std() + 1e-8)

    # Split bulk/tails
    bulk_mask = z_scores <= std_threshold
    tail_mask = ~bulk_mask

    bulk_indices = np.array(labeled_indices)[bulk_mask]
    tail_indices = np.array(labeled_indices)[tail_mask]

    n_tail = len(tail_indices)
    if n_tail == 0:
        print("[warn] No tail samples found, returning original dataset")
        return X_list, Y_list, Z_list, meta_list

    final_size = int(round(n_tail / tail_fraction))
    n_bulk = max(final_size - n_tail, 0)

    print(f"[undersampling] Bulk samples available: {len(bulk_indices)}")
    print(f"[undersampling] Tail samples: {len(tail_indices)}")
    print(f"[undersampling] Target final size: {final_size} "
          f"({n_bulk} bulk + {n_tail} tails, tail_fraction={tail_fraction:.3f})")

    # --- bulk binning ---
    bulk_z = z_scores[bulk_mask]
    bins = np.linspace(bulk_z.min(), std_threshold, bin_num + 1)
    bin_ids = np.searchsorted(bins, bulk_z, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, bin_num - 1)

    counts = np.bincount(bin_ids, minlength=bin_num)

    # --- allocation ---
    base = n_bulk // bin_num
    alloc = np.minimum(counts, base)

    leftover = n_bulk - alloc.sum()
    rng = np.random.default_rng(seed)
    if leftover > 0:
        available = counts - alloc
        if available.sum() > 0:
            probs = available / available.sum()
            extra_bins = rng.choice(bin_num, size=leftover, p=probs)
            np.add.at(alloc, extra_bins, 1)
            alloc = np.minimum(counts, alloc)

    # --- sampling ---
    chosen_bulk = []
    for b in range(bin_num):
        if alloc[b] > 0:
            bin_idxs = bulk_indices[bin_ids == b]
            chosen_bulk.extend(rng.choice(bin_idxs, size=alloc[b], replace=False))

    # Combine bulk + tails
    selected_indices = sorted(chosen_bulk + tail_indices.tolist())

    # Apply selection
    X_out = [X_list[i] for i in selected_indices]
    Y_out = [Y_list[i] for i in selected_indices]
    Z_out = [Z_list[i] for i in selected_indices] if Z_list else []
    meta_out = [meta_list[i] for i in selected_indices]

    print(f"[undersampling] Final dataset size: {len(X_out)} samples")

    return X_out, Y_out, Z_out, meta_out


def windowize_video_fast(
    video_id: str,
    df_gaze: pd.DataFrame,
    emotions: List[List[float]],
    cfg: BuildConfig,
):
    """
    Fast windowizer (no resampling):
      - Precomputes vector of window end times t_ends.
      - Uses np.searchsorted to slice raw frames within [t_end - win_sec, t_end).
      - Each window is variable-length: shape [n_frames, 6].
      - Label is V-A-D at window END time, or None.
      - Filters out windows with repetitive eye gaze data (>90% same values).
    Returns:
      X_list: List[np.ndarray (n_frames, 6)]
      Z_list: List[np.ndarray (n_frames, 6)]
      Y_list: List[Optional[np.ndarray(3,)]]
      T_ends: np.ndarray [N] window end times
    """
    t = df_gaze[cfg.timestamp_col].to_numpy(dtype=np.float64)
    if t.size == 0:
        return [], [], np.zeros((0,), dtype=np.float64)

    gaze_mat = df_gaze[list(cfg.gaze_cols)].to_numpy(dtype=np.float32)
    face_mat = df_gaze[list(cfg.face_cols)].to_numpy(dtype=np.float32)

    t_min = float(t.min())
    t_max = float(t.max())
    # Number of windows (inclusive of t_min)
    n_steps = int(math.floor((t_max - t_min) / cfg.stride_sec)) + 1
    t_ends = t_min + np.arange(n_steps, dtype=np.float64) * cfg.stride_sec
    t_starts = t_ends - cfg.win_sec

    # Precompute slice indices for all windows
    left_idx = np.searchsorted(t, t_starts, side="left")
    right_idx = np.searchsorted(t, t_ends, side="left")  # exclude frames at exactly t_end

    lookup = _emotion_lookup_fast(emotions)

    X_list: List[np.ndarray] = []
    Y_list: List[Optional[np.ndarray]] = []
    Z_list: List[Optional[np.ndarray]] = []  # for face features
    filtered_windows = 0

    for li, ri, te in zip(left_idx, right_idx, t_ends):
        if ri <= li:
            if cfg.drop_empty_windows:
                continue
            # Keep an empty window (0x6) if desired
            Xw = np.zeros((0, len(cfg.gaze_cols)), dtype=np.float32)
            Zw = np.zeros((0, len(cfg.face_cols)), dtype=np.float32)
        else:
            # Slice without copy when possible (view). `.copy()` keeps it independent.
            Xw = gaze_mat[li:ri]  # shape [n_frames, 6]
            Zw = face_mat[li:ri]  # shape [n_frames, 6]

            # Filter out windows with repetitive eye gaze data
            if _is_repetitive_gaze(Xw):
                filtered_windows += 1
                continue
        
        X_list.append(Xw)
        Z_list.append(Zw)
        Y_list.append(lookup(float(te)))

    return X_list, Y_list, Z_list, t_ends


def save_dataset_to_csv(
    X_train: List[np.ndarray], 
    Y_train: List[Optional[np.ndarray]], 
    meta_train: List[Tuple[str, float]],
    face_train: List[Optional[np.ndarray]],
    X_test: List[np.ndarray], 
    Y_test: List[Optional[np.ndarray]], 
    meta_test: List[Tuple[str, float]],
    face_test: List[Optional[np.ndarray]],
    cfg: BuildConfig,
    output_dir: Path,
):
    """
    Save the dataset to CSV files (metadata, face features, labels) and separate .npy files (gaze features).
    
    Args:
        X_train, Y_train, meta_train, face_train: Training data
        X_test, Y_test, meta_test, face_test: Testing data
        cfg: Build configuration
        output_dir: Directory to save output files
    """
    def create_dataset_df(X_list, Y_list, meta_list, face_list):
        data = []
        gaze_features_list = []  # Store gaze features separately
        
        for i, (video_id, timestamp) in enumerate(meta_list):
            if i < len(X_list) and X_list[i].size > 0:
                X_window = X_list[i]
                
                # 30 FPS, so should be exactly (30 * win_sec) frames... sometimes not quite for starting or ending.
                expected_frames = int(30 * cfg.win_sec)
                if not (expected_frames - 2 < X_window.shape[0] < expected_frames + 2):
                    # print(f"weird number of n_frames: {X_window.shape[0]}")
                    continue
                
                # if more than expected_frames, take the last expected_frames frames
                if X_window.shape[0] > expected_frames:
                    X_window = X_window[-expected_frames:]
                # if less than expected_frames, pad the beginning with the first frame
                while X_window.shape[0] < expected_frames:
                    X_window = np.vstack([X_window[0], X_window])
                
                # Store gaze features separately (will be saved as .npy)
                gaze_features_list.append(X_window.flatten())
                
                row = {
                    'video_id': video_id,
                    'window_end_time': timestamp
                }
                
                # Add face features
                # {'blink_rate': 0.0, 'blink_mean_max_ratio': 0.0, 'brow_raise_mean_max_ratio': np.float32(0.059300005), 'brow_lower_mean_max_ratio': np.float32(0.03384615), 'lip_tightness_mean_max_ratio': np.float32(0.21999998)}
                if face_list[i] is not None:
                    for j, val in enumerate(face_list[i]):
                        row[f'face_feature_{j}'] = val

                # Add labels if available
                if Y_list[i] is not None:
                    row.update({
                        'valence': Y_list[i][0],
                        'arousal': Y_list[i][1],
                        'dominance': Y_list[i][2]
                    })
                else:
                    row.update({
                        'valence': np.nan,
                        'arousal': np.nan,
                        'dominance': np.nan
                    })
                
                data.append(row)
        
        return pd.DataFrame(data), np.array(gaze_features_list)
    
    # Create train and test DataFrames and gaze features arrays
    train_df, train_gaze = create_dataset_df(X_train, Y_train, meta_train, face_train)
    test_df, test_gaze = create_dataset_df(X_test, Y_test, meta_test, face_test)

    # Save train and test CSV data (metadata, face features, labels)
    train_csv_path = output_dir / "train.csv"
    test_csv_path = output_dir / "test.csv"
    
    # Save train and test gaze features as .npy files
    train_gaze_path = output_dir / "train_gaze.npy"
    test_gaze_path = output_dir / "test_gaze.npy"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    # Save gaze features as .npy files
    np.save(train_gaze_path, train_gaze)
    np.save(test_gaze_path, test_gaze)
    
    print(f"[save] Training data saved to: {train_csv_path}")
    print(f"[save] Testing data saved to: {test_csv_path}")
    print(f"[save] Training gaze features saved to: {train_gaze_path}")
    print(f"[save] Testing gaze features saved to: {test_gaze_path}")
    print(f"[info] Train CSV shape: {train_df.shape}, Test CSV shape: {test_df.shape}")
    print(f"[info] Train gaze shape: {train_gaze.shape}, Test gaze shape: {test_gaze.shape}")
    
    print(f"\n[info] You can now load the data using:")
    print(f"      train_data = pd.read_csv('{train_csv_path}')")
    print(f"      test_data = pd.read_csv('{test_csv_path}')")
    print(f"      train_gaze = np.load('{train_gaze_path}')")
    print(f"      test_gaze = np.load('{test_gaze_path}')")


def build_dataset(
    eye_dir: Path,
    emotion_path: Path,
    cfg: BuildConfig = BuildConfig(),
    verbose: bool = True,
    debug_mode_on: bool = False
):
    # 1) Load sources
    emotions = _load_emotions(emotion_path)
    video_ids_json = set(emotions.keys())

    video_dirs = [p for p in eye_dir.iterdir() if p.is_dir()]
    
    # downsample for speed debugging
    if debug_mode_on:
        print("[debug] Debug mode ON: processing a subset of videos for speed.")
        video_dirs = random.sample(video_dirs, 10)

    video_ids_eye = set(p.name for p in video_dirs)

    shared_video_ids = sorted(video_ids_json & video_ids_eye)
    missing_in_json = sorted(video_ids_eye - video_ids_json)
    missing_in_eye = sorted(video_ids_json - video_ids_eye)

    if verbose:
        print(f"[info] EYE_GAZE_FOLDER video folders: {len(video_ids_eye)}")
        print(f"[info] Emotion JSON video_ids:       {len(video_ids_json)}")
        print(f"[info] Intersection (to process):   {len(shared_video_ids)}")
        if missing_in_json:
            print(f"[warn] {len(missing_in_json)} video_ids have eye gaze but NO emotion labels (ignored).")
        if missing_in_eye:
            print(f"[warn] {len(missing_in_eye)} video_ids have emotion labels but NO eye folder (ignored).")

    # 2) Subject split (80/20)
    subjects = sorted({_subject_of(v) for v in shared_video_ids})
    random.Random(cfg.seed).shuffle(subjects)
    split_idx = int(round(0.8 * len(subjects)))
    train_subj = set(subjects[:split_idx])
    test_subj = set(subjects[split_idx:])

    train_ids = [v for v in shared_video_ids if _subject_of(v) in train_subj]
    test_ids  = [v for v in shared_video_ids if _subject_of(v) in test_subj]

    if verbose:
        print(f"[info] Subjects total/train/test: {len(subjects)}/{len(train_subj)}/{len(test_subj)}")
        print(f"[info] Videos   train/test:       {len(train_ids)}/{len(test_ids)}")

    # 3) Process splits (fast path)
    def process_split(ids: List[str], tag: str):
        X_all: List[np.ndarray] = []
        Y_all: List[Optional[np.ndarray]] = []
        Z_all: List[Optional[np.ndarray]] = []  # for face features
        meta: List[Tuple[str, float]] = []
        n_with_label = 0
        n_total = 0

        t0 = time.perf_counter()
        for vid in tqdm(ids, desc=f"Processing {tag} videos"):
            csv_path = eye_dir / vid / cfg.csv_name
            df = _load_gaze_csv(csv_path, cfg)
            if df is None or df.empty:
                print(f"[warn] Skipping {vid}: empty or failed CSV.")
                continue

            X_list, Y_list, Z_list, t_ends = windowize_video_fast(vid, df, emotions.get(vid, []), cfg)
            if not X_list:
                continue

            X_all.extend(X_list)
            Y_all.extend(Y_list)
            Z_all.extend(Z_list)
            meta.extend([(vid, float(t)) for t in t_ends[:len(X_list)]])  # align in case empties dropped

            n_total += len(Y_list)
            n_with_label += sum(1 for y in Y_list if y is not None)

        dt = time.perf_counter() - t0
        print(f"[{tag}] TOTAL windows: {n_total}  | with label: {n_with_label} ({(100.0*n_with_label/max(n_total,1)):.1f}%)  | time: {dt:.2f}s")
        if n_total:
            print(f"[{tag}] ~{n_total/dt:.0f} windows/sec")
        return X_all, Y_all, Z_all, meta
    
    print(f"Total number of videos: {len(shared_video_ids)}")
    print(f"Total number of subjects: {len(subjects)}")

    X_train, Y_train, Z_train, meta_train = process_split(train_ids, "train")
    X_test,  Y_test,  Z_test,  meta_test  = process_split(test_ids,  "test")
    
    Y = Y_train + Y_test
    Y = [y for y in Y if y is not None]
    avg_val = np.mean([y[0] for y in Y])
    avg_arousal = np.mean([y[1] for y in Y])
    avg_dominance = np.mean([y[2] for y in Y])
    print(f"Average valence: {avg_val:.3f}, Average arousal: {avg_arousal:.3f}, Average dominance: {avg_dominance:.3f}")
    
    import pdb; pdb.set_trace()

    # Apply undersampling to both training and testing data
    print(f"[undersampling] Applying undersampling to training data...")
    X_train, Y_train, Z_train, meta_train = _apply_undersampling(
        X_train, Y_train, Z_train, meta_train, cfg.std_threshold, cfg.within_std_ratio
    )
    
    print(f"[undersampling] Applying undersampling to testing data...")
    X_test, Y_test, Z_test, meta_test = _apply_undersampling(
        X_test, Y_test, Z_test, meta_test, cfg.std_threshold, cfg.within_std_ratio
    )
    
    # Calculate face features
    face_train = []
    face_test = []

    idx_blink = cfg.face_cols.index("AU45_c")
    idx_brow = [cfg.face_cols.index("AU01_r"), cfg.face_cols.index("AU02_r")]
    idx_brow_lower = cfg.face_cols.index("AU04_r")
    idx_lip = cfg.face_cols.index("AU23_r")

    def fast_face_features(win: np.ndarray, window_duration_sec: float = cfg.win_sec):
        # win: (n_frames, n_face_features)
        if win.size == 0 or window_duration_sec <= 0:
            return {
                "blink_rate": 0.0,
                "blink_mean_max_ratio": 0.0,
                "brow_raise_mean_max_ratio": 0.0,
                "brow_lower_mean_max_ratio": 0.0,
                "lip_tightness_mean_max_ratio": 0.0,
            }
        blinks = win[:, idx_blink]
        # Find blink events (consecutive 1s)
        blink_events = []
        start = None
        for i, val in enumerate(blinks):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                blink_events.append(i - start)
                start = None
        if start is not None:
            blink_events.append(len(blinks) - start)
        blink_mean = np.mean(blink_events) if blink_events else 0.0
        blink_max = np.max(blink_events) if blink_events else 1.0
        blink_mean_max_ratio = blink_mean / max(blink_max, 1e-6)
        blink_rate = len(blink_events) / window_duration_sec

        brow_raise = win[:, idx_brow].mean(axis=1)
        brow_raise_mean_max_ratio = brow_raise.mean() / max(brow_raise.max(), 1e-6)
        brow_lower = win[:, idx_brow_lower]
        brow_lower_mean_max_ratio = brow_lower.mean() / max(brow_lower.max(), 1e-6)
        lip_tightness = win[:, idx_lip]
        lip_tightness_mean_max_ratio = lip_tightness.mean() / max(lip_tightness.max(), 1e-6)

        return [blink_rate, blink_mean_max_ratio, brow_raise_mean_max_ratio, brow_lower_mean_max_ratio, lip_tightness_mean_max_ratio]

    for win in Z_train:
        face_feats = fast_face_features(win)
        face_train.append(face_feats)

    for win in Z_test:
        face_feats = fast_face_features(win)
        face_test.append(face_feats)

    # Snapshot for debugging
    print(f"[shape] train windows: {len(X_train)} | test windows: {len(X_test)}")
    if Y_train:
        pct_labeled = 100.0 * sum(y is not None for y in Y_train) / len(Y_train)
        print(f"[label] Train labeled ratio: {pct_labeled:.1f}%")
    # Show a quick distribution of window lengths
    if X_train:
        lens = [x.shape[0] for x in X_train[:1000]]  # sample up to 1000
        print(f"[debug] Sample window lengths (frames): min={min(lens)}, p50={int(np.median(lens))}, max={max(lens)}")

    return (X_train, Y_train, meta_train, face_train), (X_test, Y_test, meta_test, face_test), cfg

# -------------------- Script entrypoint --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a dataset of eye gaze and emotion features.")
    parser.add_argument("--eye-dir", type=str, default="/data2/mjma/voices/threadward_results", help="Path to the directory containing eye gaze CSV files.")
    parser.add_argument("--emotion-json", type=str, default="/data2/mjma/voices/test_data/vad_output/arousal_valence_dominance.json", help="Path to the emotion JSON file.")
    parser.add_argument("--win-sec", type=float, default=5.0, help="History length (seconds) for each window.")
    parser.add_argument("--stride-sec", type=float, default=3.0, help="Window end-time stride (seconds).")
    parser.add_argument("--seed", type=int, default=123, help="Seed for subject split.")
    parser.add_argument("--csv-name", type=str, default="result.csv", help="Name of the per-video CSV file.")
    parser.add_argument("--drop-empty-windows", action="store_true", help="If True, drop windows with zero frames (e.g., before first timestamp).")
    parser.add_argument("--gaze-cols", type=str, nargs="+", default=["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z"], help="Names of gaze columns.")
    parser.add_argument("--face-cols", type=str, nargs="+", default=["AU01_r", "AU02_r", "AU04_r", "AU23_r", "AU45_c"], help="Names of face columns.")
    parser.add_argument("--timestamp-col", type=str, default="timestamp", help="Name of the timestamp column.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--debug-mode-on", action="store_true", help="Enable debug mode.")
    parser.add_argument("--normalize", action="store_true", help="Normalize gaze data.")

    args = parser.parse_args()
    
    if args.normalize:
        args.eye_dir = "/data2/mjma/voices/threadward_results_normalized"
        args.output_dir = f"data_normalized/{args.win_sec}_seconds/seed_{args.seed}"
        args.gaze_cols = [col + '_z' for col in args.gaze_cols] # use normalized gaze data
    else:
        args.output_dir = f"{'data' if not args.debug_mode_on else 'data_debug'}/{args.win_sec}_seconds/seed_{args.seed}"

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = BuildConfig(
        win_sec=args.win_sec,
        stride_sec=args.stride_sec,
        seed=args.seed,
        gaze_cols=tuple(args.gaze_cols),
        face_cols=tuple(args.face_cols),
        timestamp_col=args.timestamp_col,
        csv_name=args.csv_name,
        drop_empty_windows=args.drop_empty_windows,
        output_dir=args.output_dir,
        debug_mode_on=args.debug_mode_on,
    )

    eye_dir = Path(args.eye_dir)
    emotion_path = Path(args.emotion_json)

    (X_tr, Y_tr, M_tr, F_tr), (X_te, Y_te, M_te, F_te), cfg = build_dataset(
        eye_dir=eye_dir, emotion_path=emotion_path, cfg=cfg, verbose=args.verbose, debug_mode_on=args.debug_mode_on
    )

    # Save dataset to CSV files
    output_dir = Path(cfg.output_dir)
    save_dataset_to_csv(X_tr, Y_tr, M_tr, F_tr, X_te, Y_te, M_te, F_te, cfg, output_dir)

    if X_tr:
        concat_len = sum(x.shape[0] for x in X_tr[:50])
        print(f"[debug] Example stats over first 50 windows: total frames={concat_len}")

"""
python build_eyegaze_emotion_dataset.py --seed 123
python build_eyegaze_emotion_dataset.py --seed 234
python build_eyegaze_emotion_dataset.py --seed 345
python build_eyegaze_emotion_dataset.py --seed 456
python build_eyegaze_emotion_dataset.py --seed 567
python build_eyegaze_emotion_dataset.py --seed 123 --win-sec 2.0
python build_eyegaze_emotion_dataset.py --seed 234 --win-sec 2.0
python build_eyegaze_emotion_dataset.py --seed 345 --win-sec 2.0
python build_eyegaze_emotion_dataset.py --seed 456 --win-sec 2.0
python build_eyegaze_emotion_dataset.py --seed 567 --win-sec 2.0
python build_eyegaze_emotion_dataset.py --seed 123 --win-sec 10.0
python build_eyegaze_emotion_dataset.py --seed 234 --win-sec 10.0
python build_eyegaze_emotion_dataset.py --seed 345 --win-sec 10.0
python build_eyegaze_emotion_dataset.py --seed 456 --win-sec 10.0
python build_eyegaze_emotion_dataset.py --seed 567 --win-sec 10.0
slack-me "done building data"

python build_eyegaze_emotion_dataset.py --seed 123 --normalize
python build_eyegaze_emotion_dataset.py --seed 234 --normalize
python build_eyegaze_emotion_dataset.py --seed 345 --normalize
python build_eyegaze_emotion_dataset.py --seed 456 --normalize
python build_eyegaze_emotion_dataset.py --seed 567 --normalize
python build_eyegaze_emotion_dataset.py --seed 123 --win-sec 2.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 234 --win-sec 2.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 345 --win-sec 2.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 456 --win-sec 2.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 567 --win-sec 2.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 123 --win-sec 10.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 234 --win-sec 10.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 345 --win-sec 10.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 456 --win-sec 10.0 --normalize
python build_eyegaze_emotion_dataset.py --seed 567 --win-sec 10.0 --normalize
slack-me "done building normalized data"


slack-me "done building normalized data"
python baselines.py --data_folder data_normalized/123
python baselines.py --data_folder data_normalized/234
python baselines.py --data_folder data_normalized/345
python baselines.py --data_folder data_normalized/456
python baselines.py --data_folder data_normalized/567
slack-me "done running baselines on normalized data"

"""
