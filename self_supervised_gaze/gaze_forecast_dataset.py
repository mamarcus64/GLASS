# gaze_forecast_dataset.py (no-recursion, bounded resampling, vectorized repetitive check)
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import gc
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def _window_valid_mask(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # True for frames that are NOT (near-)all-zero across channels
    return ~(np.abs(arr).max(axis=1) <= eps)

def _is_repetitive_gaze(
    gaze_window: np.ndarray,
    num_sampled: int = 5,
    repetitive_fraction: float = 0.8,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    """Vectorized check: are >= repetitive_fraction of sampled frames ~ identical to the first?"""
    n = gaze_window.shape[0]
    if n <= 1:
        return False
    # If few frames, compare all to the first
    if n <= num_sampled:
        same = np.isclose(gaze_window, gaze_window[0], rtol=rtol, atol=atol, equal_nan=False).all(axis=1)
        # ignore the first frame itself when counting
        return (same[1:].mean() if same.size > 1 else 0.0) >= repetitive_fraction
    # Sample indices (excluding 0 to avoid trivially identical)
    idx = np.random.choice(np.arange(1, n), size=num_sampled, replace=False)
    sampled = gaze_window[idx]                      # (num_sampled, D)
    same = np.isclose(sampled, gaze_window[0], rtol=rtol, atol=atol, equal_nan=False).all(axis=1)
    return (same.mean() >= repetitive_fraction)

class GazeForecastDataset(Dataset):
    """
    Windows raw or normalized gaze arrays saved per video as .npy with shape (T, D).

    Debug controls:
      - max_videos: optional int cap on number of videos to index (after optional shuffle).
      - shuffle_videos: shuffle the video list before applying max_videos.

    Filtering:
      - invalid frame = (near-)all zero across channels.
      - window rejected if too many invalid frames OR repetitive per heuristic.
    """
    def __init__(
        self,
        raw_dir: str | Path,
        norm_dir: str | Path,
        use_norm: bool = False,
        past_len: int = 150,
        future_len: int = 150,
        stride: int = 30,
        videos: Optional[List[str]] = None,
        min_valid_ratio: float = 0.7,
        zero_eps: float = 1e-8,
        repetitive_num_sampled: int = 5,
        repetitive_fraction: float = 0.8,
        channels: Optional[List[int]] = None,
        frames_dir: Optional[str | Path] = None,
        cache_arrays: bool = False,
        mmap_mode: str = "r",
        # Debug / indexing controls:
        max_videos: Optional[int] = None,
        shuffle_videos: bool = False,
        shuffle_seed: int = 42,
        # NEW: bounded resampling attempts for invalid windows
        max_resample_attempts: int = 50,
        index_subsample_every: int = 1,
        max_windows_per_video: Optional[int] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.norm_dir = Path(norm_dir)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.use_norm = use_norm
        self.past_len = int(past_len)
        self.future_len = int(future_len)
        self.total_len = self.past_len + self.future_len
        self.stride = int(stride)
        self.min_valid_ratio = float(min_valid_ratio)
        self.zero_eps = float(zero_eps)
        self.repetitive_num_sampled = int(repetitive_num_sampled)
        self.repetitive_fraction = float(repetitive_fraction)
        self.channels = channels
        self.cache_arrays = cache_arrays
        self.mmap_mode = mmap_mode
        self.max_resample_attempts = int(max_resample_attempts)
        
        scan_dir = self.norm_dir if use_norm else self.raw_dir
        if videos is None:
            videos = sorted([p.stem for p in scan_dir.glob("*.npy")])
        if not videos:
            raise ValueError(f"No .npy files found in {scan_dir}")

        # Shuffle & cap (debug)
        if shuffle_videos:
            rnd = random.Random(shuffle_seed)
            rnd.shuffle(videos)
        if max_videos is not None:
            videos = videos[: int(max_videos)]
        self.videos = videos

        self._cache: Dict[str, np.ndarray] = {}
        self._index: List[Tuple[str, int]] = []

        # Build index with a progress bar
        for vid in tqdm(self.videos, desc="Indexing videos", ncols=100):
            T = self._get_length(vid)
            if T < self.total_len:
                continue
            last_start = T - self.total_len
            for start in range(0, last_start + 1, self.stride):
                self._index.append((vid, start))

        if not self._index:
            raise ValueError("No windows could be formed with the given lengths/stride.")

        print(f"[GazeForecastDataset] Videos indexed: {len(self.videos)}; windows: {len(self._index)}")

    def _arr_path(self, vid: str) -> Path:
        return (self.norm_dir if self.use_norm else self.raw_dir) / f"{vid}.npy"

    def _get_length(self, vid: str) -> int:
        path = self._arr_path(vid)
        arr = np.load(path, mmap_mode="r")
        T = int(arr.shape[0])
        del arr
        gc.collect()
        return T

    def _load_array(self, vid: str) -> np.ndarray:
        if self.cache_arrays and vid in self._cache:
            return self._cache[vid]
        path = self._arr_path(vid)
        arr = np.load(path, mmap_mode=self.mmap_mode)
        if self.channels is not None:
            arr = arr[:, self.channels]
        if self.cache_arrays:
            self._cache[vid] = arr  # NOTE: keeps FD open – not recommended with many files.
        return arr

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        # Non-recursive bounded resampling
        vid, start = self._index[idx]
        arr = self._load_array(vid)
        T = arr.shape[0]

        attempts = 0
        # First, try within the same video
        while attempts < self.max_resample_attempts:
            past = arr[start : start + self.past_len]
            future = arr[start + self.past_len : start + self.total_len]

            past = np.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
            future = np.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)

            mp = _window_valid_mask(past, eps=self.zero_eps)
            mf = _window_valid_mask(future, eps=self.zero_eps)

            bad = (
                (mp.mean() < self.min_valid_ratio)
                or (mf.mean() < self.min_valid_ratio)
                or _is_repetitive_gaze(past, self.repetitive_num_sampled, self.repetitive_fraction)
                or _is_repetitive_gaze(future, self.repetitive_num_sampled, self.repetitive_fraction)
            )

            if not bad:
                # success
                past_t   = torch.from_numpy(past.astype(np.float32))
                future_t = torch.from_numpy(future.astype(np.float32))
                mp_t     = torch.from_numpy(mp.astype(np.bool_))
                mf_t     = torch.from_numpy(mf.astype(np.bool_))
                meta = {"video": vid, "start_idx": start}
                return past_t, future_t, mp_t, mf_t, meta

            # advance within the same video (wrap safely)
            last_start = max(0, T - self.total_len)
            if last_start == 0:
                # no room to move; break to global resample
                break
            start = (start + self.stride) % (last_start + 1)
            attempts += 1

        # Global fallback: pick a random index once, then try a few times
        if len(self._index) > 1:
            rand_idx = random.randrange(0, len(self._index))
            if rand_idx == idx:
                rand_idx = (rand_idx + 1) % len(self._index)
            vid2, start2 = self._index[rand_idx]
            arr2 = self._load_array(vid2)
            T2 = arr2.shape[0]
            attempts = 0
            while attempts < self.max_resample_attempts:
                past = arr2[start2 : start2 + self.past_len]
                future = arr2[start2 + self.past_len : start2 + self.total_len]

                past = np.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
                future = np.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)

                mp = _window_valid_mask(past, eps=self.zero_eps)
                mf = _window_valid_mask(future, eps=self.zero_eps)
                bad = (
                    (mp.mean() < self.min_valid_ratio)
                    or (mf.mean() < self.min_valid_ratio)
                    or _is_repetitive_gaze(past, self.repetitive_num_sampled, self.repetitive_fraction)
                    or _is_repetitive_gaze(future, self.repetitive_num_sampled, self.repetitive_fraction)
                )
                if not bad:
                    past_t   = torch.from_numpy(past.astype(np.float32))
                    future_t = torch.from_numpy(future.astype(np.float32))
                    mp_t     = torch.from_numpy(mp.astype(np.bool_))
                    mf_t     = torch.from_numpy(mf.astype(np.bool_))
                    meta = {"video": vid2, "start_idx": start2, "skipped": False}
                    return past_t, future_t, mp_t, mf_t, meta

                last_start2 = max(0, T2 - self.total_len)
                if last_start2 == 0:
                    break
                start2 = (start2 + self.stride) % (last_start2 + 1)
                attempts += 1

        # If we’re here, we really couldn’t find a valid window — return an empty sample.
        # This keeps the DataLoader alive and the step will be effectively skipped (masked out).
        D = arr.shape[1]
        past   = np.zeros((self.past_len, D), dtype=np.float32)
        future = np.zeros((self.future_len, D), dtype=np.float32)
        mp     = np.zeros((self.past_len,), dtype=np.bool_)
        mf     = np.zeros((self.future_len,), dtype=np.bool_)

        past_t   = torch.from_numpy(past)
        future_t = torch.from_numpy(future)
        mp_t     = torch.from_numpy(mp)
        mf_t     = torch.from_numpy(mf)
        meta = {"video": vid, "start_idx": start, "skipped": True}
        return past_t, future_t, mp_t, mf_t, meta

