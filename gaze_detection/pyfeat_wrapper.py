# facial_features/pyfeat_wrapper.py
from __future__ import annotations
from pathlib import Path
import cv2, pandas as pd
from tqdm import tqdm

__all__ = ["run_pyfeat"]

def run_pyfeat(
    video_path: str | Path,
    detector,
    batch_size: int = 32,          # ← tune for your GPU / CPU
) -> pd.DataFrame:
    """Extract AU intensities using batched inference."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Py-Feat")

    columns, rows, batch, indices = None, [], [], []
    frame_idx = 0

    def _flush():
        nonlocal batch, indices, columns, rows
        if not batch:
            return
        results = detector.detect_image(
            batch, batch=True, return_fex=False  # ← single call
        )
        # results == list[dict] when batch=True
        for idx, res in zip(indices, results):
            if res and isinstance(res, dict) and "au" in res:
                if columns is None:
                    columns = ["frame", "timestamp"] + sorted(res["au"])
                rows.append(
                    [idx, idx / fps] + [res["au"][k] for k in columns[2:]]
                )
        # clear internal caches (avoid later post-proc)
        for attr in ("images", "_img_paths"):
            getattr(detector, attr, []).clear()
        batch, indices = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)
        indices.append(frame_idx)
        frame_idx += 1
        if len(batch) == batch_size:
            _flush()
        pbar.update(1)

    _flush()  # leftover frames
    pbar.close()
    cap.release()

    if not rows:
        raise RuntimeError("No faces detected in any frame.")

    return pd.DataFrame(rows, columns=columns).astype("float32", errors="ignore")
