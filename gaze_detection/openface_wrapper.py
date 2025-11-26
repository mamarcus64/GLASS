"""Run the OpenFace 2.0 **FeatureExtraction** binary and load its CSV output."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

__all__ = ["run_openface"]

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------

def _locate_csv(out_dir: Path) -> Path:
    for p in out_dir.glob("*.csv"):
        return p
    raise FileNotFoundError("OpenFace did not produce a CSV file – check logs.")


def _with_conda_lib_path(env: dict[str, str]) -> dict[str, str]:
    """Prepend ``$CONDA_PREFIX/lib`` to *LD_LIBRARY_PATH* if we’re in a conda env."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return env
    lib_path = str(Path(conda_prefix) / "lib")
    ld_path = env.get("LD_LIBRARY_PATH", "")
    if lib_path not in ld_path.split(":"):
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{ld_path}" if ld_path else lib_path
    return env

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_openface(
    video_path: str | Path,
    openface_bin: str | Path,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Execute OpenFace on *video_path* and return its per-frame DataFrame."""

    video_path = Path(video_path).expanduser().resolve()
    openface_bin = Path(openface_bin).expanduser().resolve()
    if model_dir is not None:
        model_dir = Path(model_dir).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not openface_bin.exists():
        raise FileNotFoundError(openface_bin)

    tmp_dir = Path(tempfile.mkdtemp(prefix="openface_"))
    try:
        cmd = [
            str(openface_bin),
            "-f", str(video_path),
            "-out_dir", str(tmp_dir),
            "-quiet",
            "-gaze",
            "-pose",
            "-aus",
        ]
        # if model_dir is not None:
            # cmd += ["-model_dir", str(model_dir)]

        subprocess.run(cmd, check=True, env=_with_conda_lib_path(os.environ.copy()))

        csv_path = _locate_csv(tmp_dir)
        df = pd.read_csv(csv_path)

        base_cols = [
            "frame", "timestamp",
            "gaze_0_x", "gaze_0_y", "gaze_0_z",
            "gaze_1_x", "gaze_1_y", "gaze_1_z",
            "gaze_angle_x", "gaze_angle_y",
            "pose_Tx", "pose_Ty", "pose_Tz",
            "pose_Rx", "pose_Ry", "pose_Rz",
        ]
        
        au_cols = [c for c in df.columns if c.lower().startswith("au")]
        
        cols = [c for c in base_cols + au_cols if c in df.columns]
        return df[cols].astype("float32", errors="ignore")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
