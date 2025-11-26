"""Merge OpenFace + Py‑Feat outputs and persist them to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from .openface_wrapper import run_openface
from .pyfeat_wrapper import run_pyfeat

__all__ = ["extract_facial_features"]

OutputFmt = Literal["csv", "json"]


def extract_facial_features(
    models: dict,
    save_folder: str | Path,
    video_path: str | Path,
    output_format: OutputFmt = "csv",
) -> Path:
    """Process a video and write the combined feature table."""

    video_path = Path(video_path).expanduser().resolve()
    save_folder = Path(save_folder).expanduser().resolve()
    save_folder.mkdir(parents=True, exist_ok=True)

    
    # df_pyfeat = run_pyfeat(video_path, models["pyfeat_detector"])
    df_openface = run_openface(
        video_path,
        models["openface_bin"],
        model_dir=models.get("openface_model_dir"),
    )

    # merged = pd.merge(df_openface, df_pyfeat, on=["frame", "timestamp"], how="inner")
    merged = df_openface


    out_path = save_folder / f"result.{output_format}"

    if output_format == "csv":
        merged.to_csv(out_path, index=False)
    elif output_format == "json":
        merged.to_json(out_path, orient="records", lines=True)
    else:
        raise ValueError("Unsupported output_format; choose 'csv' or 'json'.")

    return out_path