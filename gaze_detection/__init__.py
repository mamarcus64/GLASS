"""High‑level public API for the *facial_features* package.

Exports
-------
* ``extract_facial_features`` – run vision models on a single video and save per‑frame CSV or JSON.
* ``load_detector`` – convenience helper that instantiates a **Py‑Feat** ``Detector`` on a chosen device so the caller doesn't have to remember the exact constructor kwargs.
"""

from __future__ import annotations

from typing import Any

from .extractor import extract_facial_features

__all__ = [
    "extract_facial_features",
    "load_detector",
]


def load_detector(device: str = "cuda:0", batch_size: int = 1, **kwargs: Any):  # noqa: D401
    """Return a ready‑to‑run *Py‑Feat* ``Detector``.

    The function only *constructs* the model; it does **not** keep any persistent
    state between calls. Your external process manager should cache the returned
    object and pass it to :pyfunc:`extract_facial_features`.
    
    Note: Requires py-feat to be installed separately.
    """
    try:
        from feat import Detector as _FeatDetector
    except ImportError:
        raise ImportError(
            "py-feat is not installed. Install it with: pip install py-feat\n"
            "Note: py-feat is optional and only needed if you want to use PyFeat "
            "instead of OpenFace for facial feature extraction."
        )

    return _FeatDetector(device=device, batch_size=batch_size, **kwargs)
