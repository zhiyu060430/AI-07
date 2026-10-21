"""Convert pipeline values into stable JSON-ready structures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plain(obj, decimals: int = 8):
    """Return a JSON-safe copy. Floats are rounded so two processes agree."""
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.floating):
            return np.round(obj.astype(np.float64), decimals).tolist()
        if np.issubdtype(obj.dtype, np.integer):
            return obj.astype(np.int64).tolist()
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(np.round(float(obj), decimals))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        return float(np.round(obj, decimals))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): plain(value, decimals) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(value, decimals) for value in obj]
    return obj
