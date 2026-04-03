from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import tifffile
import numpy as np
import os
import json

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


# ── Metadata keys worth saving ──────────────────────────────────────────────
_META_KEYS = (
    'processing_history',
    'source',
    'hdf5_source_path',
    'reconstruction',
    'paganin',
)


def _write_metadata(base_path: str, meta: dict) -> str | None:
    """
    Write layer metadata to <base_path>.txt as formatted JSON.
    Only saves keys that are meaningful for the reconstruction history.
    Returns the metadata file path, or None if meta is empty.
    """
    payload = {k: meta[k] for k in _META_KEYS if k in meta}
    if not payload:
        return None

    meta_path = f"{base_path}.txt"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    return meta_path


def write_tiff(path: str, data: Any, meta: dict) -> list[str]:
    """Write data to a TIFF file and save layer metadata alongside it."""
    if not path.endswith('.tif'):
        path = f"{os.path.splitext(path)[0]}.tif"

    tifffile.imwrite(path, data.astype(np.float32), imagej=True)

    # napari passes layer attributes in `meta`; our custom dict is under 'metadata'
    saved = [path]
    meta_path = _write_metadata(os.path.splitext(path)[0], meta.get('metadata', {}))
    if meta_path:
        saved.append(meta_path)
    return saved


def write_raw(path: str, data: Any, meta: dict) -> list[str]:
    """Write data as .raw (dtype + shape encoded in filename) and save metadata."""
    dtype_str = str(data.dtype)

    if data.ndim == 3:
        x, y, z = data.shape[::-1]
        shape_str = f"{x}x{y}x{z}"
    else:
        shape_str = "x".join(str(s) for s in data.shape)

    base, _ = os.path.splitext(path)
    path = f"{base}_{dtype_str}_{shape_str}.raw"

    with open(path, 'wb') as f:
        f.write(data.tobytes())

    saved = [path]
    meta_path = _write_metadata(f"{base}_{dtype_str}_{shape_str}", meta.get('metadata', {}))
    if meta_path:
        saved.append(meta_path)
    return saved
