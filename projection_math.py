from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Create an XYZ rotation matrix (radians)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def transform_vertices(
    vertices: Iterable[Iterable[float]],
    rotation: Tuple[float, float, float],
    translation: Tuple[float, float, float],
    scale: float = 1.0,
) -> np.ndarray:
    verts = np.asarray(list(vertices), dtype=np.float32)
    rot = rotation_matrix_xyz(*rotation)
    translated = (verts * scale) @ rot.T + np.asarray(translation, dtype=np.float32)
    return translated


def project_point(
    point: np.ndarray,
    camera_pos: np.ndarray,
    width: int,
    height: int,
    focal_scale: float,
    depth_offset: float,
    near_plane: float = 0.02,
) -> Optional[Tuple[int, int, float]]:
    """
    Custom perspective projection used for 2.5D fish-tank style rendering.
    x_proj = x / (z + depth_offset), y_proj = y / (z + depth_offset)
    where x, y, z are camera-relative coordinates.
    """
    rel = point - camera_pos
    denom = rel[2] + depth_offset
    if denom <= near_plane:
        return None

    x_proj = rel[0] / denom
    y_proj = rel[1] / denom

    sx = int(width * 0.5 + x_proj * focal_scale * width)
    sy = int(height * 0.5 - y_proj * focal_scale * height)
    return sx, sy, float(denom)
