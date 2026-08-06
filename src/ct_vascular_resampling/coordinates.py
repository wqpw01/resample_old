"""患者物理坐标系边界转换。"""

from __future__ import annotations

import numpy as np


_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def _transform(input_coordinate_system: str) -> np.ndarray:
    if input_coordinate_system == "LPS":
        return _LPS_TO_RAS
    if input_coordinate_system == "RAS":
        return np.eye(3, dtype=np.float64)
    raise ValueError("input_coordinate_system 必须是 LPS 或 RAS")


def _xyz(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape[-1:] != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是末维为 3 的有限数值数组")
    return result


def to_ras_points(values: np.ndarray, input_coordinate_system: str) -> np.ndarray:
    return _xyz(values, "points") @ _transform(input_coordinate_system).T


def to_ras_vectors(values: np.ndarray, input_coordinate_system: str) -> np.ndarray:
    return _xyz(values, "vectors") @ _transform(input_coordinate_system).T


def to_ras_direction(direction: np.ndarray, input_coordinate_system: str) -> np.ndarray:
    values = np.asarray(direction, dtype=np.float64)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)):
        raise ValueError("direction 必须是有限的 3x3 矩阵")
    return _transform(input_coordinate_system) @ values
