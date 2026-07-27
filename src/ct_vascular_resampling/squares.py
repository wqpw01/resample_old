"""按源项目规则构造 100 mm 方形采样区域。"""

from __future__ import annotations

import numpy as np


_AXES = {
    "x": np.asarray([1.0, 0.0, 0.0]),
    "y": np.asarray([0.0, 1.0, 0.0]),
    "z": np.asarray([0.0, 0.0, 1.0]),
}
_ANGLES_DEGREES = (-5.0, 0.0, 5.0)


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-8:
        raise ValueError(f"{name} 不能为零向量")
    return vector / magnitude


def rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """使用罗德里格公式旋转向量。"""

    unit_axis = _unit(np.asarray(axis, dtype=np.float64), "axis")
    source = np.asarray(vector, dtype=np.float64)
    return (
        source * np.cos(angle_rad)
        + np.cross(unit_axis, source) * np.sin(angle_rad)
        + unit_axis * np.dot(unit_axis, source) * (1.0 - np.cos(angle_rad))
    )


def rotate_point_around_axis(point: np.ndarray, axis: np.ndarray, center: np.ndarray, angle_rad: float) -> np.ndarray:
    return rotate_vector_around_axis(np.asarray(point) - np.asarray(center), axis, angle_rad) + np.asarray(center)


def _base_axis(normal: np.ndarray, reference_axis: str) -> np.ndarray:
    if reference_axis not in _AXES:
        raise ValueError("reference_axis 必须是 x、y 或 z")
    axis = np.cross(normal, _AXES[reference_axis])
    if np.linalg.norm(axis) >= 1e-6:
        return _unit(axis, "方形底边方向")
    for candidate in _AXES.values():
        axis = np.cross(normal, candidate)
        if np.linalg.norm(axis) >= 1e-6:
            return _unit(axis, "方形底边方向")
    raise ValueError("无法构造方形底边方向")


def generate_sampling_squares(
    point: np.ndarray,
    normal: np.ndarray,
    side_length_mm: float,
    use_reverse_normal: bool,
    reference_axis: str,
) -> np.ndarray:
    """为一个表面点生成源脚本定义的 3×3×3 个方形。"""

    if side_length_mm <= 0.0:
        raise ValueError("side_length_mm 必须大于零")
    base_point = np.asarray(point, dtype=np.float64)
    if base_point.shape != (3,):
        raise ValueError("point 必须是三个数值")
    working_normal = _unit(np.asarray(normal, dtype=np.float64), "normal")
    if use_reverse_normal:
        working_normal = -working_normal

    half_side = side_length_mm / 2.0
    result: list[np.ndarray] = []
    for normal_angle in _ANGLES_DEGREES:
        x_axis = rotate_vector_around_axis(_base_axis(working_normal, reference_axis), working_normal, np.radians(normal_angle))
        x_axis = _unit(x_axis, "方形底边方向")
        bottom_left = base_point - x_axis * half_side
        bottom_right = base_point + x_axis * half_side
        top_right = bottom_right + working_normal * side_length_mm
        top_left = bottom_left + working_normal * side_length_mm
        for edge_angle in _ANGLES_DEGREES:
            edge_axis = _unit(bottom_right - bottom_left, "方形边")
            if edge_angle == 0.0:
                current = np.asarray([bottom_left, bottom_right, top_right, top_left])
            else:
                current = np.asarray(
                    [
                        rotate_point_around_axis(bottom_left, edge_axis, base_point, np.radians(edge_angle)),
                        rotate_point_around_axis(bottom_right, edge_axis, base_point, np.radians(edge_angle)),
                        rotate_point_around_axis(top_right, edge_axis, base_point, np.radians(edge_angle)),
                        rotate_point_around_axis(top_left, edge_axis, base_point, np.radians(edge_angle)),
                    ]
                )
            plane_normal = _unit(np.cross(current[1] - current[0], current[3] - current[0]), "方形法向量")
            for plane_angle in _ANGLES_DEGREES:
                if plane_angle == 0.0:
                    result.append(current)
                else:
                    result.append(
                        np.asarray(
                            [rotate_point_around_axis(vertex, plane_normal, base_point, np.radians(plane_angle)) for vertex in current]
                        )
                    )
    return np.asarray(result, dtype=np.float64)
