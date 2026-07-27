"""方形局部坐标与血管表面截面。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-8:
        raise ValueError(f"{name} 不能为零向量")
    return np.asarray(vector, dtype=np.float64) / magnitude


@dataclass(frozen=True)
class SquareFrame:
    vertices: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray
    center: np.ndarray
    width_mm: float
    length_mm: float


@dataclass(frozen=True)
class SectionContour:
    points_mm: np.ndarray
    complete: bool
    centroid_mm: np.ndarray
    area_mm2: float


def frame_from_vertices(vertices: np.ndarray) -> SquareFrame:
    """将 V1,V2,V3,V4 构造成局部二维方形坐标系。"""

    values = np.asarray(vertices, dtype=np.float64)
    if values.shape != (4, 3):
        raise ValueError("vertices 必须是 4×3 数组")
    u_vector = values[1] - values[0]
    v_vector = values[3] - values[0]
    width = float(np.linalg.norm(u_vector))
    length = float(np.linalg.norm(v_vector))
    if width < 1e-8 or length < 1e-8:
        raise ValueError("方形边长必须大于零")
    u_axis = _unit(u_vector, "方形宽度轴")
    v_axis = _unit(v_vector, "方形长度轴")
    normal = _unit(np.cross(u_axis, v_axis), "方形法向量")
    return SquareFrame(
        vertices=values,
        u_axis=u_axis,
        v_axis=v_axis,
        normal=normal,
        center=(values[0] + values[2]) / 2.0,
        width_mm=width,
        length_mm=length,
    )


def _clip_polygon(points: np.ndarray, inside, intersection) -> np.ndarray:
    if len(points) == 0:
        return points
    result: list[np.ndarray] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside != previous_inside:
            result.append(intersection(previous, current))
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(result, dtype=np.float64)


def _clip_to_square(points: np.ndarray, width: float, length: float) -> np.ndarray:
    boundaries = (
        (lambda value: value[0] >= 0.0, lambda a, b: a + (b - a) * ((0.0 - a[0]) / (b[0] - a[0]))),
        (lambda value: value[0] <= width, lambda a, b: a + (b - a) * ((width - a[0]) / (b[0] - a[0]))),
        (lambda value: value[1] >= 0.0, lambda a, b: a + (b - a) * ((0.0 - a[1]) / (b[1] - a[1]))),
        (lambda value: value[1] <= length, lambda a, b: a + (b - a) * ((length - a[1]) / (b[1] - a[1]))),
    )
    clipped = points
    for inside, intersection in boundaries:
        clipped = _clip_polygon(clipped, inside, intersection)
        if len(clipped) == 0:
            break
    return clipped


def _centroid_and_area(points: np.ndarray) -> tuple[np.ndarray, float]:
    x = points[:, 0]
    y = points[:, 1]
    next_x = np.roll(x, -1)
    next_y = np.roll(y, -1)
    cross = x * next_y - next_x * y
    signed_area = 0.5 * np.sum(cross)
    if np.isclose(signed_area, 0.0):
        return np.mean(points, axis=0), 0.0
    centroid = np.asarray(
        [
            np.sum((x + next_x) * cross) / (6.0 * signed_area),
            np.sum((y + next_y) * cross) / (6.0 * signed_area),
        ]
    )
    return centroid, abs(float(signed_area))


def intersect_mesh_with_square(mesh: trimesh.Trimesh, frame: SquareFrame) -> list[SectionContour]:
    """求网格与方形平面的闭合截面并裁剪到方形视野。"""

    section = mesh.section(plane_origin=frame.vertices[0], plane_normal=frame.normal)
    if section is None:
        return []
    contours: list[SectionContour] = []
    tolerance = 1e-8
    for path in section.discrete:
        if len(path) < 4 or not np.allclose(path[0], path[-1]):
            continue
        local = np.column_stack(
            [
                (path[:-1] - frame.vertices[0]) @ frame.u_axis,
                (path[:-1] - frame.vertices[0]) @ frame.v_axis,
            ]
        )
        complete = bool(
            np.all(local[:, 0] >= -tolerance)
            and np.all(local[:, 0] <= frame.width_mm + tolerance)
            and np.all(local[:, 1] >= -tolerance)
            and np.all(local[:, 1] <= frame.length_mm + tolerance)
        )
        clipped = _clip_to_square(local, frame.width_mm, frame.length_mm)
        if len(clipped) < 3:
            continue
        centroid, area = _centroid_and_area(clipped)
        if area > tolerance:
            contours.append(SectionContour(clipped, complete, centroid, area))
    return contours
