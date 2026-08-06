"""十二指肠中心线路径与局部切向。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import trimesh


@dataclass(frozen=True)
class CenterlinePath:
    points: np.ndarray
    tangents: np.ndarray
    cumulative_length_mm: np.ndarray
    pruned_terminal_spur_lengths_mm: tuple[float, ...] = ()


def _order_skeleton_indices(
    skeleton: np.ndarray,
    proximal_reference_points: np.ndarray,
    *,
    voxel_pitch_mm: float,
    max_terminal_spur_mm: float,
) -> tuple[np.ndarray, tuple[float, ...]]:
    values = np.asarray(skeleton, dtype=bool)
    references = np.asarray(proximal_reference_points, dtype=np.float64)
    if values.ndim != 3 or not np.any(values):
        raise ValueError("中心线骨架必须是非空三维数组")
    if references.ndim != 2 or references.shape[1] != 3 or len(references) == 0:
        raise ValueError("近端参考点必须是非空 N×3 数组")
    if voxel_pitch_mm <= 0.0 or max_terminal_spur_mm < 0.0:
        raise ValueError("体素间距必须大于零且毛刺阈值不能为负数")
    indices = np.argwhere(values)
    index_by_coordinate = {tuple(value): index for index, value in enumerate(indices)}
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
    adjacency: list[list[int]] = []
    for value in indices:
        adjacency.append(
            [
                index_by_coordinate[candidate]
                for offset in offsets
                if (candidate := tuple(value + offset)) in index_by_coordinate
            ]
        )
    active = set(range(len(indices)))

    def neighbors(index: int) -> list[int]:
        return [neighbor for neighbor in adjacency[index] if neighbor in active]

    visited: set[int] = set()
    stack = [0]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current])
    if len(visited) != len(indices):
        raise ValueError("中心线骨架不是单连通路径")
    pruned_lengths: list[float] = []
    while any(len(neighbors(index)) > 2 for index in active):
        terminal_branches: list[tuple[float, tuple[int, ...]]] = []
        endpoints = [index for index in active if len(neighbors(index)) == 1]
        for endpoint in endpoints:
            branch = [endpoint]
            previous: int | None = None
            current = endpoint
            length = 0.0
            while True:
                following = [neighbor for neighbor in neighbors(current) if neighbor != previous]
                if not following:
                    break
                if len(following) != 1:
                    break
                next_index = following[0]
                length += float(np.linalg.norm(indices[next_index] - indices[current]) * voxel_pitch_mm)
                if len(neighbors(next_index)) > 2:
                    terminal_branches.append((length, tuple(branch)))
                    break
                branch.append(next_index)
                previous, current = current, next_index
        if not terminal_branches:
            raise ValueError("中心线骨架包含无法清理的分叉")
        terminal_branches.sort(key=lambda value: value[0])
        shortest_length, shortest_branch = terminal_branches[0]
        if shortest_length > max_terminal_spur_mm + 1e-9:
            raise ValueError(
                f"中心线骨架包含分叉，最短终末支 {shortest_length:.6f} mm 超过 {max_terminal_spur_mm:.6f} mm"
            )
        if len(terminal_branches) > 1 and np.isclose(
            terminal_branches[1][0], shortest_length, rtol=0.0, atol=1e-9
        ):
            raise ValueError("中心线骨架包含无法唯一判定的等长短分叉")
        active.difference_update(shortest_branch)
        pruned_lengths.append(shortest_length)
    endpoints = [index for index in active if len(neighbors(index)) == 1]
    if len(endpoints) != 2:
        raise ValueError(f"中心线骨架必须恰好有 2 个端点，实际为 {len(endpoints)}")
    endpoint_points = indices[endpoints].astype(np.float64)
    distances, _ = cKDTree(references).query(endpoint_points, k=1)
    current = endpoints[int(np.argmin(distances))]
    ordered: list[int] = []
    previous: int | None = None
    while True:
        ordered.append(current)
        following = [neighbor for neighbor in neighbors(current) if neighbor != previous]
        if not following:
            break
        if len(following) != 1:
            raise ValueError("中心线骨架包含分叉")
        previous, current = current, following[0]
    if len(ordered) != len(active):
        raise ValueError("中心线骨架路径未覆盖全部体素")
    return indices[ordered], tuple(pruned_lengths)


def order_skeleton_indices(
    skeleton: np.ndarray,
    proximal_reference_points: np.ndarray,
    *,
    voxel_pitch_mm: float = 1.0,
    max_terminal_spur_mm: float = 5.0,
) -> np.ndarray:
    ordered, _ = _order_skeleton_indices(
        skeleton,
        proximal_reference_points,
        voxel_pitch_mm=voxel_pitch_mm,
        max_terminal_spur_mm=max_terminal_spur_mm,
    )
    return ordered


def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segments < 1e-8):
        raise ValueError("中心线路径包含重复点")
    return np.concatenate([[0.0], np.cumsum(segments)])


def _point_at_arclength(points: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    target = float(np.clip(distance, 0.0, cumulative[-1]))
    upper = int(np.searchsorted(cumulative, target, side="right"))
    if upper == 0:
        return points[0]
    if upper >= len(points):
        return points[-1]
    lower = upper - 1
    fraction = (target - cumulative[lower]) / (cumulative[upper] - cumulative[lower])
    return points[lower] + fraction * (points[upper] - points[lower])


def centerline_tangents(points: np.ndarray, window_mm: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("中心线路径必须是至少两个点的 N×3 数组")
    if window_mm <= 0.0:
        raise ValueError("window_mm 必须大于零")
    cumulative = _cumulative_lengths(values)
    if cumulative[-1] < window_mm:
        raise ValueError("中心线长度小于切向估计窗口")
    half_window = window_mm / 2.0
    tangents: list[np.ndarray] = []
    for distance in cumulative:
        if distance < half_window:
            left_distance, right_distance = 0.0, window_mm
        elif cumulative[-1] - distance < half_window:
            left_distance, right_distance = cumulative[-1] - window_mm, cumulative[-1]
        else:
            left_distance, right_distance = distance - half_window, distance + half_window
        vector = _point_at_arclength(values, cumulative, right_distance) - _point_at_arclength(
            values, cumulative, left_distance
        )
        magnitude = float(np.linalg.norm(vector))
        if magnitude < 1e-8:
            raise ValueError("中心线切向量退化")
        tangents.append(vector / magnitude)
    return np.asarray(tangents, dtype=np.float64)


def extract_duodenum_centerline(
    mesh: trimesh.Trimesh,
    stomach_reference_points: np.ndarray,
    *,
    voxel_pitch_mm: float,
    tangent_window_mm: float,
    max_terminal_spur_mm: float = 5.0,
) -> CenterlinePath:
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("十二指肠必须是非空三角网格")
    if voxel_pitch_mm <= 0.0:
        raise ValueError("voxel_pitch_mm 必须大于零")
    references = np.asarray(stomach_reference_points, dtype=np.float64)
    if references.ndim != 2 or references.shape[1] != 3 or len(references) == 0:
        raise ValueError("胃参考点必须是非空 N×3 数组")
    voxels = mesh.voxelized(pitch=voxel_pitch_mm).fill()
    skeleton = skeletonize(np.asarray(voxels.matrix, dtype=bool))
    reference_indices = voxels.points_to_indices(references)
    ordered_indices, pruned_lengths = _order_skeleton_indices(
        skeleton,
        reference_indices,
        voxel_pitch_mm=voxel_pitch_mm,
        max_terminal_spur_mm=max_terminal_spur_mm,
    )
    points = np.asarray(voxels.indices_to_points(ordered_indices), dtype=np.float64)
    cumulative = _cumulative_lengths(points)
    tangents = centerline_tangents(points, tangent_window_mm)
    return CenterlinePath(points, tangents, cumulative, pruned_lengths)
