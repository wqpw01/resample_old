"""器官表面点的确定性最远点采样。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree
import trimesh


@dataclass(frozen=True)
class RayFilterResult:
    points: np.ndarray
    normals: np.ndarray
    target_ids: tuple[str, ...]
    distances_mm: np.ndarray
    all_target_ids: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SamplingStatistics:
    requested_count: int
    candidate_count: int
    actual_count: int
    minimum_spacing_mm: float
    actual_minimum_distance_mm: float | None


@dataclass(frozen=True)
class SamplingResult:
    points: np.ndarray
    normals: np.ndarray
    stats: SamplingStatistics
    indices: np.ndarray


def sample_points_with_minimum_spacing(
    points: np.ndarray,
    normals: np.ndarray,
    count: int,
    seed: int,
    minimum_spacing_mm: float,
) -> SamplingResult:
    point_values, normal_values = _paired_arrays(points, normals)
    if count < 0:
        raise ValueError("count 不能为负数")
    if minimum_spacing_mm <= 0.0:
        raise ValueError("minimum_spacing_mm 必须大于零")
    selected: list[int] = []
    if count and len(point_values):
        distances = np.full(len(point_values), np.inf, dtype=np.float64)
        current = int(np.random.default_rng(seed).integers(len(point_values)))
        while len(selected) < min(count, len(point_values)):
            selected.append(current)
            distances = np.minimum(distances, np.linalg.norm(point_values - point_values[current], axis=1))
            distances[selected] = -np.inf
            if len(selected) >= min(count, len(point_values)):
                break
            current = int(np.argmax(distances))
            if distances[current] < minimum_spacing_mm - 1e-9:
                break
    indices = np.asarray(selected, dtype=np.int64)
    sampled_points = point_values[indices]
    sampled_normals = normal_values[indices]
    actual_minimum: float | None = None
    if len(sampled_points) >= 2:
        actual_minimum = float(np.min(cKDTree(sampled_points).query(sampled_points, k=2)[0][:, 1]))
    return SamplingResult(
        sampled_points,
        sampled_normals,
        SamplingStatistics(count, len(point_values), len(sampled_points), minimum_spacing_mm, actual_minimum),
        indices,
    )


def filter_points_by_target_rays(
    points: np.ndarray,
    normals: np.ndarray,
    targets: Mapping[str, trimesh.Trimesh],
    ray_length_mm: float,
    ray_batch_size: int = 2048,
) -> RayFilterResult:
    point_values, normal_values = _paired_arrays(points, normals)
    if ray_length_mm <= 0.0:
        raise ValueError("ray_length_mm 必须大于零")
    if ray_batch_size < 1:
        raise ValueError("ray_batch_size 必须大于零")
    if not targets:
        raise ValueError("targets 不能为空")
    unit_normals = _normalised_rows(normal_values)
    best_distances = np.full(len(point_values), np.inf, dtype=np.float64)
    best_targets = np.full(len(point_values), "", dtype=object)
    all_targets: list[set[str]] = [set() for _ in point_values]
    for target_id in sorted(targets):
        mesh = targets[target_id]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"目标网格无效: {target_id}")
        for start in range(0, len(point_values), ray_batch_size):
            stop = min(start + ray_batch_size, len(point_values))
            locations, local_ray_indices, _ = mesh.ray.intersects_location(
                ray_origins=point_values[start:stop],
                ray_directions=unit_normals[start:stop],
                multiple_hits=True,
            )
            if not len(locations):
                continue
            ray_indices = local_ray_indices + start
            distances = np.einsum("ij,ij->i", locations - point_values[ray_indices], unit_normals[ray_indices])
            valid = (distances > 1e-6) & (distances <= ray_length_mm + 1e-9)
            for ray_index, distance in zip(ray_indices[valid], distances[valid], strict=True):
                all_targets[int(ray_index)].add(target_id)
                if distance < best_distances[ray_index]:
                    best_distances[ray_index] = float(distance)
                    best_targets[ray_index] = target_id
    keep = np.isfinite(best_distances)
    return RayFilterResult(
        point_values[keep],
        unit_normals[keep],
        tuple(str(value) for value in best_targets[keep]),
        best_distances[keep],
        tuple(tuple(sorted(all_targets[index])) for index in np.flatnonzero(keep)),
    )


def filter_liver_region_one_points(
    liver_points: np.ndarray,
    liver_normals: np.ndarray,
    esophagus_points: np.ndarray,
    vena_cava_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals = _paired_arrays(liver_points, liver_normals)
    esophagus = np.asarray(esophagus_points, dtype=np.float64)
    vena_cava = np.asarray(vena_cava_points, dtype=np.float64)
    if len(esophagus) == 0 or len(vena_cava) == 0:
        raise ValueError("食管和下腔静脉模型不能为空")
    unit_normals = _normalised_rows(normals)
    x_lower, x_upper = sorted((float(np.min(points[:, 0])) + 20.0, float(np.min(vena_cava[:, 0]))))
    y_lower, y_upper = sorted((float(np.max(points[:, 1])) - 20.0, float(np.max(esophagus[:, 1]))))
    mask = (
        (points[:, 0] >= x_lower)
        & (points[:, 0] <= x_upper)
        & (points[:, 1] >= y_lower)
        & (points[:, 1] <= y_upper)
        & (unit_normals[:, 2] < 0.0)
    )
    return points[mask], unit_normals[mask]

def filter_liver_region_two_points(
    liver_points: np.ndarray,
    liver_normals: np.ndarray,
    spleen_points: np.ndarray,
    vena_cava_points: np.ndarray,
    pancreas_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals = _paired_arrays(liver_points, liver_normals)
    spleen = np.asarray(spleen_points, dtype=np.float64)
    vena_cava = np.asarray(vena_cava_points, dtype=np.float64)
    pancreas = np.asarray(pancreas_points, dtype=np.float64)
    if len(spleen) == 0 or len(vena_cava) == 0 or len(pancreas) == 0:
        raise ValueError("脾脏、下腔静脉和胰腺模型不能为空")
    unit_normals = _normalised_rows(normals)
    x_lower, x_upper = sorted((float(np.max(spleen[:, 0])), float(np.max(vena_cava[:, 0]))))
    y_limit = float(np.max(pancreas[:, 1]))
    mask = (
        (points[:, 0] >= x_lower)
        & (points[:, 0] <= x_upper)
        & (points[:, 1] < y_limit)
        & (unit_normals[:, 1] < 0.0)
        & (unit_normals[:, 2] < 0.0)
    )
    return points[mask], unit_normals[mask]


def filter_duodenum_bulb_points(
    duodenum_points: np.ndarray,
    duodenum_normals: np.ndarray,
    right_adrenal_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals = _paired_arrays(duodenum_points, duodenum_normals)
    adrenal = np.asarray(right_adrenal_points, dtype=np.float64)
    if len(adrenal) == 0:
        raise ValueError("right_adrenal_points 不能为空")
    unit_normals = _normalised_rows(normals)
    mask = points[:, 2] > np.min(adrenal[:, 2])
    return points[mask], unit_normals[mask]


def filter_esophagus_valid_segment(
    esophagus_points: np.ndarray,
    esophagus_normals: np.ndarray,
    liver_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals = _paired_arrays(esophagus_points, esophagus_normals)
    liver = np.asarray(liver_points, dtype=np.float64)
    if len(points) == 0 or liver.ndim != 2 or liver.shape[1] != 3 or len(liver) == 0:
        raise ValueError("食管和肝脏模型不能为空")
    unit_normals = _normalised_rows(normals)
    mask = (points[:, 2] >= np.min(points[:, 2])) & (points[:, 2] <= np.max(liver[:, 2]))
    return points[mask], unit_normals[mask]


def farthest_point_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """返回一次 FPS 选择的顶点索引，保证同一输入可复现。"""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points 必须是 N×3 数组")
    if count < 0:
        raise ValueError("count 不能为负数")
    if count == 0 or len(values) == 0:
        return np.empty(0, dtype=np.int64)
    if count >= len(values):
        return np.arange(len(values), dtype=np.int64)

    indices = np.empty(count, dtype=np.int64)
    distances = np.full(len(values), np.inf, dtype=np.float64)
    indices[0] = np.random.default_rng(seed).integers(len(values))
    for position in range(1, count):
        last_point = values[indices[position - 1]]
        distances = np.minimum(distances, np.linalg.norm(values - last_point, axis=1))
        indices[position] = int(np.argmax(distances))
    return indices


def sample_points_with_normals(
    points: np.ndarray,
    normals: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """以同一 FPS 索引采样顶点与对应法线。"""

    point_values = np.asarray(points, dtype=np.float64)
    normal_values = np.asarray(normals, dtype=np.float64)
    if point_values.shape != normal_values.shape or point_values.ndim != 2 or point_values.shape[1] != 3:
        raise ValueError("points 与 normals 必须是形状相同的 N×3 数组")
    indices = farthest_point_indices(point_values, count, seed)
    return point_values[indices], normal_values[indices]


def _paired_arrays(points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_values = np.asarray(points, dtype=np.float64)
    normal_values = np.asarray(normals, dtype=np.float64)
    if point_values.shape != normal_values.shape or point_values.ndim != 2 or point_values.shape[1] != 3:
        raise ValueError("points 与 normals 必须是形状相同的 N×3 数组")
    return point_values, normal_values


def _normalised_rows(normals: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float64)
    magnitudes = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(magnitudes < 1e-8):
        raise ValueError("normals 不能包含零向量")
    return values / magnitudes


def extreme_plateau_centroid(
    points: np.ndarray,
    *,
    axis: int,
    maximum: bool,
    atol_mm: float = 1e-6,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("points 必须是非空 N×3 数组")
    if axis not in (0, 1, 2) or atol_mm < 0.0:
        raise ValueError("axis 或 atol_mm 无效")
    extreme = np.max(values[:, axis]) if maximum else np.min(values[:, axis])
    mask = np.isclose(values[:, axis], extreme, rtol=0.0, atol=atol_mm)
    return np.mean(values[mask], axis=0)


def filter_pancreas_points(
    pancreas_points: np.ndarray,
    pancreas_normals: np.ndarray,
    duodenum_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """复刻胰腺相对十二指肠最高点的空间和法线筛选。"""

    points, normals = _paired_arrays(pancreas_points, pancreas_normals)
    duodenum = np.asarray(duodenum_points, dtype=np.float64)
    if len(duodenum) == 0:
        raise ValueError("duodenum_points 不能为空")
    unit_normals = _normalised_rows(normals)
    x_limit = extreme_plateau_centroid(duodenum, axis=2, maximum=True)[0]
    mask = (
        (points[:, 0] < x_limit)
        & (unit_normals[:, 2] > np.cos(np.deg2rad(105.0)))
        & (unit_normals[:, 1] > 0.0)
    )
    return points[mask], unit_normals[mask]


def filter_duodenum_upper_points(
    duodenum_points: np.ndarray,
    duodenum_normals: np.ndarray,
    right_adrenal_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return filter_duodenum_bulb_points(duodenum_points, duodenum_normals, right_adrenal_points)


def filter_duodenum_remainder_points(
    duodenum_points: np.ndarray,
    duodenum_normals: np.ndarray,
    aorta_points: np.ndarray,
    right_adrenal_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals = _paired_arrays(duodenum_points, duodenum_normals)
    aorta = np.asarray(aorta_points, dtype=np.float64)
    if len(aorta) == 0:
        raise ValueError("aorta_points 不能为空")
    adrenal = np.asarray(right_adrenal_points, dtype=np.float64)
    if len(adrenal) == 0:
        raise ValueError("right_adrenal_points 不能为空")
    unit_normals = _normalised_rows(normals)
    bulb_mask = points[:, 2] > np.min(adrenal[:, 2])
    mask = (points[:, 0] > np.max(aorta[:, 0])) & ~bulb_mask
    return points[mask], unit_normals[mask]
