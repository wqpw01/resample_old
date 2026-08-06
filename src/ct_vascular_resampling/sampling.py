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


def filter_points_by_target_rays(
    points: np.ndarray,
    normals: np.ndarray,
    targets: Mapping[str, trimesh.Trimesh],
    ray_length_mm: float,
) -> RayFilterResult:
    point_values, normal_values = _paired_arrays(points, normals)
    if ray_length_mm <= 0.0:
        raise ValueError("ray_length_mm 必须大于零")
    if not targets:
        raise ValueError("targets 不能为空")
    unit_normals = _normalised_rows(normal_values)
    best_distances = np.full(len(point_values), np.inf, dtype=np.float64)
    best_targets = np.full(len(point_values), "", dtype=object)
    for target_id in sorted(targets):
        mesh = targets[target_id]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"目标网格无效: {target_id}")
        locations, ray_indices, _ = mesh.ray.intersects_location(
            ray_origins=point_values,
            ray_directions=unit_normals,
            multiple_hits=True,
        )
        if not len(locations):
            continue
        distances = np.einsum("ij,ij->i", locations - point_values[ray_indices], unit_normals[ray_indices])
        valid = (distances > 1e-6) & (distances <= ray_length_mm + 1e-9)
        for ray_index, distance in zip(ray_indices[valid], distances[valid], strict=True):
            if distance < best_distances[ray_index]:
                best_distances[ray_index] = float(distance)
                best_targets[ray_index] = target_id
    keep = np.isfinite(best_distances)
    return RayFilterResult(
        point_values[keep],
        unit_normals[keep],
        tuple(str(value) for value in best_targets[keep]),
        best_distances[keep],
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


def build_esophagus_samples(
    points: np.ndarray,
    normals: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将食管候选按原/下移两组重排，同时严格保持目标数量。"""

    sampled_points, sampled_normals = sample_points_with_normals(points, normals, count=count, seed=seed)
    total = len(sampled_points)
    if total == 0:
        return sampled_points, sampled_normals
    half_height = (float(np.max(sampled_points[:, 2])) - float(np.min(sampled_points[:, 2]))) / 2.0
    original_count = (total + 1) // 2
    translated_count = total - original_count
    original_indices = np.linspace(0, total - 1, original_count, dtype=int)
    translated_indices = np.linspace(0, total - 1, translated_count, dtype=int) if translated_count else np.empty(0, dtype=int)
    original_points = sampled_points[original_indices]
    translated_points = sampled_points[translated_indices].copy()
    translated_points[:, 2] -= half_height
    return (
        np.vstack([original_points, translated_points]),
        np.vstack([sampled_normals[original_indices], sampled_normals[translated_indices]]),
    )
