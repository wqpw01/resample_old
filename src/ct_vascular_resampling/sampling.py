"""器官表面点的确定性最远点采样。"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


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


def filter_stomach_points(
    points: np.ndarray,
    normals: np.ndarray,
    target_voxels: np.ndarray,
    search_distance_mm: float = 10.0,
    voxel_pitch_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """复刻胃表面点到邻近器官体素的前向法线投影筛选。"""

    point_values, normal_values = _paired_arrays(points, normals)
    targets = np.asarray(target_voxels, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3 or len(targets) == 0:
        raise ValueError("target_voxels 必须是非空 N×3 数组")
    if search_distance_mm <= 0.0 or voxel_pitch_mm <= 0.0:
        raise ValueError("搜索距离和体素间距必须大于零")
    unit_normals = _normalised_rows(normal_values)
    tree = cKDTree(targets)
    keep = np.zeros(len(point_values), dtype=bool)
    for index, point in enumerate(point_values):
        neighbors = tree.query_ball_point(point, r=search_distance_mm + voxel_pitch_mm)
        if not neighbors:
            continue
        projections = (targets[neighbors] - point) @ unit_normals[index]
        keep[index] = bool(np.any((projections > 1e-6) & (projections <= search_distance_mm)))
    return point_values[keep], unit_normals[keep]


def filter_liver_points(
    liver_points: np.ndarray,
    liver_normals: np.ndarray,
    esophagus_points: np.ndarray,
    gallbladder_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """复刻肝脏的坐标与法线约束，食管偏移固定为 +10 mm。"""

    points, normals = _paired_arrays(liver_points, liver_normals)
    esophagus = np.asarray(esophagus_points, dtype=np.float64)
    gallbladder = np.asarray(gallbladder_points, dtype=np.float64)
    if len(esophagus) == 0 or len(gallbladder) == 0:
        raise ValueError("食管和胆囊模型不能为空")
    esophagus_y = esophagus[np.argmin(esophagus[:, 2]), 1]
    mask = (
        (points[:, 0] >= np.min(points[:, 0]) + 20.0)
        & (points[:, 1] <= np.max(points[:, 1]) - 20.0)
        & (points[:, 1] >= esophagus_y + 10.0)
        & (points[:, 0] <= np.min(gallbladder[:, 0]) - 35.0)
        & (normals[:, 2] < 0.0)
        & (normals[:, 0] < 0.5)
        & (normals[:, 1] < 0.5)
    )
    return points[mask], normals[mask]


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
    x_limit = duodenum[np.argmax(duodenum[:, 2]), 0]
    mask = (
        (points[:, 0] < x_limit)
        & (unit_normals[:, 2] > np.cos(np.deg2rad(105.0)))
        & (unit_normals[:, 1] > 0.0)
        & (unit_normals[:, 0] > 0.0)
    )
    return points[mask], normals[mask]


def filter_duodenum_upper_points(
    duodenum_points: np.ndarray,
    duodenum_normals: np.ndarray,
    right_adrenal_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals = _paired_arrays(duodenum_points, duodenum_normals)
    adrenal = np.asarray(right_adrenal_points, dtype=np.float64)
    if len(adrenal) == 0:
        raise ValueError("right_adrenal_points 不能为空")
    mask = points[:, 2] > np.min(adrenal[:, 2])
    return points[mask], normals[mask]


def filter_duodenum_remainder_points(
    duodenum_points: np.ndarray,
    duodenum_normals: np.ndarray,
    aorta_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals = _paired_arrays(duodenum_points, duodenum_normals)
    aorta = np.asarray(aorta_points, dtype=np.float64)
    if len(aorta) == 0:
        raise ValueError("aorta_points 不能为空")
    mask = points[:, 0] > np.max(aorta[:, 0])
    return points[mask], normals[mask]


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
