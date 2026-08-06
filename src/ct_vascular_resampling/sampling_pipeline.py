"""完整器官目录的候选点采样与方形样本扩展。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .config import SamplingConfig, SquareConfig
from .centerline import CenterlinePath, extract_duodenum_centerline
from .mesh_io import load_surface_mesh
from .sampling import (
    build_esophagus_samples,
    filter_duodenum_bulb_points,
    filter_duodenum_remainder_points,
    filter_esophagus_valid_segment,
    filter_liver_region_one_points,
    filter_liver_region_two_points,
    filter_pancreas_points,
    filter_points_by_target_rays,
    extreme_plateau_centroid,
    SamplingStatistics,
    sample_points_with_minimum_spacing,
)
from .squares import (
    DUODENUM_BULB_YAW,
    PANCREAS_SPECIAL_YAW,
    STANDARD_YAW,
    duodenum_local_frame,
    generate_pose_variants,
    ordinary_local_frame,
)


ORGAN_ORDER = ("stomach", "liver", "pancreas", "duodenum", "esophagus")
TARGET_ORGANS = (
    "adrenal_gland_left",
    "adrenal_gland_right",
    "aorta",
    "gallbladder",
    "inferior_vena_cava",
    "kidney_left",
    "kidney_right",
    "pancreas",
    "portal_vein_and_splenic_vein",
    "spleen",
    "liver",
)


@dataclass(frozen=True)
class SurfaceSamples:
    points: np.ndarray
    normals: np.ndarray
    sampling_statistics: dict[str, SamplingStatistics] = field(default_factory=dict)
    region_ids: tuple[str, ...] = ()
    target_ids: tuple[tuple[str, ...], ...] = ()
    zero_plane_anchor_world: np.ndarray | None = None
    centerline: CenterlinePath | None = None
    pancreas_special_x_limit: float | None = None


@dataclass(frozen=True)
class SquareSample:
    sample_id: str
    organ: str
    probe_point_world: np.ndarray
    input_normal_world: np.ndarray
    vertices: np.ndarray
    source_region: str = "legacy"
    yaw_policy: str = STANDARD_YAW
    roll_degrees: float = 0.0
    pitch_degrees: float = 0.0
    yaw_degrees: float = 0.0
    local_x_world: np.ndarray | None = None
    local_y_world: np.ndarray | None = None
    local_z_world: np.ndarray | None = None
    target_ids: tuple[str, ...] = ()
    duplicate_source_regions: tuple[str, ...] = ()


def _empty_samples() -> SurfaceSamples:
    return SurfaceSamples(np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64))


def _sample(
    points: np.ndarray,
    normals: np.ndarray,
    count: int,
    seed: int,
    minimum_spacing_mm: float,
    region_id: str,
    target_ids: tuple[tuple[str, ...], ...] = (),
    zero_plane_anchor_world: np.ndarray | None = None,
    centerline: CenterlinePath | None = None,
    pancreas_special_x_limit: float | None = None,
) -> SurfaceSamples:
    if len(points) == 0:
        raise ValueError(f"{region_id} 没有可构造局部坐标的合法候选")
    result = sample_points_with_minimum_spacing(
        points,
        normals,
        count=count,
        seed=seed,
        minimum_spacing_mm=minimum_spacing_mm,
    )
    selected_targets = tuple(target_ids[index] for index in result.indices) if target_ids else tuple(() for _ in result.indices)
    return SurfaceSamples(
        result.points,
        result.normals,
        {region_id: result.stats},
        tuple(region_id for _ in result.indices),
        selected_targets,
        zero_plane_anchor_world,
        centerline,
        pancreas_special_x_limit,
    )


def _valid_ordinary_indices(
    points: np.ndarray,
    normals: np.ndarray,
    anchor: np.ndarray,
    *,
    reverse_normal: bool,
) -> np.ndarray:
    valid: list[int] = []
    for index, (point, normal) in enumerate(zip(points, normals, strict=True)):
        try:
            ordinary_local_frame(point, -normal if reverse_normal else normal, anchor)
        except ValueError:
            continue
        valid.append(index)
    return np.asarray(valid, dtype=np.int64)


def _valid_duodenum_indices(points: np.ndarray, normals: np.ndarray, centerline: CenterlinePath) -> np.ndarray:
    valid: list[int] = []
    for index, (point, normal) in enumerate(zip(points, normals, strict=True)):
        try:
            duodenum_local_frame(point, normal, centerline)
        except ValueError:
            continue
        valid.append(index)
    return np.asarray(valid, dtype=np.int64)


def _merge_unique(*pairs: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    point_arrays = [points for points, _ in pairs if len(points)]
    normal_arrays = [normals for points, normals in pairs if len(points)]
    if not point_arrays:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    points = np.vstack(point_arrays)
    normals = np.vstack(normal_arrays)
    _, first_indices = np.unique(points, axis=0, return_index=True)
    retained = np.sort(first_indices)
    return points[retained], normals[retained]


def _target_metadata_for_points(
    source_points: np.ndarray,
    source_target_ids: tuple[tuple[str, ...], ...],
    selected_points: np.ndarray,
) -> tuple[tuple[str, ...], ...]:
    if len(source_points) != len(source_target_ids):
        raise ValueError("源点与目标命中元数据数量不一致")
    targets_by_point: dict[bytes, set[str]] = {}
    for point, target_ids in zip(source_points, source_target_ids, strict=True):
        targets_by_point.setdefault(np.asarray(point, dtype=np.float64).tobytes(), set()).update(target_ids)
    selected: list[tuple[str, ...]] = []
    for point in selected_points:
        key = np.asarray(point, dtype=np.float64).tobytes()
        if key not in targets_by_point:
            raise ValueError("筛选后的点无法映射回目标射线证据")
        selected.append(tuple(sorted(targets_by_point[key])))
    return tuple(selected)


def sample_organs(
    organ_models: Mapping[str, str | Path],
    settings: SamplingConfig,
    seed: int,
    input_coordinate_system: str = "RAS",
) -> dict[str, SurfaceSamples]:
    """按核心设计区域从完整器官模型目录生成带法线的采样点。"""

    meshes = {
        name: load_surface_mesh(path, input_coordinate_system=input_coordinate_system)
        for name, path in organ_models.items()
    }
    target_meshes = {name: meshes[name].mesh for name in TARGET_ORGANS}
    esophagus_anchor = extreme_plateau_centroid(meshes["esophagus"].vertices, axis=2, maximum=False)
    aorta_max_x = float(np.max(meshes["aorta"].vertices[:, 0]))
    centerline = extract_duodenum_centerline(
        meshes["duodenum"].mesh,
        meshes["stomach"].vertices,
        voxel_pitch_mm=settings.centerline_voxel_pitch_mm,
        tangent_window_mm=settings.centerline_tangent_window_mm,
        max_terminal_spur_mm=settings.centerline_max_terminal_spur_mm,
    )
    stomach_rays = filter_points_by_target_rays(
        meshes["stomach"].vertices,
        meshes["stomach"].vertex_normals,
        target_meshes,
        settings.ray_length_mm,
    )
    stomach_valid = _valid_ordinary_indices(
        stomach_rays.points,
        stomach_rays.normals,
        esophagus_anchor,
        reverse_normal=False,
    )
    liver_one = filter_liver_region_one_points(
        meshes["liver"].vertices,
        meshes["liver"].vertex_normals,
        meshes["esophagus"].vertices,
        meshes["inferior_vena_cava"].vertices,
    )
    liver_two = filter_liver_region_two_points(
        meshes["liver"].vertices,
        meshes["liver"].vertex_normals,
        meshes["spleen"].vertices,
        meshes["inferior_vena_cava"].vertices,
        meshes["pancreas"].vertices,
    )
    liver_points, liver_normals = _merge_unique(liver_one, liver_two)
    liver_valid = _valid_ordinary_indices(liver_points, liver_normals, esophagus_anchor, reverse_normal=True)
    liver_points, liver_normals = liver_points[liver_valid], liver_normals[liver_valid]
    pancreas_points, pancreas_normals = filter_pancreas_points(
        meshes["pancreas"].vertices,
        meshes["pancreas"].vertex_normals,
        meshes["duodenum"].vertices,
    )
    pancreas_valid = _valid_ordinary_indices(pancreas_points, pancreas_normals, esophagus_anchor, reverse_normal=True)
    pancreas_points, pancreas_normals = pancreas_points[pancreas_valid], pancreas_normals[pancreas_valid]
    duodenum_rays = filter_points_by_target_rays(
        meshes["duodenum"].vertices,
        meshes["duodenum"].vertex_normals,
        target_meshes,
        settings.ray_length_mm,
    )
    duodenum_valid = _valid_duodenum_indices(duodenum_rays.points, duodenum_rays.normals, centerline)
    duodenum_points = duodenum_rays.points[duodenum_valid]
    duodenum_normals = duodenum_rays.normals[duodenum_valid]
    duodenum_target_ids = tuple(duodenum_rays.all_target_ids[index] for index in duodenum_valid)
    duodenum_upper_points, duodenum_upper_normals = filter_duodenum_bulb_points(
        duodenum_points,
        duodenum_normals,
        meshes["adrenal_gland_right"].vertices,
    )
    duodenum_remainder_points, duodenum_remainder_normals = filter_duodenum_remainder_points(
        duodenum_points,
        duodenum_normals,
        meshes["aorta"].vertices,
        meshes["adrenal_gland_right"].vertices,
    )
    duodenum_upper_targets = _target_metadata_for_points(
        duodenum_points,
        duodenum_target_ids,
        duodenum_upper_points,
    )
    duodenum_remainder_targets = _target_metadata_for_points(
        duodenum_points,
        duodenum_target_ids,
        duodenum_remainder_points,
    )
    esophagus_points, esophagus_normals = filter_esophagus_valid_segment(
        meshes["esophagus"].vertices,
        meshes["esophagus"].vertex_normals,
        meshes["liver"].vertices,
    )
    esophagus_span_mm = float(np.ptp(esophagus_points[:, 2]))
    esophagus_rays = filter_points_by_target_rays(
        esophagus_points,
        esophagus_normals,
        target_meshes,
        settings.ray_length_mm,
    )

    duodenum_upper = _sample(
        duodenum_upper_points,
        duodenum_upper_normals,
        settings.point_counts["duodenum_part1"],
        seed + 3,
        settings.minimum_spacing_mm,
        "duodenum_bulb",
        target_ids=duodenum_upper_targets,
        centerline=centerline,
    )
    duodenum_remainder = _sample(
        duodenum_remainder_points,
        duodenum_remainder_normals,
        settings.point_counts["duodenum_part2"],
        seed + 4,
        settings.minimum_spacing_mm,
        "duodenum_remainder",
        target_ids=duodenum_remainder_targets,
        centerline=centerline,
    )
    if len(duodenum_upper.points) or len(duodenum_remainder.points):
        duodenum = SurfaceSamples(
            np.vstack([duodenum_upper.points, duodenum_remainder.points]),
            np.vstack([duodenum_upper.normals, duodenum_remainder.normals]),
            {**duodenum_upper.sampling_statistics, **duodenum_remainder.sampling_statistics},
            duodenum_upper.region_ids + duodenum_remainder.region_ids,
            duodenum_upper.target_ids + duodenum_remainder.target_ids,
            centerline=centerline,
        )
    else:
        duodenum = _empty_samples()
    if len(esophagus_rays.points):
        esophagus_result = build_esophagus_samples(
            esophagus_rays.points,
            esophagus_rays.normals,
            settings.point_counts["esophagus"],
            seed + 5,
            settings.minimum_spacing_mm,
            zero_plane_anchor_world=esophagus_anchor,
            translation_span_mm=esophagus_span_mm,
        )
        esophagus = SurfaceSamples(
            esophagus_result.points,
            esophagus_result.normals,
            {"esophagus": esophagus_result.stats},
            tuple("esophagus" for _ in esophagus_result.indices),
            _target_metadata_for_points(
                np.vstack(
                    [
                        esophagus_rays.points,
                        esophagus_rays.points
                        - np.asarray(
                            [0.0, 0.0, esophagus_span_mm],
                            dtype=np.float64,
                        ),
                    ]
                ),
                esophagus_rays.all_target_ids + esophagus_rays.all_target_ids,
                esophagus_result.points,
            ),
            zero_plane_anchor_world=esophagus_anchor,
        )
    else:
        esophagus = _empty_samples()
    return {
        "stomach": _sample(
            stomach_rays.points[stomach_valid],
            stomach_rays.normals[stomach_valid],
            settings.point_counts["stomach"],
            seed,
            settings.minimum_spacing_mm,
            "stomach",
            target_ids=tuple(stomach_rays.all_target_ids[index] for index in stomach_valid),
            zero_plane_anchor_world=esophagus_anchor,
            pancreas_special_x_limit=aorta_max_x,
        ),
        "liver": _sample(
            liver_points,
            liver_normals,
            settings.point_counts["liver"],
            seed + 1,
            settings.minimum_spacing_mm,
            "liver",
            zero_plane_anchor_world=esophagus_anchor,
        ),
        "pancreas": _sample(
            pancreas_points,
            pancreas_normals,
            settings.point_counts["pancreas"],
            seed + 2,
            settings.minimum_spacing_mm,
            "pancreas",
            zero_plane_anchor_world=esophagus_anchor,
            pancreas_special_x_limit=aorta_max_x,
        ),
        "duodenum": duodenum,
        "esophagus": esophagus,
    }


def _angle_token(value: float) -> str:
    rounded = int(round(value))
    prefix = "m" if rounded < 0 else "p" if rounded > 0 else "z"
    return f"{prefix}{abs(rounded):03d}"


def _validate_surface(organ: str, surface: SurfaceSamples) -> None:
    points = np.asarray(surface.points)
    normals = np.asarray(surface.normals)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{organ} 的 points 与 normals 必须是形状相同的 N×3 数组")
    if len(surface.region_ids) != len(points) or len(surface.target_ids) != len(points):
        raise ValueError(f"{organ} 的区域和目标命中元数据数量必须与采样点一致")


def _yaw_policy(
    organ: str,
    region_id: str,
    point: np.ndarray,
    target_ids: tuple[str, ...],
    pancreas_special_x_limit: float | None,
) -> str:
    if region_id == "duodenum_bulb":
        return DUODENUM_BULB_YAW
    is_pancreas_source = organ == "pancreas" or "pancreas" in target_ids
    if is_pancreas_source:
        if pancreas_special_x_limit is None:
            raise ValueError(f"{organ} 缺少胰腺特殊区的腹主动脉最大 x")
        if point[0] > pancreas_special_x_limit:
            return PANCREAS_SPECIAL_YAW
    return STANDARD_YAW


def _source_priority(sample: SquareSample) -> int:
    return 1 if sample.organ in {"liver", "pancreas"} or "supplement" in sample.source_region else 0


def _deduplicate_exact_poses(samples: list[SquareSample]) -> list[SquareSample]:
    if len(samples) < 2:
        return samples
    flattened = np.asarray([sample.vertices.reshape(-1) for sample in samples], dtype=np.float64)
    pairs = cKDTree(flattened).query_pairs(r=1e-9, p=np.inf, output_type="ndarray")
    if not len(pairs):
        return samples
    neighbours: list[list[int]] = [[] for _ in samples]
    for first, second in pairs:
        neighbours[int(first)].append(int(second))
        neighbours[int(second)].append(int(first))
    removed = np.zeros(len(samples), dtype=bool)
    retained: dict[int, SquareSample] = {}
    for winner in sorted(range(len(samples)), key=lambda index: (_source_priority(samples[index]), index)):
        if removed[winner]:
            continue
        duplicates = [index for index in neighbours[winner] if not removed[index]]
        removed[duplicates] = True
        duplicate_regions = tuple(
            dict.fromkeys(
                samples[index].source_region
                for index in duplicates
                if samples[index].source_region != samples[winner].source_region
            )
        )
        retained[winner] = replace(samples[winner], duplicate_source_regions=duplicate_regions)
    return [retained[index] for index in sorted(retained)]


def generate_square_samples(surfaces: Mapping[str, SurfaceSamples], settings: SquareConfig) -> list[SquareSample]:
    """按局部三轴语义生成姿态，并严格合并完全相同的四顶点。"""

    samples: list[SquareSample] = []
    pose_organ_order = ("stomach", "duodenum", "esophagus", "liver", "pancreas")
    for organ in pose_organ_order:
        surface = surfaces.get(organ, _empty_samples())
        if not len(surface.points):
            continue
        _validate_surface(organ, surface)
        for point_index, (point, normal, region_id, target_ids) in enumerate(
            zip(surface.points, surface.normals, surface.region_ids, surface.target_ids, strict=True)
        ):
            if organ == "duodenum":
                if surface.centerline is None:
                    raise ValueError("duodenum 缺少中心线")
                frame = duodenum_local_frame(point, normal, surface.centerline)
            else:
                if surface.zero_plane_anchor_world is None:
                    raise ValueError(f"{organ} 缺少普通 0 度面的食管极点")
                forward = -normal if organ in {"liver", "pancreas"} else normal
                frame = ordinary_local_frame(point, forward, surface.zero_plane_anchor_world)
            yaw_policy = _yaw_policy(
                organ,
                region_id,
                point,
                target_ids,
                surface.pancreas_special_x_limit,
            )
            for variant in generate_pose_variants(point, frame, settings.side_length_mm, yaw_policy):
                sample_id = (
                    f"{organ}-{point_index:06d}"
                    f"-r{_angle_token(variant.roll_degrees)}"
                    f"-p{_angle_token(variant.pitch_degrees)}"
                    f"-y{_angle_token(variant.yaw_degrees)}"
                )
                samples.append(
                    SquareSample(
                        sample_id=sample_id,
                        organ=organ,
                        probe_point_world=np.asarray(point, dtype=np.float64),
                        input_normal_world=np.asarray(normal, dtype=np.float64),
                        vertices=variant.vertices,
                        source_region=region_id,
                        yaw_policy=yaw_policy,
                        roll_degrees=variant.roll_degrees,
                        pitch_degrees=variant.pitch_degrees,
                        yaw_degrees=variant.yaw_degrees,
                        local_x_world=variant.local_frame.x_axis,
                        local_y_world=variant.local_frame.y_axis,
                        local_z_world=variant.local_frame.z_axis,
                        target_ids=target_ids,
                    )
                )
    return _deduplicate_exact_poses(samples)
