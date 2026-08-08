"""完整器官目录的候选点采样与方形样本扩展。"""

from __future__ import annotations

from collections.abc import Iterator
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
    PITCH_ANGLES_DEGREES,
    ROLL_ANGLES_DEGREES,
    STANDARD_YAW,
    YAW_ANGLES_DEGREES,
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
    region_ids: tuple[str, ...] = (),
    target_ids: tuple[tuple[str, ...], ...] = (),
    zero_plane_anchor_world: np.ndarray | None = None,
    centerline: CenterlinePath | None = None,
    pancreas_special_x_limit: float | None = None,
) -> SurfaceSamples:
    if len(points) == 0:
        raise ValueError(f"{region_id} 没有可构造局部坐标的合法候选")
    if region_ids and len(region_ids) != len(points):
        raise ValueError(f"{region_id} 的区域来源数量与候选点数量不一致")
    result = sample_points_with_minimum_spacing(
        points,
        normals,
        count=count,
        seed=seed,
        minimum_spacing_mm=minimum_spacing_mm,
    )
    selected_targets = tuple(target_ids[index] for index in result.indices) if target_ids else tuple(() for _ in result.indices)
    selected_regions = tuple(region_ids[index] for index in result.indices) if region_ids else tuple(
        region_id for _ in result.indices
    )
    return SurfaceSamples(
        result.points,
        result.normals,
        {region_id: result.stats},
        selected_regions,
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


def _merge_unique(
    *groups: tuple[np.ndarray, np.ndarray, str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    nonempty = [(points, normals, region_id) for points, normals, region_id in groups if len(points)]
    point_arrays = [points for points, _, _ in nonempty]
    normal_arrays = [normals for _, normals, _ in nonempty]
    if not point_arrays:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), ()
    points = np.vstack(point_arrays)
    normals = np.vstack(normal_arrays)
    source_regions = [region_id for values, _, region_id in nonempty for _ in values]
    _, first_indices, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    memberships: list[list[str]] = [[] for _ in range(int(np.max(inverse)) + 1)]
    for group_index, region_id in zip(inverse, source_regions, strict=True):
        if region_id not in memberships[group_index]:
            memberships[group_index].append(region_id)
    retained = np.sort(first_indices)
    region_ids = tuple("+".join(memberships[inverse[index]]) for index in retained)
    return points[retained], normals[retained], region_ids


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
        endpoint_hints_ras_mm=settings.duodenum_centerline_endpoint_hints_ras_mm,
        endpoint_match_tolerance_mm=settings.duodenum_centerline_endpoint_match_tolerance_mm,
    )
    stomach_rays = filter_points_by_target_rays(
        meshes["stomach"].vertices,
        meshes["stomach"].vertex_normals,
        target_meshes,
        settings.ray_length_mm,
        settings.ray_batch_size,
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
    liver_points, liver_normals, liver_region_ids = _merge_unique(
        (*liver_one, "liver_region_one"),
        (*liver_two, "liver_region_two"),
    )
    liver_valid = _valid_ordinary_indices(liver_points, liver_normals, esophagus_anchor, reverse_normal=True)
    liver_points, liver_normals = liver_points[liver_valid], liver_normals[liver_valid]
    liver_region_ids = tuple(liver_region_ids[index] for index in liver_valid)
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
        settings.ray_batch_size,
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
        settings.ray_batch_size,
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
            region_ids=liver_region_ids,
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
    is_pancreas_source = organ == "pancreas" or (organ == "stomach" and "pancreas" in target_ids)
    if is_pancreas_source:
        if pancreas_special_x_limit is None:
            raise ValueError(f"{organ} 缺少胰腺特殊区的腹主动脉最大 x")
        if point[0] > pancreas_special_x_limit:
            return PANCREAS_SPECIAL_YAW
    return STANDARD_YAW


def _source_priority(sample: SquareSample) -> int:
    return 1 if sample.organ in {"liver", "pancreas"} or "supplement" in sample.source_region else 0


def _deduplication_plan(
    flattened: np.ndarray,
    priorities: np.ndarray,
    source_regions: list[str],
) -> tuple[np.ndarray, dict[int, tuple[str, ...]]]:
    keep = np.ones(len(flattened), dtype=bool)
    if len(flattened) < 2:
        return keep, {}
    pairs = cKDTree(flattened).query_pairs(r=1e-9, p=np.inf, output_type="ndarray")
    if not len(pairs):
        return keep, {}
    neighbours: list[list[int]] = [[] for _ in flattened]
    for first, second in pairs:
        neighbours[int(first)].append(int(second))
        neighbours[int(second)].append(int(first))
    duplicate_regions: dict[int, tuple[str, ...]] = {}
    for winner in sorted(range(len(flattened)), key=lambda index: (int(priorities[index]), index)):
        if not keep[winner]:
            continue
        duplicates = [index for index in neighbours[winner] if keep[index]]
        keep[duplicates] = False
        duplicate_regions[winner] = tuple(
            dict.fromkeys(
                source_regions[index]
                for index in duplicates
                if source_regions[index] != source_regions[winner]
            )
        )
    return keep, duplicate_regions


def _deduplicate_exact_poses(samples: list[SquareSample]) -> list[SquareSample]:
    if not samples:
        return samples
    flattened = np.asarray([sample.vertices.reshape(-1) for sample in samples], dtype=np.float64)
    priorities = np.asarray([_source_priority(sample) for sample in samples], dtype=np.uint8)
    keep, duplicate_regions = _deduplication_plan(
        flattened,
        priorities,
        [sample.source_region for sample in samples],
    )
    return [
        replace(sample, duplicate_source_regions=duplicate_regions.get(index, ()))
        for index, sample in enumerate(samples)
        if keep[index]
    ]


def _iter_pose_candidates(
    surfaces: Mapping[str, SurfaceSamples],
    settings: SquareConfig,
) -> Iterator[SquareSample]:
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
                yield SquareSample(
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


def _candidate_pose_count(surfaces: Mapping[str, SurfaceSamples]) -> int:
    count = 0
    axis_pose_count = len(ROLL_ANGLES_DEGREES) * len(PITCH_ANGLES_DEGREES)
    for organ in ("stomach", "duodenum", "esophagus", "liver", "pancreas"):
        surface = surfaces.get(organ, _empty_samples())
        if not len(surface.points):
            continue
        _validate_surface(organ, surface)
        for point, region_id, target_ids in zip(
            surface.points,
            surface.region_ids,
            surface.target_ids,
            strict=True,
        ):
            yaw_policy = _yaw_policy(
                organ,
                region_id,
                point,
                target_ids,
                surface.pancreas_special_x_limit,
            )
            count += len(YAW_ANGLES_DEGREES[yaw_policy]) * axis_pose_count
    return count


class PoseStream:
    def __init__(
        self,
        surfaces: Mapping[str, SurfaceSamples],
        settings: SquareConfig,
        keep: np.ndarray,
        duplicate_regions: dict[int, tuple[str, ...]],
    ):
        self._surfaces = surfaces
        self._settings = settings
        self._keep = keep
        self._duplicate_regions = duplicate_regions

    def __len__(self) -> int:
        return int(np.count_nonzero(self._keep))

    def __iter__(self) -> Iterator[SquareSample]:
        for index, sample in enumerate(_iter_pose_candidates(self._surfaces, self._settings)):
            if self._keep[index]:
                yield replace(
                    sample,
                    duplicate_source_regions=self._duplicate_regions.get(index, ()),
                )


def generate_square_samples(surfaces: Mapping[str, SurfaceSamples], settings: SquareConfig) -> PoseStream:
    """构建可重复遍历的语义姿态流，并严格合并完全相同的四顶点。"""

    candidate_count = _candidate_pose_count(surfaces)
    flattened = np.empty((candidate_count, 12), dtype=np.float64)
    priorities = np.empty(candidate_count, dtype=np.uint8)
    source_regions: list[str] = []
    generated_count = 0
    for generated_count, sample in enumerate(_iter_pose_candidates(surfaces, settings), start=1):
        flattened[generated_count - 1] = sample.vertices.reshape(-1)
        priorities[generated_count - 1] = _source_priority(sample)
        source_regions.append(sample.source_region)
    if generated_count != candidate_count:
        raise RuntimeError("姿态预估数与实际生成数不一致")
    keep, duplicate_regions = _deduplication_plan(flattened, priorities, source_regions)
    return PoseStream(surfaces, settings, keep, duplicate_regions)
