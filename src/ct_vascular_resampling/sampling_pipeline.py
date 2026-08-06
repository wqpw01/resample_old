"""完整器官目录的候选点采样与方形样本扩展。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import SamplingConfig, SquareConfig
from .mesh_io import load_surface_mesh
from .sampling import (
    build_esophagus_samples,
    filter_duodenum_bulb_points,
    filter_duodenum_remainder_points,
    filter_liver_region_one_points,
    filter_liver_region_two_points,
    filter_pancreas_points,
    filter_points_by_target_rays,
    SamplingStatistics,
    sample_points_with_minimum_spacing,
)
from .squares import generate_sampling_squares


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


@dataclass(frozen=True)
class SquareSample:
    sample_id: str
    organ: str
    probe_point_world: np.ndarray
    input_normal_world: np.ndarray
    vertices: np.ndarray


def _empty_samples() -> SurfaceSamples:
    return SurfaceSamples(np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64))


def _sample(
    points: np.ndarray,
    normals: np.ndarray,
    count: int,
    seed: int,
    minimum_spacing_mm: float,
    region_id: str,
) -> SurfaceSamples:
    result = sample_points_with_minimum_spacing(
        points,
        normals,
        count=count,
        seed=seed,
        minimum_spacing_mm=minimum_spacing_mm,
    )
    return SurfaceSamples(result.points, result.normals, {region_id: result.stats})


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
    stomach_rays = filter_points_by_target_rays(
        meshes["stomach"].vertices,
        meshes["stomach"].vertex_normals,
        target_meshes,
        settings.ray_length_mm,
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
    pancreas_points, pancreas_normals = filter_pancreas_points(
        meshes["pancreas"].vertices,
        meshes["pancreas"].vertex_normals,
        meshes["duodenum"].vertices,
    )
    duodenum_rays = filter_points_by_target_rays(
        meshes["duodenum"].vertices,
        meshes["duodenum"].vertex_normals,
        target_meshes,
        settings.ray_length_mm,
    )
    duodenum_upper_points, duodenum_upper_normals = filter_duodenum_bulb_points(
        duodenum_rays.points,
        duodenum_rays.normals,
        meshes["adrenal_gland_right"].vertices,
    )
    duodenum_remainder_points, duodenum_remainder_normals = filter_duodenum_remainder_points(
        duodenum_rays.points,
        duodenum_rays.normals,
        meshes["aorta"].vertices,
        meshes["adrenal_gland_right"].vertices,
    )
    esophagus_mask = (
        (meshes["esophagus"].vertices[:, 2] >= np.min(meshes["liver"].vertices[:, 2]))
        & (meshes["esophagus"].vertices[:, 2] <= np.max(meshes["liver"].vertices[:, 2]))
    )
    esophagus_points = meshes["esophagus"].vertices[esophagus_mask]
    esophagus_normals = meshes["esophagus"].vertex_normals[esophagus_mask]
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
    )
    duodenum_remainder = _sample(
        duodenum_remainder_points,
        duodenum_remainder_normals,
        settings.point_counts["duodenum_part2"],
        seed + 4,
        settings.minimum_spacing_mm,
        "duodenum_remainder",
    )
    if len(duodenum_upper.points) or len(duodenum_remainder.points):
        duodenum = SurfaceSamples(
            np.vstack([duodenum_upper.points, duodenum_remainder.points]),
            np.vstack([duodenum_upper.normals, duodenum_remainder.normals]),
            {**duodenum_upper.sampling_statistics, **duodenum_remainder.sampling_statistics},
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
        )
        esophagus = SurfaceSamples(
            esophagus_result.points,
            esophagus_result.normals,
            {"esophagus": esophagus_result.stats},
        )
    else:
        esophagus = _empty_samples()
    return {
        "stomach": _sample(
            stomach_rays.points,
            stomach_rays.normals,
            settings.point_counts["stomach"],
            seed,
            settings.minimum_spacing_mm,
            "stomach",
        ),
        "liver": _sample(
            liver_points,
            liver_normals,
            settings.point_counts["liver"],
            seed + 1,
            settings.minimum_spacing_mm,
            "liver",
        ),
        "pancreas": _sample(
            pancreas_points,
            pancreas_normals,
            settings.point_counts["pancreas"],
            seed + 2,
            settings.minimum_spacing_mm,
            "pancreas",
        ),
        "duodenum": duodenum,
        "esophagus": esophagus,
    }


def generate_square_samples(surfaces: Mapping[str, SurfaceSamples], settings: SquareConfig) -> list[SquareSample]:
    """按器官方向配置将每个表面点扩展为 27 倍方形样本。"""

    samples: list[SquareSample] = []
    for organ in ORGAN_ORDER:
        surface = surfaces.get(organ, _empty_samples())
        use_reverse_normal, axes = settings.spec_for(organ)
        for point_index, (point, normal) in enumerate(zip(surface.points, surface.normals, strict=True)):
            for axis in axes:
                squares = generate_sampling_squares(
                    point=point,
                    normal=normal,
                    side_length_mm=settings.side_length_mm,
                    use_reverse_normal=use_reverse_normal,
                    reference_axis=axis,
                )
                retained_by_geometry_group: dict[tuple[int, int], list[np.ndarray]] = {}
                for variant_index, vertices in enumerate(squares):
                    if settings.deduplicate_degenerate_edge_angles:
                        normal_index, remainder = divmod(variant_index, 9)
                        _, plane_index = divmod(remainder, 3)
                        geometry_group = (normal_index, plane_index)
                        prior = retained_by_geometry_group.setdefault(geometry_group, [])
                        if any(np.allclose(vertices, previous, rtol=0.0, atol=1e-9) for previous in prior):
                            continue
                        prior.append(vertices)
                    samples.append(
                        SquareSample(
                            sample_id=f"{organ}-{point_index:06d}-{axis}-{variant_index:02d}",
                            organ=organ,
                            probe_point_world=np.asarray(point, dtype=np.float64),
                            input_normal_world=np.asarray(normal, dtype=np.float64),
                            vertices=vertices,
                        )
                    )
    return samples
