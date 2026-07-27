"""完整器官目录的候选点采样与方形样本扩展。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import SamplingConfig, SquareConfig
from .mesh_io import SurfaceMesh, load_surface_mesh
from .sampling import (
    build_esophagus_samples,
    filter_duodenum_remainder_points,
    filter_duodenum_upper_points,
    filter_liver_points,
    filter_pancreas_points,
    filter_stomach_points,
    sample_points_with_normals,
)
from .squares import generate_sampling_squares


ORGAN_ORDER = ("stomach", "liver", "pancreas", "duodenum", "esophagus")
STOMACH_TARGETS = (
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


@dataclass(frozen=True)
class SquareSample:
    sample_id: str
    organ: str
    probe_point_world: np.ndarray
    input_normal_world: np.ndarray
    vertices: np.ndarray


def _empty_samples() -> SurfaceSamples:
    return SurfaceSamples(np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64))


def _sample(points: np.ndarray, normals: np.ndarray, count: int, seed: int) -> SurfaceSamples:
    if len(points) == 0:
        return _empty_samples()
    sampled_points, sampled_normals = sample_points_with_normals(points, normals, count=count, seed=seed)
    return SurfaceSamples(sampled_points, sampled_normals)


def _voxel_points(mesh: SurfaceMesh, pitch_mm: float) -> np.ndarray:
    points = np.asarray(mesh.mesh.voxelized(pitch=pitch_mm).points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("目标器官体素化后没有体素点")
    return points


def sample_organs(
    organ_models: Mapping[str, str | Path],
    settings: SamplingConfig,
    seed: int,
) -> dict[str, SurfaceSamples]:
    """按源项目五类算法从完整器官模型目录生成带法线的采样点。"""

    meshes = {name: load_surface_mesh(path) for name, path in organ_models.items()}
    target_voxels = np.vstack([_voxel_points(meshes[name], settings.stomach_voxel_pitch_mm) for name in STOMACH_TARGETS])
    stomach_points, stomach_normals = filter_stomach_points(
        meshes["stomach"].vertices,
        meshes["stomach"].vertex_normals,
        target_voxels,
        settings.stomach_search_distance_mm,
        settings.stomach_voxel_pitch_mm,
    )
    liver_points, liver_normals = filter_liver_points(
        meshes["liver"].vertices,
        meshes["liver"].vertex_normals,
        meshes["esophagus"].vertices,
        meshes["gallbladder"].vertices,
    )
    pancreas_points, pancreas_normals = filter_pancreas_points(
        meshes["pancreas"].vertices,
        meshes["pancreas"].vertex_normals,
        meshes["duodenum"].vertices,
    )
    duodenum_upper_points, duodenum_upper_normals = filter_duodenum_upper_points(
        meshes["duodenum"].vertices,
        meshes["duodenum"].vertex_normals,
        meshes["adrenal_gland_right"].vertices,
    )
    duodenum_remainder_points, duodenum_remainder_normals = filter_duodenum_remainder_points(
        meshes["duodenum"].vertices,
        meshes["duodenum"].vertex_normals,
        meshes["aorta"].vertices,
    )
    esophagus_mask = (
        (meshes["esophagus"].vertices[:, 2] >= np.min(meshes["liver"].vertices[:, 2]))
        & (meshes["esophagus"].vertices[:, 2] <= np.max(meshes["liver"].vertices[:, 2]))
    )
    esophagus_points = meshes["esophagus"].vertices[esophagus_mask]
    esophagus_normals = meshes["esophagus"].vertex_normals[esophagus_mask]

    duodenum_upper = _sample(
        duodenum_upper_points,
        duodenum_upper_normals,
        settings.point_counts["duodenum_part1"],
        seed + 3,
    )
    duodenum_remainder = _sample(
        duodenum_remainder_points,
        duodenum_remainder_normals,
        settings.point_counts["duodenum_part2"],
        seed + 4,
    )
    if len(duodenum_upper.points) or len(duodenum_remainder.points):
        duodenum = SurfaceSamples(
            np.vstack([duodenum_upper.points, duodenum_remainder.points]),
            np.vstack([duodenum_upper.normals, duodenum_remainder.normals]),
        )
    else:
        duodenum = _empty_samples()
    if len(esophagus_points):
        esophagus = SurfaceSamples(
            *build_esophagus_samples(esophagus_points, esophagus_normals, settings.point_counts["esophagus"], seed + 5)
        )
    else:
        esophagus = _empty_samples()
    return {
        "stomach": _sample(stomach_points, stomach_normals, settings.point_counts["stomach"], seed),
        "liver": _sample(liver_points, liver_normals, settings.point_counts["liver"], seed + 1),
        "pancreas": _sample(pancreas_points, pancreas_normals, settings.point_counts["pancreas"], seed + 2),
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
