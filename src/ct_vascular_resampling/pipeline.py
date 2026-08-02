"""CT 与血管网格在同一采样方形中的同步重采样。"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import trimesh

from .artifacts import write_square_samples_ply, write_surface_samples_ply
from .config import DEFAULT_ORGAN_COLORS, ORGAN_BOUNDARY_IDS, CTConfig, CaseConfig, FilterConfig
from .ct_resampling import (
    CTVolume,
    diagnose_square_fov,
    hu_to_grayscale,
    load_ct,
    sample_ct_square,
    square_vertices_inside_ct,
)
from .fov_diagnostics import assess_rejected_fov
from .gallery import GalleryWriter, write_rectangles_ply
from .geometry import frame_from_vertices, intersect_mesh_with_square, mesh_bounds_may_intersect_square
from .mesh_io import load_surface_mesh
from .quality import evaluate_ct_quality
from .resampling_backend import CachedCpuBackend, create_sampling_backend, validate_backend_against_cpu
from .rendering import OrganLayer, VesselLayer, render_sample_images
from .sampling_pipeline import ORGAN_ORDER, SquareSample, SurfaceSamples, generate_square_samples, sample_organs


@dataclass(frozen=True)
class PreparedVessel:
    identifier: str
    label: str
    color: tuple[int, int, int]
    mesh: trimesh.Trimesh


@dataclass(frozen=True)
class PreparedOrgan:
    identifier: str
    label: str
    color: tuple[int, int, int]
    mesh: trimesh.Trimesh
    bounds: np.ndarray


@dataclass(frozen=True)
class RunSummary:
    case_id: str
    sampled_point_counts: dict[str, int]
    total_squares: int
    status_counts: dict[str, int]
    dry_run: bool
    indexed_feature_count: int = 0


def _render_fov_exclusion_if_needed(
    sample: SquareSample,
    hu: np.ndarray,
    volume: CTVolume,
    ct_settings: CTConfig,
    writer: GalleryWriter,
    resampling_backend: str,
) -> str | None:
    """将超出 FOV 的方形保存为带纯黑越界区域的独立 CT。"""

    if square_vertices_inside_ct(volume, sample.vertices):
        return None
    frame = frame_from_vertices(sample.vertices)
    diagnosis = diagnose_square_fov(
        volume,
        frame.vertices,
        ct_settings.output_resolution,
        probe_point_world=sample.probe_point_world,
    )
    ct_pixels = hu_to_grayscale(hu, ct_settings.window_level, ct_settings.window_width).copy()
    ct_pixels[diagnosis.out_of_bounds_mask] = 0
    return writer.write_fov_exclusion(
        sample_id=sample.sample_id,
        organ=sample.organ,
        probe_point_world=sample.probe_point_world,
        input_normal_world=sample.input_normal_world,
        frame=frame,
        fov_diagnostics=diagnosis.to_record(),
        ct_image=Image.fromarray(ct_pixels),
        resampling_backend=resampling_backend,
    )


def render_square_sample(
    sample: SquareSample,
    volume: CTVolume,
    vessels: Iterable[PreparedVessel],
    ct_settings: CTConfig,
    filter_settings: FilterConfig,
    writer: GalleryWriter,
    *,
    organs: Iterable[PreparedOrgan] = (),
) -> str:
    """将一个方形同步转换为 CT、边界、特征和最终图库状态。"""

    completed = writer.completed_status(sample.sample_id)
    if completed is not None:
        return completed
    frame = frame_from_vertices(sample.vertices)
    hu = sample_ct_square(volume, frame.vertices, ct_settings.output_resolution, ct_settings.fill_hu_value)
    return render_precomputed_square(
        sample,
        hu,
        vessels,
        ct_settings,
        filter_settings,
        writer,
        organs=organs,
        resampling_backend="cpu",
        volume=volume,
    )


def render_precomputed_square(
    sample: SquareSample,
    hu: np.ndarray,
    vessels: Iterable[PreparedVessel],
    ct_settings: CTConfig,
    filter_settings: FilterConfig,
    writer: GalleryWriter,
    *,
    organs: Iterable[PreparedOrgan] = (),
    resampling_backend: str | None = None,
    volume: CTVolume | None = None,
) -> str:
    """以已重采样的 HU 方形生成 CT、边界、特征和最终图库状态。"""

    completed = writer.completed_status(sample.sample_id)
    if completed is not None:
        return completed
    if volume is not None:
        excluded_status = _render_fov_exclusion_if_needed(
            sample,
            hu,
            volume,
            ct_settings,
            writer,
            resampling_backend or "cpu",
        )
        if excluded_status is not None:
            return excluded_status
    frame = frame_from_vertices(sample.vertices)
    ct_pixels = hu_to_grayscale(hu, ct_settings.window_level, ct_settings.window_width)
    quality = evaluate_ct_quality(ct_pixels, filter_settings)
    fov_diagnostics = None
    if not quality.accepted and volume is not None:
        fov_diagnostics = assess_rejected_fov(
            volume,
            frame.vertices,
            ct_pixels,
            filter_settings,
            quality,
            probe_point_world=sample.probe_point_world,
        ).to_record()
    vessel_layers = (
        [
            VesselLayer(vessel.identifier, vessel.label, vessel.color, intersect_mesh_with_square(vessel.mesh, frame))
            for vessel in vessels
        ]
        if quality.accepted
        else []
    )
    has_complete_vessel = any(contour.complete for layer in vessel_layers for contour in layer.contours)
    organ_layers = []
    if has_complete_vessel:
        organ_layers = [
            OrganLayer(
                organ.identifier,
                organ.label,
                organ.color,
                intersect_mesh_with_square(organ.mesh, frame),
            )
            for organ in organs
            if mesh_bounds_may_intersect_square(organ.bounds, frame)
        ]
    rendered = render_sample_images(
        ct_pixels,
        frame.width_mm,
        frame.length_mm,
        vessel_layers,
        organ_layers=organ_layers,
    )
    return writer.write_sample(
        sample_id=sample.sample_id,
        organ=sample.organ,
        probe_point_world=sample.probe_point_world,
        input_normal_world=sample.input_normal_world,
        frame=frame,
        rendered=rendered,
        quality=quality,
        resampling_backend=resampling_backend,
        fov_diagnostics=fov_diagnostics,
    )


def _write_json_atomic(destination: Path, value: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _write_run_metadata(case_directory: Path, metadata: dict) -> None:
    _write_json_atomic(case_directory / "run_metadata.json", metadata)


def _preflight(config: CaseConfig) -> None:
    if not (config.ct_path.is_file() or config.ct_path.is_dir()):
        raise FileNotFoundError(f"CT 文件不存在: {config.ct_path}")
    for identifier, path in config.organ_models.items():
        if not path.is_file():
            raise FileNotFoundError(f"器官模型不存在 ({identifier}): {path}")
    for vessel in config.vessel_models:
        if not vessel.path.is_file():
            raise FileNotFoundError(f"血管模型不存在 ({vessel.identifier}): {vessel.path}")


def _legacy_name(organ: str) -> str:
    return organ[:1].upper() + organ[1:]


def run_case(
    config: CaseConfig,
    dry_run: bool = False,
    resume: bool = True,
    steps: Iterable[str] | None = None,
    workers: int | None = None,
) -> RunSummary:
    """执行一个完整病例；干运行仅计算真实候选数，不写任何文件。"""

    selected = set(steps or {"all"})
    allowed = {"all", "sample", "square", "render", "filter", "index"}
    invalid = selected - allowed
    if invalid:
        raise ValueError(f"不支持的步骤: {', '.join(sorted(invalid))}")
    if "all" in selected:
        selected = {"sample", "square", "render", "index"}
    if "filter" in selected:
        selected.add("render")
    _preflight(config)
    surfaces = sample_organs(config.organ_models, config.sampling, config.runtime.seed)
    samples = generate_square_samples(surfaces, config.square)
    sampled_counts = {organ: len(surfaces.get(organ, SurfaceSamples([], [])).points) for organ in ORGAN_ORDER}
    if dry_run:
        return RunSummary(config.case_id, sampled_counts, len(samples), {}, True)

    case_directory = config.output_root / config.case_id
    has_run_artifacts = case_directory.exists() and any(path.name != "logs" for path in case_directory.iterdir())
    if has_run_artifacts and not resume and selected & {"sample", "square", "render"}:
        raise FileExistsError(f"输出目录已有内容，请启用恢复模式: {case_directory}")
    statuses: Counter[str] = Counter()
    if "sample" in selected:
        for organ in ORGAN_ORDER:
            surface = surfaces.get(organ, SurfaceSamples([], []))
            write_surface_samples_ply(case_directory / "ResampledpointPLY" / f"FPS-{_legacy_name(organ)}.ply", surface)
    if "square" in selected:
        for organ in ORGAN_ORDER:
            organ_squares = [sample for sample in samples if sample.organ == organ]
            write_square_samples_ply(case_directory / "squarePLY" / f"{_legacy_name(organ)}-vertex.ply", organ_squares)
        write_rectangles_ply(case_directory / "rectangles.ply", [frame_from_vertices(sample.vertices) for sample in samples])
    if "render" in selected:
        volume = load_ct(config.ct_path, dicom_series_uid=config.dicom_series_uid)
        writer = GalleryWriter(case_directory, config.case_id)
        pending_samples: list[SquareSample] = []
        for sample in samples:
            completed = writer.completed_status(sample.sample_id)
            if completed is not None:
                statuses[completed] += 1
                continue
            pending_samples.append(sample)
        vessels = [
            PreparedVessel(vessel.identifier, vessel.label, vessel.color, load_surface_mesh(vessel.path).mesh)
            for vessel in config.vessel_models
        ]
        organs = []
        for identifier in ORGAN_BOUNDARY_IDS:
            mesh = load_surface_mesh(config.organ_models[identifier]).mesh
            organs.append(PreparedOrgan(identifier, identifier, DEFAULT_ORGAN_COLORS[identifier], mesh, mesh.bounds.copy()))
        effective_workers = workers if workers is not None else config.runtime.workers
        if effective_workers < 1:
            raise ValueError("workers 必须大于零")
        backend, backend_metadata = create_sampling_backend(
            volume,
            backend=config.runtime.backend,
            gpu_device=config.runtime.gpu_device,
            gpu_batch_size=config.runtime.gpu_batch_size,
        )
        backend_metadata.update(
            {
                "case_id": config.case_id,
                "total_squares": len(samples),
                "excluded_fov_count": statuses["excluded_fov"],
                "deduplicate_degenerate_edge_angles": config.square.deduplicate_degenerate_edge_angles,
                "calibration": None,
            }
        )
        if backend.name != "cpu" and pending_samples:
            reference_backend = CachedCpuBackend(volume)
            try:
                calibration = validate_backend_against_cpu(
                    backend,
                    reference_backend,
                    np.asarray([sample.vertices for sample in pending_samples[:64]], dtype=np.float64),
                    resolution=config.ct.output_resolution,
                    fill_hu_value=config.ct.fill_hu_value,
                    window_level=config.ct.window_level,
                    window_width=config.ct.window_width,
                )
            except Exception as error:
                calibration = {"sample_count": min(64, len(samples)), "accepted": False, "error": str(error)}
            backend_metadata["calibration"] = calibration
            if not calibration["accepted"]:
                message = f"GPU 后端未通过 CPU 对照校验: {calibration}"
                if config.runtime.backend == "gpu":
                    backend.close()
                    reference_backend.close()
                    raise ValueError(message)
                backend.close()
                backend = reference_backend
                backend_metadata["selected_backend"] = "cpu"
                backend_metadata["fallback_reason"] = message
            else:
                reference_backend.close()
        def render_one(item: tuple[SquareSample, np.ndarray, str]) -> str:
            sample, hu, backend_name = item
            return render_precomputed_square(
                sample,
                hu,
                vessels,
                config.ct,
                config.filtering,
                writer,
                organs=organs,
                resampling_backend=backend_name,
                volume=volume,
            )

        def render_cpu_one(sample: SquareSample) -> str:
            hu = backend.sample_many(
                np.asarray([sample.vertices], dtype=np.float64),
                config.ct.output_resolution,
                config.ct.fill_hu_value,
            )[0]
            return render_precomputed_square(
                sample,
                hu,
                vessels,
                config.ct,
                config.filtering,
                writer,
                organs=organs,
                resampling_backend=backend.name,
                volume=volume,
            )

        try:
            if effective_workers == 1:
                executor = None
            else:
                executor = ThreadPoolExecutor(max_workers=effective_workers)
            try:
                if backend.name == "cpu":
                    cpu_batch_size = effective_workers * 4
                    for start in range(0, len(pending_samples), cpu_batch_size):
                        batch = pending_samples[start : start + cpu_batch_size]
                        if executor is None:
                            statuses.update(render_cpu_one(sample) for sample in batch)
                        else:
                            statuses.update(executor.map(render_cpu_one, batch))
                else:
                    batch_size = config.runtime.gpu_batch_size
                    for start in range(0, len(pending_samples), batch_size):
                        batch = pending_samples[start : start + batch_size]
                        try:
                            hu_batch = backend.sample_many(
                                np.asarray([sample.vertices for sample in batch], dtype=np.float64),
                                config.ct.output_resolution,
                                config.ct.fill_hu_value,
                            )
                        except Exception as error:
                            if config.runtime.backend != "auto" or backend.name == "cpu":
                                raise
                            backend.close()
                            backend = CachedCpuBackend(volume)
                            backend_metadata["selected_backend"] = "cpu"
                            backend_metadata["fallback_reason"] = f"GPU 运行失败: {error}"
                            hu_batch = backend.sample_many(
                                np.asarray([sample.vertices for sample in batch], dtype=np.float64),
                                config.ct.output_resolution,
                                config.ct.fill_hu_value,
                            )
                        items = [(sample, hu, backend.name) for sample, hu in zip(batch, hu_batch, strict=True)]
                        if executor is None:
                            statuses.update(render_one(item) for item in items)
                        else:
                            statuses.update(executor.map(render_one, items))
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
        finally:
            backend_metadata["selected_backend"] = backend.name
            backend_metadata["excluded_fov_count"] = statuses["excluded_fov"]
            _write_run_metadata(case_directory, backend_metadata)
            backend.close()
    indexed_feature_count = 0
    if "index" in selected:
        gallery_manifest = case_directory / "gallery" / "gallery.jsonl"
        label_counts: Counter[str] = Counter()
        organ_label_counts: Counter[str] = Counter()
        if gallery_manifest.is_file():
            from .registration_adapter import load_gallery_database

            database = load_gallery_database(gallery_manifest, config.registration_module_path)
            indexed_feature_count = len(database.features)
            for feature in database.features:
                label_counts.update(str(triplet.label) for triplet in feature.triplets)
            with gallery_manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        organ_label_counts.update(set(json.loads(line).get("organ_labels", [])))
        _write_json_atomic(
            case_directory / "library_summary.json",
            {
                "case_id": config.case_id,
                "gallery_manifest": "gallery/gallery.jsonl",
                "gallery_manifest_exists": gallery_manifest.is_file(),
                "indexed_feature_count": indexed_feature_count,
                "feature_label_counts": dict(sorted(label_counts.items())),
                "organ_label_counts": dict(sorted(organ_label_counts.items())),
                "organ_boundary_colors": {
                    identifier: list(DEFAULT_ORGAN_COLORS[identifier]) for identifier in ORGAN_BOUNDARY_IDS
                },
            },
        )
    return RunSummary(config.case_id, sampled_counts, len(samples), dict(statuses), False, indexed_feature_count)
