"""CT 与血管网格在同一采样方形中的同步重采样。"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np
from PIL import Image
import trimesh

from .artifacts import write_square_samples_ply, write_surface_samples_ply
from .config import (
    DEFAULT_ORGAN_COLORS,
    ORGAN_BOUNDARY_IDS,
    ORGAN_BOUNDARY_MODEL_IDS,
    CTConfig,
    CaseConfig,
    FilterConfig,
)
from .ct_resampling import (
    CTVolume,
    diagnose_square_fov,
    hu_to_grayscale,
    load_ct,
    sample_ct_square,
    square_vertices_inside_ct,
)
from .eus_organs import (
    EUS_ORGAN_GEOMETRY_SOURCES,
    EXCLUDED_ORGAN_LABELS,
    load_eus_organ_catalog,
)
from .fov_diagnostics import assess_rejected_fov
from .gallery import GalleryWriter, write_rectangles_ply
from .geometry import frame_from_vertices, intersect_mesh_with_square, mesh_bounds_may_intersect_square
from .mesh_io import load_surface_mesh
from .quality import evaluate_ct_quality
from .resampling_backend import CachedCpuBackend, create_sampling_backend, validate_backend_against_cpu
from .rendering import OrganLayer, VesselLayer, render_sample_images
from .sampling_pipeline import ORGAN_ORDER, PoseStream, SquareSample, SurfaceSamples, generate_square_samples, sample_organs
from .squares import PITCH_ANGLES_DEGREES, ROLL_ANGLES_DEGREES, YAW_ANGLES_DEGREES


CORE_DESIGN_SHA256 = "4b27aee1a6db1680e501f17bd3492a571bd169c0bf7004d79b4a512d929cc53b"


@cache
def _build_git_commit() -> str:
    candidate = os.environ.get("CT_VASCULAR_RESAMPLING_GIT_COMMIT", "").strip().lower()
    if not candidate:
        repository = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = result.stdout.strip().lower()
    if len(candidate) != 40 or any(character not in "0123456789abcdef" for character in candidate):
        raise RuntimeError("无法确定有效的构建 Git commit")
    return candidate


def _pose_metadata(sample: SquareSample) -> dict[str, object]:
    def optional_vector(value: np.ndarray | None) -> list[float] | None:
        return None if value is None else [float(item) for item in np.asarray(value, dtype=np.float64)]

    return {
        "coordinate_system": "RAS",
        "core_design_sha256": CORE_DESIGN_SHA256,
        "build_git_commit": _build_git_commit(),
        "source_region": sample.source_region,
        "yaw_policy": sample.yaw_policy,
        "angles_degrees": {
            "roll": float(sample.roll_degrees),
            "pitch": float(sample.pitch_degrees),
            "yaw": float(sample.yaw_degrees),
        },
        "local_axes_world": {
            "x": optional_vector(sample.local_x_world),
            "y": optional_vector(sample.local_y_world),
            "z": optional_vector(sample.local_z_world),
        },
        "target_ids": list(sample.target_ids),
        "duplicate_source_regions": list(sample.duplicate_source_regions),
    }


def _sha256_path(path: Path) -> str:
    source = path.resolve()
    digest = hashlib.sha256()
    if source.is_file():
        digest.update(b"file\0")
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    if source.is_dir():
        digest.update(b"directory\0")
        files = sorted(candidate for candidate in source.rglob("*") if candidate.is_file())
        for candidate in files:
            relative = candidate.relative_to(source).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with candidate.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    raise FileNotFoundError(f"输入路径不存在: {source}")


def _input_provenance(config: CaseConfig) -> dict[str, object]:
    cache: dict[Path, dict[str, object]] = {}

    def describe(path: Path) -> dict[str, object]:
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = {
                "path": str(resolved),
                "kind": "directory" if resolved.is_dir() else "file",
                "sha256": _sha256_path(resolved),
            }
        return dict(cache[resolved])

    ct = describe(config.ct_path)
    ct["dicom_series_uid"] = config.dicom_series_uid
    return {
        "ct": ct,
        "organ_models": {
            identifier: describe(path) for identifier, path in sorted(config.organ_models.items())
        },
        "vessel_models": [
            {
                "id": vessel.identifier,
                "label": vessel.label,
                "color": list(vessel.color),
                **describe(vessel.path),
            }
            for vessel in config.vessel_models
        ],
    }


def _run_protocol_metadata(
    config: CaseConfig,
    duodenum_centerline_selection: dict[str, object] | None,
) -> dict[str, object]:
    endpoint_hints = config.sampling.duodenum_centerline_endpoint_hints_ras_mm
    eus_catalog = load_eus_organ_catalog()
    protocol: dict[str, object] = {
        "coordinate_system": config.geometry.canonical_coordinate_system,
        "input_coordinate_system": config.geometry.input_coordinate_system,
        "core_design_sha256": CORE_DESIGN_SHA256,
        "build_git_commit": _build_git_commit(),
        "input_provenance": _input_provenance(config),
        "sampling_configuration": {
            "point_counts": dict(sorted(config.sampling.point_counts.items())),
            "ray_length_mm": config.sampling.ray_length_mm,
            "ray_batch_size": config.sampling.ray_batch_size,
            "minimum_spacing_mm": config.sampling.minimum_spacing_mm,
            "centerline_voxel_pitch_mm": config.sampling.centerline_voxel_pitch_mm,
            "centerline_tangent_window_mm": config.sampling.centerline_tangent_window_mm,
            "centerline_max_terminal_spur_mm": config.sampling.centerline_max_terminal_spur_mm,
            "duodenum_centerline_endpoint_hints_ras_mm": (
                {
                    "proximal": list(endpoint_hints[0]),
                    "distal": list(endpoint_hints[1]),
                }
                if endpoint_hints is not None
                else None
            ),
            "duodenum_centerline_endpoint_match_tolerance_mm": (
                config.sampling.duodenum_centerline_endpoint_match_tolerance_mm
                if endpoint_hints is not None
                else None
            ),
            "seed": config.runtime.seed,
        },
        "duodenum_centerline_selection": duodenum_centerline_selection,
        "minimum_point_spacing_mm": config.sampling.minimum_spacing_mm,
        "pose_angles_degrees": {
            "roll": list(ROLL_ANGLES_DEGREES),
            "pitch": list(PITCH_ANGLES_DEGREES),
            "yaw": {name: list(values) for name, values in YAW_ANGLES_DEGREES.items()},
        },
        "square_sampling": {
            "side_length_mm": config.square.side_length_mm,
            "output_resolution": [config.ct.output_resolution, config.ct.output_resolution],
            "interpolation": "cubic_bspline",
            "interpolation_order": 3,
            "window_level_hu": config.ct.window_level,
            "window_width_hu": config.ct.window_width,
            "fill_hu_value": config.ct.fill_hu_value,
        },
        "quality_filtering": {
            "black_threshold": config.filtering.black_threshold,
            "black_ratio_limit": config.filtering.black_ratio_limit,
            "line_min_diagonal_fraction": config.filtering.line_min_diagonal_fraction,
            "black_side_min_ratio": config.filtering.black_side_min_ratio,
            "valid_side_max_black_ratio": config.filtering.valid_side_max_black_ratio,
        },
        "fov_policy": {
            "vertex_rule": "any_square_vertex_outside_ct",
            "outside_status": "excluded_fov",
            "saved_artifacts": ["ct_png"],
            "out_of_bounds_png_value": 0,
        },
        "eus_possible_organs": {
            "schema_version": eus_catalog.schema_version,
            "sha256": eus_catalog.sha256,
            "organ_labels": sorted(eus_catalog.labels),
            "excluded_organ_labels": sorted(EXCLUDED_ORGAN_LABELS),
            "geometry_sources": dict(EUS_ORGAN_GEOMETRY_SOURCES),
        },
    }
    canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    protocol["resume_protocol_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def _validate_completed_pose(
    sample: SquareSample,
    record: dict | None,
    *,
    compatible_build_git_commits: set[str] | None = None,
) -> None:
    if record is None:
        raise ValueError(f"姿态 {sample.sample_id} 有完成状态但缺少根清单记录")
    frame = frame_from_vertices(sample.vertices)
    expected = {
        "organ": sample.organ,
        "probe_point_world": [float(value) for value in sample.probe_point_world],
        "input_normal_world": [float(value) for value in sample.input_normal_world],
        "square_vertices_world": [
            [float(value) for value in vertex] for vertex in np.asarray(sample.vertices, dtype=np.float64)
        ],
        "width_mm": float(frame.width_mm),
        "length_mm": float(frame.length_mm),
        **_pose_metadata(sample),
    }
    mismatches = [
        field
        for field, value in expected.items()
        if field != "build_git_commit" and record.get(field) != value
    ]
    expected_commit = str(expected["build_git_commit"])
    actual_commit = record.get("build_git_commit")
    allowed_commits = {expected_commit} | (compatible_build_git_commits or set())
    if actual_commit not in allowed_commits:
        mismatches.append("build_git_commit")
    if mismatches:
        raise ValueError(
            f"姿态 {sample.sample_id} 的既有几何或位姿元数据不一致: {', '.join(mismatches)}；请使用新的输出目录"
        )


def _batched(values: Iterable[SquareSample], size: int) -> Iterable[tuple[SquareSample, ...]]:
    iterator = iter(values)
    while batch := tuple(islice(iterator, size)):
        yield batch


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


def _load_prepared_organs(config: CaseConfig) -> list[PreparedOrgan]:
    organs: list[PreparedOrgan] = []
    for label, model_id in ORGAN_BOUNDARY_MODEL_IDS.items():
        mesh = load_surface_mesh(
            config.organ_models[model_id],
            input_coordinate_system=config.geometry.input_coordinate_system,
        ).mesh
        organs.append(
            PreparedOrgan(model_id, label, DEFAULT_ORGAN_COLORS[label], mesh, mesh.bounds.copy())
        )
    return organs


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
        pose_metadata=_pose_metadata(sample),
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
        pose_metadata=_pose_metadata(sample),
    )


def _write_json_atomic(destination: Path, value: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _write_run_metadata(case_directory: Path, metadata: dict) -> None:
    _write_json_atomic(case_directory / "run_metadata.json", metadata)


def _metadata_with_state(
    metadata: dict[str, object],
    *,
    state: str,
    statuses: Counter[str],
    total_squares: int,
) -> dict[str, object]:
    if state not in {"running", "interrupted", "complete"}:
        raise ValueError(f"不支持的运行元数据状态: {state}")
    result = {
        **metadata,
        "run_state": state,
        "total_squares": total_squares,
        "completed_pose_count": sum(statuses.values()),
        "status_counts": dict(sorted(statuses.items())),
        "excluded_fov_count": statuses["excluded_fov"],
    }
    recovery_history = metadata.get("recovery_history")
    if recovery_history is not None:
        if not isinstance(recovery_history, list):
            raise ValueError("recovery_history 必须是列表")
        result["recovery_history"] = recovery_history
    return result


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


@dataclass(frozen=True)
class _PosePlan:
    surfaces: dict[str, SurfaceSamples]
    samples: PoseStream
    sampled_counts: dict[str, int]
    centerline_selection: dict[str, object] | None
    protocol_metadata: dict[str, object]


def _prepare_pose_plan(config: CaseConfig) -> _PosePlan:
    _preflight(config)
    surfaces = sample_organs(
        config.organ_models,
        config.sampling,
        config.runtime.seed,
        input_coordinate_system=config.geometry.input_coordinate_system,
    )
    duodenum_centerline = surfaces.get("duodenum", SurfaceSamples([], [])).centerline
    selection_audit = duodenum_centerline.selection_audit if duodenum_centerline is not None else None
    if config.sampling.duodenum_centerline_endpoint_hints_ras_mm is not None and selection_audit is None:
        raise ValueError("病例配置了十二指肠人工端点，但采样结果缺少人工中心线选择审计")
    centerline_selection = selection_audit.to_record() if selection_audit is not None else None
    samples = generate_square_samples(surfaces, config.square)
    sampled_counts = {organ: len(surfaces.get(organ, SurfaceSamples([], [])).points) for organ in ORGAN_ORDER}
    protocol_metadata = _run_protocol_metadata(config, centerline_selection)
    return _PosePlan(surfaces, samples, sampled_counts, centerline_selection, protocol_metadata)


def recover_interrupted_run_metadata(
    config: CaseConfig,
    *,
    expected_completed_count: int,
    reason: str,
    exit_code: int,
    recovered_at_utc: str | None = None,
) -> dict[str, object]:
    """严格校验已有姿态后，为信号中断的运行重建缺失元数据。"""

    if isinstance(expected_completed_count, bool) or not isinstance(expected_completed_count, int):
        raise ValueError("expected_completed_count 必须是整数")
    if expected_completed_count < 0:
        raise ValueError("expected_completed_count 不能为负数")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason 必须是非空字符串")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ValueError("exit_code 必须是整数")
    metadata_path = config.output_root / config.case_id / "run_metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"运行元数据已存在，拒绝覆盖: {metadata_path}")
    if recovered_at_utc is None:
        recovered_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    else:
        try:
            parsed_timestamp = datetime.fromisoformat(recovered_at_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("recovered_at_utc 必须是 ISO-8601 时间") from error
        if parsed_timestamp.tzinfo is None:
            raise ValueError("recovered_at_utc 必须包含时区")

    plan = _prepare_pose_plan(config)
    writer = GalleryWriter(
        config.output_root / config.case_id,
        config.case_id,
        required_core_design_sha256=CORE_DESIGN_SHA256,
        repair_missing_state_records=False,
    )
    statuses = Counter(writer.completed_statuses.values())
    if sum(statuses.values()) != expected_completed_count:
        raise ValueError(
            f"已完成姿态数 {sum(statuses.values())} 与预期 {expected_completed_count} 不一致"
        )
    completed_build_commits = {
        str(record.get("build_git_commit"))
        for record in writer.completed_records.values()
        if isinstance(record.get("build_git_commit"), str)
    }
    expected_pose_ids: set[str] = set()
    for sample in plan.samples:
        if sample.sample_id in expected_pose_ids:
            raise ValueError(f"当前姿态流包含重复 ID: {sample.sample_id}")
        expected_pose_ids.add(sample.sample_id)
        completed = writer.completed_status(sample.sample_id)
        if completed is not None:
            _validate_completed_pose(
                sample,
                writer.completed_record(sample.sample_id),
                compatible_build_git_commits=completed_build_commits,
            )
    stale_pose_ids = set(writer.completed_statuses) - expected_pose_ids
    if stale_pose_ids:
        preview = ", ".join(sorted(stale_pose_ids)[:10])
        raise ValueError(f"既有清单包含当前姿态集合之外的陈旧 ID: {preview}")
    recovery_record = {
        "reason": reason.strip(),
        "exit_code": exit_code,
        "recovered_at_utc": recovered_at_utc,
        "completed_pose_count": expected_completed_count,
        "status_counts": dict(sorted(statuses.items())),
        "completed_build_git_commits": sorted(completed_build_commits),
        "recovery_build_git_commit": plan.protocol_metadata["build_git_commit"],
    }
    completed_backends = {
        str(record.get("resampling_backend"))
        for record in writer.completed_records.values()
        if isinstance(record.get("resampling_backend"), str)
    }
    metadata = {
        **plan.protocol_metadata,
        "case_id": config.case_id,
        "requested_backend": config.runtime.backend,
        "selected_backend": completed_backends.pop() if len(completed_backends) == 1 else "pending_resume",
        "gpu_device": config.runtime.gpu_device,
        "gpu_batch_size": config.runtime.gpu_batch_size,
        "calibration": None,
        "compatible_completed_build_git_commits": sorted(completed_build_commits),
        "recovery_history": [recovery_record],
    }
    metadata = _metadata_with_state(
        metadata,
        state="interrupted",
        statuses=statuses,
        total_squares=len(plan.samples),
    )
    _write_run_metadata(config.output_root / config.case_id, metadata)
    return metadata


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
    plan = _prepare_pose_plan(config)
    surfaces = plan.surfaces
    samples = plan.samples
    sampled_counts = plan.sampled_counts
    protocol_metadata = plan.protocol_metadata
    if dry_run:
        return RunSummary(config.case_id, sampled_counts, len(samples), {}, True)

    case_directory = config.output_root / config.case_id
    has_run_artifacts = case_directory.exists() and any(path.name != "logs" for path in case_directory.iterdir())
    if has_run_artifacts and not resume and selected & {"sample", "square", "render"}:
        raise FileExistsError(f"输出目录已有内容，请启用恢复模式: {case_directory}")
    statuses: Counter[str] = Counter()
    writer: GalleryWriter | None = None
    previous_metadata: dict[str, object] | None = None
    compatible_build_git_commits: set[str] = set()
    calibration_samples: list[SquareSample] = []
    if "render" in selected:
        writer = GalleryWriter(
            case_directory,
            config.case_id,
            required_core_design_sha256=CORE_DESIGN_SHA256,
        )
        if writer.completed_statuses:
            metadata_path = case_directory / "run_metadata.json"
            if not metadata_path.is_file():
                raise ValueError("已有完成姿态但缺少 run_metadata.json，无法验证恢复运行协议")
            try:
                previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                raise ValueError(f"无法读取既有运行元数据: {error}") from error
            if previous_metadata.get("resume_protocol_sha256") != protocol_metadata["resume_protocol_sha256"]:
                raise ValueError("当前配置、设计或构建与既有结果的运行协议不一致；请使用新的输出目录")
            raw_compatible_commits = previous_metadata.get("compatible_completed_build_git_commits", [])
            if not isinstance(raw_compatible_commits, list) or any(
                not isinstance(commit, str) or len(commit) != 40 for commit in raw_compatible_commits
            ):
                raise ValueError("既有恢复元数据的兼容构建提交列表无效")
            compatible_build_git_commits = set(raw_compatible_commits)
        pending_count = 0
        expected_pose_ids: set[str] = set()
        for sample in samples:
            if sample.sample_id in expected_pose_ids:
                raise ValueError(f"当前姿态流包含重复 ID: {sample.sample_id}")
            expected_pose_ids.add(sample.sample_id)
            completed = writer.completed_status(sample.sample_id)
            if completed is None:
                pending_count += 1
                if len(calibration_samples) < 64:
                    calibration_samples.append(sample)
            else:
                _validate_completed_pose(
                    sample,
                    writer.completed_record(sample.sample_id),
                    compatible_build_git_commits=compatible_build_git_commits,
                )
                statuses[completed] += 1
        stale_pose_ids = set(writer.completed_statuses) - expected_pose_ids
        if stale_pose_ids:
            preview = ", ".join(sorted(stale_pose_ids)[:10])
            raise ValueError(f"既有清单包含当前姿态集合之外的陈旧 ID: {preview}；请使用新的输出目录")
        if pending_count == 0:
            if sum(statuses.values()) != len(samples):
                raise RuntimeError(f"姿态状态计数不完整: {sum(statuses.values())}/{len(samples)}")
            _write_run_metadata(
                case_directory,
                _metadata_with_state(
                    previous_metadata or protocol_metadata,
                    state="complete",
                    statuses=statuses,
                    total_squares=len(samples),
                ),
            )
            selected.remove("render")
    if "sample" in selected:
        for organ in ORGAN_ORDER:
            surface = surfaces.get(organ, SurfaceSamples([], []))
            write_surface_samples_ply(case_directory / "ResampledpointPLY" / f"FPS-{_legacy_name(organ)}.ply", surface)
    if "square" in selected:
        for organ in ORGAN_ORDER:
            organ_squares = (sample for sample in samples if sample.organ == organ)
            write_square_samples_ply(case_directory / "squarePLY" / f"{_legacy_name(organ)}-vertex.ply", organ_squares)
        write_rectangles_ply(
            case_directory / "rectangles.ply",
            (frame_from_vertices(sample.vertices) for sample in samples),
        )
    if "render" in selected:
        assert writer is not None
        volume = load_ct(
            config.ct_path,
            dicom_series_uid=config.dicom_series_uid,
            input_coordinate_system=config.geometry.input_coordinate_system,
        )
        vessels = [
            PreparedVessel(
                vessel.identifier,
                vessel.label,
                vessel.color,
                load_surface_mesh(vessel.path, input_coordinate_system=config.geometry.input_coordinate_system).mesh,
            )
            for vessel in config.vessel_models
        ]
        organs = _load_prepared_organs(config)
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
                **protocol_metadata,
                "calibration": None,
            }
        )
        if previous_metadata is not None and "recovery_history" in previous_metadata:
            backend_metadata["recovery_history"] = previous_metadata["recovery_history"]
        if previous_metadata is not None and "compatible_completed_build_git_commits" in previous_metadata:
            backend_metadata["compatible_completed_build_git_commits"] = previous_metadata[
                "compatible_completed_build_git_commits"
            ]
        if backend.name != "cpu" and calibration_samples:
            reference_backend = CachedCpuBackend(volume)
            try:
                calibration = validate_backend_against_cpu(
                    backend,
                    reference_backend,
                    np.asarray([sample.vertices for sample in calibration_samples], dtype=np.float64),
                    resolution=config.ct.output_resolution,
                    fill_hu_value=config.ct.fill_hu_value,
                    window_level=config.ct.window_level,
                    window_width=config.ct.window_width,
                )
            except Exception as error:
                calibration = {"sample_count": len(calibration_samples), "accepted": False, "error": str(error)}
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
        backend_metadata["selected_backend"] = backend.name
        _write_run_metadata(
            case_directory,
            _metadata_with_state(
                backend_metadata,
                state="running",
                statuses=statuses,
                total_squares=len(samples),
            ),
        )

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

        def pending_samples() -> Iterable[SquareSample]:
            for sample in samples:
                if writer.completed_status(sample.sample_id) is None:
                    yield sample

        try:
            if effective_workers == 1:
                executor = None
            else:
                executor = ThreadPoolExecutor(max_workers=effective_workers)
            try:
                if backend.name == "cpu":
                    cpu_batch_size = effective_workers * 4
                    for batch in _batched(pending_samples(), cpu_batch_size):
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
                else:
                    batch_size = config.runtime.gpu_batch_size
                    for batch in _batched(pending_samples(), batch_size):
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
            backend_metadata["completed_pose_count"] = sum(statuses.values())
            backend_metadata["status_counts"] = dict(sorted(statuses.items()))
            _write_run_metadata(
                case_directory,
                _metadata_with_state(
                    backend_metadata,
                    state="complete" if sum(statuses.values()) == len(samples) else "interrupted",
                    statuses=statuses,
                    total_squares=len(samples),
                ),
            )
            backend.close()
        if sum(statuses.values()) != len(samples):
            raise RuntimeError(
                f"姿态状态计数不完整: {sum(statuses.values())}/{len(samples)}"
            )
    indexed_feature_count = 0
    if "index" in selected:
        gallery_manifest = case_directory / "gallery" / "gallery.jsonl"
        label_counts: Counter[str] = Counter()
        organ_label_counts: Counter[str] = Counter()
        eus_candidate_organ_label_counts: Counter[str] = Counter()
        if gallery_manifest.is_file():
            from .registration_adapter import load_gallery_database

            database = load_gallery_database(gallery_manifest, config.registration_module_path)
            indexed_feature_count = len(database.features)
            for feature in database.features:
                label_counts.update(str(triplet.label) for triplet in feature.triplets)
            with gallery_manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        organ_label_counts.update(set(record.get("organ_labels", [])))
                        eus_candidate_organ_label_counts.update(
                            set(record.get("eus_candidate_organ_labels", []))
                        )
        eus_catalog = load_eus_organ_catalog()
        _write_json_atomic(
            case_directory / "library_summary.json",
            {
                "case_id": config.case_id,
                "gallery_manifest": "gallery/gallery.jsonl",
                "gallery_manifest_exists": gallery_manifest.is_file(),
                "indexed_feature_count": indexed_feature_count,
                "feature_label_counts": dict(sorted(label_counts.items())),
                "organ_label_counts": dict(sorted(organ_label_counts.items())),
                "eus_candidate_organ_label_counts": dict(
                    sorted(eus_candidate_organ_label_counts.items())
                ),
                "eus_possible_organs": eus_catalog.to_record(),
                "organ_boundary_colors": {
                    identifier: list(DEFAULT_ORGAN_COLORS[identifier]) for identifier in ORGAN_BOUNDARY_IDS
                },
            },
        )
    return RunSummary(config.case_id, sampled_counts, len(samples), dict(statuses), False, indexed_feature_count)
